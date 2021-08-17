import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import diffsynth.util as util
from diffsynth.spectral import compute_lsd, Mfcc, loudness_loss

class EstimatorSynth(nn.Module):
    """
    audio -> Estimator -> Synth -> audio
    """
    def __init__(self, estimator, synth):
        super().__init__()
        self.estimator = estimator
        self.synth = synth
        self.mfcc = Mfcc(n_fft=1024, hop_length=256, n_mels=40, n_mfcc=20, sample_rate=16000)
        self.param_grad_log = None

    def param_loss(self, synth_output, param_dict):
        loss = 0
        for k, target in param_dict.items():
            output_name = self.synth.dag_summary[k]
            x = synth_output[output_name]
            if target.shape[1] > 1:
                x = util.resample_frames(x, target.shape[1])
            loss += F.l1_loss(x, target)
        loss = loss / len(param_dict.keys())
        return loss

    def estimate_param(self, conditioning):
        """
        Args:
            conditioning (dict): {'PARAM NAME': Conditioning Tensor, ...}

        Returns:
            torch.Tensor: estimated parameters in Tensor ranged 0~1
        """
        conditioning = self.estimator(conditioning)
        conditioning['est_param'] = torch.sigmoid(conditioning['est_param'])
        return conditioning['est_param'], conditioning

    def log_param_grad(self, params_dict):
        def save_grad(name):
            def hook(grad):
                # batch, n_frames, feat_size
                grad_v = grad.abs().mean(dim=(0, 1))
                for i, gv in enumerate(grad_v):
                    self.param_grad_log.add_entry(name+'_{0}'.format(i), gv)
            return hook

        if self.param_grad_log is not None:
            for k, v in params_dict.items():
                if v.requires_grad == True:
                    v.register_hook(save_grad(k))

    def forward(self, conditioning):
        """
        Args:
            conditioning (dict): {'PARAM NAME': Conditioning Tensor, ...}

        Returns:
            torch.Tensor: audio
        """
        audio_length = conditioning['audio'].shape[1]
        est_param, conditioning = self.estimate_param(conditioning)
        params_dict = self.synth.fill_params(est_param, conditioning)
        if self.param_grad_log is not None:
            self.log_param_grad(params_dict)

        resyn_audio, outputs = self.synth(params_dict, audio_length)
        return resyn_audio, outputs

    def get_params(self, conditioning):
        """
        Don't render audio
        """
        audio_length = conditioning['audio'].shape[1]
        est_param, conditioning = self.estimate_param(conditioning)
        params_dict = self.synth.fill_params(est_param, conditioning)
        if self.param_grad_log is not None:
            self.log_param_grad(params_dict)
        
        synth_params = self.synth.calculate_params(params_dict, audio_length)
        return synth_params, conditioning

    def audio_losses(self, target_audio, resyn_audio, **kwargs):
        # always computes mean across batch dimension
        if 'perc_model' in kwargs:
            perc_model = kwargs['perc_model']
        else:
            perc_model = None
        sw_loss = kwargs['sw_loss']
        audio_loss={}
        if kwargs['sw_w'] > 0.0 and sw_loss is not None:
            # Reconstruction loss
            spec_loss, wave_loss = sw_loss(target_audio, resyn_audio)
            audio_loss['spec'], audio_loss['wave'] = kwargs['sw_w'] * spec_loss, kwargs['sw_w'] * wave_loss
        else:
            audio_loss['spec'], audio_loss['wave'] = (0, 0)
        if kwargs['perc_w'] > 0.0 and perc_model is not None:
            audio_loss['perc'] = kwargs['perc_w']*perc_model.perceptual_loss(target_audio, resyn_audio)
        else:
            audio_loss['perc'] = 0
        if kwargs['mfcc_w'] > 0.0:
            audio_loss['mfcc'] = kwargs['mfcc_w']*F.l1_loss(self.mfcc(target_audio), self.mfcc(resyn_audio))
        else:
            audio_loss['mfcc'] = 0
        if kwargs['lsd_w'] > 0.0: # only for eval
            audio_loss['lsd'] = kwargs['lsd_w']*compute_lsd(target_audio, resyn_audio)
        else:
            audio_loss['lsd'] = 0
        if kwargs['loud_w'] > 0.0:
            audio_loss['loud'] = kwargs['loud_w'] * loudness_loss(resyn_audio, target_audio)
        else:
            audio_loss['loud'] = 0
        return audio_loss
        
    def losses(self, target, output, **loss_args):
        #default values
        args = {'param_w': 1.0, 'sw_w':1.0, 'perc_w':1.0, 'mfcc_w':1.0, 'lsd_w': 1.0, 'loud_w': 1.0, 'sw_loss': None, 'perc_model': None}
        args.update(loss_args)
        if args['param_w'] > 0.0 and 'params' in target:
            param_loss = args['param_w'] * self.param_loss(output, target['params'])
        else:
            param_loss = 0.0
        loss = self.audio_losses(target['audio'], output['output'], **args)
        loss['param'] = param_loss
        return loss

    def train_epoch(self, loader, optimizer, device, loss_weights, sw_loss=None, perc_model=None, clip=1.0, log_grad=False):
        self.train()
        sum_loss = 0
        loss_args = loss_weights.copy()
        loss_args['perc_model'] = perc_model
        loss_args['sw_loss'] = sw_loss
        count = 0
        # reset log
        self.param_grad_log = util.StatsLog() if log_grad else None
        for data_dict in loader:
            # send data to device
            if 'params' in data_dict:
                params = data_dict.pop('params')
                params = {name:tensor.to(device, non_blocking=True) for name, tensor in params.items()}
                data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
                data_dict['params'] = params
            else:
                data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}

            if loss_args['sw_w']+loss_args['perc_w']+loss_args['mfcc_w']+loss_args['lsd_w'] == 0:
                # do not render audio because reconstruction is unnecessary
                synth_params, _conditioning = self.get_params(data_dict)
                # Parameter loss
                batch_loss = loss_args['param_w'] * self.param_loss(synth_params, data_dict['params'])
            else:
                # render audio
                resyn_audio, outputs = self(data_dict)
                # Parameter loss
                losses = self.losses(data_dict, outputs, **loss_args)
                batch_loss = sum(losses.values())
            # Perform backward
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
            optimizer.step()
            sum_loss += batch_loss.detach().item()
            count += 1
        params_grad = self.param_grad_log.average() if log_grad else None
        sum_loss /= count
        return sum_loss, params_grad

    def eval_epoch(self, syn_loader, real_loader, device, sw_loss=None, perc_model=None):
        self.eval()
        # in-domain
        syn_result = util.StatsLog()
        with torch.no_grad():
            for data_dict in syn_loader:
                params = data_dict.pop('params')
                params = {name:tensor.to(device, non_blocking=True) for name, tensor in params.items()}
                data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
                data_dict['params'] = params

                resyn_audio, outputs = self(data_dict)
                # Reconstruction loss
                losses = self.losses(data_dict, outputs, sw_loss=sw_loss, perc_model=perc_model)
                syn_result.update(losses)
        syn_result_dict = {'syn/'+k: v for k, v in syn_result.average().items()}

        # out-of-domain
        real_result = util.StatsLog()
        with torch.no_grad():
            for data_dict in real_loader:
                data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
                resyn_audio, outputs = self(data_dict)
                # Reconstruction loss
                losses = self.losses(data_dict, outputs, param_w=0, sw_loss=sw_loss, perc_model=perc_model)
                real_result.update(losses)
        real_result_dict = {'real/'+k: v for k, v in real_result.average().items()}

        result = {}
        result.update(syn_result_dict)
        result.update(real_result_dict)
        return result

# only for legacy
class NoParamEstimatorSynth(EstimatorSynth):
    def __init__(self, estimator, synth):
        super().__init__(estimator, synth)
