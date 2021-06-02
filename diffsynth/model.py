import torch
import torch.nn as nn
import torch.nn.functional as F
import diffsynth.util as util
from diffsynth.spectral import compute_lsd, Mfcc

class LossLog():
    def __init__(self):
        self.stats = dict()

    def __getitem__(self, key):
        return sum(self.stats[key]) / len(self.stats[key])

    def average(self):
        return {k: sum(v)/len(v) for k, v in self.stats.items()}

    def update(self, stat_dict):
        for k, v in stat_dict.items():
            # turn into python float
            val = v.item() if isinstance(v, torch.Tensor) else v
            if k in self.stats:
                self.stats[k].append(val)
            else:
                self.stats[k] = [val,]

class EstimatorSynth(nn.Module):
    """
    audio -> Estimator -> Synth -> audio
    """
    def __init__(self, estimator, synth):
        super().__init__()
        self.estimator = estimator
        self.synth = synth
        self.mfcc = Mfcc(n_fft=1024, hop_length=256, n_mels=40, n_mfcc=20)

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
        resyn_audio, outputs = self.synth(params_dict, audio_length)
        return resyn_audio, outputs

    def get_params(self, conditioning):
        """
        Don't render audio
        """
        audio_length = conditioning['audio'].shape[1]
        est_param, conditioning = self.estimate_param(conditioning)
        params_dict = self.synth.fill_params(est_param, conditioning)
        outputs = self.synth.calculate_params(params_dict, audio_length)
        return outputs

    def losses(self, target, output, recon_loss, ae_model=None, param_w=1.0, rec_mult=1.0, mfcc_w=1.0, lsd_w=1.0, enc_w=1.0):
        target_audio = target['audio']
        resyn_audio = output['output']
        if param_w > 0.0 and 'params' in target:
            param_loss = param_w * self.param_loss(output, target['params'])
        else:
            param_loss = 0.0
        if rec_mult>0.0:
            # Reconstruction loss
            spec_loss, wave_loss = recon_loss(target_audio, resyn_audio)
            spec_loss, wave_loss = rec_mult * spec_loss, rec_mult * wave_loss
        else:
            spec_loss, wave_loss = (0, 0)
        if mfcc_w > 0.0:
            mfcc_loss = mfcc_w*F.l1_loss(self.mfcc(target_audio), self.mfcc(resyn_audio))
        else:
            mfcc_loss = 0
        if lsd_w>0.0:
            lsd_loss = lsd_w*compute_lsd(target_audio, resyn_audio)
        else:
            lsd_loss = 0
        if enc_w>0.0 and ae_model is not None:
            encoding_loss = enc_w*ae_model.encoding_loss(target_audio, resyn_audio)
        else:
            encoding_loss = 0
        return {'param': param_loss, 'spec': spec_loss, 'wave': wave_loss, 'mfcc': mfcc_loss, 'lsd': lsd_loss, 'enc': encoding_loss}

    def train_epoch(self, loader, recon_loss, optimizer, device, rec_mult=1.0, param_w=0.0, mfcc_w=0.0, enc_w=0.0, ae_model=None, clip=1.0):
        self.train()
        sum_loss = 0
        for data_dict in loader:
            # send data to device
            params = data_dict.pop('params')
            params = {name:tensor.to(device, non_blocking=True) for name, tensor in params.items()}
            data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
            data_dict['params'] = params

            if rec_mult+enc_w+mfcc_w <= 0:
                # do not render audio because reconstruction is unnecessary
                outputs = self.get_params(data_dict)
                # Parameter loss
                batch_loss = param_w * self.param_loss(outputs, params)
            else:
                # render audio
                resyn_audio, outputs = self(data_dict)
                # Parameter loss
                losses = self.losses(data_dict, outputs, recon_loss, ae_model, rec_mult=rec_mult, mfcc_w=mfcc_w, lsd_w=0, enc_w=enc_w)
                batch_loss = sum(losses)
            # Perform backward
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
            optimizer.step()
            sum_loss += batch_loss.detach().item()
        sum_loss /= len(loader)
        return sum_loss

    def eval_epoch(self, syn_loader, real_loader, recon_loss, device, ae_model=None):
        self.eval()
        # in-domain
        syn_result = LossLog()
        with torch.no_grad():
            for data_dict in syn_loader:
                params = data_dict.pop('params')
                params = {name:tensor.to(device, non_blocking=True) for name, tensor in params.items()}
                data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
                data_dict['params'] = params

                resyn_audio, outputs = self(data_dict)
                # Reconstruction loss
                losses = self.losses(data_dict, outputs, recon_loss, ae_model)
                syn_result.update(losses)
        syn_result_dict = {'syn/'+k: v for k, v in syn_result.average().items()}

        # out-of-domain
        real_result = LossLog()
        with torch.no_grad():
            for data_dict in real_loader:
                data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
                resyn_audio, outputs = self(data_dict)
                # Reconstruction loss
                losses = self.losses(data_dict, outputs, recon_loss, ae_model, param_w=0)
                real_result.update(losses)
        real_result_dict = {'real/'+k: v for k, v in real_result.average().items()}

        result = {}
        result.update(syn_result_dict)
        result.update(real_result_dict)
        return result

class NoParamEstimatorSynth(EstimatorSynth):
    """
    Ignore params
    Training set has no params and is trained by spectral loss only
    """
    def __init__(self, estimator, synth):
        super().__init__(estimator, synth)

    def train_epoch(self, loader, recon_loss, optimizer, device, rec_mult=1.0, param_w=0.0, enc_w=0.0, mfcc_w=0.0, ae_model=None, clip=1.0):
        self.train()
        sum_loss = 0
        for data_dict in loader:
            if 'params' in data_dict:
                params = data_dict.pop('params')
            data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
            resyn_audio, outputs = self(data_dict)
            # Reconstruction loss
            losses = self.losses(data_dict, outputs, recon_loss, ae_model, rec_mult=rec_mult, mfcc_w=mfcc_w, lsd_w=0, enc_w=enc_w)
            batch_loss = sum(losses)
            # Perform backward
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
            optimizer.step()
            sum_loss += batch_loss.detach().item()
            
        sum_loss /= len(loader)
        return sum_loss