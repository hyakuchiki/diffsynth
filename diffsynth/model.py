import torch
import torch.nn as nn
import torch.nn.functional as F
import diffsynth.util as util
from diffsynth.spectral import compute_lsd

class EstimatorSynth(nn.Module):
    """
    audio -> Estimator -> Synth -> audio
    """
    def __init__(self, estimator, synth):
        super().__init__()
        self.estimator = estimator
        self.synth = synth

    def param_loss(self, synth_output, param_dict):
        loss = 0
        for k, target in param_dict.items():
            output_name = self.synth.dag_summary[k]
            x = synth_output[output_name]
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

    def train_epoch(self, loader, recon_loss, optimizer, device, param_loss_w=0.0, enc_w=0.0, ae_model=None, clip=1.0):
        self.train()
        sum_loss = 0
        for data_dict in loader:
            params = data_dict.pop('params')
            params = {name:tensor.to(device, non_blocking=True) for name, tensor in params.items()}
            data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
            resyn_audio, outputs = self(data_dict)
            # Parameter loss
            param_loss = self.param_loss(outputs, params)
            # Reconstruction loss
            spec_loss, wave_loss = recon_loss(data_dict['audio'], resyn_audio)
            if enc_w>0.0:
                encoding_loss = enc_w*ae_model.encoding_loss(resyn_audio, data_dict['audio'])
            else:
                encoding_loss = 0
            batch_loss = spec_loss + wave_loss + param_loss_w * param_loss + encoding_loss
            # Perform backward
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
            optimizer.step()
            sum_loss += batch_loss.detach().item()
        sum_loss /= len(loader)
        return sum_loss

    def eval_epoch(self, loader, recon_loss, device, ae_model=None):
        self.eval()
        sum_spec_loss = 0
        sum_wave_loss = 0
        sum_param_loss = 0
        sum_lsd = 0
        sum_encoding_loss = 0
        with torch.no_grad():
            for data_dict in loader:
                params = data_dict.pop('params')
                params = {name:tensor.to(device, non_blocking=True) for name, tensor in params.items()}
                data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
                resyn_audio, outputs = self(data_dict)
                # Reconstruction loss
                param_loss = self.param_loss(outputs, params)
                # TODO: Use LSD or something instead of multiscale spec loss?
                sum_lsd += compute_lsd(data_dict['audio'], resyn_audio).item()
                spec_loss, wave_loss = recon_loss(data_dict['audio'], resyn_audio)
                sum_spec_loss += spec_loss.detach().item()
                sum_wave_loss += wave_loss.detach().item()
                sum_param_loss += param_loss.detach().item()
                if ae_model is not None:
                    sum_encoding_loss += ae_model.encoding_loss(resyn_audio, data_dict['audio']).detach().item()
        sum_spec_loss /= len(loader)
        sum_wave_loss /= len(loader)
        sum_param_loss /= len(loader)
        sum_lsd /= len(loader)
        sum_encoding_loss /= len(loader)
        return {'spec': sum_spec_loss, 'wave': sum_wave_loss, 'param': sum_param_loss, 'lsd': sum_lsd, 'enc': sum_encoding_loss}

class ParamEstimatorSynth(EstimatorSynth):
    """
    audio -> Estimator trained on parameter loss -> Synth -> audio
    """
    def __init__(self, estimator, synth):
        super().__init__(estimator, synth)

    def train_epoch(self, loader, recon_loss, optimizer, device, param_loss_w, enc_w=0.0, ae_model=None, clip=1.0):
        self.train()
        sum_loss = 0
        for data_dict in loader:
            params = data_dict.pop('params')
            params = {name:tensor.to(device, non_blocking=True) for name, tensor in params.items()}
            data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
            resyn_audio, outputs = self(data_dict)
            # Parameter loss
            batch_loss = param_loss_w * self.param_loss(outputs, params)
            # Perform backward
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
            optimizer.step()
            sum_loss += batch_loss.detach().item()
        sum_loss /= len(loader)
        return sum_loss

class NoParamEstimatorSynth(EstimatorSynth):
    """
    Ignore params
    """
    def __init__(self, estimator, synth):
        super().__init__(estimator, synth)

    def train_epoch(self, loader, recon_loss, optimizer, device, param_loss_w, clip=1.0):
        self.train()
        sum_loss = 0
        for data_dict in loader:
            params = data_dict.pop('params')
            data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
            resyn_audio, est_param = self(data_dict)
            # Reconstruction loss
            spec_loss, wave_loss = recon_loss(data_dict['audio'], resyn_audio)
            batch_loss = spec_loss + wave_loss
            # Perform backward
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
            optimizer.step()
            sum_loss += batch_loss.detach().item()
        sum_loss /= len(loader)
        return sum_loss
    
    def eval_epoch(self, loader, recon_loss, device):
        self.eval()
        sum_spec_loss = 0
        sum_wave_loss = 0
        with torch.no_grad():
            for data_dict in loader:
                params = data_dict.pop('params')
                data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
                resyn_audio, est_param = self(data_dict)
                # Reconstruction loss
                # TODO: Use LSD or something instead of multiscale spec loss?
                spec_loss, wave_loss = recon_loss(data_dict['audio'], resyn_audio)
                sum_spec_loss += spec_loss.detach().item()
                sum_wave_loss += wave_loss.detach().item()
        sum_spec_loss /= len(loader)
        sum_wave_loss /= len(loader)
        return {'spec': sum_spec_loss, 'wave': sum_wave_loss}