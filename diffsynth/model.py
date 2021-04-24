import torch
import torch.nn as nn
import torch.nn.functional as F

class EstimatorSynth(nn.Module):
    """
    audio -> Estimator -> Synth -> audio
    """
    def __init__(self, estimator, synth):
        super().__init__()
        self.estimator = estimator
        self.synth = synth

    def normalize_param(self, param_tensor):
        # min~max -> 0~1
        min_vec = self.range_vec[:, 0]
        max_vec = self.range_vec[:, 1]
        return (param_tensor - min_vec) / (max_vec - min_vec) 

    def scale_param(self, norm_tensor):
        # 0~1 -> min~max
        min_vec = self.range_vec[:, 0]
        max_vec = self.range_vec[:, 1]
        return norm_tensor * (max_vec - min_vec) + min_vec

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
        params_dict = self.synth.fill_params(est_param, conditioning, scaling=True)
        resyn_audio, outputs = self.synth(params_dict, audio_length)
        return resyn_audio, est_param

    def train_epoch(self, loader, recon_loss, optimizer, device, param_loss_w=0.0, clip=1.0):
        self.train()
        sum_loss = 0
        for data_dict in loader:
            data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
            resyn_audio, est_param = self(data_dict)
            # Parameter loss
            param_loss = F.l1_loss(est_param, data_dict['params'])
            # Reconstruction loss
            spec_loss, wave_loss = recon_loss(data_dict['audio'], resyn_audio)
            batch_loss = spec_loss + wave_loss + param_loss_w * param_loss
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
        sum_param_loss = 0
        with torch.no_grad():
            for data_dict in loader:
                data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
                resyn_audio, est_param = self(data_dict)
                # Reconstruction loss
                param_loss = F.l1_loss(est_param, data_dict['params'])
                # TODO: Use LSD or something instead of multiscale spec loss?
                spec_loss, wave_loss = recon_loss(data_dict['audio'], resyn_audio)
                sum_spec_loss += spec_loss.detach().item()
                sum_wave_loss += wave_loss.detach().item()
                sum_param_loss += param_loss.detach().item()
        sum_spec_loss /= len(loader)
        sum_wave_loss /= len(loader)
        sum_param_loss /= len(loader)
        return {'spec': sum_spec_loss, 'wave': sum_wave_loss, 'param': sum_param_loss}

class ParamEstimatorSynth(EstimatorSynth):
    """
    audio -> Estimator trained on parameter loss -> Synth -> audio
    """
    def __init__(self, estimator, synth):
        super().__init__(estimator, synth)

    def train_epoch(self, loader, recon_loss, optimizer, device, param_loss_w, clip=1.0):
        self.train()
        sum_loss = 0
        for data_dict in loader:
            data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
            est_param, _conditioning = self.estimate_param(data_dict)
            # Parameter loss
            batch_loss = F.mse_loss(est_param, data_dict['params'])
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