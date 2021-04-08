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

    def forward(self, conditioning):
        """
        Args:
            conditioning (dict): {'PARAM NAME': Conditioning Tensor, ...}

        Returns:
            torch.Tensor: audio
        """
        audio_length = conditioning['audio'].shape[1]
        conditioning = self.estimator(conditioning)
        est_param = conditioning['est_param']
        params = self.synth.fill_params(est_param, conditioning)
        resyn_audio, outputs = self.synth(params, audio_length)
        return resyn_audio

    def train_epoch(self, loader, recon_loss, optimizer, device, clip=1.0):
        self.train()
        sum_loss = 0
        for data_dict in loader:
            data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
            resyn_audio = self(data_dict)
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
        sum_loss = 0
        with torch.no_grad():
            for data_dict in loader:
                data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
                resyn_audio = self(data_dict)
                # Reconstruction loss
                spec_loss, wave_loss = recon_loss(data_dict['audio'], resyn_audio)
                batch_loss = spec_loss + wave_loss
                sum_loss += batch_loss.detach().item()
        sum_loss /= len(loader)
        return sum_loss

class ParamEstimatorSynth(nn.Module):
    """
    audio -> Estimator trained on parameter loss -> Synth -> audio
    """
    def __init__(self, estimator, synth):
        super().__init__()
        self.estimator = estimator
        self.synth = synth

        range_vec = []
        for pn, psize in self.synth.ext_param_sizes.items():
            prange = self.synth.ext_param_range[pn]
            prange = torch.tensor(prange).expand(psize, -1)
            range_vec.append(prange)
        range_vec = torch.cat(range_vec, dim=0)
        self.register_buffer('range_vec', range_vec)

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
            torch.Tensor: estimated parameters
        """
        conditioning = self.estimator(conditioning)
        # output is aimed to be 0~1 but not necessary bounded or scaled
        est_param = conditioning['est_param']
        return est_param, conditioning 
    
    def forward(self, conditioning):
        """
        Args:
            conditioning (dict): {'PARAM NAME': Conditioning Tensor, ...}

        Returns:
            torch.Tensor: audio
        """
        audio_length = conditioning['audio'].shape[1]
        est_param, conditioning = self.estimate_param(conditioning)
        scaled_param = self.scale_param(est_param)
        params = self.synth.fill_params(scaled_param, conditioning, scaling=False)
        resyn_audio, outputs = self.synth(params, audio_length)
        return resyn_audio

    def train_epoch(self, loader, recon_loss, optimizer, device, clip=1.0):
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

    def eval_epoch(self, loader, recon_loss, device):
        self.eval()
        sum_loss = 0
        with torch.no_grad():
            for data_dict in loader:
                data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
                resyn_audio = self(data_dict)
                # Reconstruction loss
                spec_loss, wave_loss = recon_loss(data_dict['audio'], resyn_audio)
                batch_loss = spec_loss + wave_loss
                sum_loss += batch_loss.detach().item()
        sum_loss /= len(loader)
        return sum_loss