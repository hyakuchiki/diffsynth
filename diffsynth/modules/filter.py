import torch
import torch.nn as nn
from diffsynth.processor import Processor
import diffsynth.util as util
import numpy as np
import math

class FIRFilter(Processor):
    """
    uses frequency sampling
    """
    
    def __init__(self, filter_size=64, name='firfilter', scale_fn=util.exp_sigmoid, initial_bias=-5.0):
        super().__init__(name)
        self.filter_size = filter_size
        self.scale_fn = scale_fn
        self.initial_bias = initial_bias

    def forward(self, audio, freq_response):
        """pass audio through FIRfilter
        Args:
            audio (torch.Tensor): [batch, n_samples]
            freq_response (torch.Tensor): frequency response (only magnitude) [batch, n_frames, filter_size // 2 + 1]

        Returns:
            [torch.Tensor]: Filtered audio. Shape [batch, n_samples]
        """
        if self.scale_fn is not None:
            freq_response = self.scale_fn(freq_response + self.initial_bias)
        return util.fir_filter(audio, freq_response, self.filter_size)
    
    def get_param_desc(self):
        return {
                'freq_response':    {'size': self.filter_size // 2 + 1, 'range': (1e-7, 2.0), 'type': 'exp_sigmoid'}, 
                'audio':            {'size':1, 'range': (-1, 1), 'type': 'raw'},
                }

# class SVFCell(nn.Module):
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, x, h_1, h_2, g, twoR, coeff_1, coeff_2):
#         # parameter [batch_size]
#         y_bp = coeff_2 * (x - h_2) + coeff_1 * h_1
#         y_lp = g * y_bp + h_2
#         y_hp = x - y_lp - twoR * y_bp
#         h_1 = 2 * y_bp - h_1
#         h_2 = 2 * y_lp - h_2
#         return y_bp, y_lp, y_hp, h_1, h_2

class SVFLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # self.cell = cell
    
    def forward(self, audio, g, twoR, mix):
        """pass audio through SVF
        Args:
            audio (torch.Tensor): [batch_size, n_samples]
            All filter parameters are [batch_size, n_samples, 1or3]
            g (torch.Tensor): Cutoff parameter
            twoR (torch.Tensor): Damping parameter
            mix (torch.Tensor): Mixing coefficient of bp, lp and hp

        Returns:
            [torch.Tensor]: Filtered audio. Shape [batch, n_samples]
        """
        batch_size, seq_len = audio.shape

        T = 1.0 / (1.0 + g * (g + twoR))
        H = T.unsqueeze(-1) * torch.cat([torch.ones_like(g), -g, g, twoR*g+1], dim=-1).reshape(batch_size, seq_len, 2, 2)

        # Y = gHBx + Hs
        gHB = g * T * torch.cat([torch.ones_like(g), g], dim=-1)
        # [batch_size, n_samples, 2]
        gHBx = gHB * audio.unsqueeze(-1)
        
        Y = torch.empty(batch_size, seq_len, 2, device=audio.device)
        # initialize filter state
        state = torch.ones(batch_size, 2, device=audio.device)
        for t in range(seq_len):
            Y[:, t] = gHBx[:, t] + torch.bmm(H[:, t], state.unsqueeze(-1)).squeeze(-1)
            state = 2 * Y[:, t] - state

        # HP = x - LP - 2R*BP
        y_hps = audio - twoR.squeeze(-1) * Y[:, :, 0] -  Y[:, :, 1]
        
        y_mixed = twoR.squeeze(-1) * mix[:, :, 0] * Y[:, :, 0] + mix[:, :, 1] * Y[:, :, 1] + mix[:, :, 2] * y_hps
        return y_mixed

class SVFilter(Processor):
    def __init__(self, name='svf'):
        super().__init__(name)
        self.svf = torch.jit.script(SVFLayer())

    def forward(self, audio, g, twoR, mix):
        batch_size, seq_len = audio.shape
        g = torch.clamp(g, min=1e-6, max=1)
        twoR = torch.clamp(twoR, min=1e-6, max=np.sqrt(2))
        if g.ndim == 2: # not time changing
            g[:, None, :].expand(-1, seq_len, -1)
        elif g.shape[1] != seq_len:
            g = util.resample_frames(g, seq_len)

        if twoR.ndim == 2: # not time changing
            twoR[:, None, :].expand(-1, seq_len, -1)
        elif twoR.shape[1] != seq_len:
            twoR = util.resample_frames(twoR, seq_len)

        # normalize mixing coefficient
        mix = mix / mix.sum(dim=-1, keepdim=True)
        if mix.ndim == 2: # not time changing
            mix[:, None, :].expand(-1, seq_len, -1)
        elif mix.shape[1] != seq_len:
            mix = util.resample_frames(mix, seq_len)

        filt_audio = self.svf(audio, g, twoR, mix)
        return filt_audio

    def get_param_desc(self):
        return {
                'audio':    {'size': 1, 'range': (-1, 1), 'type': 'sigmoid'},
                'g':        {'size': 1, 'range': (1e-6, 1), 'type': 'sigmoid'}, 
                'twoR':     {'size': 1, 'range': (1e-6, np.sqrt(2)), 'type': 'sigmoid'},
                'mix':      {'size': 3, 'range': (0, 1.0), 'type': 'sigmoid'}
                }