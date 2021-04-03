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
    
    def __init__(self, filter_size=64, scale_fn=util.exp_sigmoid, name='firfilter', initial_bias=-5.0):
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
    
    def get_param_sizes(self):
        return {'freq_response': self.filter_size // 2 + 1, 'audio': None}

class SVF(Processor):
    def __init__(self, name='svf'):
        super().__init__(name)

    def step_svf(self, x, h_1, h_2, g, twoR):
        # parameter [batch_size, 1]
        coeff_1 = 1.0 / (1.0 + g * (g + twoR))
        coeff_2 = g * coeff_1
        y_bp = coeff_2 * (x - h_2) + coeff_1 * h_1
        y_lp = g * y_bp + h_2
        y_hp = x - y_lp - twoR * y_bp
        h_1 = 2 * y_bp - h_1
        h_2 = 2 * y_lp - h_2
        return y_bp, y_lp, y_hp, h_1, h_2

    def forward(self, audio, g, twoR, mix):
        """pass audio through SVF
        Args:
            audio (torch.Tensor): [batch_size, n_samples]
            All filter parameters are [batch_size, (n_frames), 1]
            g (torch.Tensor): Cutoff parameter
            twoR (torch.Tensor): Damping parameter
            mix (torch.Tensor): Mixing coefficient of bp, lp and hp

        Returns:
            [torch.Tensor]: Filtered audio. Shape [batch, n_samples]
        """
        
        batch_size, seq_len = audio.shape
        if g.ndim == 2: # not time changing
            g[:, None, :].expand(-1, seq_len, -1)
        elif g.shape[1] != seq_len:
            g = util.resample_frames(g, seq_len)

        if twoR.ndim == 2: # not time changing
            twoR[:, None, :].expand(-1, seq_len, -1)
        elif twoR.shape[1] != seq_len:
            twoR = util.resample_frames(twoR, seq_len)

        if mix.ndim == 2: # not time changing
            mix[:, None, :].expand(-1, seq_len, -1)
        elif mix.shape[1] != seq_len:
            mix = util.resample_frames(mix, seq_len)

        # clamp
        g = torch.clamp(g, min=1e-8)
        twoR = torch.clamp(g, min=1e-8)

        # initial filter state
        h_1 = torch.zeros(batch_size, device=audio.device)
        h_2 = torch.zeros(batch_size, device=audio.device)
        y = torch.empty_like(audio)

        for t in range(seq_len):
            y_bp, y_lp, y_hp, h_1, h_2 = self.step_svf(audio[:, t], h_1, h_2, g[:, t, 0], twoR[:, t, 0])
            y[:, t] = mix[:, t, 0] * y_bp + mix[:, t, 1] * y_lp + mix[:, t, 2] * y_hp
        return y