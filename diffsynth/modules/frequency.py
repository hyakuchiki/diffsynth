import numpy as np
import torch
import torch.nn.functional as F
from diffsynth.processor import Processor

class FreqMultiplier(Processor):
    
    def __init__(self, mult_low=0.5, mult_high=8, name='frq', mult_scale_fn=torch.sigmoid):
        super().__init__(name=name)
        # coarse: 0.5, 1, 2, 3, 4, ..., 31
        multipliers = torch.arange(0, 8, dtype=torch.float)
        multipliers[0] = 0.5
        self.register_buffer('multipliers', multipliers)
        self.mult_low = mult_low
        self.mult_high = mult_high
        self.mult_scale_fn = mult_scale_fn

    def forward(self, base_freq, mult):
        if self.mult_scale_fn is not None:
            mult = self.mult_scale_fn(mult)
        multiplier = mult * (self.mult_high - self.mult_low) + self.mult_low
        frq = base_freq * multiplier
        return frq

    def get_param_sizes(self):
        return {'base_freq': 1, 'mult': 1}

    def get_param_range(self):
        param_range = {'base_freq': (30, 8000), 'mult': (0, 1)}
        if self.mult_scale_fn is not None:
            param_range['mult'] = (-np.inf, np.inf)
        return param_range

class FreqKnobsCoarse(Processor):
    # DX7 oscillator frequency knobs without fine
    
    def __init__(self, low_frq=30, high_frq=8000, name='frq', coarse_scale_fn='gumbel', detune_scale_fn=torch.sigmoid):
        super().__init__(name=name)
        # coarse: 0.5, 1, 2, 3, 4, ..., 31
        multipliers = torch.arange(0, 8, dtype=torch.float)
        multipliers[0] = 0.5
        self.register_buffer('multipliers', multipliers)
        self.low_frq = low_frq
        self.high_frq = high_frq
        self.coarse_scale_fn = coarse_scale_fn
        self.detune_scale_fn = detune_scale_fn

    def forward(self, base_freq, coarse, detune):
        if self.coarse_scale_fn == 'gumbel':
            # coarse - logits over multipliers
            one_hot = F.gumbel_softmax(coarse, tau=1, hard=True)
            coarse_value = (one_hot * self.multipliers).sum(dim=-1)
        elif self.coarse_scale_fn is not None:
            coarse_value = self.coarse_scale_fn(coarse)
        else:
            coarse_value = torch.argmax(coarse, dim=-1)
        if self.detune_scale_fn is not None:
            detune = self.detune_scale_fn(detune)
        
        coarse_value = coarse_value.unsqueeze(-1)
        frq = base_freq * coarse_value
        frq = (frq + (detune-0.5)*14) #Hz
        return frq

    def get_param_sizes(self):
        return {'base_freq': 1, 'coarse': 8, 'detune': 1}

    def get_param_range(self):
        param_range = {'base_freq': (self.low_frq, self.high_frq), 'coarse': (-np.inf, np.inf), 'detune': (0, 1)}
        if self.coarse_scale_fn is not None:
            param_range['coarse'] = (-np.inf, np.inf)
        if self.detune_scale_fn is not None:
            param_range['detune'] = (-np.inf, np.inf)
        return param_range