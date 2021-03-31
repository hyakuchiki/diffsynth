import numpy as np
import torch
import torch.nn.functional as F
from diffsynth.processor import Processor

class FreqKnobsCoarse(Processor):
    # DX7 oscillator frequency knobs without fine
    
    def __init__(self, low_frq=30, high_frq=8000, name='frq'):
        super().__init__(name=name)
        # coarse: 0.5, 1, 2, 3, 4, ..., 31
        multipliers = torch.arange(0, 8, dtype=torch.float)
        multipliers[0] = 0.5
        self.register_buffer('multipliers', multipliers)
        self.low_frq = low_frq
        self.high_frq = high_frq

    def forward(self, base_freq, coarse, detune):
        # coarse - logits over multipliers
        one_hot = F.gumbel_softmax(coarse, tau=1, hard=True)
        coarse_value = (one_hot * self.multipliers).sum(dim=-1).unsqueeze(-1)
        frq = base_freq * coarse_value
        frq = (frq + (detune-0.5)*14) #Hz
        return frq

    def get_param_sizes(self):
        return {'base_freq': 1, 'coarse': 8, 'detune': 1}

    def get_param_range(self):
        return {'base_freq': (self.low_frq, self.high_frq), 'coarse': (0, np.inf), 'detune': (0, 1)}