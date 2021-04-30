import numpy as np
import torch
import torch.nn.functional as F
from diffsynth.processor import Processor

class FreqMultiplier(Processor):
    
    def __init__(self, mult_low=0.5, mult_high=8, name='frq'):
        super().__init__(name=name)
        self.mult_low = mult_low
        self.mult_high = mult_high
        self.param_desc = {
                'base_freq':{'size': 1, 'range': (32.7, 2093), 'type': 'freq_sigmoid'}, 
                'mult':     {'size': 1, 'range': (self.mult_low, self.mult_high), 'type': 'sigmoid'},
                }

    def forward(self, base_freq, mult):
        frq = base_freq * mult
        return frq

class FreqKnobsCoarse(Processor):
    # DX7 oscillator frequency knobs without fine
    
    def __init__(self, name='frq', coarse_scale_fn='gumbel'):
        super().__init__(name=name)
        # coarse: 0.5, 1, 2, 3, 4, ..., 31
        multipliers = torch.arange(0, 8, dtype=torch.float)
        multipliers[0] = 0.5
        self.register_buffer('multipliers', multipliers)
        self.coarse_scale_fn = coarse_scale_fn
        self.param_desc = {
                'base_freq':{'size': 1, 'range': (32.7, 2093), 'type': 'freq_sigmoid'}, 
                'coarse':   {'size': 8, 'range': (-np.inf, np.inf), 'type': 'raw'},
                'detune':   {'size': 8, 'range': (-7, 7), 'type': 'sigmoid'},
            }

    def forward(self, base_freq, coarse, detune):
        if self.coarse_scale_fn == 'gumbel':
            # coarse - logits over multipliers
            one_hot = F.gumbel_softmax(coarse, tau=1, hard=True, dim=-1)
            coarse_value = (one_hot * self.multipliers).sum(dim=-1)
        elif self.coarse_scale_fn is not None:
            coarse_value = self.coarse_scale_fn(coarse)
        else:
            one_hot = torch.argmax(coarse, dim=-1)
            coarse_value = (one_hot * self.multipliers).sum(dim=-1)
        
        coarse_value = coarse_value.unsqueeze(-1)
        frq = base_freq * coarse_value
        frq = (frq + detune) #Hz
        return frq