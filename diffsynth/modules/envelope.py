import numpy as np
import torch
from diffsynth.processor import Processor

def soft_clamp_min(x, min_v, T=100):
    return torch.sigmoid((min_v-x)*T)*(min_v-x)+x

class ADSREnvelope(Processor):
    def __init__(self, n_frames=250, name='env', low=0.0, high=1.0):
        super().__init__(name=name)
        self.n_frames = int(n_frames)
        self.param_names = ['total_level', 'attack', 'decay', 'sus_level', 'release']
        self.low = low
        self.high = high

    def forward(self, total_level, attack, decay, sus_level, release, note_off=0.8, n_frames=None):
        """generate envelopes from parameters

        Args:
            total_level (torch.Tensor): peak of the signal 0~1 (batch, 1, 1)
            attack (torch.Tensor): relative attack point 0~1 (batch, 1, 1)
            decay (torch.Tensor): actual decay point is attack+decay (batch, 1, 1)
            sus_level (torch.Tensor): sustain level 0~1 (batch, 1, 1)
            release (torch.Tensor): release point is attack+decay+release (batch, 1, 1)
            note_off (float or torch.Tensor, optional): note off position. Defaults to 0.8.
            n_frames (int, optional): number of frames. Defaults to None.

        Returns:
            torch.Tensor: envelope signal (batch_size, n_frames, 1)
        """

        batch_size = attack.shape[0]
        if n_frames is None:
            n_frames = self.n_frames
        # batch, n_frames, 1
        x = torch.linspace(0, 1.0, n_frames).repeat(batch_size, 1).unsqueeze(-1)
        x = x.to(attack.device)
        attack = attack * note_off
        A = x / (attack)
        A = torch.clamp(A, max=1.0)
        D = (x - attack) * (sus_level - 1) / decay
        D = torch.clamp(D, max=0.0)
        D = soft_clamp_min(D, sus_level-1)
        S = (x - note_off) * (-sus_level / release)
        S = torch.clamp(S, max=0.0)
        S = soft_clamp_min(S, -sus_level)
        signal = (total_level*(A+D+S))*(self.high-self.low) + self.low
        return torch.clamp(signal, min=self.low)

    def get_param_desc(self):
        return {
                'total_level':  {'size':1, 'range': (0, 1), 'type': 'sigmoid'}, 
                'attack':       {'size':1, 'range': (0, 1), 'type': 'sigmoid'},
                'decay':        {'size':1, 'range': (0, 1), 'type': 'sigmoid'},
                'sus_level':    {'size':1, 'range': (0, 1), 'type': 'sigmoid'},
                'release':      {'size':1, 'range': (0, 1), 'type': 'sigmoid'},
                'note_off':     {'size':1, 'range': (0, 1), 'type': 'sigmoid'},
                }