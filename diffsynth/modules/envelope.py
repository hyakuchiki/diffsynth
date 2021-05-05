import numpy as np
import torch
from diffsynth.processor import Processor

def soft_clamp_min(x, min_v, T=100):
    return torch.sigmoid((min_v-x)*T)*(min_v-x)+x

class ADSREnvelope(Processor):
    def __init__(self, n_frames=250, name='env', min_value=0.0, max_value=1.0, channels=1):
        super().__init__(name=name)
        self.n_frames = int(n_frames)
        self.param_names = ['total_level', 'attack', 'decay', 'sus_level', 'release']
        self.min_value = min_value
        self.max_value = max_value
        self.channels = channels
        self.param_desc = {
                'floor':        {'size':self.channels, 'range': (0, 1), 'type': 'sigmoid'}, 
                'peak':         {'size':self.channels, 'range': (0, 1), 'type': 'sigmoid'}, 
                'attack':       {'size':self.channels, 'range': (0, 1), 'type': 'sigmoid'},
                'decay':        {'size':self.channels, 'range': (0, 1), 'type': 'sigmoid'},
                'sus_level':    {'size':self.channels, 'range': (0, 1), 'type': 'sigmoid'},
                'release':      {'size':self.channels, 'range': (0, 1), 'type': 'sigmoid'},
                'noise_mag':    {'size':self.channels, 'range': (0, 0.1), 'type': 'sigmoid'},
                'note_off':     {'size':self.channels, 'range': (0, 1), 'type': 'sigmoid'},
                }

    def forward(self, floor, peak, attack, decay, sus_level, release, noise_mag=0.0, note_off=0.8, n_frames=None):
        """generate envelopes from parameters

        Args:
            floor (torch.Tensor): floor level of the signal 0~1, 0=min_value (batch, 1, channels)
            peak (torch.Tensor): peak level of the signal 0~1, 1=max_value (batch, 1, channels)
            attack (torch.Tensor): relative attack point 0~1 (batch, 1, channels)
            decay (torch.Tensor): actual decay point is attack+decay (batch, 1, channels)
            sus_level (torch.Tensor): sustain level 0~1 (batch, 1, channels)
            release (torch.Tensor): release point is attack+decay+release (batch, 1, channels)
            note_off (float or torch.Tensor, optional): note off position. Defaults to 0.8.
            n_frames (int, optional): number of frames. Defaults to None.

        Returns:
            torch.Tensor: envelope signal (batch_size, n_frames, 1)
        """
        torch.clamp(floor, min=0, max=1)
        torch.clamp(peak, min=0, max=1)
        torch.clamp(attack, min=0, max=1)
        torch.clamp(decay, min=0, max=1)
        torch.clamp(sus_level, min=0, max=1)
        torch.clamp(release, min=0, max=1)

        batch_size = attack.shape[0]
        if n_frames is None:
            n_frames = self.n_frames
        # batch, n_frames, 1
        x = torch.linspace(0, 1.0, n_frames)[None, :, None].repeat(batch_size, 1, self.channels)
        x = x.to(attack.device)
        attack = attack * note_off
        A = x / (attack)
        A = torch.clamp(A, max=1.0)
        D = (x - attack) * (sus_level - 1) / (decay+1e-5)
        D = torch.clamp(D, max=0.0)
        D = soft_clamp_min(D, sus_level-1)
        S = (x - note_off) * (-sus_level / (release+1e-5))
        S = torch.clamp(S, max=0.0)
        S = soft_clamp_min(S, -sus_level)
        peak = peak * self.max_value + (1 - peak) * self.min_value
        floor = floor * self.max_value + (1 - floor) * self.min_value
        signal = (A + D + S + torch.randn_like(A)*noise_mag)*(peak - floor) + floor
        return torch.clamp(signal, min=self.min_value, max=self.max_value)