import numpy as np
import torch
from diffsynth.processor import Processor

def soft_clamp_min(x, min_v, T=100):
    return torch.sigmoid((min_v-x)*T)*(min_v-x)+x

class ADSREnvelope(Processor):
    def __init__(self, n_frames=250, name='env', param_scale_fn=torch.sigmoid):
        super().__init__(name=name)
        self.n_frames = int(n_frames)
        self.param_names = ['total_level', 'attack', 'decay', 'sus_level', 'release']
        self.param_scale_fn = param_scale_fn

    def forward(self, total_level, attack, decay, sus_level, release, note_off=0.8, n_frames=None):
        """generate envelopes from parameters

        Args:
            total_level (torch.Tensor): 0~1 factor multiplied over the entire signal (batch, 1, 1)
            attack (torch.Tensor): relative attack point 0~1 (batch, 1, 1)
            decay (torch.Tensor): actual decay point is attack+decay (batch, 1, 1)
            sus_level (torch.Tensor): sustain level 0~1 (batch, 1, 1)
            release (torch.Tensor): release point is attack+decay+release (batch, 1, 1)
            note_off (float, optional): note off position. Defaults to 0.8.
            n_frames (int, optional): number of frames. Defaults to None.

        Returns:
            torch.Tensor: envelope signal (batch_size, n_frames, 1)
        """
        if self.param_scale_fn is not None:
            total_level = self.param_scale_fn(total_level)
            attack = self.param_scale_fn(attack)
            decay = self.param_scale_fn(decay)
            sus_level = self.param_scale_fn(sus_level)
            release = self.param_scale_fn(release)

        batch_size = attack.shape[0]
        if n_frames is None:
            n_frames = self.n_frames
        # batch, n_frames, 1
        x = torch.linspace(0, 1.0, n_frames).repeat(batch_size, 1).unsqueeze(-1)
        x = x.to(attack.device)
        A = x / (attack*note_off)
        A = torch.clamp(A, max=1.0)
        D = (x - attack) * (sus_level - 1) / decay
        D = torch.clamp(D, max=0.0)
        D = soft_clamp_min(D, sus_level-1)
        S = (x - note_off) * (-sus_level / release)
        S = torch.clamp(S, max=0.0)
        S = soft_clamp_min(S, -sus_level)
        return torch.clamp(total_level*(A+D+S), min=0.0)

    def get_param_sizes(self):
        return {'total_level': 1, 'attack': 1, 'decay': 1, 'sus_level': 1, 'release': 1, 'note_off': 1}
    
    def get_param_range(self):
        if self.param_scale_fn is not None:
            param_range = {pn: (np.inf, -np.inf) for pn in self.param_names}
        else:
            param_range = {pn: (0, 1) for pn in self.param_names}
        param_range['note_off'] = (0,1)
        return param_range