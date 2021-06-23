import numpy as np
import torch
from diffsynth.processor import Processor

def soft_clamp_min(x, min_v, T=100):
    x = torch.sigmoid((min_v-x)*T)*(min_v-x)+x
    return x

class LFO(Processor):
    def __init__(self, n_frames=250, n_secs=4, channels=1, rate_range=(1, 100), level_range=(0, 1), name='lfo'):
        super().__init__(name=name)
        self.n_secs = n_secs
        self.n_frames = n_frames
        self.channels = channels
        self.param_desc = {
            'rate':     {'size': self.channels, 'range': rate_range,  'type': 'sigmoid'},
            'level':    {'size': self.channels, 'range': level_range, 'type': 'sigmoid'}, 
            }
    
    def forward(self, rate, level, n_frames=None):
        """
        Args:
            rate (torch.Tensor): in Hz (batch, 1, self.channels)
            level (torch.Tensor): LFO level (batch, 1, self.channels)
            n_frames (int, optional): number of frames to generate. Defaults to None.

        Returns:
            torch.Tensor: lfo signal (batch_size, n_frames, self.channels)
        """
        if n_frames is None:
            n_frames = self.n_frames
        
        batch_size = rate.shape[0]
        final_phase = rate * self.n_secs * np.pi * 2
        x = torch.linspace(0, 1, n_frames, device=rate.device)[None, :, None].repeat(batch_size, 1, self.channels) # batch, n_frames, channels
        phase = x * final_phase
        wave = level * torch.sin(phase)
        return wave