import torch
from diffsynth.modules.synth import Processor

def soft_clamp_min(x, min_v, T=100):
    x = torch.sigmoid((min_v-x)*T)*(min_v-x)+x
    return x

class LFO(Processor):
    def __init__(self, n_secs=4, sample_rate=16000, name='lfo'):
        super().__init__(name=name)
        self.n_secs = n_secs
        self.sample_rate = sample_rate
        self.n_samples = int(n_secs*sample_rate)
    
    def forward(self, rate, level, n_samples=None):
        """

        Args:
            rate (torch.Tensor): in Hz (batch, 1)
            level (torch.Tensor): LFO level (batch, 1)
            n_samples (int, optional): number of samples to generate. Defaults to None.

        Returns:
            torch.Tensor: lfo signal (batch_size, n_samples)
        """
        if n_samples is None:
            n_samples = self.n_samples
        
        batch_size = rate.shape[0]
        phase_velocity = rate / self.sample_rate
        final_phase = phase_velocity * n_samples
        x = torch.linspace(0, 1, n_samples).repeat(batch_size, 1)
        phase = x * final_phase
        wave = level * torch.sin(phase)
        return wave

    def get_param_sizes(self):
        return {'rate': 1, 'level': 1}
    
    def get_param_range(self):
        return {'rate': (1, self.sample_rate/4), 'level': (0, 1)}