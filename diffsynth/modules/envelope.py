import torch
from diffsynth.modules.synth import Processor

def soft_clamp_min(x, min_v, T=100):
    x = torch.sigmoid((min_v-x)*T)*(min_v-x)+x
    return x

class ADSREnvelope(Processor):
    def __init__(self, n_secs=4, sample_rate=16000, name='env'):
        super().__init__(name=name)
        self.n_secs = n_secs
        self.sample_rate = sample_rate
        self.n_samples = int(n_secs*sample_rate)
    
    def forward(self, total_level, attack, decay, sus_level, release, note_off=0.8, n_samples=None):
        """generate envelopes from parameters

        Args:
            total_level (torch.Tensor): 0~1 factor multiplied over the entire signal (batch, 1)
            attack (torch.Tensor): relative attack point 0~1 (batch, 1)
            decay (torch.Tensor): actual decay point is attack+decay (batch, 1)
            sus_level (torch.Tensor): sustain level 0~1 (batch, 1)
            release (torch.Tensor): release point is attack+decay+release (batch, 1)
            note_off (float, optional): note off position. Defaults to 0.8.
            n_samples (int, optional): number of samples. Defaults to None.

        Returns:
            torch.Tensor: envelope signal (batch_size, n_samples)
        """ 
        batch_size = attack.shape[0]
        if n_samples is None:
            n_samples = self.n_samples
        x = torch.linspace(0, 1.0, n_samples).repeat(batch_size, 1)
        A = x / attack
        A = torch.clamp(A, max=1.0)
        D = (x-attack) * (sus_level-1) / decay
        D = torch.clamp(D, max=0.0)
        D = soft_clamp_min(D, sus_level-1)
        S = (x-note_off) * (-sus_level / release)
        S = torch.clamp(S, max=0.0)
        S = soft_clamp_min(S, -sus_level)
        return total_level*(A+D+S)

    def get_param_sizes(self):
        return {'total_level': 1, 'attack': 1, 'decay': 1, 'sus_level': 1, 'release': 1, 'note_off': 1}
    
    def get_param_range(self):
        return {'total_level': (0, 1), 'attack': (0, 1), 'decay': (0, 1), 'sus_level': (0, 1), 'release': (0, 1), 'note_off': (0, 1)}