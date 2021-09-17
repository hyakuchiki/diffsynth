import torch
import torch.nn as nn
from diffsynth.processor import Gen, FREQ_RANGE
import diffsynth.util as util
import numpy as np

class FM2(Gen):
    """
    FM Synth with one carrier and one modulator both sine waves
    """
    def __init__(self, sample_rate=16000, max_mod_index=14, name='fm2'):
        super().__init__(name=name)
        self.sample_rate = sample_rate
        self.mod_ratio = np.log(max_mod_index+1)
        self.param_desc = {
                'mod_amp':      {'size': 1, 'range': (0, 1), 'type': 'sigmoid'},
                'mod_freq':     {'size': 1, 'range': FREQ_RANGE, 'type': 'freq_sigmoid'}, 
                'car_amp':      {'size': 1, 'range': (0, 1), 'type': 'sigmoid'},
                'car_freq':     {'size': 1, 'range': FREQ_RANGE, 'type': 'freq_sigmoid'}
                }

    def forward(self, mod_amp, mod_freq, car_amp, car_freq, n_samples):
        # https://sound.stackexchange.com/questions/31709/what-is-the-level-of-frequency-modulation-of-many-synthesizers
        mod_amp = torch.exp(mod_amp**3.4*self.mod_ratio) - 1

        mod_signal = util.sin_synthesis(mod_freq, mod_amp, n_samples, self.sample_rate)
        car_signal = util.sin_synthesis(car_freq, car_amp, n_samples, self.sample_rate, mod_signal)
        return car_signal

class FM3(Gen):
    """
    Osc1 -> Osc2 -> Osc3 -> output
    All sin waves
    """
    def __init__(self, sample_rate=16000, max_mod_index=14, name='fm3'):
        super().__init__(name=name)
        self.sample_rate = sample_rate
        # self.mod_ratio = np.log(max_mod_index+1)
        self.max_mod_index = max_mod_index
        self.param_desc = {
                'amp_1':      {'size': 1, 'range': (0, 1), 'type': 'sigmoid'},
                'freq_1':     {'size': 1, 'range': FREQ_RANGE, 'type': 'freq_sigmoid'},
                'amp_2':      {'size': 1, 'range': (0, 1), 'type': 'sigmoid'},
                'freq_2':     {'size': 1, 'range': FREQ_RANGE, 'type': 'freq_sigmoid'}, 
                'amp_3':      {'size': 1, 'range': (0, 1), 'type': 'sigmoid'},
                'freq_3':     {'size': 1, 'range': FREQ_RANGE, 'type': 'freq_sigmoid'}
                }

    def forward(self, amp_1, freq_1, amp_2, freq_2, amp_3, freq_3, n_samples):
        audio_1 = util.sin_synthesis(freq_1, amp_1, n_samples, self.sample_rate)
        audio_2 = util.sin_synthesis(freq_2, amp_2, n_samples, self.sample_rate, fm_signal=audio_1 * self.max_mod_index)
        audio_3 = util.sin_synthesis(freq_3, amp_3, n_samples, self.sample_rate, fm_signal=audio_2 * self.max_mod_index)
        return audio_3