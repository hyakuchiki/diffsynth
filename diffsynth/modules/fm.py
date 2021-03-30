import torch
import torch.nn as nn
from diffsynth.processor import Gen
import diffsynth.util as util
import numpy as np

class FM2(Gen):
    """
    FM Synth with one carrier and one modulator both sine waves
    """
    def __init__(self, n_samples=64000, sample_rate=16000, amp_scale_fn=util.exp_sigmoid, freq_scale_fn=util.frequencies_sigmoid, max_mod_index=14, name='fm'):
        super().__init__(name=name)
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.amp_scale_fn = amp_scale_fn
        self.freq_scale_fn = freq_scale_fn
        self.mod_ratio = np.log(max_mod_index+1)


    def forward(self, mod_amp, mod_freq, car_amp, car_freq, n_samples=None):
        if self.amp_scale_fn is not None:
            car_amp = self.amp_scale_fn(car_amp)
        if self.freq_scale_fn is not None: # input is 0~1
            mod_freq = self.freq_scale_fn(mod_freq) 
            car_freq = self.freq_scale_fn(car_freq)
        if n_samples is None:   
            n_samples = self.n_samples
        
        # https://sound.stackexchange.com/questions/31709/what-is-the-level-of-frequency-modulation-of-many-synthesizers
        mod_amp = torch.exp(mod_amp**3.4*self.mod_ratio) - 1

        mod_signal = util.sin_synthesis(mod_freq, mod_amp, n_samples, self.sample_rate)
        car_signal = util.sin_synthesis(car_freq, car_amp, n_samples, self.sample_rate, mod_signal)
        return car_signal

    def get_param_sizes(self):
        return {'mod_amp': 1, 'car_amp': 1, 'mod_freq': 1, 'car_freq': 1}

    def get_param_range(self):
        param_range = {}
        if self.freq_scale_fn is None:
            param_range['mod_freq'] = (30, self.sample_rate/2)
            param_range['car_freq'] = (30, self.sample_rate/2)
        else:
            param_range['mod_freq'] = (-np.inf, np.inf)
            param_range['car_freq'] = (-np.inf, np.inf)
        if self.amp_scale_fn is None:
            param_range['mod_amp'] = (0, 1)
            param_range['car_amp'] = (0, 1)
        else:
            param_range['mod_amp'] = (-np.inf, np.inf)
            param_range['car_amp'] = (-np.inf, np.inf)
        return param_range

class FM3(Gen):
    """
    Osc1 -> Osc2 -> Osc3 -> output
    All sin waves
    """
    def __init__(self, n_samples=64000, sample_rate=16000, amp_scale_fn=util.exp_sigmoid, freq_scale_fn=util.frequencies_sigmoid, max_mod_index=14, name='fm'):
        super().__init__(name=name)
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.amp_scale_fn = amp_scale_fn
        self.freq_scale_fn = freq_scale_fn
        self.mod_ratio = np.log(max_mod_index+1)
        # 1, len_wavetable=2048
        self.register_buffer('waveform', torch.sin(torch.linspace(-np.pi, np.pi, 64).unsqueeze(0)))

    def forward(self, amp_1, freq_1, amp_2, freq_2, amp_3, freq_3, n_samples=None):
        if self.amp_scale_fn is not None:
            amp_3 = self.amp_scale_fn(amp_3)
        if self.freq_scale_fn is not None: # input is 0~1
            freq_1 = self.freq_scale_fn(freq_1) 
            freq_2 = self.freq_scale_fn(freq_2)
            freq_3 = self.freq_scale_fn(freq_3)
        if n_samples is None:   
            n_samples = self.n_samples
    
        # https://sound.stackexchange.com/questions/31709/what-is-the-level-of-frequency-modulation-of-many-synthesizers
        amp_1 = torch.exp(amp_1**3.4*self.mod_ratio) - 1
        amp_2 = torch.exp(amp_2**3.4*self.mod_ratio) - 1
        audio_1 = util.sin_synthesis(freq_1, amp_1, n_samples, self.sample_rate)
        audio_2 = util.sin_synthesis(freq_2, amp_2, n_samples, self.sample_rate, fm_signal=audio_1)
        audio_3 = util.sin_synthesis(freq_3, amp_3, n_samples, self.sample_rate, fm_signal=audio_2)
        return audio_3

    def get_param_sizes(self):
        return {'amp_1': 1, 'amp_2': 1, 'amp_3': 1, 'freq_1': 1, 'freq_2': 1, 'freq_3': 1}

    def get_param_range(self):
        param_range = {}
        if self.freq_scale_fn is None:
            param_range['freq_1'] = (30, self.sample_rate/2)
            param_range['freq_2'] = (30, self.sample_rate/2)
            param_range['freq_3'] = (30, self.sample_rate/2)
        else:
            param_range['freq_1'] = (-np.inf, np.inf)
            param_range['freq_2'] = (-np.inf, np.inf)
            param_range['freq_3'] = (-np.inf, np.inf)
        if self.amp_scale_fn is None:
            param_range['amp_1'] = (0, 1)
            param_range['amp_2'] = (0, 1)
            param_range['amp_3'] = (0, 1)
        else:
            param_range['amp_1'] = (-np.inf, np.inf)
            param_range['amp_2'] = (-np.inf, np.inf)        
            param_range['amp_3'] = (-np.inf, np.inf)        
        return param_range