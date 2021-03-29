import torch
import torch.nn as nn
from diffsynth.processor import Gen
import diffsynth.util as util
import numpy as np
from diffsynth.util import midi_to_hz

class Additive(Gen):
    """Synthesize audio with a bank of harmonic sinusoidal oscillators.
    code mostly borrowed from DDSP"""

    def __init__(self, n_samples=64000, sample_rate=16000, scale_fn=util.exp_sigmoid, normalize_below_nyquist=True, name='harmonic', n_harmonics=64):
        super().__init__(name=name)
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.scale_fn = scale_fn
        self.normalize_below_nyquist = normalize_below_nyquist
        self.n_harmonics = n_harmonics

    def forward(self, amplitudes, harmonic_distribution, f0_hz, n_samples=None):
        """Synthesize audio with additive synthesizer from controls.

        Args:
        amplitudes: Amplitude tensor of shape [batch, n_frames, 1].
        harmonic_distribution: Tensor of shape [batch, n_frames, n_harmonics].
        f0_hz: The fundamental frequency in Hertz. Tensor of shape [batch,
            n_frames, 1].

        Returns:
        signal: A tensor of harmonic waves of shape [batch, n_samples].
        """
        if n_samples is None:
            n_samples = self.n_samples
        # Scale the amplitudes.
        if self.scale_fn is not None:
            amplitudes = self.scale_fn(amplitudes)
            harmonic_distribution = self.scale_fn(harmonic_distribution)
        if len(f0_hz.shape) < 3: # when given as a condition
            f0_hz = f0_hz[:, :, None]
        # Bandlimit the harmonic distribution.
        if self.normalize_below_nyquist:
            n_harmonics = int(harmonic_distribution.shape[-1])
            harmonic_frequencies = util.get_harmonic_frequencies(f0_hz, n_harmonics)
            harmonic_distribution = util.remove_above_nyquist(harmonic_frequencies, harmonic_distribution, self.sample_rate)

        # Normalize
        harmonic_distribution /= torch.sum(harmonic_distribution, axis=-1, keepdim=True)

        signal = util.harmonic_synthesis(frequencies=f0_hz, amplitudes=amplitudes, harmonic_distribution=harmonic_distribution, n_samples=n_samples, sample_rate=self.sample_rate)
        return signal
    
    def get_param_sizes(self):
        return {'amplitudes': 1, 'harmonic_distribution': self.n_harmonics, 'f0_hz': 1}
    
    def get_param_range(self):
        param_range = {}
        param_range['f0_hz'] = (0, self.sample_rate/2)
        if self.scale_fn is None:
            param_range['amplitudes'] = (0, 1)
            param_range['harmonic_distribution'] = (0, 1)
        else:
            param_range['amplitudes'] = (-np.inf, np.inf)
            param_range['harmonic_distribution'] = (-np.inf, np.inf)
        return param_range

class Sinusoids(Gen):
    def __init__(self, n_samples=64000, sample_rate=16000, amp_scale_fn=util.exp_sigmoid, freq_scale_fn=util.frequencies_sigmoid, name='sinusoids', n_sinusoids=64):
        super().__init__(name=name)
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.amp_scale_fn = amp_scale_fn
        self.freq_scale_fn = freq_scale_fn
        self.n_sinusoids = n_sinusoids

    def forward(self, amplitudes, frequencies, n_samples=None):
        """Synthesize audio with sinusoid oscillators

        Args:
        amplitudes: Amplitude tensor of shape [batch, n_frames, n_sinusoids].
        frequencies: Tensor of shape [batch, n_frames, n_sinusoids].

        Returns:
        signal: A tensor of harmonic waves of shape [batch, n_samples].
        """
        if n_samples is None:
            n_samples = self.n_samples
        # Scale the amplitudes.
        if self.amp_scale_fn is not None:
            amplitudes = self.amp_scale_fn(amplitudes)
        if self.freq_scale_fn is not None:
            frequencies = self.freq_scale_fn(frequencies)

        # resample to n_samples
        amplitudes_envelope = util.resample_frames(amplitudes, n_samples)
        frequency_envelope = util.resample_frames(frequencies, n_samples)

        signal = util.oscillator_bank(frequency_envelope, amplitudes_envelope, self.sample_rate)
        return signal
    
    def get_param_sizes(self):
        return {'amplitudes': self.n_sinusoids, 'frequencies': self.n_sinusoids}

    def get_param_range(self):
        param_range = {}
        param_range['f0_hz'] = (0, self.sample_rate/2)
        if self.amp_scale_fn is None:
            param_range['amplitudes'] = (0, 2) # 2?
        else:
            param_range['amplitudes'] = (-np.inf, np.inf)
        if self.freq_scale_fn is None:
            param_range['frequencies'] = (0, self.sample_rate/2)
        else:
            param_range['frequencies'] = (-np.inf, np.inf)
        return param_range

class FilteredNoise(Gen):
    """
    taken from ddsp-pytorch and ddsp
    uses frequency sampling
    """
    
    def __init__(self, filter_size=257, n_samples=64000, scale_fn=util.exp_sigmoid, name='noise', initial_bias=-5.0, amplitude=1.0):
        super().__init__(name=name)
        self.filter_size = filter_size
        self.n_samples = n_samples
        self.scale_fn = scale_fn
        self.initial_bias = initial_bias
        self.amplitude = amplitude

    def forward(self, freq_response, n_samples=None):
        """generate Gaussian white noise through FIRfilter
        Args:
            freq_response (torch.Tensor): frequency response (only magnitude) [batch, n_frames, filter_size // 2 + 1]

        Returns:
            [torch.Tensor]: Filtered audio. Shape [batch, n_samples]
        """
        if n_samples is None:
            n_samples = self.n_samples

        batch_size = freq_response.shape[0]
        if self.scale_fn:
            freq_response = self.scale_fn(freq_response + self.initial_bias)

        audio = (torch.rand(batch_size, n_samples)*2.0-1.0).to(freq_response.device) * self.amplitude
        filtered = util.fir_filter(audio, freq_response, self.filter_size)
        return filtered
    
    def get_param_sizes(self):
        return {'freq_response': self.filter_size // 2 + 1}

    def get_param_range(self):
        param_range = {}
        if self.scale_fn is None:
            param_range['freq_response'] = (0, 2) # 2?
        else:
            param_range['freq_response'] = (-np.inf, np.inf)
        return param_range

class Wavetable(Gen):
    """Synthesize audio from a wavetable (series of single cycle waveforms).
    wavetable is parameterized
    code mostly borrowed from DDSP
    """

    def __init__(self, len_waveform, n_samples=64000, sample_rate=16000, scale_fn=util.exp_sigmoid, name='wavetable'):
        super().__init__(name=name)
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.scale_fn = scale_fn
        self.len_waveform = len_waveform
    
    def forward(self, amplitudes, wavetable, f0_hz, n_samples=None):
        """forward pass

        Args:
            amplitudes: (batch_size, n_frames)
            wavetable ([type]): (batch_size, n_frames, len_waveform)
            f0_hz ([type]): frequency of oscillator at each frame (batch_size, n_frames)

        Returns:
            signal: synthesized signal ([batch_size, n_samples])
        """
        if n_samples is None:
            n_samples = self.n_samples
        if self.scale_fn is not None:
            amplitudes = self.scale_fn(amplitudes)
        
        signal = util.wavetable_synthesis(f0_hz, amplitudes, wavetable, n_samples, self.sample_rate)
        return signal

    def get_param_sizes(self):
        return {'amplitudes': 1, 'wavetable': self.len_waveform, 'f0_hz': 1}

    def get_param_range(self):
        param_range = {}
        param_range['f0_hz'] = (0, self.sample_rate/2)
        param_range['wavetable'] = (-1, 1)
        if self.scale_fn is None:
            param_range['amplitudes'] = (0, 1)
        else:
            param_range['amplitudes'] = (-np.inf, np.inf)
        return param_range

class SawOscillator(Gen):
    """Synthesize audio from a saw oscillator
    """

    def __init__(self, n_samples=64000, sample_rate=16000, scale_fn=util.exp_sigmoid, name='wavetable'):
        super().__init__(name=name)
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.scale_fn = scale_fn
        self.waveform = torch.linspace(1.0, -1.0, 2048) # saw waveform, will interpolate later anyways
    
    def forward(self, amplitudes, f0_hz, n_samples=None):
        """forward pass of saw oscillator

        Args:
            amplitudes: (batch_size, n_frames, 1)
            f0_hz: frequency of oscillator at each frame (batch_size, n_frames, 1)

        Returns:
            signal: synthesized signal ([batch_size, n_samples])
        """
        if n_samples is None:   
            n_samples = self.n_samples
        if self.scale_fn is not None:
            amplitudes = self.scale_fn(amplitudes)
        
        signal = util.wavetable_synthesis(f0_hz, amplitudes, self.waveform, n_samples, self.sample_rate)
        return signal

    def get_param_sizes(self):
        return {'amplitudes': 1, 'f0_hz': 1}

    def get_param_range(self):
        param_range = {}
        param_range['f0_hz'] = (0, self.sample_rate/2)
        if self.scale_fn is None:
            param_range['amplitudes'] = (0, 1)
        else:
            param_range['amplitudes'] = (-np.inf, np.inf)
        return param_range

class SineOscillator(Gen):
    """Synthesize audio from a saw oscillator
    """

    def __init__(self, n_samples=64000, sample_rate=16000, scale_fn=util.exp_sigmoid, name='sin'):
        super().__init__(name=name)
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.scale_fn = scale_fn
        self.waveform = torch.sin(torch.linspace(-np.pi, np.pi, 2048))
    
    def forward(self, amplitudes, f0_hz, n_samples=None):
        """forward pass of saw oscillator

        Args:
            amplitudes: (batch_size, n_frames, 1)
            f0_hz: frequency of oscillator at each frame (batch_size, n_frames, 1)

        Returns:
            signal: synthesized signal ([batch_size, n_samples])
        """
        if n_samples is None:   
            n_samples = self.n_samples
        if self.scale_fn is not None:
            amplitudes = self.scale_fn(amplitudes)
        
        signal = util.wavetable_synthesis(f0_hz, amplitudes, self.waveform, n_samples, self.sample_rate)
        return signal

    def get_param_sizes(self):
        return {'amplitudes': 1, 'f0_hz': 1}

    def get_param_range(self):
        param_range = {}
        param_range['f0_hz'] = (0, self.sample_rate/2)
        if self.scale_fn is None:
            param_range['amplitudes'] = (0, 1)
        else:
            param_range['amplitudes'] = (-np.inf, np.inf)
        return param_range

class SimpleFMOsc(Gen):
    """FM Synth with one carrier and one modulator both sine waves

    """
    def __init__(self, n_samples=64000, sample_rate=16000, amp_scale_fn=util.exp_sigmoid, freq_scale_fn=util.frequencies_sigmoid, name='fm'):
        super().__init__(name=name)
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.amp_scale_fn = amp_scale_fn
        self.freq_scale_fn = freq_scale_fn
        # 1, len_wavetable=2048
        self.register_buffer('waveform', torch.sin(torch.linspace(-np.pi, np.pi, 64).unsqueeze(0)))

    def forward(self, mod_amp, mod_freq, car_amp, car_freq, n_samples=None):
        if self.amp_scale_fn is not None:
            mod_amp = self.amp_scale_fn(mod_amp)
            car_amp = self.amp_scale_fn(car_amp)
        if self.freq_scale_fn is not None: # input is 0~1
            mod_freq = self.freq_scale_fn(mod_freq) #0-8.2Hz 1-12543Hz
            car_freq = self.freq_scale_fn(car_freq)
        if n_samples is None:   
            n_samples = self.n_samples
        
        waveform = self.waveform.expand(mod_amp.shape[0], -1)
        mod_signal = util.wavetable_synthesis(mod_freq, mod_amp, waveform, n_samples, self.sample_rate)
        car_signal = util.wavetable_synthesis(car_freq, car_amp, waveform, n_samples, self.sample_rate, mod_signal)
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
#TODO: wavetable scanner
