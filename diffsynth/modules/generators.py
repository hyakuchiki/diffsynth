import torch
import torch.nn as nn
from diffsynth.processor import Gen
import diffsynth.util as util
import numpy as np
from diffsynth.util import midi_to_hz

class Additive(Gen):
    """Synthesize audio with a bank of harmonic sinusoidal oscillators.
    code mostly borrowed from DDSP"""

    def __init__(self, n_samples=16000, sample_rate=16000, normalize_below_nyquist=True, name='harmonic', n_harmonics=64):
        super().__init__(name=name)
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.normalize_below_nyquist = normalize_below_nyquist
        self.n_harmonics = n_harmonics
        self.param_desc = {
                'amplitudes':               {'size': 1, 'range': (0, 1), 'type': 'exp_sigmoid'},
                'harmonic_distribution':    {'size': self.n_harmonics, 'range': (0, 1), 'type': 'exp_sigmoid'}, 
                'f0_hz':                    {'size': 1, 'range': (32.7, 2093), 'type': 'freq_sigmoid'}
                }

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

class Sinusoids(Gen):
    def __init__(self, n_samples=16000, sample_rate=16000, name='sinusoids', n_sinusoids=64):
        super().__init__(name=name)
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.n_sinusoids = n_sinusoids
        self.param_desc = {
                'amplitudes':   {'size': self.n_sinusoids, 'range': (0, 2), 'type': 'exp_sigmoid'},
                'frequencies':  {'size': self.n_sinusoids, 'range': (32.7, 2093), 'type': 'freq_sigmoid'}, 
                }

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

        # resample to n_samples
        amplitudes_envelope = util.resample_frames(amplitudes, n_samples)
        frequency_envelope = util.resample_frames(frequencies, n_samples)

        signal = util.oscillator_bank(frequency_envelope, amplitudes_envelope, self.sample_rate)
        return signal

class FilteredNoise(Gen):
    """
    taken from ddsp-pytorch and ddsp
    uses frequency sampling
    """
    
    def __init__(self, filter_size=257, n_samples=16000, scale_fn=util.exp_sigmoid, name='noise', initial_bias=-5.0, amplitude=1.0):
        super().__init__(name=name)
        self.filter_size = filter_size
        self.n_samples = n_samples
        self.scale_fn = scale_fn
        self.initial_bias = initial_bias
        self.amplitude = amplitude
        self.param_desc = {
                'freq_response':    {'size': self.filter_size // 2 + 1, 'range': (1e-7, 2.0), 'type': 'exp_sigmoid'}, 
                }

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

class Wavetable(Gen):
    """Synthesize audio from a wavetable (series of single cycle waveforms).
    wavetable is parameterized
    code mostly borrowed from DDSP
    """

    def __init__(self, len_waveform, n_samples=16000, sample_rate=16000, name='wavetable'):
        super().__init__(name=name)
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.len_waveform = len_waveform
        self.param_desc = {
                'amplitudes':   {'size': 1, 'range': (0, 1.0), 'type': 'sigmoid'}, 
                'wavetable':    {'size': self.len_waveform, 'range': (-1, 1), 'type': 'sigmoid'}, 
                'f0_hz':        {'size': 1, 'range': (32.7, 2093), 'type': 'freq_sigmoid'}, 
                }

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
        
        signal = util.wavetable_synthesis(f0_hz, amplitudes, wavetable, n_samples, self.sample_rate)
        return signal

class SawOscillator(Gen):
    """Synthesize audio from a saw oscillator
    """

    def __init__(self, n_samples=16000, sample_rate=16000, name='wavetable'):
        super().__init__(name=name)
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        # saw waveform
        waveform = torch.roll(torch.linspace(1.0, -1.0, 64), 32) # aliasing?
        self.register_buffer('waveform', waveform)
        self.param_desc = {
                'amplitudes':   {'size': 1, 'range': (0, 1.0), 'type': 'sigmoid'}, 
                'f0_hz':        {'size': 1, 'range': (32.7, 2093), 'type': 'freq_sigmoid'}, 
                }
    
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
        
        signal = util.wavetable_synthesis(f0_hz, amplitudes, self.waveform, n_samples, self.sample_rate)
        return signal

class SineOscillator(Gen):
    """Synthesize audio from a saw oscillator
    """

    def __init__(self, n_samples=16000, sample_rate=16000, name='sin'):
        super().__init__(name=name)
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.param_desc = {
                'amplitudes':   {'size': 1, 'range': (0, 1.0), 'type': 'sigmoid'}, 
                'frequencies':  {'size': 1, 'range': (32.7, 2093), 'type': 'freq_sigmoid'}, 
                }

    def forward(self, amplitudes, frequencies, n_samples=None):
        """forward pass of saw oscillator

        Args:
            amplitudes: (batch_size, n_frames, 1)
            f0_hz: frequency of oscillator at each frame (batch_size, n_frames, 1)

        Returns:
            signal: synthesized signal ([batch_size, n_samples])
        """
        if n_samples is None:   
            n_samples = self.n_samples

        signal = util.sin_synthesis(frequencies, amplitudes, n_samples, self.sample_rate)
        return signal
                
#TODO: wavetable scanner
