import torch
import torch.nn as nn
from diffsynth.processor import Gen, FREQ_RANGE
import diffsynth.util as util
import numpy as np

def low_pass(freq, cutoff, q):
    ratio = freq/cutoff
    s = ratio*1j
    freq_response = abs(1/(s**2 + 1/q*s+1))
    return freq_response

class Harmor(Gen):
    """
    Subtractive synth-like additive synth
    Mixes 3 oscillators
    Each can be interpolated between saw <-> square
    Each has separate amplitude/frequencies or not (sep_f0s)
    Then a low-pass filter applied to all
    """

    def __init__(self, n_samples=16000, sample_rate=16000, name='harmor', n_harmonics=24, sep_amp=False, n_oscs=2):
        super().__init__(name=name)
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics
        self.sep_amp = sep_amp
        self.n_oscs = n_oscs

        # harmonic distribution of saw/sqr waves
        k = torch.arange(1, n_harmonics+1)
        saw_harm_dist = 2 / np.pi * (1/k)
        self.register_buffer('saw_harm_dist', saw_harm_dist)
        odd = torch.ones(n_harmonics)
        odd[1::2] = 0
        sqr_harm_dist = 4/np.pi * (1/k) * odd
        self.register_buffer('sqr_harm_dist', sqr_harm_dist)
        n_amps = self.n_oscs if self.sep_amp else 1
        self.param_desc = {
            'amplitudes':       {'size': n_amps, 'range': (0, 1),    'type': 'sigmoid'},
            'osc_mix':          {'size': self.n_oscs, 'range': (0, 1),    'type': 'sigmoid'}, 
            'f0_hz':            {'size': 1, 'range': FREQ_RANGE, 'type': 'freq_sigmoid'},
            'f0_mult':          {'size': self.n_oscs-1, 'range': (1, 8), 'type': 'sigmoid'},
            'cutoff':           {'size': 1, 'range': (30.0, self.sample_rate/2), 'type': 'freq_sigmoid'},
            'q':                {'size': 1, 'range': (0.0, 2.0), 'type': 'sigmoid'}
            }
        

    def forward(self, amplitudes, osc_mix, f0_hz, f0_mult, cutoff, q, n_samples=None):
        """Synthesize audio with additive synthesizer from controls.

        Args:
        amplitudes: Amplitudes tensor of shape. [batch, n_frames, self.n_oscs or 1]
        osc_mix: saw<->sqr mix. [batch, n_frames, self.n_oscs]
        f0_hz: f0 of each oscillators. [batch, n_frames, 1]
        f0_mult: f0 of each oscillators. [batch, n_frames or 1, self.n_oscs-1]
        cutoff: cutoff frequency in hz. [batch, n_frames, 1]
        q: resonance param 0~around 1.5 is ok. [batch, n_frames or 1, 1]

        Returns:
        signal: A tensor of harmonic waves of shape [batch, n_samples].
        """
        if n_samples is None:
            n_samples = self.n_samples

        batch, n_frames, _ = f0_hz.shape
        first_mult = torch.ones(batch, n_frames, 1).to(f0_hz.device)
        f0_mult = f0_mult.expand(-1, n_frames, -1)
        f0_mult = torch.cat([first_mult, f0_mult], dim=-1)
        f0_hz = f0_hz.expand(-1, -1, self.n_oscs)
        f0_hz = f0_hz * f0_mult

        if not self.sep_amp:
            amplitudes = amplitudes.expand(-1, -1, self.n_oscs)
        harm_dist = (1-osc_mix).unsqueeze(-1) * self.saw_harm_dist + osc_mix.unsqueeze(-1) * self.sqr_harm_dist
        audio = 0
        amps = []
        frqs = []
        for k in range(self.n_oscs):
            # create harmonic distributions for each oscs from osc_mix and f0
            harmonic_amplitudes = amplitudes[:, :, k:k+1] * harm_dist[:, :, k, :]
            harmonic_frequencies = util.get_harmonic_frequencies(f0_hz[:, :, k:k+1], self.n_harmonics)
            amps.append(harmonic_amplitudes)
            frqs.append(harmonic_frequencies)
        amplitude_envelopes = torch.cat(amps, dim=-1)
        frequency_envelopes = torch.cat(frqs, dim=-1)
        lowpass_multiplier = low_pass(frequency_envelopes, cutoff, q)
        filt_amplitude = lowpass_multiplier * amplitude_envelopes

        filt_amplitude = util.resample_frames(filt_amplitude, n_samples)
        frequency_envelopes = util.resample_frames(frequency_envelopes, n_samples)
        # TODO: Phaser?

        # removes sinusoids above nyquist freq.
        audio = util.oscillator_bank(frequency_envelopes, filt_amplitude, sample_rate=self.sample_rate)
        return audio