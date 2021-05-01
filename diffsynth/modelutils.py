import numpy as np
import torch
from diffsynth.modules.generators import SineOscillator, SawOscillator
from diffsynth.modules.fm import FM2, FM3
from diffsynth.modules.envelope import ADSREnvelope
from diffsynth.synthesizer import Synthesizer
from diffsynth.modules.frequency import FreqKnobsCoarse, FreqMultiplier
from diffsynth.modules.filter import SVFilter
from diffsynth.modules.harmor import Harmor

def construct_synths(name):
    static_params = []
    if name == 'fm2_fixed':
        fmosc = FM2(n_samples=16000, name='fm2')
        dag = [
            (fmosc, {'mod_amp': 'envm', 'car_amp': 'envc', 'mod_freq': 'FRQ_M', 'car_freq': 'FRQ_C'})
        ]
        fixed_params = {'FRQ_M': torch.ones(1)*440, 'FRQ_C': torch.ones(1)*440}
    elif name == 'fm2_free':
        fmosc = FM2(n_samples=16000, name='fm2')
        dag = [
            (fmosc, {'mod_amp': 'AMP_M', 'car_amp': 'AMP_C', 'mod_freq': 'FRQ_M', 'car_freq': 'FRQ_C'})
        ]
        static_params=['FRQ_M', 'FRQ_C']
        fixed_params = {}
    elif name == 'fm3_free':
        fmosc = FM3(n_samples=16000, name='fm3')
        dag = [
            (fmosc, {'amp_1': 'AMP_1', 'amp_2': 'AMP_2', 'amp_3': 'AMP_3', 'freq_1': 'FRQ_1', 'freq_2': 'FRQ_2', 'freq_3': 'FRQ_3'})
        ]
        static_params=['FRQ_1', 'FRQ_2', 'FRQ_3']
        fixed_params = {}
    elif name == 'sin':
        sin = SineOscillator(n_samples=16000, name='sin')
        dag = [
            (sin, {'amplitudes': 'AMP','frequencies': 'FRQ'})
        ]
        static_params=['FRQ']
        fixed_params = {}
    elif name == 'harmor_fixed':
        harmor = Harmor(n_samples=16000, name='harmor', n_oscs=1)
        dag = [
            (harmor, {'amplitudes': 'AMP', 'osc_mix': 'M_OSC', 'f0_hz': 'BFRQ', 'f0_mult': 'DUMMY', 'cutoff': 'CUTOFF', 'q': 'Q_FILT'})
        ]
        fixed_params = {'BFRQ': torch.ones(1)*440}
        static_params = ['M_OSC', 'Q_FILT', 'DUMMY']
    elif name == 'harmor_free':
        harmor = Harmor(n_samples=16000, name='harmor', n_oscs=1)
        dag = [
            (harmor, {'amplitudes': 'AMP', 'osc_mix': 'M_OSC', 'f0_hz': 'BFRQ', 'f0_mult': 'DUMMY', 'cutoff': 'CUTOFF', 'q': 'Q_FILT'})
        ]
        fixed_params = {}
        static_params = ['M_OSC', 'Q_FILT', 'DUMMY', 'BFRQ']
    elif name == 'harmor_2oscfree':
        harmor = Harmor(n_samples=16000, name='harmor', sep_amp=False, n_oscs=2)
        dag = [
            (harmor, {'amplitudes': 'AMP', 'osc_mix': 'M_OSC', 'f0_hz': 'BFRQ', 'f0_mult': 'MULT', 'cutoff': 'CUTOFF', 'q': 'Q_FILT'}),
            ]
        fixed_params = {}
        static_params=['BFRQ', 'M_OSC', 'MULT', 'Q_FILT']
    synth = Synthesizer(dag, fixed_params=fixed_params, static_params=static_params)

    return synth

