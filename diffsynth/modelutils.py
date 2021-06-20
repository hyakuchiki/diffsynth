import numpy as np
import torch
from diffsynth.modules.generators import SineOscillator, SawOscillator
from diffsynth.processor import Add, Mix
from diffsynth.modules.fm import FM2, FM3
from diffsynth.modules.envelope import ADSREnvelope
from diffsynth.synthesizer import Synthesizer
from diffsynth.modules.frequency import FreqKnobsCoarse, FreqMultiplier
from diffsynth.modules.filter import SVFilter
from diffsynth.modules.harmor import Harmor
from diffsynth.modules.delay import ChorusFlanger

def construct_synths(name, n_samples=64000, sr=16000):
    static_params = []
    if name == 'fm2_fixed':
        fmosc = FM2(n_samples=n_samples, sample_rate=sr, name='fm2')
        dag = [
            (fmosc, {'mod_amp': 'envm', 'car_amp': 'envc', 'mod_freq': 'FRQ_M', 'car_freq': 'FRQ_C'})
        ]
        fixed_params = {'FRQ_M': torch.ones(1)*440, 'FRQ_C': torch.ones(1)*440}
        static_params=['FRQ_M', 'FRQ_C']
    elif name == 'fm2_free':
        fmosc = FM2(n_samples=n_samples, sample_rate=sr, name='fm2')
        dag = [
            (fmosc, {'mod_amp': 'AMP_M', 'car_amp': 'AMP_C', 'mod_freq': 'FRQ_M', 'car_freq': 'FRQ_C'})
        ]
        static_params=['FRQ_M', 'FRQ_C']
        fixed_params = {}
    elif name == 'fm2_half':
        fmosc = FM2(n_samples=n_samples, sample_rate=sr, name='fm2')
        dag = [
            (fmosc, {'mod_amp': 'AMP_M', 'car_amp': 'AMP_C', 'mod_freq': 'FRQ_M', 'car_freq': 'FRQ_C'})
        ]
        static_params=['FRQ_M']
        fixed_params = {'FRQ_C': torch.ones(1)*440}
    elif name == 'fm2_free_env':
        fmosc = FM2(n_samples=n_samples, sample_rate=sr, name='fm2')
        envm = ADSREnvelope(name='envm')
        envc = ADSREnvelope(name='envc')
        dag = [
            (envm, {'floor': 'AMP_FLOOR', 'peak': 'PEAK_M', 'attack': 'AT_M', 'decay': 'DE_M', 'sus_level': 'SU_M', 'release': 'RE_M', 'note_off': 'NO'}),
            (envc, {'floor': 'AMP_FLOOR', 'peak': 'PEAK_C', 'attack': 'AT_C', 'decay': 'DE_C', 'sus_level': 'SU_C', 'release': 'RE_C', 'note_off': 'NO'}),
            (fmosc, {'mod_amp': 'envm', 'car_amp': 'envc', 'mod_freq': 'FRQ_M', 'car_freq': 'FRQ_C'})
        ]
        fixed_params = {'AMP_FLOOR':torch.zeros(1), 'NO': torch.ones(1)*0.8}
        static_params=['FRQ_M', 'FRQ_C', 'PEAK_M', 'AT_M', 'DE_M', 'SU_M', 'RE_M', 'PEAK_C', 'AT_C', 'DE_C', 'SU_C', 'RE_C', 'AMP_FLOOR', 'NO']
    elif name == 'fm3_free':
        fmosc = FM3(n_samples=n_samples, sample_rate=sr, name='fm3')
        dag = [
            (fmosc, {'amp_1': 'AMP_1', 'amp_2': 'AMP_2', 'amp_3': 'AMP_3', 'freq_1': 'FRQ_1', 'freq_2': 'FRQ_2', 'freq_3': 'FRQ_3'})
        ]
        static_params=['FRQ_1', 'FRQ_2', 'FRQ_3']
        fixed_params = {}
    elif name == 'fm2x2_free':
        fm2_1 = FM2(n_samples=n_samples, sample_rate=sr, name='fm2_1')
        fm2_2 = FM2(n_samples=n_samples, sample_rate=sr, name='fm2_2')
        mix = Mix(name='add')
        dag = [
            (fm2_1, {'mod_amp': 'AMP_1', 'car_amp': 'AMP_2', 'mod_freq': 'FRQ_1', 'car_freq': 'FRQ_2'}),
            (fm2_2, {'mod_amp': 'AMP_3', 'car_amp': 'AMP_4', 'mod_freq': 'FRQ_3', 'car_freq': 'FRQ_4'}),
            (mix, {'signal_a': 'fm2_1', 'signal_b': 'fm2_2', 'mix_a': 'MIX_A', 'mix_b': 'MIX_B'})
        ]
        static_params=['FRQ_1', 'FRQ_2', 'FRQ_3', 'FRQ_4', 'MIX_A', 'MIX_B']
        fixed_params = {'MIX_A': torch.ones(1)*0.5, 'MIX_B': torch.ones(1)*0.5}
    elif name == 'fm6_free':
        fm3_1 = FM3(n_samples=n_samples, sample_rate=sr, name='fm3_1')
        fm3_2 = FM3(n_samples=n_samples, sample_rate=sr, name='fm3_2')
        add = Add(name='add')
        dag = [
            (fm3_1, {'amp_1': 'AMP_1', 'amp_2': 'AMP_2', 'amp_3': 'AMP_3', 'freq_1': 'FRQ_1', 'freq_2': 'FRQ_2', 'freq_3': 'FRQ_3'}),
            (fm3_2, {'amp_1': 'AMP_4', 'amp_2': 'AMP_5', 'amp_3': 'AMP_6', 'freq_1': 'FRQ_4', 'freq_2': 'FRQ_5', 'freq_3': 'FRQ_6'}),
            (add, {'signal_a': 'fm3_1', 'signal_b': 'fm3_2'})
        ]
        static_params=['FRQ_1', 'FRQ_2', 'FRQ_3', 'FRQ_4', 'FRQ_5', 'FRQ_6']
        fixed_params = {}
    elif name == 'sin':
        sin = SineOscillator(n_samples=n_samples, sample_rate=sr, name='sin')
        dag = [
            (sin, {'amplitudes': 'AMP','frequencies': 'FRQ'})
        ]
        static_params=['FRQ']
        fixed_params = {}
    elif name == 'harmor_fixed':
        harmor = Harmor(n_samples=n_samples, sample_rate=sr, name='harmor', n_oscs=1)
        dag = [
            (harmor, {'amplitudes': 'AMP', 'osc_mix': 'M_OSC', 'f0_hz': 'BFRQ', 'f0_mult': 'DUMMY', 'cutoff': 'CUTOFF', 'q': 'Q_FILT'})
        ]
        fixed_params = {'BFRQ': torch.ones(1)*440}
        static_params = ['M_OSC', 'Q_FILT', 'DUMMY']
    elif name == 'harmor_1oscfree':
        harmor = Harmor(n_samples=n_samples, sample_rate=sr, name='harmor', n_oscs=1)
        dag = [
            (harmor, {'amplitudes': 'AMP', 'osc_mix': 'M_OSC', 'f0_hz': 'BFRQ', 'f0_mult': 'DUMMY', 'cutoff': 'CUTOFF', 'q': 'Q_FILT'})
        ]
        fixed_params = {}
        static_params = ['M_OSC', 'Q_FILT', 'DUMMY', 'BFRQ']
    elif name == 'harmor_2oscfree':
        harmor = Harmor(n_samples=n_samples, sample_rate=sr, name='harmor', sep_amp=True, n_oscs=2)
        dag = [
            (harmor, {'amplitudes': 'AMP', 'osc_mix': 'M_OSC', 'f0_hz': 'BFRQ', 'f0_mult': 'MULT', 'cutoff': 'CUTOFF', 'q': 'Q_FILT'}),
            ]
        fixed_params = {}
        static_params=['BFRQ', 'M_OSC', 'MULT', 'Q_FILT']
    elif name == 'harmor_cf':
        harmor = Harmor(n_samples=n_samples, sample_rate=sr, name='harmor', sep_amp=True, n_oscs=2)
        cf = ChorusFlanger(name='cf', sr=sr)
        dag = [
            (harmor, {'amplitudes': 'AMP', 'osc_mix': 'M_OSC', 'f0_hz': 'BFRQ', 'f0_mult': 'MULT', 'cutoff': 'CUTOFF', 'q': 'Q_FILT'}),
            (cf, {'audio': 'harmor', 'delay_ms': 'CF_DELAY', 'rate': 'CF_RATE', 'depth': 'CF_DEPTH', 'mix': 'CF_MIX'})
            ]
        fixed_params = {}
        static_params=['BFRQ', 'M_OSC', 'MULT', 'Q_FILT', 'CF_DELAY', 'CF_RATE', 'CF_DEPTH', 'CF_MIX']
    synth = Synthesizer(dag, fixed_params=fixed_params, static_params=static_params)

    return synth

