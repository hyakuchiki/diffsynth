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
from diffsynth.modules.delay import ModulatedDelay, ChorusFlanger
from diffsynth.modules.reverb import DecayReverb

def construct_synths(name, sr=16000):
    static_params = []
    if name == 'fm2_fixed':
        fmosc = FM2(sample_rate=sr, name='fm2')
        dag = [
            (fmosc, {'mod_amp': 'envm', 'car_amp': 'envc', 'mod_freq': 'FRQ_M', 'car_freq': 'FRQ_C'})
        ]
        fixed_params = {'FRQ_M': torch.ones(1)*440, 'FRQ_C': torch.ones(1)*440}
        static_params=['FRQ_M', 'FRQ_C']
    elif name == 'fm2_free':
        fmosc = FM2(sample_rate=sr, name='fm2')
        dag = [
            (fmosc, {'mod_amp': 'AMP_M', 'car_amp': 'AMP_C', 'mod_freq': 'FRQ_M', 'car_freq': 'FRQ_C'})
        ]
        static_params=['FRQ_M', 'FRQ_C']
        fixed_params = {}
    elif name == 'fm2_half':
        fmosc = FM2(sample_rate=sr, name='fm2')
        dag = [
            (fmosc, {'mod_amp': 'AMP_M', 'car_amp': 'AMP_C', 'mod_freq': 'FRQ_M', 'car_freq': 'FRQ_C'})
        ]
        static_params=['FRQ_M']
        fixed_params = {'FRQ_C': torch.ones(1)*440}
    elif name == 'fm2_free_env':
        fmosc = FM2(sample_rate=sr, name='fm2')
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
        fmosc = FM3(sample_rate=sr, name='fm3')
        dag = [
            (fmosc, {'amp_1': 'AMP_1', 'amp_2': 'AMP_2', 'amp_3': 'AMP_3', 'freq_1': 'FRQ_1', 'freq_2': 'FRQ_2', 'freq_3': 'FRQ_3'})
        ]
        static_params=['FRQ_1', 'FRQ_2', 'FRQ_3']
        fixed_params = {}
    elif name == 'fm2x2_free':
        fm2_1 = FM2(sample_rate=sr, name='fm2_1')
        fm2_2 = FM2(sample_rate=sr, name='fm2_2')
        mix = Mix(name='add')
        dag = [
            (fm2_1, {'mod_amp': 'AMP_1', 'car_amp': 'AMP_2', 'mod_freq': 'FRQ_1', 'car_freq': 'FRQ_2'}),
            (fm2_2, {'mod_amp': 'AMP_3', 'car_amp': 'AMP_4', 'mod_freq': 'FRQ_3', 'car_freq': 'FRQ_4'}),
            (mix, {'signal_a': 'fm2_1', 'signal_b': 'fm2_2', 'mix_a': 'MIX_A', 'mix_b': 'MIX_B'})
        ]
        static_params=['FRQ_1', 'FRQ_2', 'FRQ_3', 'FRQ_4', 'MIX_A', 'MIX_B']
        fixed_params = {'MIX_A': torch.ones(1)*0.5, 'MIX_B': torch.ones(1)*0.5}
    elif name == 'fm6_free':
        fm3_1 = FM3(sample_rate=sr, name='fm3_1')
        fm3_2 = FM3(sample_rate=sr, name='fm3_2')
        add = Add(name='add')
        dag = [
            (fm3_1, {'amp_1': 'AMP_1', 'amp_2': 'AMP_2', 'amp_3': 'AMP_3', 'freq_1': 'FRQ_1', 'freq_2': 'FRQ_2', 'freq_3': 'FRQ_3'}),
            (fm3_2, {'amp_1': 'AMP_4', 'amp_2': 'AMP_5', 'amp_3': 'AMP_6', 'freq_1': 'FRQ_4', 'freq_2': 'FRQ_5', 'freq_3': 'FRQ_6'}),
            (add, {'signal_a': 'fm3_1', 'signal_b': 'fm3_2'})
        ]
        static_params=['FRQ_1', 'FRQ_2', 'FRQ_3', 'FRQ_4', 'FRQ_5', 'FRQ_6']
        fixed_params = {}
    elif name == 'sin':
        sin = SineOscillator(sample_rate=sr, name='sin')
        dag = [
            (sin, {'amplitudes': 'AMP','frequencies': 'FRQ'})
        ]
        static_params=['FRQ']
        fixed_params = {}
    elif name == 'harmor_fixed':
        harmor = Harmor(sample_rate=sr, name='harmor', n_oscs=1)
        dag = [
            (harmor, {'amplitudes': 'AMP', 'osc_mix': 'M_OSC', 'f0_hz': 'BFRQ', 'f0_mult': 'DUMMY', 'cutoff': 'CUTOFF', 'q': 'Q_FILT'})
        ]
        fixed_params = {'BFRQ': torch.ones(1)*440}
        static_params = ['M_OSC', 'Q_FILT', 'BFRQ', 'DUMMY']
    elif name == 'harmor_1oscfree':
        harmor = Harmor(sample_rate=sr, name='harmor', n_oscs=1)
        dag = [
            (harmor, {'amplitudes': 'AMP', 'osc_mix': 'M_OSC', 'f0_hz': 'BFRQ', 'f0_mult': 'DUMMY', 'cutoff': 'CUTOFF', 'q': 'Q_FILT'})
        ]
        fixed_params = {}
        static_params = ['M_OSC', 'Q_FILT', 'DUMMY', 'BFRQ']
    elif name == 'harmor_2oscfree':
        harmor = Harmor(sample_rate=sr, name='harmor', sep_amp=True, n_oscs=2)
        dag = [
            (harmor, {'amplitudes': 'AMP', 'osc_mix': 'M_OSC', 'f0_hz': 'BFRQ', 'f0_mult': 'MULT', 'cutoff': 'CUTOFF', 'q': 'Q_FILT'}),
            ]
        fixed_params = {}
        static_params=['BFRQ', 'M_OSC', 'MULT', 'Q_FILT']
    elif name == 'harmor_cf':
        harmor = Harmor(sample_rate=sr, name='harmor', sep_amp=True, n_oscs=2)
        md = ModulatedDelay(name='md', sr=sr)
        dag = [
            (harmor, {'amplitudes': 'AMP', 'osc_mix': 'M_OSC', 'f0_hz': 'BFRQ', 'f0_mult': 'MULT', 'cutoff': 'CUTOFF', 'q': 'Q_FILT'}),
            (md, {'audio': 'harmor', 'delay_ms': 'MD_DELAY', 'phase': 'MD_PHASE', 'depth': 'MD_DEPTH', 'mix': 'MD_MIX'})
            ]
        fixed_params = {}
        static_params=['BFRQ', 'M_OSC', 'MULT', 'Q_FILT', 'MD_DELAY', 'MD_DEPTH', 'MD_MIX']
    elif name == 'harmor_cffixed':
        harmor = Harmor(sample_rate=sr, name='harmor', sep_amp=True, n_oscs=2)
        cf = ChorusFlanger(name='cf', sr=sr, delay_range=(1.0, 40.0))
        dag = [
            (harmor, {'amplitudes': 'AMP', 'osc_mix': 'M_OSC', 'f0_hz': 'BFRQ', 'f0_mult': 'MULT', 'cutoff': 'CUTOFF', 'q': 'Q_FILT'}),
            (cf, {'audio': 'harmor', 'delay_ms': 'CF_DELAY', 'rate': 'CF_RATE', 'depth': 'CF_DEPTH', 'mix': 'CF_MIX'})
            ]
        fixed_params = {'CF_RATE': torch.ones(1), 'CF_DEPTH': torch.ones(1)*0.1}
        static_params=['BFRQ', 'M_OSC', 'MULT', 'Q_FILT', 'CF_DELAY', 'CF_MIX', 'CF_RATE', 'CF_DEPTH']
    elif name == 'harmor_cffenv':
        harmor = Harmor(name='harmor', sep_amp=True, n_oscs=2)
        enva = ADSREnvelope(name='enva', max_value=0.6, channels=2)
        envc = ADSREnvelope(name='envc', channels=1)
        cf = ChorusFlanger(name='cf', sr=sr, delay_range=(1.0, 40.0))
        dag = [
            (enva, {'floor': 'AMP_FLOOR', 'peak': 'PEAK_A', 'attack': 'AT_A', 'decay': 'DE_A', 'sus_level': 'SU_A', 'release': 'RE_A', 'note_off': 'NO', 'noise_mag': 'NOISE_A'}),
            (envc, {'floor': 'CUT_FLOOR', 'peak': 'PEAK_C', 'attack': 'AT_C', 'decay': 'DE_C', 'sus_level': 'SU_C', 'release': 'RE_C', 'note_off': 'NO', 'noise_mag': 'NOISE_C'}),
            (harmor, {'amplitudes': 'enva', 'osc_mix': 'M_OSC', 'f0_hz': 'BFRQ', 'f0_mult': 'MULT', 'cutoff': 'envc', 'q': 'Q_FILT'}),
            (cf, {'audio': 'harmor', 'delay_ms': 'CF_DELAY', 'rate': 'CF_RATE', 'depth': 'CF_DEPTH', 'mix': 'CF_MIX'})
            ]
        fixed_params = {'AMP_FLOOR':torch.zeros(1), 'NO': torch.ones(1)*0.75, 'CF_RATE': torch.ones(1), 'CF_DEPTH': torch.ones(1)*0.1, 'NOISE_A':torch.zeros(1), 'NOISE_C':torch.zeros(1)}
        static_params=['AMP_FLOOR', 'PEAK_A', 'AT_A', 'DE_A', 'SU_A', 'RE_A', 'CUT_FLOOR', 'PEAK_C', 'AT_C', 'DE_C', 'SU_C', 'RE_C', 'BFRQ', 'MULT', 'M_OSC', 'Q_FILT', 'NOISE_A', 'NOISE_C', 'CF_DELAY', 'CF_MIX', 'NO', 'CF_DEPTH', 'CF_RATE']
    elif name == 'harmor_rev':
        harmor = Harmor(sample_rate=sr, name='harmor', sep_amp=True, n_oscs=2)
        reverb = DecayReverb(name='reverb', ir_length=16000)
        dag = [
            (harmor, {'amplitudes': 'AMP', 'osc_mix': 'M_OSC', 'f0_hz': 'BFRQ', 'f0_mult': 'MULT', 'cutoff': 'CUTOFF', 'q': 'Q_FILT'}),
            (reverb, {'audio': 'harmor', 'gain': 'RE_GAIN', 'decay': 'RE_DECAY'})
            ]
        fixed_params = {}
        static_params=['BFRQ', 'M_OSC', 'MULT', 'Q_FILT', 'RE_GAIN', 'RE_DECAY']
    synth = Synthesizer(dag, fixed_params=fixed_params, static_params=static_params)

    return synth

