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
    if name == 'fixedfm2':
        fmosc = FM2(n_samples=16000)
        envm = ADSREnvelope(name='envm')
        envc = ADSREnvelope(name='envc')
        dag = [
            (envm, {'total_level': 'TL_M', 'attack': 'AT_M', 'decay': 'DE_M', 'sus_level': 'SU_M', 'release': 'RE_M', 'note_off': 'NO_M'}),
            (envc, {'total_level': 'TL_C', 'attack': 'AT_C', 'decay': 'DE_C', 'sus_level': 'SU_C', 'release': 'RE_C', 'note_off': 'NO_C'}),
            (fmosc, {'mod_amp': 'envm', 'car_amp': 'envc', 'mod_freq': 'FRQ_M', 'car_freq': 'FRQ_C'})
        ]
        fixed_params = {'NO_M': torch.ones(1)*0.8, 'NO_C': torch.ones(1)*0.8, 'FRQ_M': torch.ones(1)*440, 'FRQ_C': torch.ones(1)*440}
    elif name == 'fixedfm3':
        fmosc = FM3(n_samples=16000)
        env1 = ADSREnvelope(name='env1')
        env2 = ADSREnvelope(name='env2')
        env3 = ADSREnvelope(name='env3')
        dag = [
            (env1, {'total_level': 'TL_1', 'attack': 'AT_1', 'decay': 'DE_1', 'sus_level': 'SU_1', 'release': 'RE_1', 'note_off': 'NO_1'}),
            (env2, {'total_level': 'TL_2', 'attack': 'AT_2', 'decay': 'DE_2', 'sus_level': 'SU_2', 'release': 'RE_2', 'note_off': 'NO_2'}),
            (env3, {'total_level': 'TL_3', 'attack': 'AT_3', 'decay': 'DE_3', 'sus_level': 'SU_3', 'release': 'RE_3', 'note_off': 'NO_3'}),
            (fmosc, {'amp_1': 'env1', 'amp_2': 'env2', 'amp_3': 'env3', 'freq_1': 'FRQ_1', 'freq_2': 'FRQ_2', 'freq_3': 'FRQ_3'})
        ]
        fixed_params = {'NO_1': torch.ones(1)*0.8, 'NO_2': torch.ones(1)*0.8, 'NO_3': torch.ones(1)*0.8, 'FRQ_1': torch.ones(1)*440, 'FRQ_2': torch.ones(1)*440, 'FRQ_3': torch.ones(1)*440}
    elif name == 'coarsefm2':
        fmosc = FM2(n_samples=16000)
        envm = ADSREnvelope(name='envm')
        envc = ADSREnvelope(name='envc')
        frqm = FreqKnobsCoarse(name='frqm')
        dag = [
            (frqm,  {'base_freq': 'BFRQ', 'coarse': 'FRQM_C', 'detune': 'FRQM_D'}),
            (envm, {'total_level': 'TL_M', 'attack': 'AT_M', 'decay': 'DE_M', 'sus_level': 'SU_M', 'release': 'RE_M', 'note_off': 'NO_M'}),
            (envc, {'total_level': 'TL_C', 'attack': 'AT_C', 'decay': 'DE_C', 'sus_level': 'SU_C', 'release': 'RE_C', 'note_off': 'NO_C'}),
            (fmosc, {'mod_amp': 'envm', 'car_amp': 'envc', 'mod_freq': 'frqm', 'car_freq': 'BFRQ'})
        ]
        fixed_params = {'NO_M': torch.ones(1)*0.8, 'NO_C': torch.ones(1)*0.8, 'BFRQ': torch.ones(1)*440}
    elif name == 'fm2':
        fmosc = FM2(n_samples=16000)
        envm = ADSREnvelope(name='envm')
        envc = ADSREnvelope(name='envc')
        frqm = FreqMultiplier(name='frqm')
        dag = [
            (frqm,  {'base_freq': 'BFRQ', 'mult': 'FRQM_M'}),
            (envm, {'total_level': 'TL_M', 'attack': 'AT_M', 'decay': 'DE_M', 'sus_level': 'SU_M', 'release': 'RE_M', 'note_off': 'NO_M'}),
            (envc, {'total_level': 'TL_C', 'attack': 'AT_C', 'decay': 'DE_C', 'sus_level': 'SU_C', 'release': 'RE_C', 'note_off': 'NO_C'}),
            (fmosc, {'mod_amp': 'envm', 'car_amp': 'envc', 'mod_freq': 'frqm', 'car_freq': 'BFRQ'})
        ]
        fixed_params = {'NO_M': torch.ones(1)*0.8, 'NO_C': torch.ones(1)*0.8, 'BFRQ': torch.ones(1)*440}
    elif name == 'fm2_noenv':
        fmosc = FM2(n_samples=16000)
        frqm = FreqMultiplier(name='frqm')
        dag = [
            (frqm,  {'base_freq': 'BFRQ', 'mult': 'FRQM_M'}),
            (fmosc, {'mod_amp': 'MOD_A', 'car_amp': 'CAR_A', 'mod_freq': 'frqm', 'car_freq': 'BFRQ'})
        ]
        fixed_params = {'BFRQ': torch.ones(1)*440}
    elif name == 'coarsesin':
        sinosc = SineOscillator(n_samples=16000)
        env = ADSREnvelope(name='env')
        frq = FreqKnobsCoarse(name='frq')
        dag = [
            (frq,  {'base_freq': 'BFRQ', 'coarse': 'FRQ_C', 'detune': 'FRQ_D'}),
            (env, {'total_level': 'TL', 'attack': 'AT', 'decay': 'DE', 'sus_level': 'SU', 'release': 'RE', 'note_off': 'NO'}),
            (sinosc, {'amplitudes': 'env', 'frequencies': 'frq'})
        ]
        fixed_params = {'NO': torch.ones(1)*0.8, 'BFRQ': torch.ones(1)*440}
    elif name == 'coarsesin_noenv':
        sinosc = SineOscillator(n_samples=16000)
        frq = FreqKnobsCoarse(name='frq')
        dag = [
            (frq,  {'base_freq': 'BFRQ', 'coarse': 'FRQ_C', 'detune': 'FRQ_D'}),
            (sinosc, {'amplitudes': 'SIN_A', 'frequencies': 'frq'})
        ]
        fixed_params = {'BFRQ': torch.ones(1)*440}
    elif name == 'sin':
        sinosc = SineOscillator(n_samples=16000)
        env = ADSREnvelope(name='env')
        frq = FreqMultiplier(name='frq')
        dag = [
            (frq,  {'base_freq': 'BFRQ', 'mult': 'FRQ_M'}),
            (env, {'total_level': 'TL', 'attack': 'AT', 'decay': 'DE', 'sus_level': 'SU', 'release': 'RE', 'note_off': 'NO'}),
            (sinosc, {'amplitudes': 'env', 'frequencies': 'frq'})
        ]
        fixed_params = {'NO': torch.ones(1)*0.8, 'BFRQ': torch.ones(1)*440}
    elif name == 'sin_noenv':
        sinosc = SineOscillator(n_samples=16000)
        frq = FreqMultiplier(name='frq')
        dag = [
            (frq,  {'base_freq': 'BFRQ', 'mult': 'FRQ_M'}),
            (sinosc, {'amplitudes': 'SIN_A', 'frequencies': 'frq'})
        ]
        fixed_params = {'BFRQ': torch.ones(1)*440}
    elif name == 'saw_svf':
        sawosc = SawOscillator(n_samples=16000, name='sawosc')
        svf = SVFilter(name='svf')
        dag = [
            (sawosc, {'amplitudes': 'AMP', 'f0_hz': 'BFRQ'}),
            (svf, {'audio': 'sawosc', 'g': 'SVF_G', 'twoR': 'SVF_R', 'mix': 'SVF_M'})
        ]
        fixed_params = {'BFRQ': torch.ones(1)*440, 'AMP': torch.ones(1)*0.8}
    elif name == 'harmor':
        harmor = Harmor(n_samples=16000, name='harmor', n_oscs=1)
        dag = [
            (harmor, {'amplitudes': 'AMP', 'osc_mix': 'M_OSC', 'f0_hz': 'BFRQ', 'cutoff': 'CUTOFF', 'q': 'Q_FILT'})
        ]
        fixed_params = {'BFRQ': torch.ones(1)*440}
        static_params = ['M_OSC', 'Q_FILT']
    elif name == 'harmor_free':
        harmor = Harmor(n_samples=16000, name='harmor', n_oscs=1)
        dag = [
            (harmor, {'amplitudes': 'AMP', 'osc_mix': 'M_OSC', 'f0_hz': 'BFRQ', 'cutoff': 'CUTOFF', 'q': 'Q_FILT'})
        ]
        fixed_params = {}
        static_params = ['M_OSC', 'Q_FILT', 'BFRQ']
    elif name == 'harmor_2oscfree':
        harmor = Harmor(n_samples=16000, name='harmor', sep_amp=False, sep_f0s=True, n_oscs=2)
        dag = [
            (harmor, {'amplitudes': 'AMP', 'osc_mix': 'M_OSC', 'f0_hz': 'BFRQ', 'cutoff': 'CUTOFF', 'q': 'Q_FILT'}),
            ]
        fixed_params = {}
        static_params=['BFRQ', 'M_OSC', 'Q_FILT']
    synth = Synthesizer(dag, fixed_params=fixed_params, static_params=static_params)

    return synth

