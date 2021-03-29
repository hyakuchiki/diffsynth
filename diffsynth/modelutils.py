import torch
from diffsynth.modules.generators import SimpleFMOsc
from diffsynth.modules.envelope import ADSREnvelope
from diffsynth.synthesizer import Synthesizer

def construct_synths(name, device='cpu'):
    if name == 'fixedfm2':
        fmosc = SimpleFMOsc(n_samples=16000, amp_scale_fn=None, freq_scale_fn=None).to(device)
        envm = ADSREnvelope(name='envm').to(device)
        envc = ADSREnvelope(name='envc').to(device)
        dag = [
            (envm, {'total_level': 'TL_M', 'attack': 'AT_M', 'decay': 'DE_M', 'sus_level': 'SU_M', 'release': 'RE_M', 'note_off': 'NO_M'}),
            (envc, {'total_level': 'TL_C', 'attack': 'AT_C', 'decay': 'DE_C', 'sus_level': 'SU_C', 'release': 'RE_C', 'note_off': 'NO_C'}),
            (fmosc, {'mod_amp': 'envm', 'car_amp': 'envc', 'mod_freq': 'FRQ_M', 'car_freq': 'FRQ_C'})
        ]
        fixed_params = {'NO_M': torch.ones(1)*0.8, 'NO_C': torch.ones(1)*0.8, 'FRQ_M': torch.ones(1)*440, 'FRQ_C': torch.ones(1)*440}

    if fixed_params is not None:
        fixed_params = {k: v.to(device) for k, v in fixed_params.items()}
    synth = Synthesizer(dag, fixed_params=fixed_params).to(device)
    return synth

