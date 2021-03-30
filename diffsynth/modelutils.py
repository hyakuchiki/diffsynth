import torch
from diffsynth.modules.fm import FM2, FM3
from diffsynth.modules.envelope import ADSREnvelope
from diffsynth.synthesizer import Synthesizer

def construct_synths(name, device='cpu'):
    if name == 'fixedfm2':
        fmosc = FM2(n_samples=16000, amp_scale_fn=None, freq_scale_fn=None).to(device)
        envm = ADSREnvelope(name='envm').to(device)
        envc = ADSREnvelope(name='envc').to(device)
        dag = [
            (envm, {'total_level': 'TL_M', 'attack': 'AT_M', 'decay': 'DE_M', 'sus_level': 'SU_M', 'release': 'RE_M', 'note_off': 'NO_M'}),
            (envc, {'total_level': 'TL_C', 'attack': 'AT_C', 'decay': 'DE_C', 'sus_level': 'SU_C', 'release': 'RE_C', 'note_off': 'NO_C'}),
            (fmosc, {'mod_amp': 'envm', 'car_amp': 'envc', 'mod_freq': 'FRQ_M', 'car_freq': 'FRQ_C'})
        ]
        fixed_params = {'NO_M': torch.ones(1)*0.8, 'NO_C': torch.ones(1)*0.8, 'FRQ_M': torch.ones(1)*440, 'FRQ_C': torch.ones(1)*440}
    elif name == 'fixedfm3':
        fmosc = FM3(n_samples=16000, amp_scale_fn=None, freq_scale_fn=None).to(device)
        env1 = ADSREnvelope(name='env1').to(device)
        env2 = ADSREnvelope(name='env2').to(device)
        env3 = ADSREnvelope(name='env3').to(device)
        dag = [
            (env1, {'total_level': 'TL_1', 'attack': 'AT_1', 'decay': 'DE_1', 'sus_level': 'SU_1', 'release': 'RE_1', 'note_off': 'NO_1'}),
            (env2, {'total_level': 'TL_2', 'attack': 'AT_2', 'decay': 'DE_2', 'sus_level': 'SU_2', 'release': 'RE_2', 'note_off': 'NO_2'}),
            (env3, {'total_level': 'TL_3', 'attack': 'AT_3', 'decay': 'DE_3', 'sus_level': 'SU_3', 'release': 'RE_3', 'note_off': 'NO_3'}),
            (fmosc, {'amp_1': 'env1', 'amp_2': 'env2', 'amp_3': 'env3', 'freq_1': 'FRQ_1', 'freq_2': 'FRQ_2', 'freq_3': 'FRQ_3'})
        ]
        fixed_params = {'NO_1': torch.ones(1)*0.8, 'NO_2': torch.ones(1)*0.8, 'NO_3': torch.ones(1)*0.8, 'FRQ_1': torch.ones(1)*440, 'FRQ_2': torch.ones(1)*440, 'FRQ_3': torch.ones(1)*440}
    if fixed_params is not None:
        fixed_params = {k: v.to(device) for k, v in fixed_params.items()}
    synth = Synthesizer(dag, fixed_params=fixed_params).to(device)
    return synth

