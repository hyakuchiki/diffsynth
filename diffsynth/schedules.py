"""

For scheduling AE parameters

"""
import functools

def linear_anneal(i, end_value, start_value, start, warm):
    l = max(i - start, 0)
    value = (end_value-start_value) * (float(l) / float(max(warm, l))) + start_value
    return value

# loss weights and other parameters used during training
required_args =    ['param_w', # parameter loss
                    'sw_w', # spectral/waveform loss
                    'enc_w', # ae encoding loss
                    'mfcc_w', # MFCC L1 loss
                    'lsd_w', # log spectral distortion
                    'loud_w', # loudness L1 loss
                    'cls_w', # classifier loss (domain adversarial)
                    'acc_w', # classifier accuracy (not a loss)
                    'grl' # grl gradient backwards scale
                    ]

class ParamScheduler():
    def __init__(self, schedule_dict):
        self.sched = schedule_dict
        self.unit = self.sched.pop('unit')
        for k in required_args:
            if k not in self.sched:
                self.sched[k] = 0.0

    def get_parameters(self, cur_epoch, dl_size=None):
        cur_param = {}
        i = cur_epoch if self.unit=='epochs' else cur_epoch*dl_size
        for param_name, param_func in self.sched.items():
            cur_param[param_name] = param_func(i=i) if callable(param_func) else param_func
        return cur_param

SCHEDULE_REGISTRY = {}

# switch completely from param to spectral loss
switch_1 = {
    'unit': 'epochs',
    # parameter loss weight
    'param_w': functools.partial(linear_anneal, end_value=0.0, start_value=10.0, start=50, warm=150),
    # reconstruction (spectral/wave) loss weight
    'sw_w': functools.partial(linear_anneal, end_value=1.0, start_value=0.0, start=50, warm=150),
}
SCHEDULE_REGISTRY['switch_1'] = switch_1

both_1 = {
    'unit': 'epochs',
    'param_w': functools.partial(linear_anneal, end_value=5.0, start_value=10.0, start=50, warm=150),
    'sw_w': functools.partial(linear_anneal, end_value=0.5, start_value=0.0, start=50, warm=150),
}
SCHEDULE_REGISTRY['both_1'] = both_1

only_param = {
    'unit': 'epochs',
    'param_w': 10.0,
}
SCHEDULE_REGISTRY['only_param'] = only_param

only_sw= {
    'unit': 'epochs',
    'sw_w': 1.0,
}
SCHEDULE_REGISTRY['only_sw'] = only_sw

dann_1 = {
    'unit': 'epochs',
    'param_w': 10.0,
    'cls_w': 10.0,
    'grl': 1.0,
}
SCHEDULE_REGISTRY['dann_1'] = dann_1

dann_2 = {
    'unit': 'epochs',
    'param_w': 10.0,
    'cls_w': 1.0,
    'grl': 1.0,
}
SCHEDULE_REGISTRY['dann_2'] = dann_2

dann_3 = {
    'unit': 'epochs',
    'param_w': 10.0,
    'cls_w': 1.0,
    'grl': functools.partial(linear_anneal, end_value=1.0, start_value=0.0, start=10, warm=100),
}
SCHEDULE_REGISTRY['dann_3'] = dann_3

# DOESNT WORK WELL

only_sw = {
    'unit': 'epochs',
    'sw_w': 1.0,
}
SCHEDULE_REGISTRY['only_sw'] = only_sw

only_enc = {
    'unit': 'epochs',
    'enc_w': 10.0,
}
SCHEDULE_REGISTRY['only_enc'] = only_enc

# switch completely from param to spectral loss and embedding loss
switch_e = {
    'unit': 'epochs',
    # parameter loss weight
    'param_w': functools.partial(linear_anneal, end_value=0.0, start_value=10.0, start=50, warm=150),
    # reconstruction (spectral/wave) loss weight
    'sw_w': functools.partial(linear_anneal, end_value=1.0, start_value=0.0, start=50, warm=150),
    # perceptual loss based on ae
    'enc_w': functools.partial(linear_anneal, end_value=10.0, start_value=0.0, start=50, warm=150),
}
SCHEDULE_REGISTRY['switch_e'] = switch_e

switch_only_e = {
    'unit': 'epochs',
    'param_w': functools.partial(linear_anneal, end_value=0.0, start_value=10.0, start=50, warm=150),
    'enc_w': functools.partial(linear_anneal, end_value=10.0, start_value=0.0, start=50, warm=150),
}
SCHEDULE_REGISTRY['switch_only_e'] = switch_only_e

switch_mfcc = {
    'unit': 'epochs',
    'param_w': functools.partial(linear_anneal, end_value=0.0, start_value=10.0, start=50, warm=150),
    'mfcc_w': functools.partial(linear_anneal, end_value=10.0, start_value=0.0, start=50, warm=150),
}
SCHEDULE_REGISTRY['switch_mfcc'] = switch_mfcc

switch_ld = {
    'unit': 'epochs',
    # parameter loss weight
    'param_w': functools.partial(linear_anneal, end_value=0.0, start_value=10.0, start=50, warm=150),
    # reconstruction (spectral/wave) loss weight
    'sw_w': functools.partial(linear_anneal, end_value=1.0, start_value=0.0, start=50, warm=150),
    'loud_w': functools.partial(linear_anneal, end_value=1.0, start_value=0.0, start=50, warm=150),
}
SCHEDULE_REGISTRY['switch_ld'] = switch_ld

switch_ld2 = {
    'unit': 'epochs',
    # parameter loss weight
    'param_w': functools.partial(linear_anneal, end_value=0.0, start_value=10.0, start=50, warm=150),
    # reconstruction (spectral/wave) loss weight
    'sw_w': functools.partial(linear_anneal, end_value=1.0, start_value=0.0, start=50, warm=150),
    'loud_w': functools.partial(linear_anneal, end_value=0.1, start_value=0.0, start=50, warm=150),
}
SCHEDULE_REGISTRY['switch_ld2'] = switch_ld2