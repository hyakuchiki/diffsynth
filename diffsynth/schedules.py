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
                    'perc_w', # perceptual loss
                    ]

class ParamSchedule():
    def __init__(self, sched_cfg):
        self.sched = {}
        for param_name, param_sched in sched_cfg.items():
            if param_name == 'name':
                continue 
            if isinstance(param_sched, float):
                self.sched[param_name] = param_sched
                continue
            if param_sched['type'] == 'linear':
                self.sched[param_name] = functools.partial(linear_anneal, 
                    start=param_sched['start'], 
                    warm=param_sched['warm'],
                    start_value=param_sched['start_v'],
                    end_value=param_sched['end_v'])
            else:
                raise ValueError()
        for k in required_args:
            if k not in self.sched:
                self.sched[k] = 0.0

    def get_parameters(self, cur_step):
        cur_param = {}
        for param_name, param_func in self.sched.items():
            cur_param[param_name] = param_func(i=cur_step) if callable(param_func) else param_func
        return cur_param


# Below is defunct

SCHEDULE_REGISTRY = {}

class ParamScheduler():
    def __init__(self, name):
        self.sched = SCHEDULE_REGISTRY[name]
        for k in required_args:
            if k not in self.sched:
                self.sched[k] = 0.0

    def get_parameters(self, cur_step):
        cur_param = {}
        for param_name, param_func in self.sched.items():
            cur_param[param_name] = param_func(i=cur_step) if callable(param_func) else param_func
        return cur_param

switch_1 = {
    # parameter loss weight
    'param_w': functools.partial(linear_anneal, end_value=0.0, start_value=10.0, start=12500, warm=37500),
    # reconstruction (spectral/wave) loss weight
    'sw_w': functools.partial(linear_anneal, end_value=1.0, start_value=0.0, start=12500, warm=37500),
}
SCHEDULE_REGISTRY['switch_1'] = switch_1

# even weights
even_1 = {
    # parameter loss weight
    'param_w': functools.partial(linear_anneal, end_value=5.0, start_value=10.0, start=12500, warm=37500),
    # reconstruction (spectral/wave) loss weight
    'sw_w': functools.partial(linear_anneal, end_value=0.5, start_value=0.0, start=12500, warm=37500),
}
SCHEDULE_REGISTRY['even_1'] = even_1

# switch completely from param to spectral loss and perceptual loss
switch_p = {
    # parameter loss weight
    'param_w': functools.partial(linear_anneal, end_value=0.0, start_value=10.0, start=12500, warm=37500),
    # reconstruction (spectral/wave) loss weight
    'sw_w': functools.partial(linear_anneal, end_value=1.0, start_value=0.0, start=12500, warm=37500),
    # perceptual loss based on ae
    'perc_w': functools.partial(linear_anneal, end_value=0.01, start_value=0.0, start=12500, warm=37500),
}
SCHEDULE_REGISTRY['switch_p'] = switch_p

only_param = {
    'param_w': 10.0,
}
SCHEDULE_REGISTRY['only_param'] = only_param

sw_param = {
    'param_w': 5.0,
    'sw_w': 0.5
}
SCHEDULE_REGISTRY['sw_param'] = sw_param

only_sw = {
    'sw_w': 1.0,
}
SCHEDULE_REGISTRY['only_sw'] = only_sw

only_perc = {
    'perc_w': 10.0,
}
SCHEDULE_REGISTRY['only_perc'] = only_perc

sw_perc = {
    'sw_w': 1.0,
    'perc_w': 10.0
}
SCHEDULE_REGISTRY['sw_perc'] = sw_perc