"""

For scheduling AE parameters

"""
import functools

def linear_anneal(i, end_value, start_value, start, warm):
    l = max(i - start, 0)
    value = (end_value-start_value) * (float(l) / float(max(warm, l))) + start_value
    return value

SCHEDULE_REGISTRY = {}

switch_1 = {
    'unit': 'epochs',
    # parameter loss weight
    'param': functools.partial(linear_anneal, end_value=1.0, start_value=0.0, start=50, warm=150),
    # reconstruction (spectral) loss weight
    'recon': functools.partial(linear_anneal, end_value=0, start_value=1.0, start=50, warm=150),
}

SCHEDULE_REGISTRY['switch_1'] = switch_1

both_1 = {
    'unit': 'epochs',
    'param': functools.partial(linear_anneal, end_value=0.5, start_value=0.0, start=50, warm=150),
    'recon': functools.partial(linear_anneal, end_value=0.5, start_value=1.0, start=50, warm=150),
}

SCHEDULE_REGISTRY['both_1'] = both_1

class ParamScheduler():
    def __init__(self, schedule_dict):
        self.sched = schedule_dict
        if self.sched is not None:
            self.unit = self.sched.pop('unit')

    def get_parameters(self, cur_epoch, dl_size=None):
        cur_param = {}
        if self.sched is None:
            return {}
        i = cur_epoch if self.unit=='epochs' else cur_epoch*dl_size
        for param_name, param_func in self.sched.items():
            cur_param[param_name] = param_func(i=i) if callable(param_func) else param_func
        return cur_param