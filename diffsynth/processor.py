import torch
import torch.nn as nn
import diffsynth.util as util
import math

SCALE_FNS = {
    'sigmoid': lambda x, low, high: x*(high-low) + low,
    'freq_sigmoid': lambda x, low, high: util.unit_to_hz(x, low, high, clip=False),
    'exp_sigmoid': lambda x, low, high: util.exp_scale(x, math.log(10.0), high, 1e-7+low),
}

class Processor(nn.Module):
    def __init__(self, name):
        """ Initialize as module """
        super().__init__()
        self.name = name
        self.param_desc = {}

    def process(self, scaled_params=[], **kwargs):
        # scaling each parameter according to the property
        # input is 0~1
        for k in kwargs.keys():
            if k not in self.param_desc or k in scaled_params:
                continue
            desc = self.param_desc[k]
            scale_fn = SCALE_FNS[desc['type']]
            p_range = desc['range']
            # if (kwargs[k] > 1).any():
            #     raise ValueError('parameter to be scaled is not 0~1')
            kwargs[k] = scale_fn(kwargs[k], p_range[0], p_range[1])
        return self(**kwargs)

    def forward(self):
        raise NotImplementedError

class Gen(Processor):
    def __init__(self, name):
        super().__init__(name)

class Add(Processor):
    def __init__(self, name='add', n_samples=64000):
        super().__init__(name=name)
        self.param_desc = {
            'signal_a': {'size':1, 'range': (-1, 1), 'type': 'raw'}, 'signal_b': {'size':1, 'range': (-1, 1), 'type': 'raw'}
            }

    def forward(self, signal_a, signal_b):
        # kinda sucks can only add two
        return signal_a+signal_b

class Mix(Processor):
    def __init__(self, name='add'):
        super().__init__(name=name)
        self.param_desc = {
                'signal_a': {'size':1, 'range': (-1, 1), 'type': 'raw'}, 
                'signal_b': {'size':1, 'range': (-1, 1), 'type': 'raw'},
                'mix_a': {'size':1, 'range': (0, 1), 'type': 'sigmoid'},
                'mix_b': {'size':1, 'range': (0, 1), 'type': 'sigmoid'},
                }    

    def forward(self, signal_a, signal_b, mix_a, mix_b):
        # kinda sucks can only add two
        return mix_a*signal_a+mix_b*signal_b