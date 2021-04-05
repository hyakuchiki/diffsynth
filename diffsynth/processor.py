import torch
import torch.nn as nn

class Processor(nn.Module):
    def __init__(self, name):
        """ Initialize as module """
        super().__init__()
        self.name = name
    def get_param_desc(self):
        raise NotImplementedError
    def forward(self):
        raise NotImplementedError

class Gen(Processor):
    def __init__(self, name):
        super().__init__(name)

class Add(Processor):
    def __init__(self, name='add', n_samples=64000):
        super().__init__(name=name)
    
    def forward(self, signal_a, signal_b):
        # kinda sucks can only add two
        return signal_a+signal_b

    def get_param_desc(self):
        return {'signal_a': {'size':1, 'range': (-1, 1), 'type': 'raw'}, 'signal_b': {'size':1, 'range': (-1, 1), 'type': 'raw'}}

class Mix(Processor):
    def __init__(self, name='add'):
        super().__init__(name=name)
    
    def forward(self, signal_a, signal_b, mix_a, mix_b):
        # kinda sucks can only add two
        return mix_a*signal_a+mix_b*signal_b

    def get_param_desc(self):
        return {
                'signal_a': {'size':1, 'range': (-1, 1), 'type': 'raw'}, 
                'signal_b': {'size':1, 'range': (-1, 1), 'type': 'raw'},
                'mix_a': {'size':1, 'range': (0, 1), 'type': 'sigmoid'},
                'mix_b': {'size':1, 'range': (0, 1), 'type': 'sigmoid'},
                }