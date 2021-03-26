import torch
import torch.nn as nn
from diffsynth.modules.synth import Processor
import diffsynth.util as util
import numpy as np
import math

class FIRFilter(Processor):
    """
    uses frequency sampling
    """
    
    def __init__(self, filter_size=64, scale_fn=util.exp_sigmoid, name='firfilter', initial_bias=-5.0):
        super().__init__()
        self.filter_size = filter_size
        self.scale_fn = scale_fn
        self.initial_bias = initial_bias

    def forward(self, audio, freq_response):
        """pass audio through FIRfilter
        Args:
            audio (torch.Tensor): [batch, n_samples]
            freq_response (torch.Tensor): frequency response (only magnitude) [batch, n_frames, filter_size // 2 + 1]

        Returns:
            [torch.Tensor]: Filtered audio. Shape [batch, n_samples]
        """
        if self.scale_fn is not None:
            freq_response = self.scale_fn(freq_response + self.initial_bias)
        return util.fir_filter(audio, freq_response, self.filter_size)
    
    def get_param_sizes(self):
        return {'freq_response': self.filter_size // 2 + 1, 'audio': None}