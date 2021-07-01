import torch
import torch.nn as nn
import torch.nn.functional as F

from diffsynth.layers import Resnet1D, MLP

class Perceptual(nn.Module):
    def perceptual_loss(self, target_audio, input_audio):
        raise NotImplementedError
    
# class PerceptualClassifier(Perceptual):
#     # take melspectrogram