import torch
import torchcrepe
import torch.nn.functional as F

from .perceptual import Perceptual
from diffsynth.util import slice_windows

class CREPELoss(Perceptual):
    def __init__(self, model_spec='tiny'):
        super().__init__()
        torchcrepe.load.model('cpu', model_spec)
        self.model = torchcrepe.infer.model

    def process_frames(self, frames):
        # https://github.com/maxrmorrison/torchcrepe/blob/08b36ebe8b443ac1d2a9655192e268b3f1b19f34/torchcrepe/core.py#L625
        frames = frames - frames.mean(dim=-1, keepdim=True)
        frames = frames / torch.clamp(frames.std(dim=-1, keepdim=True), min=1e-10)
        return frames

    def perceptual_loss(self, target_audio, input_audio):
        target_frames = self.process_frames(slice_windows(target_audio, 1024, 512, window=None))
        input_frames = self.process_frames(slice_windows(input_audio, 1024, 512, window=None))
        # [batch, n_frames, 1024]-> batch', 1024
        target_frames = target_frames.flatten(0,1)
        target_embed = self.model.embed(target_frames)
        input_frames = input_frames.flatten(0,1)
        input_embed = self.model.embed(input_frames)
        return F.l1_loss(target_embed, input_embed)