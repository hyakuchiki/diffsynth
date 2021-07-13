import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffsynth.spectral import MelSpec
from diffsynth.layers import MLP, Normalize2d, Resnet2D
from diffsynth.transforms import LogTransform

class Perceptual(nn.Module):
    def perceptual_loss(self, target_audio, input_audio):
        raise NotImplementedError
    
class PerceptualClassifier(Perceptual):
    # take melspectrogram
    def __init__(self, output_dims, n_samples, n_mels=128, n_downsample=3, stride=(2,2), res_depth=3, channels=32, dilation_growth_rate=2, m_conv=1.0, n_fft=1024, hop=256, norm='batch', sample_rate=16000):
        super().__init__()
        self.n_mels = n_mels
        self.channels = channels
        self.logmel = nn.Sequential(MelSpec(n_fft=n_fft, hop_length=hop, n_mels=n_mels, sample_rate=sample_rate, power=2), LogTransform())
        self.norm = Normalize2d(norm) if norm else None

        spec_len = math.ceil((n_samples - n_fft) / hop) + 1
        self.spec_len_target = 2**(int(math.log2(spec_len))+1) # power of 2
        kernel_size = [s*2 for s in stride]
        pad = [s//2 for s in stride]
        input_size = (n_mels, self.spec_len_target)
        final_size = input_size
        blocks = []
        for i in range(n_downsample):
            block = nn.Sequential(
                # downsampling conv, output size is L_in/stride
                nn.Conv2d(1 if i == 0 else channels, channels, kernel_size, stride, pad),
                # ResNet with growing dilation, doesn't change size
                Resnet2D(channels, res_depth, m_conv, dilation_growth_rate, reverse_dilation=False),
            )
            blocks.append(block)
            final_size = (final_size[0] // stride[0], final_size[1] // stride[1])
        
        self.convmodel = nn.Sequential(*blocks)
        print('output dims after convolution', final_size)

        self.mlp = MLP(final_size[0] * final_size[1] * channels, 64, loop=2)
        self.out = nn.Linear(64, output_dims)
    
    def perceptual_loss(self, target_audio, input_audio, layers=(2, )):
        self.eval()
        batch_size = input_audio.shape[0]
        audios = torch.cat([input_audio, target_audio], dim=0)
        specs = self.logmel(audios).unsqueeze(1)
        loss = 0
        out = specs
        for i, m in enumerate(self.convmodel):
            out = m(out)
            if i in layers:
                loss += F.l1_loss(out[:batch_size], out[batch_size:])
        return loss

    def transform(self, audio):
        spec = self.logmel(audio)
        if self.norm is not None:
            spec = self.norm(spec)
        # (batch, n_mels, time)
        batch_size, n_mels, n_frames = spec.shape
        padded_spec = F.pad(spec, (0, self.spec_len_target-n_frames))
        return padded_spec

    def forward(self, audio):
        x = self.transform(audio).unsqueeze(1)
        x = self.convmodel(x)
        x = x.flatten(1, -1)
        out = self.mlp(x)
        out = self.out(out)
        return out
    
    def train_epoch(self, loader, optimizer, device, clip=1.0):
        self.train()
        sum_loss = 0
        count = 0
        for data_dict in loader:
            # send data to device
            data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
            target = data_dict['label']
            audio = data_dict['audio']
            logits = self(audio)
            batch_loss = F.cross_entropy(logits, target)
            # Perform backward
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
            optimizer.step()
            sum_loss += batch_loss.detach().item()
            count += 1
        sum_loss /= count
        return sum_loss

    def eval_epoch(self, loader, device):
        self.eval()
        sum_correct = 0
        count = 0
        with torch.no_grad():
            for data_dict in loader:
                # send data to device
                data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
                audio = data_dict['audio']
                target = data_dict['label']
                logits = self(audio)
                sum_correct += (torch.argmax(logits, dim=-1) == target).sum().item()
                count += audio.shape[0]
        return sum_correct/count # accuracy