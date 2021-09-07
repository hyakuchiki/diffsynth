import math
import torch
import torch.nn as nn

from diffsynth.util import midi_to_hz, hz_to_midi
from diffsynth.layers import Resnet1D, Normalize2d
from diffsynth.transforms import LogTransform
from nnAudio.Spectrogram import MelSpectrogram, MFCC

class MFCCEstimator(nn.Module):
    def __init__(self, output_dim, n_mels=128, n_mfccs=30, n_fft=1024, hop=256, sample_rate=16000, num_layers=2, hidden_size=512, dropout_p=0.0, norm='instance'):
        super().__init__()
        self.mfcc = MFCC(sr=sample_rate, n_mfcc=n_mfccs, norm='ortho', verbose=True, hop_length=hop, n_fft=n_fft, n_mels=n_mels, center=True, sample_rate=sample_rate)
        self.norm = Normalize2d(norm) if norm else None
        self.gru = nn.GRU(n_mfccs, hidden_size, num_layers=num_layers, dropout=dropout_p, batch_first=True)
        self.output_dim = output_dim
        self.out = nn.Linear(hidden_size, output_dim)
    
    def forward(self, audio):
        x = self.mfcc(audio)
        x = self.norm(x) if self.norm else x
        x = x.permute(0, 2, 1).contiguous()
        # batch_size, n_frames, n_mfcc = x.shape
        output, _hidden = self.gru(x)
        # output: [batch_size, n_frames, self.output_dim]
        output = self.out(output)
        return torch.sigmoid(output)

class MelEstimator(nn.Module):
    def __init__(self, output_dim, n_mels=128, n_fft=1024, hop=256, sample_rate=16000, channels=64, kernel_size=7, strides=[2,2,2], num_layers=1, hidden_size=512, dropout_p=0.0, norm='batch'):
        super().__init__()
        self.n_mels = n_mels
        self.channels = channels
        self.logmel = nn.Sequential(MelSpectrogram(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, hop_length=hop, center=True, power=1.0, htk=True, trainable_mel=False, trainable_STFT=False), LogTransform())
        self.norm = Normalize2d(norm) if norm else None
        # Regular Conv
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(1, channels, kernel_size,
                        padding=kernel_size // 2,
                        stride=strides[0]), nn.BatchNorm1d(channels), nn.ReLU())]
            + [nn.Sequential(nn.Conv1d(channels, channels, kernel_size,
                         padding=kernel_size // 2,
                         stride=strides[i]), nn.BatchNorm1d(channels), nn.ReLU())
                         for i in range(1, len(strides))])
        self.l_out = self.get_downsampled_length()[-1] # downsampled in frequency dimension
        print('output dims after convolution', self.l_out)
        self.gru = nn.GRU(self.l_out * channels, hidden_size, num_layers=num_layers, dropout=dropout_p, batch_first=True)
        self.out = nn.Linear(hidden_size, output_dim)
        self.output_dim = output_dim

    def forward(self, audio):
        x = self.logmel(audio)
        x = self.norm(x)
        batch_size, n_mels, n_frames = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, self.n_mels).unsqueeze(1)
        # x: [batch_size*n_frames, 1, n_mels]
        for i, conv in enumerate(self.convs):
            x = conv(x)
        x = x.view(batch_size, n_frames, self.channels, self.l_out)
        x = x.view(batch_size, n_frames, -1)
        output, _hidden = self.gru(x)
        # output: [batch_size, n_frames, self.output_dim]
        output = self.out(output)
        return torch.sigmoid(output)

    def get_downsampled_length(self):
        l = self.n_mels
        lengths = [l]
        for conv in self.convs:
            conv_module = conv[0]
            l = (l + 2 * conv_module.padding[0] - conv_module.dilation[0] * (conv_module.kernel_size[0] - 1) - 1) // conv_module.stride[0] + 1
            lengths.append(l)
        return lengths

frame_setting_stride = {
    # n_downsample, stride
    "coarse": (5, 4), # hop: 1024
    "fine": (9, 2), # 512, too deep?
    "finer": (8, 2), # 256
    "finest": (6, 2) # 64
}

class FrameDilatedConvEstimator(nn.Module):
    """
    Process raw waveform
    Similar to Jukebox
    """
    def __init__(self, output_dim, frame_setting='finer', res_depth=4, channels=32, dilation_growth_rate=3, m_conv=1.0):
        """
        Args:
            output_dims (int): output channels
            res_depth (int, optional): depth of each resnet. Defaults to 4.
            channels (int, optional): conv channels. Defaults to 32.
            dilation_growth_rate (int, optional): exponential growth of dilation. Defaults to 3.
            m_conv (float, optional): multiplier for resnet channels. Defaults to 1.0.
        """
        super().__init__()
        self.n_downsample, self.stride = frame_setting_stride[frame_setting]
        blocks = []
        kernel_size, pad = self.stride * 2, self.stride // 2
        for i in range(self.n_downsample):
            block = nn.Sequential(
                # downsampling conv, output size is L_in/stride
                nn.Conv1d(1 if i == 0 else channels, channels, kernel_size, self.stride, pad),
                # ResNet with growing dilation
                Resnet1D(channels, res_depth, m_conv, dilation_growth_rate),
            )
            blocks.append(block)
        # # doesn't change size
        # block = nn.Conv1d(channels, output_dims, 3, 1, 1) # output:(batch, output_dims, n_frames)
        # blocks.append(block)
        self.model = nn.Sequential(*blocks)
        self.out = nn.Linear(channels, output_dim)
        self.output_dim = output_dim

    def get_z_frames(self, n_samples):
        n_frames = n_samples // (self.stride ** self.n_downsample)
        return n_frames

    def forward(self, audio):
        batch_size, n_samples = audio.shape
        x = audio.unsqueeze(1)
        x = self.model(x) # (batch, channels, n_frames)
        x = x.permute(0, 2, 1)
        assert x.shape == (batch_size, self.get_z_frames(n_samples), self.output_dims)
        x = torch.sigmoid(self.out(x))
        return x