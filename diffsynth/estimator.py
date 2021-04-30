import math
import torch
import torch.nn as nn

from diffsynth.util import midi_to_hz, hz_to_midi
from diffsynth.layers import Resnet1D, Resnet2D, MLP, Normalize2d, CoordConv1D
from diffsynth.spectral import MelSpec, Spec, Mfcc
from diffsynth.transforms import LogTransform

def get_window_hop(enc_frame_setting):
    if enc_frame_setting not in ['coarse', 'fine', 'finer']:
        raise ValueError(
            '`enc_frame_setting` currently limited to coarse, fine, finer')
    # copied from ddsp
    # this only works when x.shape[-1] = 64000
    z_audio_spec = {
        'coarse': { # 62 or something
            'n_fft': 2048,
            'overlap': 0.5
        },
        'fine': {
            'n_fft': 1024,
            'overlap': 0.5
        },
        'finer': {
            'n_fft': 1024,
            'overlap': 0.75
        },
    }
    n_fft = z_audio_spec[enc_frame_setting]['n_fft']
    hop_length = int((1 - z_audio_spec[enc_frame_setting]['overlap']) * n_fft)
    return n_fft, hop_length

class Estimator(nn.Module):
    def __init__(self, f0_encoder=None, noise_prob=0.3, noise_mag=0.1):
        super().__init__()
        self.f0_encoder = f0_encoder
        self.noise_prob = noise_prob
        self.noise_mag = noise_mag

    def forward(self, conditioning):
        conditioning = self.fill_f0(conditioning)
        # add noise
        x = conditioning['audio']
        if self.training and self.noise_prob > 0:
            mask = torch.rand(x.shape[0], 1, device=x.device) < self.noise_prob
            x = x + (mask * torch.randn_like(x) * self.noise_mag)
        conditioning['audio'] = x
        param_tensor = self.compute_params(conditioning)
        conditioning['est_param'] = param_tensor
        return conditioning

    def fill_f0(self, conditioning):
        if self.f0_encoder:
            # Use frequency conditioning created by the f0_encoder, not the dataset.
            # Overwrite `f0_scaled` and `f0_hz`. 'f0_scaled' is a value in [0, 1]
            # corresponding to midi values [0..127]
            conditioning['f0_scaled'] = self.f0_encoder(conditioning)
            conditioning['f0_hz'] = midi_to_hz(conditioning['f0_scaled'] * 127.0)
        else:
            if 'f0_hz' in conditioning:
                # conditioning['f0_hz'] = fix_f0(conditioning['f0_hz'])
                if len(conditioning['f0_hz'].shape) == 2:
                    # [batch, n_frames, feature_size=1]
                    conditioning['f0_hz'] = conditioning['f0_hz'][:, :, None]
                conditioning['f0_scaled'] = hz_to_midi(conditioning['f0_hz']) / 127.0
            else:
                # doesn't fill f0 info
                pass
        return conditioning

class MFCCEstimator(Estimator):
    def __init__(self, output_dims, input_dims, frame_setting='finer', n_mfccs=30, sample_rate=16000, num_layers=2, hidden_size=512, dropout_p=0.0, norm='instance', f0_encoder=None, noise_prob=0.0, noise_mag=0.0):
        super().__init__(f0_encoder, noise_prob, noise_mag)
        n_fft, hop = get_window_hop(frame_setting)
        self.frame_setting = frame_setting
        self.mfcc = Mfcc(n_fft, hop, 128, n_mfccs, f_min=20, sample_rate=sample_rate)
        self.norm = Normalize2d(norm) if norm else None
        self.gru = nn.GRU(n_mfccs, hidden_size, num_layers=num_layers, dropout=dropout_p, batch_first=True)
        self.out = nn.Linear(hidden_size, output_dims)
    
    def compute_params(self, conditioning):
        audio = conditioning['audio']
        x = self.mfcc(audio)
        x = self.norm(x) if self.norm else x
        x = x.permute(0, 2, 1).contiguous()
        # batch_size, n_frames, _n_mfcc = x.shape
        output, _hidden = self.gru(x)
        return self.out(output)

"""
deprecated estimators
use static parameters only or doesn't have GRUs

"""
class DilatedConvEstimator(Estimator):
    """
    Similar to Jukebox
    Static Parameter
    """
    def __init__(self, output_dims, input_dims, n_downsample=6, stride=4, res_depth=4, channels=32, dilation_growth_rate=3, m_conv=1.0, f0_encoder=None, noise_prob=0.3, noise_mag=0.1):
        """
        Args:
            output_dims (int): output dimension
            input_dims (int): input dimension (length of audio in samples)
            n_downsample (int, optional): times to downsample. Defaults to 6.
            stride (int, optional): downsampling rate / conv stride. Defaults to 4.
            res_depth (int, optional): depth of each resnet. Defaults to 4.
            channels (int, optional): conv channels. Defaults to 32.
            dilation_growth_rate (int, optional): exponential growth of dilation. Defaults to 3.
            m_conv (float, optional): multiplier for resnet channels. Defaults to 1.0.
        """
        super().__init__(f0_encoder, noise_prob, noise_mag)
        self.output_dims = output_dims
        self.n_downsample = n_downsample
        self.stride = stride
        
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
        # output:(batch, channels, n_frames)
        self.convmodel = nn.Sequential(*blocks)
        z_frames = input_dims // (self.stride ** self.n_downsample)
        self.mlp = MLP(z_frames*channels, channels)
        self.out = nn.Linear(channels, output_dims)
    
    def compute_params(self, conditioning):
        x = conditioning['audio']
        batch_size, n_samples = x.shape
        x = x.unsqueeze(1)
        x = self.convmodel(x) # (batch, channels, n_frames)
        x = x.flatten(1, 2)
        out = self.mlp(x)
        out = self.out(out).unsqueeze(1) # (batch, 1, n_params)
        return out

class MelConvEstimator(Estimator):
    """
    Static Parameter
    """
    def __init__(self, output_dims, n_samples, n_mels=128, n_fft=2048, hop=512, n_downsample=3, stride=(4,2), norm='batch', res_depth=2, channels=32, dilation_growth_rate=3, m_conv=1.0, sr=16000, f0_encoder=None, noise_prob=0.3, noise_mag=0.1):
        super().__init__(f0_encoder, noise_prob, noise_mag)

        self.logmel = nn.Sequential(MelSpec(n_fft=n_fft, hop_length=hop, n_mels=n_mels, sample_rate=sr), LogTransform())
        self.norm = Normalize2d(norm) if norm else None

        # for n_samples = 16384 and n_fft=2048, hop=512, this is 29
        spec_len = math.ceil((n_samples - n_fft) / hop) + 1
        # so the input is (n_mels, 29)
        final_size = (n_mels, spec_len)

        blocks = []
        kernel_size = [s*2 for s in stride]
        pad = [s//2 for s in stride]
        for i in range(n_downsample):
            block = nn.Sequential(
                # downsampling conv, output size is L_in/stride
                nn.Conv2d(1 if i == 0 else channels, channels, kernel_size, stride, pad),
                # ResNet with growing dilation, doesn't change size
                Resnet2D(channels, res_depth, m_conv, dilation_growth_rate),
            )
            blocks.append(block)
            final_size = (final_size[0] // stride[0], final_size[1] // stride[1])

        # output:(batch, channels, final_size[0], final_size[1])
        self.convmodel = nn.Sequential(*blocks)
        print('output dims after convolution', final_size)

        self.mlp = MLP(final_size[0] * final_size[1] * channels, channels)
        self.out = nn.Linear(channels, output_dims)


    def compute_params(self, conditioning):
        audio = conditioning['audio']
        """computes melspectrogram of input audio"""
        x = self.logmel(audio)
        x = self.norm(x) if self.norm else x # batch, n_mels, time
        batch_size, _n_mels, n_frames  = x.shape
        x = self.convmodel(x.unsqueeze(1)) #batch, channel=1, n_mels, time
        x = x.flatten(1, -1)
        out = self.mlp(x)
        out = self.out(out).unsqueeze(1) # (batch, 1, n_params)
        return out

class FrameMelConvEstimator(Estimator):
    """
    Frame by Frame Parameters
    """
    def __init__(self, output_dims, n_mels=128, channels=64, kernel_size=7, strides=[2,2,2,2], n_fft=1024, hop=256, norm='batch', sr=16000, f0_encoder=None, noise_prob=0.0, noise_mag=0.0):
        super().__init__(f0_encoder, noise_prob, noise_mag)
    
        self.n_mels = n_mels
        self.channels = channels
        self.logmel = nn.Sequential(MelSpec(n_fft=n_fft, hop_length=hop, n_mels=n_mels, sample_rate=sr), LogTransform())
        self.norm = Normalize2d(norm) if norm else None
        # Regular Conv
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(1, channels, kernel_size,
                        padding=kernel_size // 2,
                        stride=strides[0]), nn.BatchNorm1d(channels), nn.ReLU())]
            + [nn.Sequential(nn.Conv1d(channels, channels, kernel_size,
                         padding=kernel_size // 2,
                         stride=strides[i]), nn.BatchNorm1d(channels), nn.ReLU())
                         for i in range(1, len(strides) - 1)]
            + [nn.Sequential(nn.Conv1d(channels, channels, kernel_size,
                         padding=kernel_size // 2,
                         stride=strides[-1]))])
        #  CoordConv
        # self.convs = nn.ModuleList(
        #     [nn.Sequential(CoordConv1D(1, channels, n_mels, kernel_size=1,
        #                 padding=0,
        #                 stride=1))]
        #     + [nn.Sequential(nn.Conv1d(channels, channels, kernel_size,
        #                  padding=kernel_size // 2,
        #                  stride=strides[i]), nn.BatchNorm1d(channels), nn.ReLU())
        #                  for i in range(0, len(strides) - 1)]
        #     + [nn.Sequential(nn.Conv1d(channels, channels, kernel_size,
        #                  padding=kernel_size // 2,
        #                  stride=strides[-1]))])
        self.l_out = self.get_downsampled_length()[-1] # downsampled in frequency dimension
        self.mlp = MLP(self.l_out * channels, channels, loop=2)
        self.out = nn.Linear(channels, output_dims)    

    def compute_params(self, conditioning):
        audio = conditioning['audio']
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
        output = self.mlp(x)
        output = self.out(output)
        # output: [batch_size, n_frames, latent_size]
        return output

    def get_downsampled_length(self):
        l = self.n_mels
        lengths = [l]
        for conv in self.convs:
            conv_module = conv[0]
            l = (l + 2 * conv_module.padding[0] - conv_module.dilation[0] * (conv_module.kernel_size[0] - 1) - 1) // conv_module.stride[0] + 1
            lengths.append(l)
        return lengths

class FrameSpecConvEstimator(Estimator):
    """
    Use regular spectrograms
    Frame by Frame Parameters
    """
    def __init__(self, output_dims, channels=32, kernel_size=7, strides=[2,2,2,2], n_fft=1024, hop=256, norm='batch', sr=16000, f0_encoder=None, noise_prob=0.0, noise_mag=0.0):
        super().__init__(f0_encoder, noise_prob, noise_mag)
    
        self.channels = channels
        self.log_spec = nn.Sequential(Spec(n_fft=n_fft, hop_length=hop), LogTransform())
        self.norm = Normalize2d(norm) if norm else None
        self.fft_dims = n_fft // 2 + 1
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
        self.mlp = MLP(self.l_out * channels, channels, loop=2)
        self.out = nn.Linear(channels, output_dims)

    def compute_params(self, conditioning):
        audio = conditioning['audio']
        x = self.log_spec(audio)
        x = self.norm(x)
        batch_size, fft_dims, n_frames = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, self.fft_dims).unsqueeze(1)
        # x: [batch_size*n_frames, 1, n_mels]
        for i, conv in enumerate(self.convs):
            x = conv(x)
        x = x.view(batch_size, n_frames, self.channels, self.l_out)
        x = x.view(batch_size, n_frames, -1)
        output = self.mlp(x)
        output = self.out(output)
        # output: [batch_size, n_frames, latent_size]
        return output

    def get_downsampled_length(self):
        l = self.fft_dims
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

class FrameDilatedConvEstimator(Estimator):
    """
    Frame by Frame Parameters
    Similar to Jukebox
    """
    def __init__(self, output_dims, frame_setting='finer', res_depth=4, channels=32, dilation_growth_rate=3, m_conv=1.0, f0_encoder=None, noise_prob=0.0, noise_mag=0.0):
        """
        Args:
            output_dims (int): output channels
            res_depth (int, optional): depth of each resnet. Defaults to 4.
            channels (int, optional): conv channels. Defaults to 32.
            dilation_growth_rate (int, optional): exponential growth of dilation. Defaults to 3.
            m_conv (float, optional): multiplier for resnet channels. Defaults to 1.0.
        """
        super().__init__(f0_encoder, noise_prob, noise_mag)
        self.n_downsample, self.stride = frame_setting_stride[frame_setting]
        self.output_dims = output_dims
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
        self.out = nn.Linear(channels, output_dims)   

    def get_z_frames(self, n_samples):
        n_frames = n_samples // (self.stride ** self.n_downsample)
        return n_frames

    def compute_params(self, conditioning):
        x = conditioning['audio']
        batch_size, n_samples = x.shape
        x = x.unsqueeze(1)
        x = self.model(x) # (batch, channels, n_frames)
        x = x.permute(0, 2, 1)
        x = self.out(x)
        assert x.shape == (batch_size, self.get_z_frames(n_samples), self.output_dims)
        return x