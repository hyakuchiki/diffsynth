import torch
import torch.nn as nn

from diffsynth.util import midi_to_hz, hz_to_midi
from diffsynth.layers import Resnet1D, MLP

class Estimator(nn.Module):
    def __init__(self, f0_encoder=None):
        super().__init__()
        self.f0_encoder = f0_encoder

    def forward(self, conditioning):
        conditioning = self.fill_f0(conditioning)
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

class DilatedConvEstimator(Estimator):
    """
    Similar to Jukebox
    """
    def __init__(self, output_dims, input_dims, n_downsample=6, stride=4, res_depth=4, channels=32, dilation_growth_rate=3, m_conv=1.0, f0_encoder=None):
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
        super().__init__(f0_encoder)
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
        out = torch.sigmoid(self.out(out)).unsqueeze(1) # (batch, 1, n_frames)
        return out