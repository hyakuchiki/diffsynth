import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffsynth.loss import SpecWaveLoss
from diffsynth.spectral import MelSpec
from diffsynth.layers import Resnet2D, MLP, Normalize2d
from diffsynth.transforms import LogTransform


def get_mel_ae(encoder_dims, latent_size, n_samples, n_mels=128, n_fft=2048, hop=512):
    spec_len = math.ceil((n_samples - n_fft) / hop) + 1
    spec_len_target = 2**(int(math.log2(spec_len))+1)
    mel_size = [n_mels, spec_len_target]
    encoder = MelEncoder(encoder_dims, mel_size)
    decoder = MelDecoder(latent_size, encoder.final_size)
    # encoder = MelConvEncoder(mel_size, encoder_dims, n_layers=4)
    # decoder = MelConvDecoder(latent_size, encoder.final_size, mel_size, n_layers=4)
    return MelAE(encoder, decoder, encoder_dims, latent_size, n_samples, n_mels, n_fft, hop)

class MelAE(nn.Module):
    def __init__(self, encoder, decoder, encoder_dims, latent_size, n_samples, n_mels=128, n_fft=2048, hop=512, sr=16000, norm='batch'):
        super().__init__()
        self.logmel = nn.Sequential(MelSpec(n_fft=n_fft, hop_length=hop, n_mels=n_mels, sample_rate=sr), LogTransform())
        self.norm = Normalize2d(norm) if norm else None
        # for n_samples = 16384 and n_fft=2048, hop=512, this is 29
        spec_len = math.ceil((n_samples - n_fft) / hop) + 1
        self.spec_len_target = 2**(int(math.log2(spec_len))+1)

        self.encoder = encoder
        self.decoder = decoder
        self.latent_size = latent_size
        self.encoder_dims = encoder_dims
        self.map_latent = nn.Linear(encoder_dims, latent_size)
        self.apply(self.init_parameters)
    
    def init_parameters(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.uniform_(-0.01, 0.01)
            m.bias.data.fill_(0.01)

    def get_transform(self, data_dict):
        spec = self.logmel(data_dict['audio'])
        if self.norm is not None:
            spec = self.norm(spec)
        # (batch, n_mels, time)
        batch_size, n_mels, n_frames = spec.shape
        padded_spec = F.pad(spec, (0, self.spec_len_target-n_frames))
        return padded_spec

    def forward(self, mel):
        # Encode the inputs
        encoder_output = self.encoder(mel)
        # regularization if vae
        z_tilde = self.map_latent(encoder_output)
        # Decode the samples to get synthesis parameters
        recon_mel = self.decoder(z_tilde)
        return recon_mel

    def train_epoch(self, loader, recon_loss, optimizer, device, clip=1.0):
        self.train()
        sum_loss = 0
        for data_dict in loader:
            data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
            mel = self.get_transform(data_dict)
            recon_mel = self(mel)
            # Reconstruction loss
            batch_loss = F.l1_loss(recon_mel, mel)
            # Perform backward
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
            optimizer.step()
            sum_loss += batch_loss.detach().item()
        sum_loss /= len(loader)
        return sum_loss
    
    def eval_epoch(self, loader, recon_loss, device):
        self.eval()
        sum_loss = 0
        with torch.no_grad():
            for data_dict in loader:
                data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
                mel = self.get_transform(data_dict)
                recon_mel = self(mel)
                batch_loss = F.mse_loss(recon_mel, mel)
                sum_loss += batch_loss.detach().item()
        sum_loss /= len(loader)
        return sum_loss

class MelEncoder(nn.Module):
    def __init__(self, encoder_dims, input_size, n_downsample=3, stride=(2,2), res_depth=3, channels=32, dilation_growth_rate=2, m_conv=1.0):
        super().__init__()

        final_size = input_size
        kernel_size = [s*2 for s in stride]
        pad = [s//2 for s in stride]
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

        self.final_size = final_size
        # output:(batch, channels, final_size[0], final_size[1])
        self.convmodel = nn.Sequential(*blocks)
        print('output dims after convolution', final_size)

        self.mlp = MLP(final_size[0] * final_size[1] * channels, 256)
        self.out = nn.Linear(256, encoder_dims)

    def forward(self, mel):
        mel = mel.unsqueeze(1)
        x = self.convmodel(mel) #batch, channels, final_size[0], final_size[1]
        x = x.flatten(1, -1)
        out = self.mlp(x)
        out = self.out(out) # (batch, n_params)
        return out

class MelDecoder(nn.Module):
    def __init__(self, latent_size, final_size, n_downsample=3, stride=(2,2), res_depth=3, channels=32, dilation_growth_rate=2, m_conv=1.0):
        super().__init__()
        # final_size : CNN
        self.out = nn.Linear(latent_size, 256)
        self.mlp = MLP(256, final_size[0] * final_size[1] * channels)

        self.channels = channels
        self.final_size = final_size
        blocks = []
        kernel_size = [s*2 for s in stride]
        pad = [s//2 for s in stride]
        out_size = final_size
        for i in range(n_downsample):
            block = nn.Sequential(
                # ResNet with growing dilation, doesn't change size
                Resnet2D(channels, res_depth, m_conv, dilation_growth_rate, reverse_dilation=True),
                # upsampling conv, output size is L_in*stride
                nn.ConvTranspose2d(channels, 1 if i == n_downsample-1 else channels, kernel_size, stride, pad),
            )
            blocks.append(block)
            out_size = (out_size[0] * stride[0], out_size[1] * stride[1])

        # output:(batch, channels, final_size[0], final_size[1])
        self.convmodel = nn.Sequential(*blocks)
        print('reconstructed size', out_size)

    def forward(self, z):
        x = self.out(z)
        x = self.mlp(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, self.channels, self.final_size[0], self.final_size[1])
        x = self.convmodel(x)
        x = x.squeeze(1)
        return x

# recipe from flowsynth...
class MelConvEncoder(nn.Module):
    def __init__(self, in_size, encoder_dims, kernel=5, channels=32, n_layers = 5, hidden_size = 512, n_mlp = 2):
        super().__init__()
        size = in_size.copy()
        modules = nn.Sequential()
        stride = 2
        for l in range(n_layers):
            dil = 2 ** l
            pad = 3 * (dil + 1)
            in_s = 1 if l==0 else channels
            out_s = 1 if l==(n_layers - 1) else channels
            modules.add_module('conv%i'%l, nn.Conv2d(in_s, out_s, kernel, stride, pad, dil))
            if (l < n_layers-1):
                modules.add_module('bn%i'%l, nn.BatchNorm2d(out_s))
                modules.add_module('relu%i'%l, nn.ReLU())
                modules.add_module('drop%i'%l, nn.Dropout2d(p=.25))
            size[0] = int((size[0]+2*pad-(dil*(kernel-1)+1))/stride+1)
            size[1] = int((size[1]+2*pad-(dil*(kernel-1)+1))/stride+1)
        
        self.final_size = size
        self.cnn = modules
        self.mlp = nn.Sequential()
        for l in range(n_mlp):
            in_s = (l==0) and (size[0] * size[1]) or hidden_size
            out_s = (l == n_mlp - 1) and encoder_dims or hidden_size
            self.mlp.add_module('h%i'%l, nn.Linear(in_s, out_s))
            if (l < n_mlp - 1):
                self.mlp.add_module('b%i'%l, nn.BatchNorm1d(out_s))
                self.mlp.add_module('a%i'%l, nn.ReLU())
                self.mlp.add_module('d%i'%l, nn.Dropout(p=.25))
    
    def forward(self, mel):
        mel = mel.unsqueeze(1)
        x = self.cnn(mel) #batch, channels, final_size[0], final_size[1]
        x = x.flatten(1, -1)
        out = self.mlp(x)
        return out

class MelConvDecoder(nn.Module):
    def __init__(self, latent_size, enc_final_size, out_size, kernel=5, channels = 32, n_layers = 5, hidden_size = 512, n_mlp = 2):
        super().__init__()
        # Create modules
        self.enc_final_size = enc_final_size
        self.out_size = out_size
        size = enc_final_size.copy()
        stride = 2
        self.mlp = nn.Sequential()
        """ First go through MLP """
        for l in range(n_mlp):
            in_s = (l==0) and (latent_size) or hidden_size
            out_s = (l == n_mlp - 1) and enc_final_size[0]*enc_final_size[1] or hidden_size
            self.mlp.add_module('h%i'%l, nn.Linear(in_s, out_s))
            if (l < n_layers - 1):
                self.mlp.add_module('b%i'%l, nn.BatchNorm1d(out_s))
                self.mlp.add_module('a%i'%l, nn.ReLU())
                self.mlp.add_module('d%i'%l, nn.Dropout(p=.25))
        modules = nn.Sequential()
        """ Then do a CNN """
        for l in range(n_layers):
            dil = 2 ** ((n_layers - 1) - l)
            pad = 3 * (dil + 1)
            out_pad = (pad % 2)
            in_s = (l==0) and 1 or channels
            out_s = (l == n_layers - 1) and 1 or channels
            modules.add_module('c2%i'%l, nn.ConvTranspose2d(in_s, out_s, kernel, stride, pad, output_padding=out_pad, dilation = dil))
            if (l < n_layers - 1):
                modules.add_module('b2%i'%l, nn.BatchNorm2d(out_s))
                modules.add_module('a2%i'%l, nn.ReLU())
                modules.add_module('a2%i'%l, nn.Dropout2d(p=.25))
            size[0] = int((size[0] - 1) * stride - (2 * pad) + dil * (kernel - 1) + out_pad + 1)
            size[1] = int((size[1] - 1) * stride - (2 * pad) + dil * (kernel - 1) + out_pad + 1)
        self.cnn = modules
        self.dec_final_size = size

    def forward(self, z):
        out = self.mlp(z)
        out = out.view(-1, 1, self.enc_final_size[0], self.enc_final_size[1])
        out = self.cnn(out)
        out = out[:, :, :self.out_size[0], :self.out_size[1]].squeeze(1)
        return out