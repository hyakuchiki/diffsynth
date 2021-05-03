import torch
import torch.nn as nn
import torch.nn.functional as F

from diffsynth.layers import Resnet1D, MLP

frame_setting_stride = {
    # n_downsample, stride
    "coarse": (5, 4), # hop: 1024
    "fine": (9, 2), # 512, too deep?
    "finer": (4, 4), # 256
    "finest": (6, 2) # 64
}

def get_wave_ae(z_steps, encoder_dims, latent_size, res_depth, channels, dil_rate):
    encoder = DilatedConvEncoder(z_steps, encoder_dims, res_depth, channels, dil_rate)
    decoder = DilatedConvDecoder(z_steps, latent_size, res_depth, channels, dil_rate)
    model = AE(encoder, decoder, encoder_dims, latent_size)
    return model

class DilatedConvEncoder(nn.Module):
    """
    Doesn't use sliding windows
    Similar to Jukebox
    """
    def __init__(self, frame_setting, encoder_dims, res_depth=4, channels=32, dilation_growth_rate=3, m_conv=1.0):
        """
        Args:
            encoder_dims (int): output channels
            n_downsample (int, optional): times to downsample. Defaults to 4.
            stride (int, optional): downsampling rate / conv stride. Defaults to 4.
            res_depth (int, optional): depth of each resnet. Defaults to 4.
            channels (int, optional): conv channels. Defaults to 32.
            dilation_growth_rate (int, optional): exponential growth of dilation. Defaults to 3.
            m_conv (float, optional): multiplier for resnet channels. Defaults to 1.0.
        """
        super().__init__()
        self.n_downsample, self.stride = frame_setting_stride[frame_setting]
        self.encoder_dims = encoder_dims
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
        # doesn't change size
        block = nn.Conv1d(channels, encoder_dims, 3, 1, 1) # output:(batch, encoder_dims, n_frames)
        blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def get_z_frames(self, n_samples):
        n_frames = n_samples // (self.stride ** self.n_downsample)
        return n_frames

    def forward(self, x):
        """
        x: raw audio
        """
        batch_size, n_samples = x.shape
        x = x.unsqueeze(1)
        out = self.model(x) # (batch, encoder_dims, n_frames)
        out = out.permute(0, 2, 1)
        # assert x.shape == (batch_size, self.get_z_frames(n_samples), self.encoder_dims)
        return out

class DilatedConvDecoder(nn.Module):
    """
    Outputs raw audio
    Similar to Jukebox by OpenAI
    """
    def __init__(self, frame_setting, latent_dims, res_depth=4, channels=32, dilation_growth_rate=3, m_conv=1.0):
        """
        Args:
            latent_dims (int): input channels  
            res_depth (int, optional): depth of each resnet. Defaults to 4.
            channels (int, optional): conv channels. Defaults to 32.
            dilation_growth_rate (int, optional): exponential growth of dilation. Defaults to 3.
            m_conv (float, optional): multiplier for resnet channels. Defaults to 1.0.
        """
        super().__init__()
        self.n_downsample, self.stride = frame_setting_stride[frame_setting]
        self.latent_dims = latent_dims
        blocks = []
        kernel_size, pad = self.stride * 2, self.stride // 2
        # doesn't change size
        block = nn.Conv1d(latent_dims, channels, 3, 1, 1) # output:(batch, channels, n_frames)
        blocks.append(block)
        for i in range(self.n_downsample):
            block = nn.Sequential(
                # ResNet with growing dilation
                Resnet1D(channels, res_depth, m_conv, dilation_growth_rate, reverse_dilation=True),
                # upsampling deconv, output size is L_in*stride
                nn.ConvTranspose1d(channels, 1 if i == (self.n_downsample - 1) else channels, kernel_size, self.stride, pad),
            )
            blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, z):
        """doesnt use transforms
        """
        # batch, n_frames, latent_dims
        z = z.permute(0, 2, 1)
        resyn_audio = self.model(z) # (batch, 1, n_samples)
        resyn_audio = resyn_audio.squeeze(1)
        return resyn_audio

class AE(nn.Module):
    
    def __init__(self, encoder, decoder, encoder_dims, latent_dims):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dims = latent_dims
        self.encoder_dims = encoder_dims
        self.map_latent = nn.Linear(encoder_dims, latent_dims)
        self.apply(self.init_parameters)
    
    def init_parameters(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.uniform_(-0.01, 0.01)
            m.bias.data.fill_(0.01)

    def forward(self, data):
        audio = data['audio']
        # Encode the inputs
        encoder_output = self.encoder(audio)
        # regularization if vae
        z_tilde = self.map_latent(encoder_output)
        # Decode the samples to get synthesis parameters
        resyn_audio = self.decoder(z_tilde)
        return resyn_audio

    def encode_audio(self, audio):
        encoder_output = self.encoder(audio)
        z_tilde = self.map_latent(encoder_output)
        return z_tilde

    def encoding_loss(self, input_audio, target_audio):
        batch_size = input_audio.shape[0]
        audios = torch.cat([input_audio, target_audio], dim=0)
        encodings = self.encode_audio(audios)
        input_encoding = encodings[:batch_size]
        target_encoding = encodings[batch_size:]
        return F.l1_loss(input_encoding, target_encoding)

    def train_epoch(self, loader, recon_loss, optimizer, device, clip=1.0):
        self.train()
        sum_loss = 0
        for data_dict in loader:
            data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
            resyn_audio = self(data_dict)
            # Reconstruction loss
            spec_loss, wave_loss = recon_loss(data_dict['audio'], resyn_audio)
            batch_loss = spec_loss + wave_loss
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
                resyn_audio = self(data_dict)
                spec_loss, wave_loss = recon_loss(data_dict['audio'], resyn_audio)
                batch_loss = spec_loss + wave_loss
                sum_loss += batch_loss.detach().item()
        sum_loss /= len(loader)
        return sum_loss