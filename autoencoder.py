import os, glob, argparse
import tqdm
import librosa
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader, Dataset, ConcatDataset, random_split
from torch.utils.tensorboard import SummaryWriter

from diffsynth.layers import Resnet1D
from diffsynth.loss import SpecWaveLoss
from trainutils import save_to_board

frame_setting_stride = {
    # n_downsample, stride
    "coarse": (5, 4), # hop: 1024
    "fine": (9, 2), # 512, too deep?
    "finer": (4, 4), # 256
    "finest": (6, 2) # 64
}

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
        super(AE, self).__init__()
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

class BasicWaveDataset(Dataset):
    def __init__(self, base_dir, sample_rate=16000, length=1.0):
        self.base_dir = base_dir
        self.raw_files = sorted(glob.glob(os.path.join(base_dir, '*.wav')))
        print('loaded {0} files'.format(len(self.raw_files)))
        self.length = length
        self.sample_rate = sample_rate
    
    def __getitem__(self, idx):
        output = {}
        raw_path = self.raw_files[idx]
        output['audio'], _sr = librosa.load(raw_path, sr=self.sample_rate, duration=self.length)
        return output

    def __len__(self):
        return len(self.raw_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir',   type=str,   help='')
    parser.add_argument('dataset',      type=str,   help='directory of dataset')
    # AE params
    parser.add_argument('--latent_size',    type=int,   default=8,             help='')
    parser.add_argument('--z_steps',        type=str,   default='finer',         help='')
    parser.add_argument('--encoder_dims',   type=int,   default=64,             help='')
    # wave encoder
    parser.add_argument('--enc_res_depth',  type=int,   default=3,             help='')
    parser.add_argument('--enc_channels',   type=int,   default=32,             help='')
    parser.add_argument('--enc_dil_rate',   type=int,   default=3,              help='')

    # loss weights
    parser.add_argument('--mag_w',          type=float, default=1.0,            help='')
    parser.add_argument('--log_mag_w',      type=float, default=1.0,            help='')
    parser.add_argument('--l1_w',           type=float, default=1.0,            help='')
    parser.add_argument('--l2_w',           type=float, default=0.0,            help='')
    parser.add_argument('--linf_w',         type=float, default=0.0,            help='')
    # fft loss
    parser.add_argument('--fft_sizes',      type=int,   default=[64, 128, 256, 512, 1024, 2048], nargs='*', help='')
    parser.add_argument('--hop_lengths',      type=int,   default=None, nargs='*', help='')
    parser.add_argument('--win_lengths',      type=int,   default=None, nargs='*', help='')
    # optim. arguments
    parser.add_argument('--batch_size',     type=int,   default=128,             help='')
    parser.add_argument('--epochs',         type=int,   default=200,            help='')
    parser.add_argument('--clip',           type=float, default=1.0,            help='')
    parser.add_argument('--lr',             type=float, default=1e-3,           help='')
    parser.add_argument('--patience',       type=int,   default=10,             help='')
    parser.add_argument('--subset',         type=int,   default=None,           help='')
    parser.add_argument('--plot_interval',  type=int,   default=10,             help='')

    parser.add_argument('--nbworkers',      type=int,   default=4,              help='')
    args = parser.parse_args()

    device = 'cuda'

    # load nsynth dataset
    dset = BasicWaveDataset(args.dataset, sample_rate=16000, length=1.024)
    dset_l = len(dset)
    splits=[.8, .1, .1]
    split_sizes = [int(dset_l*splits[0]), int(dset_l*splits[1])]
    split_sizes.append(dset_l - split_sizes[0] - split_sizes[1])
    dset_train, dset_valid, dset_test = random_split(dset, lengths=split_sizes, generator=torch.Generator().manual_seed(0))

    if args.subset is not None:
        indices = np.random.choice(len(dset_train), args.subset, replace=False)
        dset_train = Subset(dset_train, indices)
    
    dl_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.nbworkers, pin_memory=True)
    dl_valid = DataLoader(dset_valid, batch_size=args.batch_size, num_workers=args.nbworkers, pin_memory=True)
    dl_test = DataLoader(dset_test, batch_size=args.batch_size, num_workers=args.nbworkers, pin_memory=True)
    testbatch = next(iter(dl_test))
    testbatch = {name:tensor.to(device) for name, tensor in testbatch.items()}

    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.output_dir, purge_step=0)
    writer.add_text('args', str(args.__dict__))

    encoder = DilatedConvEncoder(args.z_steps, args.encoder_dims, args.enc_res_depth, args.enc_channels, args.enc_dil_rate)
    decoder = DilatedConvDecoder(args.z_steps, args.latent_size, args.enc_res_depth, args.enc_channels, args.enc_dil_rate)
    model = AE(encoder, decoder, args.encoder_dims, args.latent_size).to(device)

    recon_loss = SpecWaveLoss(args.fft_sizes, args.hop_lengths, args.win_lengths, mag_w=args.mag_w, log_mag_w=args.log_mag_w, l1_w=args.l1_w, l2_w=args.l2_w, linf_w=args.linf_w, norm=None)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True, threshold=1e-5)

    best_loss = np.inf
    for i in tqdm.tqdm(range(args.epochs)):
        train_loss = model.train_epoch(loader=dl_train, recon_loss=recon_loss, optimizer=optimizer, device=device)
        valid_loss = model.eval_epoch(loader=dl_valid, recon_loss=recon_loss, device=device)

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], i)
        writer.add_scalar('train/loss', train_loss, i)
        writer.add_scalar('valid/loss', valid_loss, i)

        scheduler.step(valid_loss)
        tqdm.tqdm.write('Epoch: {0:03} Train: {1:.4f} Valid: {2:.4f}'.format(i, train_loss, valid_loss))

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model, os.path.join(model_dir, 'state_dict.pth'))

        if (i+1)%args.plot_interval == 0:
            model.eval()
            with torch.no_grad():
                resyn_audio = model(testbatch)
                save_to_board(i, writer, testbatch['audio'], resyn_audio, 8)
