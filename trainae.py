import os, glob, argparse, json
import tqdm
import librosa
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader, Dataset, ConcatDataset, random_split
from torch.utils.tensorboard import SummaryWriter

from diffsynth.loss import SpecWaveLoss
from trainutils import save_to_board, save_to_board_mel
from train import BatchPTDataset
from diffsynth.perceptual.ae import get_wave_ae
from diffsynth.perceptual.melae import get_mel_ae, MelAE

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
    parser.add_argument('--data_type',    type=str,   default='wave',             help='')
    parser.add_argument('--latent_size',    type=int,   default=8,             help='')
    parser.add_argument('--encoder_dims',   type=int,   default=64,             help='')
    # wave encoder
    parser.add_argument('--waveae',         action='store_true', help='waveform ae')
    parser.add_argument('--z_steps',        type=str,   default='finer',         help='')
    parser.add_argument('--res_depth',  type=int,   default=3,             help='')
    parser.add_argument('--channels',   type=int,   default=32,             help='')
    parser.add_argument('--dil_rate',   type=int,   default=3,              help='')

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
    
    # load waveform dataset
    if args.data_type == 'wave':
        dset = BasicWaveDataset(args.dataset, sample_rate=16000, length=1.024)
    elif args.data_type == 'pt':
        dset = BatchPTDataset(args.dataset, params=False)
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
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    if args.waveae:
        model = get_wave_ae(args.z_steps, args.encoder_dims, args.latent_size, args.res_depth, args.channels, args.dil_rate, testbatch['audio'].shape[-1]).to(device)
    else:
        model = get_mel_ae(args.encoder_dims, args.latent_size, testbatch['audio'].shape[-1]).to(device)
    
    # not used for mel
    recon_loss = SpecWaveLoss(args.fft_sizes, args.hop_lengths, args.win_lengths, mag_w=args.mag_w, log_mag_w=args.log_mag_w, l1_w=args.l1_w, l2_w=args.l2_w, linf_w=args.linf_w, norm=None)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True, threshold=1e-7)

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
            torch.save(model.state_dict(), os.path.join(model_dir, 'state_dict.pth'))

        if (i+1)%args.plot_interval == 0:
            model.eval()
            with torch.no_grad():
                if isinstance(model, MelAE):
                    mel = model.get_transform(testbatch['audio'])
                    recon_mel = model(mel)
                    save_to_board_mel(i, writer, mel, recon_mel, 8)
                else:
                    resyn_audio = model(testbatch)
                    save_to_board(i, writer, testbatch['audio'], resyn_audio, 8)
