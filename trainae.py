import os, glob, argparse, json
import tqdm
import librosa
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from diffsynth.loss import SpecWaveLoss
from trainutils import save_to_board, save_to_board_mel, get_loaders, WaveParamDataset
from diffsynth.perceptual.ae import get_wave_ae
from diffsynth.perceptual.melae import get_mel_ae, MelAE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir',   type=str,   help='')
    parser.add_argument('dataset',      type=str,   help='directory of dataset')
    # AE params
    parser.add_argument('--latent_size',    type=int,   default=8,             help='')
    parser.add_argument('--encoder_dims',   type=int,   default=64,             help='')
    # wave encoder
    parser.add_argument('--waveae',         action='store_true', help='waveform ae')
    parser.add_argument('--z_steps',        type=str,   default='finer',         help='')
    parser.add_argument('--res_depth',  type=int,   default=3,             help='')
    parser.add_argument('--channels',   type=int,   default=32,             help='')
    parser.add_argument('--dil_rate',   type=int,   default=3,              help='')

    # loss weights
    # fft loss if waveae
    parser.add_argument('--mag_w',          type=float, default=1.0,            help='')
    parser.add_argument('--log_mag_w',      type=float, default=1.0,            help='')
    parser.add_argument('--l1_w',           type=float, default=1.0,            help='')
    parser.add_argument('--l2_w',           type=float, default=0.0,            help='')
    parser.add_argument('--linf_w',         type=float, default=0.0,            help='')
    parser.add_argument('--fft_sizes',      type=int,   default=[64, 128, 256, 512, 1024, 2048], nargs='*', help='')
    parser.add_argument('--hop_lengths',      type=int,   default=None, nargs='*', help='')
    parser.add_argument('--win_lengths',      type=int,   default=None, nargs='*', help='')
    # optim. arguments
    parser.add_argument('--batch_size',     type=int,   default=128,             help='')
    parser.add_argument('--epochs',         type=int,   default=400,            help='')
    parser.add_argument('--clip',           type=float, default=1.0,            help='')
    parser.add_argument('--lr',             type=float, default=1e-5,           help='')
    parser.add_argument('--patience',       type=int,   default=20,             help='')
    parser.add_argument('--subset',         type=int,   default=None,           help='')
    parser.add_argument('--plot_interval',  type=int,   default=10,             help='')

    parser.add_argument('--nbworkers',      type=int,   default=4,              help='')
    args = parser.parse_args()

    device = 'cuda'
    
    # load waveform dataset
    dset = WaveParamDataset(args.dataset, sample_rate=16000, length=4.0, params=False)
    dsets, loaders = get_loaders(dset, args.batch_size, subset_train=args.subset, splits=[.8, .1, .1], nbworkers=args.nbworkers)
    dset_train, dset_valid, dset_test = dsets
    train_loader, valid_loader, test_loader = loaders

    testbatch = next(iter(valid_loader))
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
    for i in tqdm.tqdm(range(1, args.epochs+1)):
        train_loss = model.train_epoch(loader=train_loader, recon_loss=recon_loss, optimizer=optimizer, device=device)
        valid_loss = model.eval_epoch(loader=valid_loader, recon_loss=recon_loss, device=device)

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], i)
        writer.add_scalar('train/loss', train_loss, i)
        writer.add_scalar('valid/loss', valid_loss, i)

        scheduler.step(valid_loss)
        tqdm.tqdm.write('Epoch: {0:03} Train: {1:.4f} Valid: {2:.4f}'.format(i, train_loss, valid_loss))

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(model_dir, 'state_dict.pth'))

        if i % args.plot_interval == 0:
            model.eval()
            with torch.no_grad():
                if isinstance(model, MelAE):
                    mel = model.get_transform(testbatch['audio'])
                    recon_mel = model(mel)
                    save_to_board_mel(i, writer, mel, recon_mel, 8)
                else:
                    resyn_audio = model(testbatch)
                    save_to_board(i, 'audio', writer, testbatch['audio'], resyn_audio, 8)
