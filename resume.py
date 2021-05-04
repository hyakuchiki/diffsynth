import os, tqdm, glob, argparse, json
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset, DataLoader, Dataset, ConcatDataset, random_split
from torch.utils.tensorboard import SummaryWriter

from diffsynth.estimator import MFCCEstimator, MelEstimator
from diffsynth.model import NoParamEstimatorSynth
from diffsynth.loss import SpecWaveLoss
from diffsynth.modelutils import construct_synths
from trainutils import save_to_board
from diffsynth.perceptual.ae import get_wave_ae
from diffsynth.perceptual.melae import get_mel_ae
from trainae import BasicWaveDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=str,   help='')
    parser.add_argument('load_dir',   type=str,   help='')
    parser.add_argument('dataset',      type=str,   help='directory of dataset')

    parser.add_argument('--subset',     type=int,   default=None,           help='')
    parser.add_argument('--epochs',     type=int,   default=400,    help='directory of dataset')
    parser.add_argument('--batch_size', type=int,   default=128,     help='directory of dataset')
    parser.add_argument('--lr',         type=float, default=1e-3,   help='directory of dataset')

    parser.add_argument('--fft_sizes',        type=int,   default=[32, 64, 128, 256, 512, 1024], nargs='*', help='')
    parser.add_argument('--hop_lengths',      type=int,   default=None, nargs='*', help='')
    parser.add_argument('--win_lengths',      type=int,   default=None, nargs='*', help='')

    parser.add_argument('--mag_w',          type=float, default=1.0,            help='')
    parser.add_argument('--log_mag_w',      type=float, default=1.0,            help='')
    parser.add_argument('--l1_w',           type=float, default=0.0,            help='')
    parser.add_argument('--l2_w',           type=float, default=0.0,            help='')
    parser.add_argument('--linf_w',         type=float, default=0.0,            help='')

    parser.add_argument('--noise_prob',     type=float, default=0.0,            help='')
    parser.add_argument('--noise_mag',      type=float, default=0.1,            help='')

    parser.add_argument('--patience',       type=int,   default=10,             help='')
    parser.add_argument('--plot_interval',  type=int,   default=10,             help='')
    parser.add_argument('--nbworkers',      type=int,   default=4,              help='')
    args = parser.parse_args()

    device = 'cuda'

    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=args.output_dir, purge_step=0)
    writer.add_text('args', str(args.__dict__))
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    pre_args_file = os.path.join(args.load_dir, 'args.txt')
    with open(pre_args_file) as f:
        pre_args = json.load(f)
        pre_args = SimpleNamespace(**pre_args)

    # load OoD data
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
    
    synth = construct_synths(pre_args.synth)
    if pre_args.estimator == 'mfccgru':
        estimator = MFCCEstimator(synth.ext_param_size, noise_prob=pre_args.noise_prob, noise_mag=pre_args.noise_mag).to(device)
    elif pre_args.estimator == 'melgru':
        estimator = MelEstimator(synth.ext_param_size, noise_prob=pre_args.noise_prob, noise_mag=pre_args.noise_mag).to(device)
    model = NoParamEstimatorSynth(estimator, synth)
    model.load_state_dict(torch.load(os.path.join(args.load_dir, 'model/state_dict.pth')))
    model = model.to(device)

    recon_loss = SpecWaveLoss(args.fft_sizes, args.hop_lengths, args.win_lengths, mag_w=args.mag_w, log_mag_w=args.log_mag_w, l1_w=args.l1_w, l2_w=args.l2_w, linf_w=args.linf_w, norm=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True, threshold=1e-5)


    with torch.no_grad():
        # save_batch(testbatch['audio'], resyn_audio, i+1, plot_dir, audio_dir)
        resyn_audio, _output = model(testbatch)
        save_to_board(0, writer, testbatch['audio'], resyn_audio, 8)

    best_loss = np.inf
    for i in tqdm.tqdm(range(1, args.epochs+1)):
        train_loss = model.train_epoch(loader=dl_train, recon_loss=recon_loss, optimizer=optimizer, device=device, param_loss_w=0.0)
        valid_losses = model.eval_epoch(loader=dl_valid, recon_loss=recon_loss, device=device)
        tqdm.tqdm.write('Epoch: {0:03} Train: {1:.4f} Valid: {2:.4f}'.format(i, train_loss, valid_losses['spec']))
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], i)
        writer.add_scalar('train/loss', train_loss, i)
        for k in valid_losses:
            writer.add_scalar('valid/'+k, valid_losses[k], i)
        if valid_losses['spec'] < best_loss:
            best_loss = valid_losses['spec']
            torch.save(model.state_dict, os.path.join(model_dir, 'state_dict.pth'))
        if (i + 1) % args.plot_interval == 0:
            # plot spectrograms
            model.eval()
            with torch.no_grad():
                # save_batch(testbatch['audio'], resyn_audio, i+1, plot_dir, audio_dir)
                resyn_audio, _output = model(testbatch)
                save_to_board(i, writer, testbatch['audio'], resyn_audio, 8)