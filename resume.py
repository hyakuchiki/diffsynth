import os, tqdm, glob, argparse, json
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset, DataLoader, Dataset, ConcatDataset, random_split
from torch.utils.tensorboard import SummaryWriter

from diffsynth.estimator import MFCCEstimator, MelEstimator
from diffsynth.model import NoParamEstimatorSynth, EstimatorSynth
from diffsynth.loss import SpecWaveLoss
from diffsynth.modelutils import construct_synths
from trainutils import save_to_board
from diffsynth.perceptual.ae import get_wave_ae
from diffsynth.perceptual.melae import get_mel_ae
from train import WaveParamDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=str,   help='')
    parser.add_argument('load_dir',   type=str,   help='')

    parser.add_argument('--train_data', type=str,   default='real', help='synth, real, or mix')

    parser.add_argument('--epochs',     type=int,   default=400,    help='')
    parser.add_argument('--batch_size', type=int,   default=64,     help='')
    parser.add_argument('--lr',         type=float, default=1e-4,   help='')
    parser.add_argument('--decay_rate',     type=float, default=1.0,            help='')
    # Multiscale fft params
    parser.add_argument('--fft_sizes',        type=int,   default=[32, 64, 128, 256, 512, 1024], nargs='*', help='')
    parser.add_argument('--hop_lengths',      type=int,   default=None, nargs='*', help='')
    parser.add_argument('--win_lengths',      type=int,   default=None, nargs='*', help='')
    # spectral loss weights
    parser.add_argument('--mag_w',          type=float, default=1.0,            help='')
    parser.add_argument('--log_mag_w',      type=float, default=1.0,            help='')
    parser.add_argument('--l1_w',           type=float, default=0.0,            help='')
    parser.add_argument('--l2_w',           type=float, default=0.0,            help='')
    parser.add_argument('--linf_w',         type=float, default=0.0,            help='')
    # param loss weight
    parser.add_argument('--p_w',            type=float, default=0.0,            help='')

    parser.add_argument('--noise_prob',     type=float, default=0.0,            help='')
    parser.add_argument('--noise_mag',      type=float, default=0.1,            help='')

    parser.add_argument('--plot_interval',  type=int,   default=10,             help='')
    parser.add_argument('--nbworkers',      type=int,   default=4,              help='')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(seed=0) # subset

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
    
    # load dataloaders
    datasets_file = os.path.join(args.load_dir, 'datasets.pt')

    syn_dset_train, syn_dset_valid, syn_dset_test, real_dset_train, real_dset_valid, real_dset_test = torch.load(datasets_file)
    assert len(syn_dset_train) == len(real_dset_train)

    if args.train_data == 'real':
        train_loader = DataLoader(real_dset_train, shuffle=True, batch_size=args.batch_size, num_workers=4)
    elif args.train_data == 'synth':
        train_loader = DataLoader(syn_dset_train, shuffle=True, batch_size=args.batch_size, num_workers=4)
    elif args.train_data == 'mix':
        indices = np.random.choice(len(syn_dset_train)+len(real_dset_train), len(syn_dset_train), replace=False)
        # Mix each roughly evenly
        train_dset = Subset(ConcatDataset([syn_dset_train, real_dset_train]), indices)
        train_loader = DataLoader(train_dset, shuffle=True, batch_size=args.batch_size, num_workers=4)

    syn_valid_loader = DataLoader(syn_dset_valid, batch_size=args.batch_size, num_workers=4)
    syn_test_loader = DataLoader(syn_dset_test, batch_size=args.batch_size, num_workers=4)

    real_valid_loader = DataLoader(real_dset_valid, batch_size=args.batch_size, num_workers=4)
    real_test_loader = DataLoader(real_dset_test, batch_size=args.batch_size, num_workers=4)

    syn_testbatch = next(iter(syn_valid_loader))
    syn_testbatch.pop('params')
    syn_testbatch = {name:tensor.to(device) for name, tensor in syn_testbatch.items()}

    real_testbatch = next(iter(real_valid_loader))
    real_testbatch = {name:tensor.to(device) for name, tensor in real_testbatch.items()}

    synth = construct_synths(pre_args.synth)
    if pre_args.estimator == 'mfccgru':
        estimator = MFCCEstimator(synth.ext_param_size, noise_prob=pre_args.noise_prob, noise_mag=pre_args.noise_mag)
    elif pre_args.estimator == 'melgru':
        estimator = MelEstimator(synth.ext_param_size, noise_prob=pre_args.noise_prob, noise_mag=pre_args.noise_mag)
    
    if args.p_w > 0:
        assert args.train_data == 'synth'
        model = EstimatorSynth(estimator, synth)
    else:
        model = NoParamEstimatorSynth(estimator, synth)

    model.load_state_dict(torch.load(os.path.join(args.load_dir, 'model/statedict_200.pth')))
    model = model.to(device)

    recon_loss = SpecWaveLoss(args.fft_sizes, args.hop_lengths, args.win_lengths, mag_w=args.mag_w, log_mag_w=args.log_mag_w, l1_w=args.l1_w, l2_w=args.l2_w, linf_w=args.linf_w, norm=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)

    # initial state (epoch=0)
    with torch.no_grad():
        model.eval()
        resyn_audio, _output = model(syn_testbatch)
        save_to_board(0, 'syn', writer, syn_testbatch['audio'], resyn_audio, 8)
        resyn_audio, _output = model(real_testbatch)
        save_to_board(0, 'real', writer, real_testbatch['audio'], resyn_audio, 8)
        valid_losses = model.eval_epoch(syn_loader=syn_valid_loader, real_loader=real_valid_loader, recon_loss=recon_loss, device=device)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], 0)
        for k in valid_losses:
            writer.add_scalar('valid/'+k, valid_losses[k], 0)
        print('Initial stats: real/lsd: {0:.4f}'.format(valid_losses['real/lsd']))


    best_loss = np.inf
    monitor='real/lsd'
    for i in tqdm.tqdm(range(1, args.epochs+1)):
        train_loss = model.train_epoch(loader=train_loader, recon_loss=recon_loss, optimizer=optimizer, device=device, param_loss_w=args.p_w)
        valid_losses = model.eval_epoch(syn_loader=syn_valid_loader, real_loader=real_valid_loader, recon_loss=recon_loss, device=device)
        tqdm.tqdm.write('Epoch: {0:03} Train: {1:.4f} Valid: {2:.4f}'.format(i, train_loss, valid_losses[monitor]))
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], i)
        writer.add_scalar('train/loss', train_loss, i)
        scheduler.step()
        for k in valid_losses:
            writer.add_scalar('valid/'+k, valid_losses[k], i)
        if valid_losses[monitor] < best_loss:
            best_loss = valid_losses[monitor]
            torch.save(model.state_dict, os.path.join(model_dir, 'state_dict.pth'))
        if i % args.plot_interval == 0:
            # plot spectrograms
            model.eval()
            with torch.no_grad():
                resyn_audio, _output = model(syn_testbatch)
                save_to_board(i, 'syn', writer, syn_testbatch['audio'], resyn_audio, 8)
                resyn_audio, _output = model(real_testbatch)
                save_to_board(i, 'real', writer, real_testbatch['audio'], resyn_audio, 8)