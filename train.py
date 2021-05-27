import os, tqdm, glob, argparse, json
from types import SimpleNamespace
import numpy as np
import librosa
import torch
from torch.utils.tensorboard import SummaryWriter


from diffsynth.loss import SpecWaveLoss
from diffsynth.estimator import MFCCEstimator, MelEstimator
from diffsynth.model import EstimatorSynth
from diffsynth.modelutils import construct_synths
from trainutils import save_to_board, get_loaders, WaveParamDataset
from diffsynth.perceptual.ae import get_wave_ae
from diffsynth.perceptual.melae import get_mel_ae
from diffsynth.schedules import SCHEDULE_REGISTRY, ParamScheduler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir',   type=str,   help='')
    parser.add_argument('syn_dataset',      type=str,   help='directory of synthetic dataset')
    parser.add_argument('real_dataset',      type=str,   help='directory of real dataset')
    parser.add_argument('synth',        type=str,   help='synth name')
    parser.add_argument('--estimator',  type=str,   default='melgru', help='estimator name')
    parser.add_argument('--epochs',     type=int,   default=200,    help='directory of dataset')
    parser.add_argument('--batch_size', type=int,   default=64,     help='directory of dataset')
    parser.add_argument('--lr',         type=float, default=1e-3,   help='directory of dataset')
    parser.add_argument('--decay_rate',     type=float, default=1.0,            help='')
    # Multiscale fft params
    parser.add_argument('--fft_sizes',        type=int,   default=[32, 64, 128, 256, 512, 1024], nargs='*', help='')
    parser.add_argument('--hop_lengths',      type=int,   default=None, nargs='*', help='')
    parser.add_argument('--win_lengths',      type=int,   default=None, nargs='*', help='')
    # spectral loss weights
    parser.add_argument('--mag_w',          type=float, default=1.0,            help='')
    parser.add_argument('--log_mag_w',      type=float, default=1.0,            help='')
    # param loss weights
    parser.add_argument('--p_w',            type=float, default=10.0,            help='')
    # encoding loss
    parser.add_argument('--enc_w',          type=float, default=0.0,            help='')
    parser.add_argument('--ae_dir',         type=str,   default=None,            help='')
    # waveform loss weights (not used)
    parser.add_argument('--l1_w',           type=float, default=0.0,            help='')
    parser.add_argument('--l2_w',           type=float, default=0.0,            help='')
    parser.add_argument('--linf_w',         type=float, default=0.0,            help='')
    # weight schedule/annealing (ignores above values if specified)
    parser.add_argument('--loss_sched',     type=str,   default=None,           help='')

    parser.add_argument('--noise_prob',     type=float, default=0.0,            help='')
    parser.add_argument('--noise_mag',      type=float, default=0.1,            help='')

    parser.add_argument('--plot_interval',  type=int,   default=10,             help='')
    parser.add_argument('--nbworkers',      type=int,   default=4,              help='')
    args = parser.parse_args()

    torch.manual_seed(0) # just fixes the dataset/dataloader and initial weights
    np.random.seed(seed=0) # subset

    device = 'cuda'
    # output dir
    os.makedirs(args.output_dir, exist_ok=True)
    # audio_dir = os.path.join(args.output_dir, 'audio')
    # plot_dir = os.path.join(args.output_dir, 'plot')
    model_dir = os.path.join(args.output_dir, 'model')
    # os.makedirs(audio_dir, exist_ok=True)
    # os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=args.output_dir, purge_step=0)
    writer.add_text('args', str(args.__dict__))
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    # load synthetic dataset
    syn_dset = WaveParamDataset(args.syn_dataset, params=True)
    syn_dsets, syn_loaders = get_loaders(syn_dset, args.batch_size, splits=[.8, .1, .1], nbworkers=args.nbworkers)
    syn_dset_train, syn_dset_valid, syn_dset_test = syn_dsets
    syn_train_loader, syn_valid_loader, syn_test_loader = syn_loaders
 
    # load real (out-of-domain) dataset (nsynth, etc)
    # just for monitoring during train.py
    real_dset = WaveParamDataset(args.real_dataset, params=False)
    # the real dataset should be the same size as the synth. dataset
    real_dsets, real_loaders = get_loaders(real_dset, args.batch_size, subset_train=len(syn_dset_train), splits=[.8, .1, .1], nbworkers=args.nbworkers)
    real_dset_train, real_dset_valid, real_dset_test = real_dsets
    real_train_loader, real_valid_loader, real_test_loader = real_loaders
    assert len(syn_dset_train) == len(real_dset_train)

    syn_testbatch = next(iter(syn_valid_loader))
    syn_testbatch.pop('params')
    syn_testbatch = {name:tensor.to(device) for name, tensor in syn_testbatch.items()}

    real_testbatch = next(iter(real_valid_loader))
    real_testbatch = {name:tensor.to(device) for name, tensor in real_testbatch.items()}

    torch.save([syn_dset_train, syn_dset_valid, syn_dset_test, real_dset_train, real_dset_valid, real_dset_test], os.path.join(args.output_dir, 'datasets.pt'))

    # create model
    synth = construct_synths(args.synth)
    if args.estimator == 'mfccgru':
        estimator = MFCCEstimator(synth.ext_param_size, noise_prob=args.noise_prob, noise_mag=args.noise_mag).to(device)
    elif args.estimator == 'melgru':
        estimator = MelEstimator(synth.ext_param_size, noise_prob=args.noise_prob, noise_mag=args.noise_mag).to(device)
    
    model = EstimatorSynth(estimator, synth).to(device)

    # spectral loss (+waveform loss)
    recon_loss = SpecWaveLoss(args.fft_sizes, args.hop_lengths, args.win_lengths, mag_w=args.mag_w, log_mag_w=args.log_mag_w, l1_w=args.l1_w, l2_w=args.l2_w, linf_w=args.linf_w, norm=None)  
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)

    # encoding (perceptual) loss
    if args.ae_dir:
        args_file = os.path.join(args.ae_dir, 'args.txt')
        with open(args_file) as f:
            ae_args = json.load(f)
            ae_args = SimpleNamespace(**ae_args)
        ## construct ae model
        if ae_args.waveae:
            ae_model = get_wave_ae(ae_args.z_steps, ae_args.encoder_dims, ae_args.latent_size, ae_args.res_depth, ae_args.channels, ae_args.dil_rate)
        else:
            ae_model = get_mel_ae(ae_args.encoder_dims, ae_args.latent_size, syn_testbatch['audio'].shape[-1])
        ae_model.load_state_dict(torch.load(os.path.join(args.ae_dir, 'model/state_dict.pth')))
        ae_model = ae_model.eval().to(device)
    else:
        print('no autoencoder for perceptual loss')
        ae_model = None

    if args.loss_sched is not None:
        loss_mult_sched = ParamScheduler(SCHEDULE_REGISTRY[args.loss_sched])
    else:
        # multipliers for each loss during training
        mult = {'param': 1.0, 'recon': 1.0}
        loss_mult_sched = ParamScheduler(mult) # always 1.0 for both

    # initial state (epoch=0)
    with torch.no_grad():
        model.eval()
        resyn_audio, _output = model(syn_testbatch)
        save_to_board(0, 'syn', writer, syn_testbatch['audio'], resyn_audio, 8)
        resyn_audio, _output = model(real_testbatch)
        save_to_board(0, 'real', writer, real_testbatch['audio'], resyn_audio, 8)
        valid_losses = model.eval_epoch(syn_loader=syn_valid_loader, real_loader=real_valid_loader, recon_loss=recon_loss, device=device, ae_model=ae_model)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], 0)
        for k in valid_losses:
            writer.add_scalar('valid/'+k, valid_losses[k], 0)
    
    best_loss = np.inf
    monitor='real/lsd'
    for i in tqdm.tqdm(range(1, args.epochs+1)):
        loss_mult = loss_mult_sched.get_parameters(i)
        p_w = args.p_w * loss_mult['param']
        if 'enc' in loss_mult:
            enc_w = args.enc_w * loss_mult['enc']
        train_loss = model.train_epoch(loader=syn_train_loader, recon_loss=recon_loss, optimizer=optimizer, device=device, rec_mult=loss_mult['recon'], param_loss_w=p_w, enc_w=args.enc_w, ae_model=ae_model)
        valid_losses = model.eval_epoch(syn_loader=syn_valid_loader, real_loader=real_valid_loader, recon_loss=recon_loss, device=device, ae_model=ae_model)
        tqdm.tqdm.write('Epoch: {0:03} Train: {1:.4f} Valid: {2:.4f}'.format(i, train_loss, valid_losses[monitor]))
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], i)
        writer.add_scalar('train/loss', train_loss, i)
        scheduler.step()
        for k in valid_losses:
            writer.add_scalar('valid/'+k, valid_losses[k], i)
        if valid_losses[monitor] < best_loss:
            best_loss = valid_losses[monitor]
            torch.save(model.state_dict(), os.path.join(model_dir, 'state_dict.pth'))
        if i % args.plot_interval == 0:
            # plot spectrograms
            torch.save(model.state_dict(), os.path.join(model_dir, 'statedict_{0:03}.pth'.format(i)))
            model.eval()
            with torch.no_grad():
                resyn_audio, _output = model(syn_testbatch)
                save_to_board(i, 'syn', writer, syn_testbatch['audio'], resyn_audio, 8)
                resyn_audio, _output = model(real_testbatch)
                save_to_board(i, 'real', writer, real_testbatch['audio'], resyn_audio, 8)