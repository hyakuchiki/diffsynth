import os, tqdm, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter


from diffsynth.loss import SpecWaveLoss
from diffsynth.estimator import DilatedConvEstimator, MelConvEstimator
from diffsynth.model import EstimatorSynth
from diffsynth.modelutils import construct_synths
from trainutils import save_to_board

def train_epoch(model, loader, recon_loss, optimizer, device, clip=1.0):
    model.train()
    sum_loss = 0
    for data_dict in loader:
        data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
        resyn_audio = model(data_dict)
        # Reconstruction loss
        spec_loss, wave_loss = recon_loss(data_dict['audio'], resyn_audio)
        batch_loss = spec_loss + wave_loss
        # Perform backward
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        sum_loss += batch_loss.detach().item()
    sum_loss /= len(loader)
    return sum_loss

def eval_epoch(model, loader, recon_loss, device):
    model.eval()
    sum_loss = 0
    with torch.no_grad():
        for data_dict in loader:
            data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
            resyn_audio = model(data_dict)
            # Reconstruction loss
            spec_loss, wave_loss = recon_loss(data_dict['audio'], resyn_audio)
            batch_loss = spec_loss + wave_loss
            sum_loss += batch_loss.detach().item()
    sum_loss /= len(loader)
    return sum_loss

class BatchPTDataset(Dataset):
    def __init__(self, base_dir):
        self.audio_dir = os.path.join(base_dir, 'audio')
        self.audio_files = sorted(glob.glob(os.path.join(self.audio_dir, '*.pt')))
        self.last_idx = int(self.audio_files[-1].split('_')[-1][:-3])
        self.param_dir = os.path.join(base_dir, 'param')
        self.param_files = sorted(glob.glob(os.path.join(self.param_dir, '*.pt')))
    
    def __getitem__(self, idx):
        file_idx = idx // 64
        in_idx = idx % 64
        audios = torch.load(self.audio_files[file_idx])
        params = torch.load(self.param_files[file_idx])
        audio = audios[in_idx]
        
        audio = torch.nn.functional.pad(audio, (0, 16384 - audios.shape[1]))
        return {'audio': audio, 'params': params[in_idx]}

    def __len__(self):
        return self.last_idx+1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir',   type=str,   help='')
    parser.add_argument('dataset',      type=str,   help='directory of dataset')
    parser.add_argument('synth',        type=str,   help='synth name')
    parser.add_argument('--estimator',  type=str,   default='wave', help='estimator name')
    parser.add_argument('--epochs',     type=int,   default=400,    help='directory of dataset')
    parser.add_argument('--batch_size', type=int,   default=128,     help='directory of dataset')
    parser.add_argument('--lr',         type=float, default=1e-3,   help='directory of dataset')
    # loss
    parser.add_argument('--fft_sizes',        type=int,   default=[64, 128, 256, 512, 1024, 2048], nargs='*', help='')
    parser.add_argument('--hop_lengths',      type=int,   default=None, nargs='*', help='')
    parser.add_argument('--win_lengths',      type=int,   default=None, nargs='*', help='')
    parser.add_argument('--mag_w',          type=float, default=1.0,            help='')
    parser.add_argument('--log_mag_w',      type=float, default=1.0,            help='')
    parser.add_argument('--l1_w',           type=float, default=0.0,            help='')
    parser.add_argument('--l2_w',           type=float, default=0.0,            help='')
    parser.add_argument('--linf_w',         type=float, default=0.0,            help='')
    args = parser.parse_args()

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

    dset = BatchPTDataset(args.dataset)
    dset_len = len(dset)
    tr_len = int(0.8*dset_len)
    dset_train, dset_valid = random_split(dset, lengths=[tr_len, dset_len-tr_len], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(dset_train, batch_size=args.batch_size, num_workers=4)
    valid_loader = DataLoader(dset_valid, batch_size=args.batch_size, num_workers=4)
    # create model
    synth = construct_synths(args.synth)
    if args.estimator == 'wave':
        estimator = DilatedConvEstimator(synth.ext_param_size, 16384).to(device)
    elif args.estimator == 'melconv':
        estimator = MelConvEstimator(synth.ext_param_size, 16384).to(device)
    model = EstimatorSynth(estimator, synth).to(device)
    testbatch = next(iter(valid_loader))
    testbatch = {name:tensor.to(device) for name, tensor in testbatch.items()}
    
    recon_loss = SpecWaveLoss(args.fft_sizes, args.hop_lengths, args.win_lengths, mag_w=args.mag_w, log_mag_w=args.log_mag_w, l1_w=args.l1_w, l2_w=args.l2_w, linf_w=args.linf_w, norm=None)  
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True, threshold=1e-5)
    
    best_loss = np.inf
    for i in tqdm.tqdm(range(args.epochs)):
        train_loss = train_epoch(model, loader=train_loader, recon_loss=recon_loss, optimizer=optimizer, device=device)
        valid_loss = eval_epoch(model, loader=valid_loader, recon_loss=recon_loss, device=device)
        tqdm.tqdm.write('Epoch: {0:03} Train: {1:.4f} Valid: {2:.4f}'.format(i, train_loss, valid_loss))
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], i)
        writer.add_scalar('loss/train', train_loss, i)
        writer.add_scalar('loss/valid', valid_loss, i)
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model, os.path.join(model_dir, 'state_dict.pth'))
        if (i + 1) % 10 == 0:
            # plot spectrograms
            model.eval()
            with torch.no_grad():
                # save_batch(testbatch['audio'], resyn_audio, i+1, plot_dir, audio_dir)
                resyn_audio = model(testbatch)
                save_to_board(i, writer, testbatch['audio'], resyn_audio, 8)