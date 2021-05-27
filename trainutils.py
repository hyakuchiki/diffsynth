import os, glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset, Dataset, DataLoader, random_split

class WaveParamDataset(Dataset):
    def __init__(self, base_dir, sample_rate=16000, length=4.0, params=True):
        self.audio_dir = os.path.join(base_dir, 'audio')
        self.raw_files = sorted(glob.glob(os.path.join(self.audio_dir, '*.wav')))
        print('loaded {0} files'.format(len(self.raw_files)))
        self.length = length
        self.sample_rate = sample_rate
        self.params = params
        if params:
            self.param_dir = os.path.join(base_dir, 'param')
            assert os.path.exists(self.param_dir)
            self.param_files = sorted(glob.glob(os.path.join(self.param_dir, '*.pt')))
    
    def __getitem__(self, idx):
        raw_path = self.raw_files[idx]
        audio, _sr = librosa.load(raw_path, sr=self.sample_rate, duration=self.length)
        assert audio.shape[0] == self.length * self.sample_rate
        if self.params:
            params = torch.load(self.param_files[idx])
            return {'audio': audio, 'params': params}
        else:
            return {'audio': audio}

    def __len__(self):
        return len(self.raw_files)

def plot_spec(y, ax, sr=16000):
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax)

def get_loaders(dset, batch_size, subset_train=None, splits=[.8, .1, .1], nbworkers=4):
    dset_l = len(dset)
    split_sizes = [int(dset_l*splits[0]), int(dset_l*splits[1])]
    split_sizes.append(dset_l - split_sizes[0] - split_sizes[1])
    dset_train, dset_valid, dset_test = random_split(dset, lengths=split_sizes, generator=torch.Generator().manual_seed(0))
    if subset_train is not None:
        indices = np.random.choice(len(dset_train), subset_train, replace=False)
        dset_train = Subset(dset_train, indices)
    
    dl_train = DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=nbworkers, pin_memory=True)
    dl_valid = DataLoader(dset_valid, batch_size=batch_size, num_workers=nbworkers, pin_memory=True)
    dl_test = DataLoader(dset_test, batch_size=batch_size, num_workers=nbworkers, pin_memory=True)
    return (dset_train, dset_valid, dset_test), (dl_train, dl_valid, dl_test)

def plot_recons(x, x_tilde, plot_dir, name=None, epochs=None, sr=16000, num=6, save=True):
    """Plot spectrograms/waveforms of original/reconstructed audio

    Args:
        x (numpy array): [batch, n_samples]
        x_tilde (numpy array): [batch, n_samples]
        sr (int, optional): sample rate. Defaults to 16000.
        dir (str): plot directory.
        name (str, optional): file name.
        epochs (int, optional): no. of epochs.
        num (int, optional): number of spectrograms to plot. Defaults to 6.
    """
    fig, axes = plt.subplots(num, 4, figsize=(15, 30))
    for i in range(num):
        plot_spec(x[i], axes[i, 0], sr)
        plot_spec(x_tilde[i], axes[i, 1], sr)
        axes[i, 2].plot(x[i])
        axes[i, 2].set_ylim(-1,1)
        axes[i, 3].plot(x_tilde[i])
        axes[i, 3].set_ylim(-1,1)
    if save:
        if epochs:
            fig.savefig(os.path.join(plot_dir, 'epoch{:0>3}_recons.png'.format(epochs)))
            plt.close(fig)
        else:
            fig.savefig(os.path.join(plot_dir, name+'.png'))
            plt.close(fig)
    else:
        return fig

def save_to_board(i, name, writer, orig_audio, resyn_audio, plot_num=4, sr=16000):
    orig_audio = orig_audio.detach().cpu()
    resyn_audio = resyn_audio.detach().cpu()
    for j in range(plot_num):
        if i == 0:
            writer.add_audio('{0}_orig/{1}'.format(name, j), orig_audio[j].unsqueeze(0), i, sample_rate=sr)
        writer.add_audio('{0}_resyn/{1}'.format(name, j), resyn_audio[j].unsqueeze(0), i, sample_rate=sr)
    fig = plot_recons(orig_audio.detach().cpu().numpy(), resyn_audio.detach().cpu().numpy(), '', sr=sr, num=plot_num, save=False)
    writer.add_figure('plot_recon_{0}'.format(name), fig, i)

def save_to_board_mel(i, writer, orig_mel, recon_mel, plot_num=8):
    orig_mel = orig_mel.detach().cpu()
    recon_mel = recon_mel.detach().cpu()

    fig, axes = plt.subplots(2, plot_num, figsize=(30, 8))
    for j in range(plot_num):
        axes[0, j].imshow(orig_mel[j], aspect=0.25)
        axes[1, j].imshow(recon_mel[j], aspect=0.25)
    fig.tight_layout()
    writer.add_figure('plot_recon', fig, i)