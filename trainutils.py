import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_spec(y, ax, sr=16000):
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax)

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

def save_to_board(i, writer, orig_audio, resyn_audio, plot_num=4, sr=16000):
    orig_audio = orig_audio.detach().cpu()
    resyn_audio = resyn_audio.detach().cpu()
    for j in range(plot_num):
        writer.add_audio('audio_orig/{0}'.format(j), orig_audio[j].unsqueeze(0), i, sample_rate=sr)
        writer.add_audio('audio_resyn/{0}'.format(j), resyn_audio[j].unsqueeze(0), i, sample_rate=sr)
    fig = plot_recons(orig_audio.detach().cpu().numpy(), resyn_audio.detach().cpu().numpy(), '', sr=sr, num=plot_num, save=False)
    writer.add_figure('plot_recon', fig, i)

def save_to_board_mel(i, writer, orig_mel, recon_mel, plot_num=8):
    orig_mel = orig_mel.detach().cpu()
    recon_mel = recon_mel.detach().cpu()

    fig, axes = plt.subplots(2, plot_num, figsize=(30, 8))
    for j in range(plot_num):
        axes[0, j].imshow(orig_mel[j], aspect=0.25)
        axes[1, j].imshow(recon_mel[j], aspect=0.25)
    fig.tight_layout()
    writer.add_figure('plot_recon', fig, i)