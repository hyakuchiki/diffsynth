import torch
import torch.nn as nn
from diffsynth.spectral import multiscale_fft, compute_loudness
from diffsynth.util import log_eps
import torch.nn.functional as F
import functools

def spectrogram_loss(x_audio, target_audio, fft_sizes=[64, 128, 256, 512, 1024, 2048], hop_ls=None, win_ls=None, log_mag_w=0.0, mag_w=1.0, norm=None):
    x_specs = multiscale_fft(x_audio, fft_sizes, hop_ls, win_ls)
    target_specs = multiscale_fft(target_audio, fft_sizes, hop_ls, win_ls)
    loss = 0.0
    spec_loss = {}
    log_spec_loss = {}
    for n_fft, x_spec, target_spec in zip(fft_sizes, x_specs, target_specs):
        spec_norm = norm['spec'][n_fft] if norm is not None else 1.0
        log_spec_norm = norm['logspec'][n_fft] if norm is not None else 1.0
        if mag_w > 0:
            spec_loss[n_fft] = mag_w * torch.mean(torch.abs(x_spec - target_spec)) / spec_norm
        if log_mag_w > 0:
            log_spec_loss[n_fft] = log_mag_w * torch.mean(torch.abs(log_eps(x_spec) - log_eps(target_spec))) / log_spec_norm
    return {'spec':spec_loss, 'logspec':log_spec_loss}

def waveform_loss(x_audio, target_audio, l1_w=0, l2_w=1.0, linf_w=0, linf_k=1024, norm=None):
    norm = {'l1':1.0, 'l2':1.0} if norm is None else norm
    l1_loss = l1_w * torch.mean(torch.abs(x_audio - target_audio)) / norm['l1'] if l1_w > 0 else 0.0
    # mse loss
    l2_loss = l2_w * torch.mean((x_audio - target_audio)**2) / norm['l2'] if l2_w > 0 else 0.0
    if linf_w > 0:
        # actually gets k elements
        residual = (x_audio - target_audio)**2
        values, _ = torch.topk(residual, linf_k, dim=-1)
        linf_loss = torch.mean(values) / norm['l2']
    else:
        linf_loss = 0.0
    return {'l1':l1_loss, 'l2':l2_loss, 'linf':linf_loss}

class SpecWaveLoss():
    """
    loss for reconstruction with multiscale spectrogram loss and waveform loss
    """
    def __init__(self, fft_sizes=[64, 128, 256, 512, 1024, 2048], hop_lengths=None, win_lengths=None, mag_w=1.0, log_mag_w=1.0, l1_w=0, l2_w=0.0, linf_w=0.0, linf_k=1024, norm=None):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_lengths = hop_lengths
        self.win_lengths = win_lengths
        self.mag_w = mag_w
        self.log_mag_w = log_mag_w
        self.l1_w=l1_w
        self.l2_w=l2_w
        self.linf_w=linf_w
        self.spec_loss = functools.partial(spectrogram_loss, fft_sizes=fft_sizes, hop_ls=hop_lengths, win_ls=win_lengths, log_mag_w=log_mag_w, mag_w=mag_w, norm=norm)
        self.wave_loss = functools.partial(waveform_loss, l1_w=l1_w, l2_w=l2_w, linf_w=linf_w, linf_k=linf_k, norm=norm)
        
    def __call__(self, x_audio, target_audio):
        if (self.mag_w + self.log_mag_w) > 0:
            spec_losses = self.spec_loss(x_audio, target_audio)
            multi_spec_loss = sum(spec_losses['spec'].values()) + sum(spec_losses['logspec'].values())
            multi_spec_loss /= (len(self.fft_sizes)*(self.mag_w + self.log_mag_w))
        else: # no spec loss
            multi_spec_loss = torch.tensor([0.0], device=x_audio.device)
        if (self.l1_w + self.l2_w + self.linf_w) > 0:
            wave_losses = self.wave_loss(x_audio, target_audio)
            waveform_loss = wave_losses['l1'] + wave_losses['l2'] + wave_losses['linf']
            waveform_loss /= (self.l1_w + self.l2_w + self.linf_w)
        else: # no waveform loss
            waveform_loss = torch.tensor([0.0], device=x_audio.device)
        return multi_spec_loss, waveform_loss

def calculate_norm(loader, fft_sizes, hop_ls, win_ls):
    """
    calculate stats for scaling losses
    based on jukebox
    doesn't really work
    """
    n, spec_n = 0, 0
    spec_total = {n_fft: 0.0 for n_fft in fft_sizes}
    log_spec_total = {n_fft: 0.0 for n_fft in fft_sizes}
    total, total_sq, l1_total = 0.0, 0.0, 0.0
    print('calculating bandwidth')
    for data_dict in loader:
        x_audio = data_dict['audio']
        total = torch.sum(x_audio)
        total_sq = torch.sum(x_audio**2)
        l1_total = torch.sum(torch.abs(x_audio))
        x_specs = multiscale_fft(x_audio, fft_sizes, hop_ls, win_ls)
        for n_fft, spec in zip(fft_sizes, x_specs): 
            # spec: power spectrogram [batch_size, n_bins, time]
            spec_total[n_fft] += torch.mean(spec)
            # probably not right
            log_spec_total[n_fft] += torch.mean(torch.abs(log_eps(spec)))
        n += x_audio.shape[0] * x_audio.shape[1]
        spec_n += 1
    
    print('done.')
    mean = total / n
    for n_fft in fft_sizes:
        spec_total[n_fft] /= spec_n
        log_spec_total[n_fft] /= spec_n

    return {'l2': total_sq/n - mean**2, 'l1': l1_total/n, 'spec': spec_total, 'logspec': log_spec_total}
