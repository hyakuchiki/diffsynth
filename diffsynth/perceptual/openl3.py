import os
import numpy as np
import librosa
import torch
import torch.nn.functional as F
import torchopenl3 as ol3
from torchopenl3.models import CustomSTFT

from .perceptual import Perceptual
from diffsynth.util import slice_windows

HOP_48K = 242
NDFT_48K_MEL = 2048
ORIG_SIZE = (257, 197)

def custom_pad(x, n_dft, n_hop, sr):
    """
    Taken from torchopenl3
    Pad sequence.
    Implemented similar to keras version used in kapre=0.1.4
    """
    # x: (batch, 1, 16000)
    if sr % n_hop == 0:
        pad_along_width = max(n_dft - n_hop, 0)
    else:
        pad_along_width = max(n_dft - (sr % n_hop), 0)

    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    x = F.pad(x, (pad_left, pad_right))
    return x

def amplitude_to_decibel(x, amin=1e-10, dynamic_range=80.0):
    """
    Taken from torchopenl3
    Convert (linear) amplitude to decibel (log10(x)).
    Implemented similar to kapre=0.1.4
    """

    log_spec = (
        10 * torch.log(torch.clamp(x, min=amin)) / np.log(10).astype(np.float32)
    )
    if x.ndim > 1:
        axis = tuple(range(x.ndim)[1:])
    else:
        axis = None

    log_spec = log_spec - torch.amax(log_spec, dim=axis, keepdims=True)
    log_spec = torch.clamp(log_spec, min=-1 * dynamic_range)
    return log_spec

class PerceptualOpenl3(Perceptual):
    def __init__(self, base_dir, input_repr='mel256', data='music', embed_size=512, sr=16000, hop_s=1.0):
        super().__init__()
        assert input_repr in ['mel256', 'mel128']
        assert data in ['music', 'env']
        assert embed_size in [512, 6144]
        self.orig_sr = sr
        self.hop_s = hop_s
        model_name='torchopenl3_{0}_{1}_{2}.pth.tar'.format(input_repr, data, embed_size)
        print('loading', model_name)
        # model = ol3.models.PytorchOpenl3('_', input_repr, embed_size)
        # model = model.load_state_dict(torch.load(os.path.join(base_dir,model_name)))
        model = ol3.core.load_audio_embedding_model(input_repr, data, embed_size)
        self.model = model.eval().requires_grad_(False)
        self.model.speclayer = torch.nn.Identity()

        self.n_mels = int(input_repr[3:])
        mult = self.orig_sr/48000
        self.n_dft = int(mult * NDFT_48K_MEL) # 682
        self.hop_len = int(mult * HOP_48K) # 80
        mel_fb = librosa.filters.mel(
                    sr=self.orig_sr,
                    n_fft=self.n_dft,
                    n_mels=self.n_mels,
                    fmin=0,
                    fmax=24000,
                    htk=True,
                    norm=1,
                ) # lots of empty banks
        self.register_buffer("mel_fb", torch.tensor(mel_fb, requires_grad=False))

        self.pad_freq = ORIG_SIZE[0] - (self.n_dft//2+1)
        self.remainder_w = ((16000 - self.hop_len) // self.hop_len + 1) - ORIG_SIZE[1]
        assert self.remainder_w >= 0
        self.lin_spec = CustomSTFT(
            n_dft=self.n_dft,
            n_hop=self.hop_len,
            power_spectrogram=2.0,
            return_decibel_spectrogram=False,
        )

    def stft_mel(self, x):
        # batch, 1, frame_size
        x = slice_windows(x, self.orig_sr, int(self.orig_sr*self.hop_s)).flatten(0,1).unsqueeze(1)
        x = custom_pad(x, self.n_dft, self.hop_len, self.orig_sr)
        spec = self.lin_spec(x)
        assert spec.shape[2] >= ORIG_SIZE[1], spec.shape[2]
        spec = spec[:, :, :ORIG_SIZE[1], 0]
        # [batch_slices, 257, 197, 1]
        melspec = torch.matmul(self.mel_fb, spec)
        melspec = torch.sqrt(melspec)
        return amplitude_to_decibel(melspec)

    def get_embed(self, x):
        mel = self.stft_mel(x)
        return self.model(mel.contiguous())

    def perceptual_loss(self, target_audio, input_audio):
        # output: (batch, freq, time, 1)
        target_embed = self.get_embed(target_audio)
        input_embed = self.get_embed(input_audio)
        return (1 - F.cosine_similarity(target_embed, input_embed, dim=1)).mean()

def load_openl3_model(base_dir, input_repr='mel256', data='music', embed_size=512):
    assert input_repr in ['mel256', 'mel128', 'linear']
    assert data in ['music', 'env']
    assert embed_size in [512, 6144]
    # model_name='torchopenl3_{0}_{1}_{2}.pth.tar'.format(input_repr, data, embed_size)
    # print('loading', model_name)
    # model = ol3.models.PytorchOpenl3('_', input_repr, embed_size)
    # model = model.load_state_dict(torch.load(os.path.join(base_dir,model_name)))
    model = ol3.core.load_audio_embedding_model(input_repr, data, embed_size)
    return model.eval().requires_grad_(False)

def openl3_loss(model, target_audio, input_audio, hop_size=1.0):
    # target/input : (batch, n_samples)
    # resampled to 48000Hz and sliced to be [batch*n_slice, 48000*1(window_size)]
    target_audio = ol3.utils.preprocess_audio_batch(target_audio, sr=16000, center=False, hop_size=hop_size, sampler='julian')
    input_audio  = ol3.utils.preprocess_audio_batch(input_audio, sr=16000, center=False, hop_size=hop_size, sampler='julian')

    target_embed = model(target_audio.contiguous())
    input_embed = model(input_audio.contiguous())
    return (1 - F.cosine_similarity(target_embed, input_embed, dim=1)).mean()
