import numpy as np
import torch
import torch.nn as nn
import torchcrepe

def compute_f0(audio, sample_rate):
    """ For preprocessing
    Args:
        audio: torch.Tensor of single audio example. Shape [audio_length,].
        sample_rate: Sample rate in Hz.

    Returns:
        f0_hz: Fundamental frequency in Hz. Shape [1 + int(time // hop_length,]
    """
    audio = audio.unsqueeze(0)

    # Compute f0 with torchcrepe.
    # uses viterbi by default
    # pad=False is probably center=False
    # [output_shape=(1, 1 + int(time // hop_length))]
    f0_hz = torchcrepe.predict(audio, sample_rate, hop_length=256, pad=False, device='cuda', batch_size=2048, model='full', fmin=32, fmax=2000)[0]
    # Postprocessing on f0_hz

    return f0_hz

if __name__ == "__main__":
    import argparse, os, glob
    import librosa
    import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir',  type=str,   help='')
    parser.add_argument('--duration',  type=float, default=4.0, help='')
    parser.add_argument('--overwrite',  action='store_true')
    args = parser.parse_args()
    
    audio_dir = os.path.join(args.base_dir, 'audio')
    f0_dir = os.path.join(args.base_dir, 'f0')
    os.makedirs(f0_dir, exist_ok=True)
    raw_files = sorted(glob.glob(os.path.join(audio_dir, '*.wav')))

    for rf in tqdm.tqdm(raw_files):
        basename = os.path.basename(rf)
        f0_file = os.path.join(f0_dir, basename[:-4]+'.pt')
        if not args.overwrite and os.path.exists(f0_file):
            continue
        audio, _sr = librosa.load(rf, sr=16000, duration=args.duration)
        f0 = compute_f0(torch.from_numpy(audio), 16000)
        torch.save(f0, f0_file)