import os, tqdm, glob, argparse, json
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset, DataLoader, Dataset, ConcatDataset, random_split

from diffsynth.estimator import MFCCEstimator, MelEstimator
from diffsynth.model import EstimatorSynth
from diffsynth.loss import SpecWaveLoss
from diffsynth.modelutils import construct_synths
from train import WaveParamDataset
from trainutils import plot_spec
import soundfile as sf

def write_plot_audio(y, name):
    # y; numpy array of audio
    # write audio file
    sf.write('{0}.ogg'.format(name), y, 16000)
    fig, ax = plt.subplots(figsize=(1.5, 1))
    ax.axis('off')
    plot_spec(y, ax, 16000)
    fig.savefig('{0}.png'.format(name))
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('load_dir',         type=str,   help='')
    parser.add_argument('dataset_dir',      type=str,   help='directory of saved dataset')
    parser.add_argument('--batch_size',     type=int,   default=32,     help='')
    parser.add_argument('--epoch',          type=int,   default=None,     help='')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(seed=0) # subset
    device = 'cuda'

    output_dir = args.load_dir.replace('results', 'output')

    audio_dir = os.path.join(output_dir, 'audio')
    os.makedirs(audio_dir, exist_ok=True)
    model_dir = os.path.join(args.load_dir, 'model')

    pre_args_file = os.path.join(args.load_dir, 'args.txt')
    with open(pre_args_file) as f:
        pre_args = json.load(f)
        if 'load_dir' in pre_args:
            pre_args_file = os.path.join(pre_args['load_dir'], 'args.txt')
            with open(pre_args_file) as f:
                pre_args = json.load(f)

        pre_args = SimpleNamespace(**pre_args)

    # load test sets
    syn_dset_train, syn_dset_valid, syn_dset_test, real_dset_train, real_dset_valid, real_dset_test = torch.load(args.dataset_dir)
    if args.dataset_dir[-2:] != 'pt': # test on something else
        real_dset_test = WaveParamDataset(args.dataset_dir, params=True)
        print('loaded directory with {0} files for real data'.format(len(real_dset_test)))

    synth = construct_synths(pre_args.synth)
    if pre_args.estimator == 'mfccgru':
        estimator = MFCCEstimator(synth.ext_param_size)
    elif pre_args.estimator == 'melgru':
        estimator = MelEstimator(synth.ext_param_size)
    
    model = EstimatorSynth(estimator, synth)
    if args.epoch is None:
        model_name = 'model/state_dict.pth'
    else:
        model_name = 'model/statedict_{0}.pth'.format(args.epoch)
    model.load_state_dict(torch.load(os.path.join(args.load_dir, model_name)))

    model.to(device)

    recon_loss = SpecWaveLoss(l1_w=0.0, l2_w=0.0, norm=None)
    
    syn_test_loader = DataLoader(syn_dset_test, batch_size=args.batch_size, num_workers=0)
    real_test_loader = DataLoader(real_dset_test, batch_size=args.batch_size, num_workers=0)

    syn_testbatch = next(iter(syn_test_loader))
    syn_testbatch.pop('params')
    syn_testbatch = {name:tensor.to(device) for name, tensor in syn_testbatch.items()}

    real_testbatch = next(iter(real_test_loader))
    real_testbatch = {name:tensor.to(device) for name, tensor in real_testbatch.items()}

    with torch.no_grad():
        model = model.eval()
        # render audio and plot spectrograms?
        syn_resyn_audio, _output = model(syn_testbatch)
        for i in range(args.batch_size):
            resyn_audio = syn_resyn_audio[i].detach().cpu().numpy()
            write_plot_audio(resyn_audio, os.path.join(audio_dir, 'synth_{0:03}_resyn'.format(i)))
            orig_audio = syn_testbatch['audio'][i].detach().cpu().numpy()
            write_plot_audio(orig_audio, os.path.join(audio_dir, 'synth_{0:03}_orig'.format(i)))
        real_resyn_audio, _output = model(real_testbatch)
        for i in range(args.batch_size):
            resyn_audio = real_resyn_audio[i].detach().cpu().numpy()
            write_plot_audio(resyn_audio, os.path.join(audio_dir, 'real_{0:03}_resyn'.format(i)))
            orig_audio = real_testbatch['audio'][i].detach().cpu().numpy()
            write_plot_audio(orig_audio, os.path.join(audio_dir, 'real_{0:03}_orig'.format(i)))
        
        print('finished writing audio')
        # get objective measure
        test_losses = model.eval_epoch(syn_loader=syn_test_loader, real_loader=real_test_loader, recon_loss=recon_loss, device=device)
        results_str = 'Test loss: '
        for k in test_losses:
            results_str += '{0}: {1:.3f} '.format(k, test_losses[k])
        print(results_str)
        with open(os.path.join(output_dir, 'test_loss.json'), 'w') as f:
            json.dump(test_losses, f)

