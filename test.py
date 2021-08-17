import os, tqdm, glob, argparse, json, pickle
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset, DataLoader, Dataset, ConcatDataset, random_split

from diffsynth.estimator import MFCCEstimator, MelEstimator
from diffsynth.model import EstimatorSynth
from diffsynth.loss import SpecWaveLoss
from diffsynth.modelutils import construct_synths
from diffsynth import util
from train import WaveParamDataset
from trainutils import plot_spec, load_model, plot_param_dist
import soundfile as sf

def write_plot_audio(y, name):
    # y; numpy array of audio
    # write audio file
    sf.write('{0}.wav'.format(name), y, 16000)
    fig, ax = plt.subplots(figsize=(1.5, 1), tight_layout=True)
    # ax.axis('off')
    plot_spec(y, ax, 16000)
    fig.savefig('{0}.png'.format(name))
    plt.close(fig)

def test_model(model, syn_loader, real_loader, device, sw_loss=None, perc_model=None):
    model.eval()
    # in-domain
    syn_result = util.StatsLog()
    param_stats = [util.StatsLog(), util.StatsLog()]
    with torch.no_grad():
        for data_dict in syn_loader:
            params = data_dict.pop('params')
            params = {name:tensor.to(device, non_blocking=True) for name, tensor in params.items()}
            data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
            data_dict['params'] = params

            resyn_audio, outputs = model(data_dict)
            # parameter values
            monitor_params = [model.synth.dag_summary[k] for k in params.keys()]
            for pname, pvalue in outputs.items():
                if pname in monitor_params:
                    # pvalue: batch, n_frames, param_dim>=1
                    pvs = pvalue.mean(dim=1)
                    for i, pv in enumerate(pvs.unbind(-1)):
                        param_stats[0].add_entry(pname+'{0}'.format(i), pv)

            # Reconstruction loss
            losses = model.losses(data_dict, outputs, sw_loss=sw_loss, perc_model=perc_model)
            syn_result.update(losses)
    syn_result_dict = {'syn/'+k: v for k, v in syn_result.average().items()}
    
    # out-of-domain
    real_result = util.StatsLog()
    with torch.no_grad():
        for data_dict in real_loader:
            data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}

            resyn_audio, outputs = model(data_dict)
            # parameter values
            monitor_params = [model.synth.dag_summary[k] for k in params.keys()]
            for pname, pvalue in outputs.items():
                if pname in monitor_params:
                    # pvalue: batch, n_frames, param_dim>=1
                    pvs = pvalue.mean(dim=1)
                    for i, pv in enumerate(pvs.unbind(-1)):
                        param_stats[1].add_entry(pname+'{0}'.format(i), pv)

            # Reconstruction loss
            losses = model.losses(data_dict, outputs, param_w=0, sw_loss=sw_loss, perc_model=perc_model)
            real_result.update(losses)
    real_result_dict = {'real/'+k: v for k, v in real_result.average().items()}
    
    result = {}
    result.update(syn_result_dict)
    result.update(real_result_dict)
    return result, param_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('load_dir',         type=str,   help='')
    parser.add_argument('dataset_dir',      type=str,   help='directory of saved dataset')
    parser.add_argument('--write_audio',    action='store_true')
    parser.add_argument('--batch_size',     type=int,   default=64,     help='')
    parser.add_argument('--epoch',          type=int,   default=None,     help='')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(seed=0) # subset
    device = 'cuda'

    # directory for audio/spectrogram output
    output_dir = args.load_dir.replace('results', 'output')
    os.makedirs(output_dir, exist_ok=True)
    # directory for ground-truth
    target_dir = os.path.join(os.path.split(output_dir)[0], 'target')
    os.makedirs(target_dir, exist_ok=True)

    model = load_model(args.load_dir, args.epoch)
    model.to(device)

    # load test sets
    syn_dset_train, syn_dset_valid, syn_dset_test, real_dset_train, real_dset_valid, real_dset_test = torch.load(args.dataset_dir)
    if args.dataset_dir[-2:] != 'pt': # test on something else
        real_dset_test = WaveParamDataset(args.dataset_dir, params=True)
        print('loaded directory with {0} files for real data'.format(len(real_dset_test)))

    syn_test_loader = DataLoader(syn_dset_test, batch_size=args.batch_size, num_workers=0)
    real_test_loader = DataLoader(real_dset_test, batch_size=args.batch_size, num_workers=0)

    syn_testbatch = next(iter(syn_test_loader))
    syn_testbatch.pop('params')
    syn_testbatch = {name:tensor.to(device) for name, tensor in syn_testbatch.items()}
    real_testbatch = next(iter(real_test_loader))
    real_testbatch = {name:tensor.to(device) for name, tensor in real_testbatch.items()}

    sw_loss = SpecWaveLoss(l1_w=0.0, l2_w=0.0, norm=None)
    with torch.no_grad():
        model = model.eval()
        if args.write_audio:
            # render audio and plot spectrograms?
            syn_resyn_audio, _output = model(syn_testbatch)
            for i in range(args.batch_size):
                resyn_audio = syn_resyn_audio[i].detach().cpu().numpy()
                write_plot_audio(resyn_audio, os.path.join(output_dir, 'synth_{0:03}'.format(i)))
                orig_audio = syn_testbatch['audio'][i].detach().cpu().numpy()
                write_plot_audio(orig_audio, os.path.join(target_dir, 'synth_{0:03}'.format(i)))
            real_resyn_audio, _output = model(real_testbatch)
            for i in range(args.batch_size):
                resyn_audio = real_resyn_audio[i].detach().cpu().numpy()
                write_plot_audio(resyn_audio, os.path.join(output_dir, 'real_{0:03}'.format(i)))
                orig_audio = real_testbatch['audio'][i].detach().cpu().numpy()
                write_plot_audio(orig_audio, os.path.join(target_dir, 'real_{0:03}'.format(i)))
            print('finished writing audio')
        
        # get objective measure
        test_losses, param_stats = test_model(model, syn_loader=syn_test_loader, real_loader=real_test_loader, sw_loss=sw_loss, device=device)
        results_str = 'Test loss: '
        for k in test_losses:
            results_str += '{0}: {1:.3f} '.format(k, test_losses[k])
        print(results_str)
        with open(os.path.join(output_dir, 'test_loss.json'), 'w') as f:
            json.dump(test_losses, f)
        # plot parameter stats
        fig_1 = plot_param_dist(param_stats[0].stats)
        fig_1.savefig(os.path.join(output_dir, 'id_params_dist.png'))        
        fig_2 = plot_param_dist(param_stats[1].stats)
        fig_2.savefig(os.path.join(output_dir, 'ood_params_dist.png'))
        with open(os.path.join(output_dir, 'params_dists.pkl'), 'wb') as f:
            pickle.dump(param_stats, f)