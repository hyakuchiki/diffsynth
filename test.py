import os, argparse, json, pickle, re
import matplotlib.pyplot as plt
import torch

from diffsynth.loss import SpecWaveLoss
from diffsynth import util
from diffsynth.model import EstimatorSynth
from plot import plot_spec, plot_param_dist
import soundfile as sf

import hydra
import pytorch_lightning as pl

def write_plot_audio(y, name):
    # y; numpy array of audio
    # write audio file
    sf.write('{0}.wav'.format(name), y, 16000)
    fig, ax = plt.subplots(figsize=(1.5, 1), tight_layout=True)
    ax.axis('off')
    plot_spec(y, ax, 16000)
    fig.savefig('{0}.png'.format(name))
    plt.close(fig)

def test_model(model, id_loader, ood_loader, device, sw_loss=None, perc_model=None):
    model.eval()
    # in-domain
    syn_result = util.StatsLog()
    param_stats = [util.StatsLog(), util.StatsLog()]
    with torch.no_grad():
        for data_dict in id_loader:
            params = data_dict.pop('params')
            params = {name:tensor.to(device, non_blocking=True) for name, tensor in params.items()}
            data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
            data_dict['params'] = params

            resyn_audio, outputs = model(data_dict)
            # parameter values
            monitor_params = list(params.keys())
            for pname, pvalue in outputs.items():
                if pname in monitor_params:
                    # pvalue: batch, n_frames, param_dim>=1
                    pvs = pvalue.mean(dim=1)
                    for i, pv in enumerate(pvs.unbind(-1)):
                        param_stats[0].add_entry(pname+'{0}'.format(i), pv)

            # Reconstruction loss
            losses = model.train_losses(data_dict, outputs, sw_loss=sw_loss, perc_model=perc_model)
            losses.update(model.monitor_losses(data_dict, outputs))
            syn_result.update(losses)
    syn_result_dict = {'id/'+k: v for k, v in syn_result.average().items()}
    
    # out-of-domain
    real_result = util.StatsLog()
    with torch.no_grad():
        for data_dict in ood_loader:
            data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}

            resyn_audio, outputs = model(data_dict)
            # parameter values
            monitor_params = list(params.keys())
            for pname, pvalue in outputs.items():
                if pname in monitor_params:
                    # pvalue: batch, n_frames, param_dim>=1
                    pvs = pvalue.mean(dim=1)
                    for i, pv in enumerate(pvs.unbind(-1)):
                        param_stats[1].add_entry(pname+'{0}'.format(i), pv)

            # Reconstruction loss
            losses = model.train_losses(data_dict, outputs, sw_loss=sw_loss, perc_model=perc_model)
            losses.update(model.monitor_losses(data_dict, outputs))
            real_result.update(losses)
    real_result_dict = {'ood/'+k: v for k, v in real_result.average().items()}
    
    result = {}
    result.update(syn_result_dict)
    result.update(real_result_dict)
    return result, param_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt',             type=str,   help='')
    parser.add_argument('--batch_size',     type=int,   default=64, help='')
    parser.add_argument('--write_audio',    action='store_true')
    args = parser.parse_args()

    pl.seed_everything(0, workers=True)
    device = 'cuda'

    ckpt_dir = args.ckpt
    config_dir = re.sub(r'tb_logs.*', '.hydra', ckpt_dir)
    # initialize model
    hydra.initialize(config_path=config_dir, job_name="test")
    cfg = hydra.compose(config_name="config")

    model = EstimatorSynth(cfg.model, cfg.synth, cfg.schedule)
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(None)
    id_test_loader, ood_test_loader = datamodule.test_dataloader()
    model = EstimatorSynth.load_from_checkpoint(ckpt_dir).to(device)

    # directory for audio/spectrogram output
    output_dir = re.sub(r'tb_logs.*', 'test/output', ckpt_dir)
    os.makedirs(output_dir, exist_ok=True)
    # directory for ground-truth
    target_dir = re.sub(r'tb_logs.*', 'test/target', ckpt_dir)
    os.makedirs(target_dir, exist_ok=True)

    id_testbatch = next(iter(id_test_loader))
    id_testbatch.pop('params')
    id_testbatch = {name:tensor.to(device) for name, tensor in id_testbatch.items()}
    ood_testbatch = next(iter(ood_test_loader))
    ood_testbatch = {name:tensor.to(device) for name, tensor in ood_testbatch.items()}

    sw_loss = SpecWaveLoss(l1_w=0.0, l2_w=0.0, norm=None)
    with torch.no_grad():
        model = model.eval()
        if args.write_audio:
            # render audio and plot spectrograms?
            id_resyn_audio, _output = model(id_testbatch)
            for i in range(args.batch_size):
                resyn_audio = id_resyn_audio[i].detach().cpu().numpy()
                write_plot_audio(resyn_audio, os.path.join(output_dir, 'id_{0:03}'.format(i)))
                orig_audio = id_testbatch['audio'][i].detach().cpu().numpy()
                write_plot_audio(orig_audio, os.path.join(target_dir, 'id_{0:03}'.format(i)))
            ood_resyn_audio, _output = model(ood_testbatch)
            for i in range(args.batch_size):
                resyn_audio = ood_resyn_audio[i].detach().cpu().numpy()
                write_plot_audio(resyn_audio, os.path.join(output_dir, 'ood_{0:03}'.format(i)))
                orig_audio = ood_testbatch['audio'][i].detach().cpu().numpy()
                write_plot_audio(orig_audio, os.path.join(target_dir, 'ood_{0:03}'.format(i)))
            print('finished writing audio')
        
        # get objective measure
        test_losses, param_stats = test_model(model, id_loader=id_test_loader, ood_loader=ood_test_loader, device=device, sw_loss=sw_loss)
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