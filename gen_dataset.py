import torch
import soundfile as sf
import tqdm
import argparse, os
from diffsynth.modelutils import construct_synth_from_conf
from omegaconf import OmegaConf

def make_dirs(base_dir, synth_name):
    dat_dir = os.path.join(base_dir, synth_name)
    audio_dir = os.path.join(dat_dir, 'audio')
    param_dir = os.path.join(dat_dir, 'param')
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(param_dir, exist_ok=True)
    return audio_dir, param_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir',  type=str,   help='')
    parser.add_argument('synth_conf',   type=str,   help='')
    parser.add_argument('--data_size',  type=int,   default=20000)
    parser.add_argument('--audio_len',  type=float, default=4.0)
    parser.add_argument('--sr',         type=int,   default=16000)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--save_param', action='store_true')
    args = parser.parse_args()

    conf = OmegaConf.load(args.synth_conf)
    synth = construct_synth_from_conf(conf).to('cuda')

    audio_dir, param_dir = make_dirs(args.dataset_dir, conf.name)

    n_samples = int(args.audio_len * args.sr)
    count = 0
    break_flag = False
    skip_count = 0
    if args.save_param:
        save_params = conf.save_params # harmor_q, harmor_cutoff, etc.
    else: # save all external params
        rev_dag_summary = {v: k for k,v in synth.dag_summary.items()} # HARM_Q: harmor_q
        save_params = [rev_dag_summary[k] for k in synth.ext_param_sizes.keys()]
    with torch.no_grad():
        with tqdm.tqdm(total=args.data_size) as pbar:
            while True:
                if break_flag:
                    break
                audio, output = synth.uniform(args.batch_size, n_samples, 'cuda')
                params = {k: output[synth.dag_summary[k]].cpu() for k in save_params}
                for j in range(args.batch_size):
                    if count >= args.data_size:
                        break_flag=True
                        break
                    aud = audio[j]
                    # remove silence
                    if aud.abs().max() < 0.05:
                        skip_count += 1
                        continue
                    p = {k:pv[j] for k, pv in params.items()}
                    param_path = os.path.join(param_dir, '{0:05}.pt'.format(count))
                    torch.save(p, param_path)
                    audio_path = os.path.join(audio_dir, '{0:05}.wav'.format(count))
                    sf.write(audio_path, aud.cpu().numpy(), samplerate=args.sr)
                    count+=1
                    pbar.update(1)
    print('skipped {0} quiet sounds'.format(skip_count))