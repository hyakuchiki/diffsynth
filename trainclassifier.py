import os, json, argparse
import tqdm
import numpy as np
import librosa

import torch
from torch.utils.data import Dataset

from trainutils import get_loaders
from diffsynth.perceptual.perceptual import PerceptualClassifier

class NSynthDataset(Dataset):
    def __init__(self, base_dir, sample_rate=16000, length=4.0):
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, 'audio')
        self.length = length
        self.sample_rate = sample_rate
        # load json file that comes with nsynth dataset
        with open(os.path.join(self.base_dir, 'examples.json')) as f:
            self.json_dict = json.load(f)
        self.json_keys = list(self.json_dict.keys())
        # restrict the dataset to some categories
        self.nb_files = len(self.json_keys)
    
    def __getitem__(self, index):
        output = {}
        note = self.json_dict[self.json_keys[index]]
        file_name = os.path.join(self.raw_dir, note['note_str']+'.wav')
        output['label'] = int(note['instrument_family'])# 0=bass, 1=brass, 2=flute, etc.
        output['audio'], _sr = librosa.load(file_name, sr=self.sample_rate, duration=self.length)
        return output

    def __len__(self):
        return self.nb_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir',   type=str,   help='')
    parser.add_argument('dataset', type=str,   help='directory of dataset')
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--epochs',     type=int,   default=200)
    parser.add_argument('--decay_rate', type=float, default=1.0, help='')
    parser.add_argument('--length',     type=float, default=4.0,    help='')
    parser.add_argument('--sr',         type=int,   default=16000,  help='')
    args = parser.parse_args()

    torch.manual_seed(0) # just fixes the dataset/dataloader and initial weights
    np.random.seed(seed=0) # subset

    device = 'cuda'
    # output dir
    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    # load dataset
    dset = NSynthDataset(args.dataset, sample_rate=args.sr, length=args.length)
    dsets, loaders = get_loaders(dset, args.batch_size, splits=[.8, .2, 0.0], nbworkers=4)
    dset_train, dset_valid, _dset_test = dsets
    train_loader, valid_loader, _test_loader = loaders
    testbatch = next(iter(valid_loader))

    # load model
    model = PerceptualClassifier(11, testbatch['audio'].shape[-1]).to(device) # 11 classes to classify
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)

    best_acc= 0
    # training loop
    for i in tqdm.tqdm(range(1, args.epochs+1)):
        train_loss = model.train_epoch(loader=train_loader, optimizer=optimizer, device=device)
        valid_acc = model.eval_epoch(loader=valid_loader, device=device)
        scheduler.step()
        tqdm.tqdm.write('Epoch: {0:03} Train: {1:.4f} Valid: {2:.4f}'.format(i, train_loss, valid_acc))
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(model_dir, 'state_dict.pth'))
        if i % 10 == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, 'statedict_{0:03}.pth'.format(i)))