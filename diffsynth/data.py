import os, glob
import librosa
import torch
from torch.utils.data import Subset, Dataset, DataLoader, random_split
import pytorch_lightning as pl
import numpy as np

class WaveParamDataset(Dataset):
    def __init__(self, base_dir, sample_rate=16000, length=4.0, params=True, domain=0):
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

class IdOodDataModule(pl.LightningDataModule):
    def __init__(self, id_dir, ood_dir, train_type, batch_size, sample_rate=16000, length=4.0, num_workers=8, splits=[.8, .1, .1]):
        super().__init__()
        self.id_dir = id_dir
        self.ood_dir = ood_dir
        assert train_type in ['id', 'ood']
        self.train_type = train_type
        self.splits = splits
        self.sr = sample_rate
        self.l = length
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def create_split(self, dataset):
        dset_l = len(dataset)
        split_sizes = [int(dset_l*self.splits[0]), int(dset_l*self.splits[1])]
        split_sizes.append(dset_l - split_sizes[0] - split_sizes[1])
        # should be seeded fine but probably better to split test set in some other way
        dset_train, dset_valid, dset_test = random_split(dataset, lengths=split_sizes)
        return {'train': dset_train, 'valid': dset_valid, 'test': dset_test}

    def setup(self, stage):
        id_dat = WaveParamDataset(self.id_dir, self.sr, self.l, True, 0)
        id_datasets = self.create_split(id_dat)
        # ood should be the same size as in-domain
        ood_dat = WaveParamDataset(self.ood_dir, self.sr, self.l, False, 1)
        indices = np.random.choice(len(ood_dat), len(id_dat), replace=False)
        ood_dat = Subset(ood_dat, indices)
        ood_datasets = self.create_split(ood_dat)
        assert len(id_datasets['train']) == len(ood_datasets['train'])
        self.train_dataset = id_datasets['train'] if self.train_type=='id' else ood_datasets['train']
        self.id_datasets = id_datasets
        self.ood_datasets = ood_datasets

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return [DataLoader(self.id_datasets["valid"], batch_size=self.batch_size, num_workers=self.num_workers),
                DataLoader(self.ood_datasets["valid"], batch_size=self.batch_size, num_workers=self.num_workers)]

    def test_dataloader(self):
        return [DataLoader(self.id_datasets["test"], batch_size=self.batch_size, num_workers=self.num_workers),
                DataLoader(self.ood_datasets["test"], batch_size=self.batch_size, num_workers=self.num_workers)]