import os, glob, functools
import librosa
import torch
from torch.utils.data import Subset, Dataset, DataLoader, random_split, ConcatDataset, SubsetRandomSampler, BatchSampler
import pytorch_lightning as pl
import numpy as np
from diffsynth.f0 import process_f0

def mix_iterable(dl_a, dl_b):
    for i, j in zip(dl_a, dl_b):
        yield i
        yield j

class ReiteratableWrapper():
    def __init__(self, f, length):
        self._f = f
        self.length = length

    def __iter__(self):
        # make generator
        return self._f()

    def __len__(self):
        return self.length

class WaveParamDataset(Dataset):
    def __init__(self, base_dir, sample_rate=16000, length=4.0, params=True, f0=False):
        self.base_dir = base_dir
        self.audio_dir = os.path.join(base_dir, 'audio')
        self.raw_files = sorted(glob.glob(os.path.join(self.audio_dir, '*.wav')))
        print('loaded {0} files'.format(len(self.raw_files)))
        self.length = length
        self.sample_rate = sample_rate
        self.params = params
        self.f0 = f0
        if f0:
            self.f0_dir = os.path.join(base_dir, 'f0')
            assert os.path.exists(self.f0_dir)
            # all the f0 files should already be written
            # with the same name as the audio
            self.f0_files = sorted(glob.glob(os.path.join(self.f0_dir, '*.pt')))
        if params:
            self.param_dir = os.path.join(base_dir, 'param')
            assert os.path.exists(self.param_dir)
            # all the files should already be written
            self.param_files = sorted(glob.glob(os.path.join(self.param_dir, '*.pt')))
    
    def __getitem__(self, idx):
        raw_path = self.raw_files[idx]
        audio, _sr = librosa.load(raw_path, sr=self.sample_rate, duration=self.length)
        assert audio.shape[0] == self.length * self.sample_rate
        data = {'audio': audio}
        if self.f0:
            f0, periodicity = torch.load(self.f0_files[idx])
            f0_hz = process_f0(f0, periodicity)
            data['f0_hz'] = f0_hz.unsqueeze(-1)
        if self.params:
            params = torch.load(self.param_files[idx])
            data['params'] = params
        return data

    def __len__(self):
        return len(self.raw_files)

class IdOodDataModule(pl.LightningDataModule):
    def __init__(self, id_dir, ood_dir, train_type, batch_size, sample_rate=16000, length=4.0, num_workers=8, splits=[.8, .1, .1], f0=False):
        super().__init__()
        self.id_dir = id_dir
        self.ood_dir = ood_dir
        assert train_type in ['id', 'ood', 'mixed']
        self.train_type = train_type
        self.splits = splits
        self.sr = sample_rate
        self.l = length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.f0 = f0
    
    def create_split(self, dataset):
        dset_l = len(dataset)
        split_sizes = [int(dset_l*self.splits[0]), int(dset_l*self.splits[1])]
        split_sizes.append(dset_l - split_sizes[0] - split_sizes[1])
        # should be seeded fine but probably better to split test set in some other way
        dset_train, dset_valid, dset_test = random_split(dataset, lengths=split_sizes)
        return {'train': dset_train, 'valid': dset_valid, 'test': dset_test}

    def setup(self, stage):
        id_dat = WaveParamDataset(self.id_dir, self.sr, self.l, True, self.f0)
        id_datasets = self.create_split(id_dat)
        # ood should be the same size as in-domain
        ood_dat = WaveParamDataset(self.ood_dir, self.sr, self.l, False, self.f0)
        indices = np.random.choice(len(ood_dat), len(id_dat), replace=False)
        ood_dat = Subset(ood_dat, indices)
        ood_datasets = self.create_split(ood_dat)
        self.id_datasets = id_datasets
        self.ood_datasets = ood_datasets
        assert len(id_datasets['train']) == len(ood_datasets['train'])
        if self.train_type == 'mixed':
            dat_len = len(id_datasets['train'])
            indices = np.random.choice(dat_len, dat_len//2, replace=False)
            self.train_set = ConcatDataset([Subset(id_datasets['train'], indices), Subset(ood_datasets['train'], indices)])

    def train_dataloader(self):
        if self.train_type=='id':
            return DataLoader(self.id_datasets['train'], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)
        elif self.train_type=='ood':
            return DataLoader(self.ood_datasets['train'], batch_size=self.batch_size,
                            num_workers=self.num_workers, shuffle=True)
        elif self.train_type=='mixed':
            id_indices = list(range(len(self.train_set)//2))
            ood_indices = list(range(len(self.train_set)//2, len(self.train_set)))
            id_samp = SubsetRandomSampler(id_indices)
            ood_samp = SubsetRandomSampler(ood_indices)
            id_batch_samp = BatchSampler(id_samp, batch_size=self.batch_size, drop_last=False)
            ood_batch_samp = BatchSampler(ood_samp, batch_size=self.batch_size, drop_last=False)
            generator = functools.partial(mix_iterable, id_batch_samp, ood_batch_samp)
            b_sampler = ReiteratableWrapper(generator, len(id_batch_samp)+len(ood_batch_samp))
            return DataLoader(self.train_set, batch_sampler=b_sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        return [DataLoader(self.id_datasets["valid"], batch_size=self.batch_size, num_workers=self.num_workers),
                DataLoader(self.ood_datasets["valid"], batch_size=self.batch_size, num_workers=self.num_workers)]

    def test_dataloader(self):
        return [DataLoader(self.id_datasets["test"], batch_size=self.batch_size, num_workers=self.num_workers),
                DataLoader(self.ood_datasets["test"], batch_size=self.batch_size, num_workers=self.num_workers)]