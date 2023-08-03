"""
MaskCycleGAN-VC models as described in https://arxiv.org/pdf/2102.12841.pdf
this code copy from https://github.com/GANtastic3/MaskCycleGAN-VC
"""

from torch.utils.data.dataset import Dataset
import numpy as np


class VCDataset(Dataset):
    def __init__(self, ds, n_frames=64, valid=False):
        self.n_frames = n_frames
        self.valid = valid
        self.length = len(ds)

        train_data_index_subset = np.arange(len(ds))
        np.random.shuffle(train_data_index_subset)

        train_data = list()

        for index in train_data_index_subset:
            data = ds[index]
            frames_total = data.shape[1]
            assert frames_total >= n_frames
            start = np.random.randint(frames_total - n_frames + 1)
            end = start + n_frames
            train_data.append(data[:, start:end])

        train_data = np.array(train_data)
        self.dataset = train_data

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
