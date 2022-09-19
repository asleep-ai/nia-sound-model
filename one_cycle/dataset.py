import os
import pandas as pd
import glob
from torchvision.io import read_image
import torch
import json
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import numpy as np


class SleepSoundDataset(Dataset):
    def __init__(self, root, train):
        self.data_root = root
        self.train = 'train' if train else 'test'
        self.data_dir = os.path.join(self.data_root, self.train)

    def __len__(self):
        return 8 if self.train == 'train' else 2

    def __getitem__(self, idx):
        sound_paths = sorted(glob.glob(os.path.join(self.data_dir, 'sound/*.npy')))
        sound_path = sound_paths[idx]
        sound = torch.from_numpy(np.load(sound_path)).float()

        label_paths = sorted(glob.glob(os.path.join(self.data_dir, 'label/*.json')))
        label_path = label_paths[idx]
        with open(label_path) as f:
            label_data = json.load(f)
        ahi = label_data['Report']['Obstructive_Apnea_Index']
        label = 1 if float(ahi) > 15 else 0
        label = torch.tensor(label)

        # Get feature from sound
        sound = torch.mean(input=sound, dim=[0, 2])
        return sound, label

    # def __getitem__(self, idx):
    #     sound_paths = sorted(glob.glob(os.path.join(self.data_dir, 'sound/*.npy')))
    #     sound_path = sound_paths[idx]
    #     sound = torch.from_numpy(np.load(sound_path)).float()

    #     label_paths = sorted(glob.glob(os.path.join(self.data_dir, 'label/*.json')))
    #     label_path = label_paths[idx]
    #     with open(label_path) as f:
    #         label_data = json.load(f)
    #     ahi = label_data['Report']['Obstructive_Apnea_Index']
    #     label = [1., 0.] if float(ahi) > 15 else [0., 1.]
    #     label = torch.tensor(label)

    #     # Get feature from sound
    #     sound = torch.mean(input=sound, dim=[0, 2])
    #     return sound, label


if __name__ == '__main__':

    training_data = SleepSoundDataset(
        root="data",
        train=True
    )

    test_data = SleepSoundDataset(
        root="data",
        train=False
    )
    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
    for X, y in train_dataloader:
        print(X.shape, y.shape)
        print(X, y)
        break

