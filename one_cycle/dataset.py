import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from EDF import EDFReader

class SleepSoundDataset(Dataset):
    def __init__(self, root, train):
        self.data_root = root
        mode = 'train' if train else 'test'
        self.data_dir = os.path.join(self.data_root, mode)

        # List all the files
        self.label_file_list = self.get_label_file_list(self.data_dir)
        self.data_file_list = self.get_data_file_list(self.data_dir, self.label_file_list)

        # Read all data and label
        self.mel_data, self.label_data = [], []
        for (data_file, label_file) in zip(self.data_file_list, self.label_file_list):
            mel_data, label_data = self.load_data(data_file, label_file)
            self.mel_data.append(mel_data)
            self.label_data.append(label_data)

        # Data length
        self.len = len(self.mel_data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        mel_data = self.mel_data[idx]
        label_data = self.label_data[idx]
        return mel_data, label_data

    def get_label_file_list(self, label_dir):
        label_file_list = [os.path.join(label_dir, x) for x in os.listdir(label_dir) if x.endswith('.json')]
        return label_file_list

    def get_data_file_list(self, data_dir, label_file_list):
        data_file_list = []
        for label_file in label_file_list:
            label_file = os.path.basename(label_file)[:-5]
            data_file = label_file + '-raw.edf'
            data_file = os.path.join(data_dir, data_file)
            assert os.path.exists(data_file), f'{data_file} does not exist.'
            data_file_list.append(data_file)
        return data_file_list

    def load_data(self, edf_file, label_file):
        """Load EDF file and label file to return Mel Spectrogram data and OSA label."""

        # Load data
        mel_data = []
        edfObj = EDFReader(edf_file)
        for i in range(edfObj.meas_info['n_records']):
            data = edfObj.readBlock(i)
            data = data[4:]     # Take only Mel data
            mel_data.append(data)
        mel_data = np.concatenate(mel_data, axis=1)

        # Data preprocessing
        mel_data = self.data_preprocess(mel_data)
        mel_data = torch.from_numpy(mel_data).float()

        # Load label
        with open(label_file) as f:
            label_data = json.load(f)
        label = label_data['Test_Result']['OSA_Risk']
        label = 1 if label == 'Y' else 0
        label = torch.tensor(label).long()

        return mel_data, label

    def data_preprocess(self, mel_data: np.ndarray) -> np.ndarray:
        mel_data = mel_data.mean(axis=1)
        return mel_data

if __name__ == '__main__':
    root = '/HDD/nia/data'

    # Train
    training_data = SleepSoundDataset(root=root, train=True)
    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
    for X, y in train_dataloader:
        print(X.shape, y.shape)
        print(X, y)
        break

    # Test
    test_data = SleepSoundDataset(root=root, train=False)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    for X, y in test_dataloader:
        print(X.shape, y.shape)
        print(X, y)
        break
