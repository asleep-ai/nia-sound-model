import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class SleepSoundDataset(Dataset):
    def __init__(self, npy_file, get_name=False):
        # Read data
        database = np.load(npy_file, allow_pickle=True).item()
        self.data = database['x']
        self.labels = database['y']

        # Return name of data or not
        self.get_name = get_name
        if self.get_name:
            self.names = database['name']

        # Data length
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        x = torch.from_numpy(x).float()
        y = torch.tensor(y).long()

        if self.get_name:
            return x, y, self.names[idx]
        else:
            return x, y


if __name__ == '__main__':
    root = '/data/.data/20221215_modified'

    # Valid
    valid_data = SleepSoundDataset('data/valid.npy', get_name=True)
    valid_loader = DataLoader(valid_data, batch_size=2, shuffle=False)
    for X, y, names in valid_loader:
        print(X.shape, y.shape)
        print(X, y)
        print(list(names))
        break
