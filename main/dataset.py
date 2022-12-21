import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class SleepSoundDataset(Dataset):
    def __init__(self, npy_file):
        # Read data
        database = np.load(npy_file, allow_pickle=True).item()
        self.data = database['x']
        self.labels = database['y']

        # Data length
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]

        return data, label


if __name__ == '__main__':
    root = '/data/.data/20221215_modified'

    # Valid
    valid_data = SleepSoundDataset('data/valid.npy')
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False)
    for X, y in valid_loader:
        print(X.shape, y.shape)
        print(X, y)
        break
