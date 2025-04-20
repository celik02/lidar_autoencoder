import torch
import csv
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os


class LidarDataset(Dataset):
    def __init__(self, csv_path_list, transform=None):
        self.min_range = 0.4
        self.scans = []
        if isinstance(csv_path_list, str):
            csv_path_list = [csv_path_list]
        for lidar_path in csv_path_list:
            temp = self._load_csv(lidar_path)
            self.scans.extend(temp)

        self.transform = transform

    def _load_csv(self, csv_dir):
        """
        Loads every .csv in csv_dir as one datapoint.
        """
        dfs = []
        for fname in sorted(os.listdir(csv_dir)):
            if not fname.lower().endswith('.csv'):
                continue
            fullpath = os.path.join(csv_dir, fname)
            # skip the 3 header lines, name the two columns
            df = pd.read_csv(fullpath, skiprows=3, names=['angle', 'range'], encoding='latin1', dtype={'angle': float, 'range': float})
            df['range'] = df['range'].replace([np.inf, -np.inf], self.min_range)
            df['filename'] = fname
            dfs.append(df)
        return dfs

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        df = self.scans[idx]
        angles = torch.from_numpy(df['angle'].values).float()
        ranges = torch.from_numpy(df['range'].values).float()
        sample = torch.stack((angles, ranges), dim=1)  # shape = [N_points, 2]
        if self.transform:
            sample = self.transform(sample)
        return sample


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class SaltPepperNoise(object):
    """Randomly replace some range values with min or max."""
    def __init__(self, prob=0.01, min_val=0.5, max_val=10.0):
        self.prob = prob
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, sample):
        # sample: [N,2] tensor
        N = sample.size(0)
        mask = torch.rand(N) < self.prob
        # choose pepper or salt
        salt = torch.rand(mask.sum()) < 0.5
        vals = torch.where(salt, self.max_val, self.min_val)
        sample[mask, 1] = vals
        return sample


class RandomCircularShift(object):
    """Roll the entire scan by a random offset."""
    def __init__(self, max_shift=None):
        self.max_shift = max_shift

    def __call__(self, sample):
        N = sample.size(0)
        shift = np.random.randint(0, self.max_shift or N)
        return torch.roll(sample, shifts=shift, dims=0)


class RandomScale(object):
    """Multiply ranges by a random factor around 1."""
    def __init__(self, mean=1.0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        factor = np.random.normal(self.mean, self.std)
        sample[:, 1] = sample[:, 1] * factor
        return sample


if __name__ == "__main__":
    # Example usage
    dataset = LidarDataset(csv_path_list='lidardata2/')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in dataloader:
        print(batch)
        print(len(batch))
        # print(batch.shape)
        print(batch[0].shape)
        input("Press Enter to continue...")
