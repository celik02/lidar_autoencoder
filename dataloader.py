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
            df = pd.read_csv(fullpath, skiprows=3, names=['angle', 'range'])
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
