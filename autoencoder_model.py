import torch
import torch.nn as nn
import torch.nn.functional as F


class LidarAutoencoder(nn.Module):
    def __init__(self, in_channels=2, latent_channels=2, kernel_size=3,  latent_length=50):
        super().__init__()
        pad = kernel_size // 2
        # Encoder: in_channels→latent_channels
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size, padding=pad, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size, padding=pad, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, latent_channels, kernel_size, padding=pad, padding_mode='circular'),
            nn.ReLU(inplace=True),
        )
        # **POOL to reduce N → latent_length**
        self.pool = nn.AdaptiveAvgPool1d(latent_length)

        # Decoder: latent_channels→in_channels
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_channels, 32, kernel_size, padding=pad, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 16, kernel_size, padding=pad, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, in_channels, kernel_size, padding=pad, padding_mode='circular'),
        )

    def forward(self, x):
        # x: [B, N_points, 2] → [B, 2, N_points]
        x = x.permute(0, 2, 1)
        z = self.encoder(x)
        z = self.pool(z)                   # [B, C_latent, 50]
        # **UPSAMPLE back to original N_in**
        z = F.interpolate(z, size=x.size(2),
                          mode='linear', align_corners=True)
        out = self.decoder(z)
        # back to [B, N_points, 2]
        return out.permute(0, 2, 1)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataloader import LidarDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cvs_path_list = ['lidardata/', 'lidardata2/']
    ds = LidarDataset(csv_path_list=cvs_path_list)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2)

    model = LidarAutoencoder(latent_channels=1).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # one training epoch
    model.train()
    for i in range(30):
        print(f"Epoch {i+1}")
        for batch in loader:
            batch = batch.to(device)               # [B, N, 2]
            recon = model(batch)                   # [B, N, 2]
            loss = criterion(recon, batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
            print(f"loss: {loss.item():.4f}")
