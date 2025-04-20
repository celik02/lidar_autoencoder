import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from autoencoder_model import LidarAutoencoder
from dataloader import LidarDataset


class LitLidarAutoencoder(pl.LightningModule):
    def __init__(
        self,
        csv_path_list,
        in_channels: int = 2,
        latent_channels: int = 1,
        kernel_size: int = 3,
        latent_length: int = 50,
        batch_size: int = 8,
        lr: float = 1e-4,
        val_split: float = 0.2,
        num_workers: int = 2,
    ):
        super().__init__()
        # save all init args to self.hparams automatically
        self.save_hyperparameters()
        # model + loss
        self.model = LidarAutoencoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            kernel_size=kernel_size,
            latent_length=latent_length,
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        recon = self(batch)
        loss = self.criterion(recon, batch)
        # log train loss on epoch level
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        recon = self(batch)
        loss = self.criterion(recon, batch)
        # log val loss on epoch level
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def prepare_data(self):
        # nothing to download; instantiate full dataset
        self.dataset = LidarDataset(csv_path_list=self.hparams.csv_path_list)

    def setup(self, stage=None):
        # split once
        n = len(self.dataset)
        val_n = int(n * self.hparams.val_split)
        train_n = n - val_n
        self.train_ds, self.val_ds = random_split(
            self.dataset, [train_n, val_n],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )


if __name__ == "__main__":
    # paths to your lidar folders
    csv_paths = ["lidardata/", "lidardata2/"]

    model = LitLidarAutoencoder(csv_path_list=csv_paths)

    # logger + checkpoint callback
    logger = TensorBoardLogger("tb_logs", name="lidar_ae")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", save_top_k=3, mode="min"
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=30,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model)
