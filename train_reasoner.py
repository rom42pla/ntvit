from datetime import datetime
from datasets.dreamer import DREAMERDataset
from datasets.amigos import AMIGOSDataset
from models.ae import AutoEncoder

import os
from os import makedirs
from os.path import join, isdir
import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

from models.reasoner import Reasoner

if __name__ == "__main__":
    ##############################
    # ARGS
    ##############################
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices={"dreamer", "amigos"},
        default="dreamer",
        help="Type of dataset",
    )
    parser.add_argument("--layers", type=int, default=4, help="Number of layers to use")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate of the optimizer",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    args = parser.parse_args()

    date = datetime.now().strftime("%Y%m%d_%H%M")
    project_name = f"eeg2ecg"
    subtitle = f"{date}_r_{args.dataset_type}"
    project_path = f"logs/{subtitle}"
    if not isdir(project_path):
        makedirs(project_path)

    torch.set_float32_matmul_precision("high")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "gpu not available"

    if args.dataset_type == "dreamer":
        dataset_path = "../../datasets/dreamer"
        dataset = DREAMERDataset(
            path=dataset_path,
            discretize_labels=True,
            normalize_eegs=True,
            window_size=2,
            window_stride=1,
        )
    elif args.dataset_type == "amigos":
        dataset_path = "../../datasets/amigos"
        dataset = AMIGOSDataset(
            path=dataset_path,
            discretize_labels=True,
            normalize_eegs=True,
            window_size=2,
            window_stride=1,
        )
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [0.8, 0.2])

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=os.cpu_count() - 1,
        pin_memory=True,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=os.cpu_count() - 1,
        pin_memory=True,
    )

    ecgs_ae = AutoEncoder.load_from_checkpoint(join("models", "weights", "dreamer-ae-ecgs.ckpt"), map_location=device)
    eegs_ae = AutoEncoder.load_from_checkpoint(join("models", "weights", "dreamer-ae-eegs.ckpt"), map_location=device)
    reasoner = Reasoner(
        eegs_ae=eegs_ae,
        ecgs_ae=ecgs_ae,
    ).to(device)

    wandb.login()
    wandb_logger = WandbLogger(
        project=project_name,
        name=subtitle,
        save_dir=project_path,
        log_model=False,
    )
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu" if device == "cuda" else "cpu",
        precision="16-mixed" if device == "cuda" else 32,
        logger=wandb_logger,
        max_epochs=2000,
        callbacks=[
            # EarlyStopping(monitor="val/loss", min_delta=0, patience=20),
            # ModelCheckpoint(
            #     monitor="val/loss",
            #     dirpath=project_path,
            #     filename=f"{args.dataset_type}-reasoner-" + "{epoch:02d}",
            # ),
        ],
    )
    trainer.fit(reasoner, dataloader_train, dataloader_val)
