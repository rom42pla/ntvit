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
    parser.add_argument(
        "--signal_type",
        type=str,
        choices={"eegs", "ecgs"},
        default="ecgs",
        help="Type of signal to be used",
    )
    parser.add_argument("--layers", type=int, default=4, help="Number of layers to use")
    parser.add_argument("--h_dim", type=int, default=1024, help="Size of the latent")
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
    subtitle = f"{date}_ae_{args.signal_type}_{args.dataset_type}"
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
    if args.signal_type == "eegs":
        channels = len(dataset.eeg_electrodes)
        sampling_rate = dataset.eeg_sampling_rate
        min_frequency, max_frequency = 0, 30
    else:
        channels = len(dataset.ecg_electrodes)
        sampling_rate = dataset.ecg_sampling_rate
        min_frequency, max_frequency = 1, 60
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [0.8, 0.2])

    def custom_collate(batch):
        key = "eegs" if args.signal_type == "eegs" else "ecgs"
        return torch.stack([torch.as_tensor(sample[key]) for sample in batch])

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=os.cpu_count() - 1,
        pin_memory=True,
        collate_fn=custom_collate,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=os.cpu_count() - 1,
        pin_memory=True,
        collate_fn=custom_collate,
    )

    model = AutoEncoder(
        seconds=dataset.window_size,
        channels=channels,
        sampling_rate=sampling_rate,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        layers=args.layers,
        h_dim=args.h_dim,
        learning_rate=args.learning_rate,
    )

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
        max_epochs=1000,
        callbacks=[
            EarlyStopping(monitor="val/loss", min_delta=0, patience=20),
            ModelCheckpoint(
                monitor="val/loss",
                dirpath=project_path,
                filename=f"{args.dataset_type}-ae-{args.signal_type}-" + "{epoch:02d}",
            ),
        ],
    )
    trainer.fit(model, dataloader_train, dataloader_val)
