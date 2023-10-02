from datetime import datetime
from datasets.dreamer import DREAMERDataset
from datasets.amigos import AMIGOSDataset

import os
from os import makedirs
from os.path import join, isdir
import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

if __name__ == "__main__":
    ##############################
    # ARGS
    ##############################
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type",
                        type=str,
                        choices={"dreamer", "amigos"},
                        default="dreamer",
                        help="Type of dataset")
    parser.add_argument("--model_type",
                        type=str,
                        choices={"cnn", "attn"},
                        default="attn",
                        help="Type of model used")
    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        help="Batch size")
    args = parser.parse_args()

    date = datetime.now().strftime("%Y%m%d_%H%M")
    project_name = f"eeg2ecg"
    project_path = join(".")
    if not isdir(project_path):
        makedirs(project_path)
    
    torch.set_float32_matmul_precision("high")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "gpu not available"

    if args.dataset_type == "dreamer":
        dataset_path = "../../datasets/dreamer"
        dataset = DREAMERDataset(
            path=dataset_path,
            discretize_labels=True, normalize_eegs=True,
            window_size=2, window_stride=1,
        )
    elif args.dataset_type == "amigos":
        dataset_path = "../../datasets/amigos"
        dataset = AMIGOSDataset(
            path=dataset_path,
            discretize_labels=True, normalize_eegs=True,
            window_size=2, window_stride=1,
        )
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [0.8, 0.2])
    
    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=os.cpu_count()-1, pin_memory=True,
    )
    dataloader_val = DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=os.cpu_count()-1, pin_memory=True,
    )

    if args.model_type == "cnn":
        from models.base import EEG2ECGModel
        model = EEG2ECGModel(
            eeg_channels=len(dataset.eeg_electrodes),
            eeg_sampling_rate=dataset.eeg_sampling_rate,
            ecg_channels=len(dataset.ecg_electrodes),
            ecg_sampling_rate=dataset.ecg_sampling_rate,
            spectrogram_scale=8,
            encoder_layers=4,
            kernel_size=7,
            seconds=dataset.window_size,
        )
    elif args.model_type == "attn":
        from models.attn import EEG2ECGModel
        model = EEG2ECGModel(
            eeg_channels=len(dataset.eeg_electrodes),
            eeg_sampling_rate=dataset.eeg_sampling_rate,
            ecg_channels=len(dataset.ecg_electrodes),
            ecg_sampling_rate=dataset.ecg_sampling_rate,
            spectrogram_scale=4,
            layers=2,
            h_dim=512,
            seconds=dataset.window_size,
        )
    assert model.eeg_samples == dataset.eeg_samples_per_window
    assert model.ecg_samples == dataset.ecg_samples_per_window

    wandb.login()
    wandb_logger = WandbLogger(project=project_name, name=f"{date}_{args.dataset_type}", save_dir=project_path, log_model=True)
    trainer = pl.Trainer(
        devices=1, accelerator="gpu" if device=="cuda" else "cpu",
        precision="16-mixed" if device=="cuda" else 32,
        logger=wandb_logger,
        max_epochs=1000,
        # limit_train_batches=0.1,
    )
    trainer.fit(model, dataloader_train, dataloader_val)