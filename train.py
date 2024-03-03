from datetime import datetime
from dataset import EEG2fMRIPreprocessedDataset

import random
from tqdm import tqdm
import os
from os import makedirs
from os.path import join, isdir
import argparse
import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

from ntvit import NTViT
from utils import download_from_wandb, get_loso_runs, get_kfold_runs, set_seed

if __name__ == "__main__":
    ##############################
    # ARGS
    ##############################
    parser = argparse.ArgumentParser()
    # dataset params
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices={"noddi", "oddball"},
        default="noddi",
        required=True,
        help="Type of dataset",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset folder",
    )
    parser.add_argument(
        "--dont_normalize_eegs",
        action="store_true",
        help="Whether not to normalize the EEGs in [-1, 1]",
    )
    parser.add_argument(
        "--dont_normalize_fmris",
        action="store_true",
        help="Whether not to normalize the fMRIs in [0, 1]",
    )
    parser.add_argument(
        "--validation",
        type=str,
        choices={"loso", "kfold"},
        default="loso",
        help="The validation scheme to use",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="The number of folds in the kfold cross-validation",
    )
    # model params
    parser.add_argument("--eeg_patches_size", type=int, default=8, help="Size of the patches of the EEG spectrogram in the encoder")
    parser.add_argument("--fmri_patches_size", type=int, default=8, help="Size of the fMRI patches in the decoder")
    parser.add_argument("--fmris_downsampling_factor", type=int, default=1, help="Size of the downsampling of the fMRI volumes")
    parser.add_argument("--spectrogram_scale", type=float, default=2, help="Stride of the STFT window when generating the EEG spectrogram")
    parser.add_argument("--activation", type=str, default="softplus", choices={"softplus", "gelu", "selu"}, help="Type of activation function to use")
    parser.add_argument("--layers", type=int, default=1, help="Number of layers to use")
    parser.add_argument("--h_dim", type=int, default=256, help="Size of the latent")
    parser.add_argument(
        "--use_cm_loss",
        action="store_true",
        help="Whether to use the Center of Mass loss",
    )
    parser.add_argument(
        "--use_kl_loss",
        action="store_true",
        help="Whether to use the Kullback-Leibler loss",
    )
    parser.add_argument(
        "--use_disp_loss",
        action="store_true",
        help="Whether to use the displacement loss",
    )
    parser.add_argument(
        "--use_discriminator",
        action="store_true",
        help="Whether to add a discriminator module",
    )
    parser.add_argument(
        "--use_domain_matching",
        action="store_true",
        help="Whether to use the Domain Matching component",
    )
    parser.add_argument(
        "--alpha_disc",
        type=float,
        default=1.,
        help="The alpha used in the discriminator loss",
    )
    # regularization
    parser.add_argument("--dropout", type=float, default=0.2, help="The amount of dropout to use")
    parser.add_argument("--input_noise", type=float, default=0.2, help="The amount of gaussian noise to add to the EEG spectrogram")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="The amount of weight decay to use with AdamW")
    # trainer params
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Learning rate of the optimizer",
    )
    parser.add_argument(
        "--use_lr_scheduler",
        action="store_true",
        help="Whether to use cosine annealing with warm restarts",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=20,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=None,
        help="Maximum number of LOSO runs to do",
    )
    parser.add_argument(
        "--plot_images",
        action="store_true",
        help="Whether to plot images in the logger",
    )
    parser.add_argument(
        "--save_weights",
        action="store_true",
        help="Whether to save the weights of the model",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Whether to enable early stopping",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=random.randint(0, 2**32),
        help="The seed to use for reproducibility purposes",
    )
    parser.add_argument(
        "--label",
        type=str,
        help="The label of the experiment",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    
    # sets the seed
    set_seed(args.seed)

    # working directories
    date = datetime.now().strftime("%Y%m%d_%H%M")
    project_name = f"eeg2fmri"
    subtitle = f"{date}_ds={args.dataset_type}_val={args.validation}"
    if args.label:
        subtitle += f"_lb={args.label}"
    project_path = f"logs/{subtitle}"
    if not isdir(project_path):
        makedirs(project_path)

    torch.set_float32_matmul_precision("high")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset parsing
    dataset = EEG2fMRIPreprocessedDataset(path=args.dataset_path, normalize_eegs=not args.dont_normalize_eegs, normalize_fmris=not args.dont_normalize_fmris)
    subject_ids = dataset.subject_ids
    
    # logger setup
    wandb.login()
    wandb_logger = WandbLogger(
        project=project_name,
        name=subtitle,
        save_dir=project_path,
        log_model=False,
    )

    # loops through the splits
    if args.validation == "loso":
        runs = get_loso_runs(dataset)
    elif args.validation == "kfold":
        runs = get_kfold_runs(dataset, k=args.k)
    if args.max_runs:
        runs = runs[:args.max_runs]
    for i_run, run in tqdm(enumerate(runs), desc="subjects", total=len(runs)):
        # dataset 
        train_indices, val_indices = run["train_indices"], run["val_indices"]
        dataset_train = Subset(dataset, indices=train_indices)
        dataset_val = Subset(dataset, indices=val_indices)
        # dataloaders
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=os.cpu_count() - 1,
            pin_memory=True,
        )
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=os.cpu_count() - 1,
            pin_memory=True,
        )
        # model
        model = NTViT(
            fmris_shape=dataset.fmris_shape,
            eegs_seconds=dataset.eegs_seconds,
            eegs_channels=len(dataset.eegs_electrodes),
            eegs_sampling_rate=dataset.eegs_sampling_rate,
            fmris_downsampling_factor=args.fmris_downsampling_factor,
            normalized_fmris=not args.dont_normalize_fmris,
            # normalized_eegs=not args.dont_normalize_eegs,
            spectrogram_scale=args.spectrogram_scale,
            fmri_patches_size=args.fmri_patches_size,
            eeg_patches_size=args.eeg_patches_size,
            use_domain_matching=args.use_domain_matching,
            use_discriminator=args.use_discriminator,
            alpha_disc=args.alpha_disc,
            activation=args.activation,
            layers=args.layers,
            h_dim=args.h_dim,
            dropout=args.dropout,
            input_noise=args.input_noise,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            use_lr_scheduler=args.use_lr_scheduler,
            use_kl_loss=args.use_kl_loss,
            use_cm_loss=args.use_cm_loss,
            use_disp_loss=args.use_disp_loss,
            plot_images=args.plot_images,
            logger_prefix=f"run_{run['subject_id']}",
        )
        # callbacks
        callbacks = []
        if args.save_weights:
            callbacks.append(
                ModelCheckpoint(
                    monitor=f"run_{run['subject_id']}/val/ssim",
                    dirpath=project_path,
                    filename=f"{args.dataset_type}-" + "{epoch:02d}-" + f"run={run['subject_id']}", 
                    mode="max",
                ),
            )
        if args.early_stopping:
            callbacks.append(
                EarlyStopping(monitor="val/loss_G", min_delta=0, patience=10, verbose=True, mode="min"),
            )
        # trainer setup
        trainer = pl.Trainer(
            devices=1,
            accelerator="gpu" if device == "cuda" else "cpu",
            precision="16-mixed" if device == "cuda" else 32,
            logger=wandb_logger,
            log_every_n_steps=5,
            max_epochs=args.max_epochs,
            callbacks=callbacks,
            enable_checkpointing=args.save_weights,
            enable_model_summary=len(runs) == 1,
        )
        trainer.fit(model, dataloader_train, dataloader_val)
    download_from_wandb(
        user_id=wandb_logger.experiment.entity,
        project_id=wandb_logger.experiment.project,
        run_id=wandb_logger.experiment.id,
        output_folder=join("logs", subtitle),
    )
