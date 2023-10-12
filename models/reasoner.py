from typing import Any, Dict, Optional, Union
from math import ceil, floor, log2
from einops.layers.torch import Rearrange
from torch import nn
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import torchaudio
import torchmetrics
import wandb

from models.ae import AutoEncoder

try:
    from plots import (
        plot_reconstructed_ecgs_waveforms,
        plot_reconstructed_spectrograms,
    )
    from models.sdtw import SoftDTW
except:
    print("error loading libraries")


class Reasoner(pl.LightningModule):
    def __init__(
        self,
        eegs_ae,
        ecgs_ae,
        layers: int = 3,
        learning_rate: float = 1e-3,
        activation_fn: str = "leaky_relu",
        norm_fn: str = None,
        dropout: float = 0.0,
    ):
        super(Reasoner, self).__init__()

        self.automatic_optimization = False

        ##################################
        ##################################
        # OPTIMIZER
        ##################################
        ##################################
        self.learning_rate = learning_rate

        ##################################
        ##################################
        # NEURAL NETWORK
        # PARAMETERS
        ##################################
        ##################################
        self.layers = layers

        ####################################
        # MODULES
        ####################################
        if activation_fn == "gelu":
            self.activation_fn = nn.GELU()
        elif activation_fn == "leaky_relu":
            self.activation_fn = nn.LeakyReLU()
        elif activation_fn == "selu":
            self.activation_fn = nn.SELU()
        if norm_fn == "instance":
            self.norm_fn = nn.InstanceNorm2d
        elif norm_fn == "batch":
            self.norm_fn = nn.BatchNorm2d
        elif norm_fn in {None, "none"}:
            self.norm_fn = None
        self.eegs_ae = eegs_ae
        self.ecgs_ae = ecgs_ae
        # for module in [self.eegs_ae, self.ecgs_ae]:
        #     for p in module.parameters():
        #         p.requires_grad = False
        self.h_dim = eegs_ae.h_dim
        self.reasoner = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim * 2),
            self.activation_fn,
            nn.Linear(self.h_dim * 2, self.h_dim * 2),
            self.activation_fn,
            nn.Linear(self.h_dim * 2, self.h_dim * 2),
            self.activation_fn,
            nn.Linear(self.h_dim * 2, self.h_dim),
        )

        self.save_hyperparameters(ignore=["eegs_ae", "ecgs_ae"])

    def configure_optimizers(self):
        # optimizers
        opt = torch.optim.AdamW(
            # self.reasoner.parameters(),
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.5, 0.999),
            weight_decay=0.01,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt,
            T_0=10,
            # eta_min=self.learning_rate * 1e-1,
            eta_min=1e-5,
            verbose=False,
        )
        return [opt], [
            {"scheduler": sch, "interval": "step"},
        ]

    def shared_step(self, batch, batch_idx):
        has_trainer = False
        try:
            self.trainer
            has_trainer = True
        except Exception:
            print("trainer not found")

        # retrieves the optimizers
        if self.training and has_trainer:
            opt = self.optimizers()
            sch = self.lr_schedulers()

        outs: Dict[str, Any] = {}
        outs.update(self(batch))

        # computes the reconstruction loss between latent representations
        # outs["loss_rec_latent"] = self.reconstruction_loss(
        #     outs["eegs_latent_transformed"], outs["ecgs_latent"], norm=2
        # )
        outs["loss_rec_spec"] = self.reconstruction_loss(
            outs["ecgs_mel_spectrogram_gen"], outs["ecgs_mel_spectrogram_gt"], norm=1
        )
        outs["loss"] = sum(v for k, v in outs.items() if k.startswith("loss"))

        if self.training and has_trainer:
            opt.zero_grad(set_to_none=True)
            self.manual_backward(outs["loss"])
            self.clip_gradients(
                opt,
                gradient_clip_val=1.0,
                gradient_clip_algorithm="norm",
            )
            opt.step()
            sch.step(self.current_epoch + batch_idx / self.get_dataloader_length())
            outs["lr"] = sch.get_last_lr()[-1]

        # asserts that no loss is corrupted
        for loss_name, loss in [
            (k, v) for k, v in outs.items() if k.startswith("loss")
        ]:
            assert not torch.isnan(
                loss
            ).any(), f"{loss_name} has become None at epoch {self.current_epoch} and step {batch_idx}"

        ##################################
        ##################################
        # LOGGING
        ##################################
        ##################################
        if has_trainer and batch_idx == 0:
            outs["images/mel_rec"] = wandb.Image(
                plot_reconstructed_spectrograms(
                    sg_pred=outs["ecgs_mel_spectrogram_gen"][0],
                    sg_gt=outs["ecgs_mel_spectrogram_gt"][0],
                    vmin=0,
                ),
                caption="generated Mel spectrograms",
            )
            outs["images/latent"] = wandb.Image(
                plot_reconstructed_spectrograms(
                    sg_pred=outs["eegs_latent_transformed"][:8].unsqueeze(1),
                    sg_gt=outs["ecgs_latent"][:8].unsqueeze(1),
                    vmin=0,
                ),
                caption="generated latents",
            )
            self.log_dict(
                {
                    f"{self.get_phase_name()}/{k}": v
                    for k, v in outs.items()
                    if (isinstance(v, torch.Tensor) and v.numel() == 1)
                    or isinstance(v, (float, int))
                },
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            try:
                wandb.log(
                    {
                        f"{self.get_phase_name()}/{k}": v
                        for k, v in outs.items()
                        if isinstance(v, wandb.Image)
                    }
                )
            except:
                pass
        return outs

    def forward(self, batch):
        ##################################
        ##################################
        # INPUTS
        ##################################
        ##################################
        assert isinstance(batch, dict)
        eegs_gt = batch["eegs"].to(self.device)
        assert not torch.isnan(eegs_gt).any(), "there are nans in the eegs"
        ecgs_gt = batch["ecgs"].to(self.device)
        assert not torch.isnan(ecgs_gt).any(), "there are nans in the input ecgs"
        batch_size = eegs_gt.shape[0]

        ##################################
        ##################################
        # SPECTROGRAM
        ##################################
        ##################################
        eegs_mel_spectrogram_gt = self.eegs_ae.waveform_to_mel_spectrogram(
            waveform=eegs_gt,
            limit_outputs=True,
        )
        ecgs_mel_spectrogram_gt = self.ecgs_ae.waveform_to_mel_spectrogram(
            waveform=ecgs_gt,
            limit_outputs=True,
        )

        ##################################
        ##################################
        # ENCODER
        ##################################
        ##################################
        with profiler.record_function("encoder".upper()):
            eegs_latent = self.eegs_ae.encode_mel_spectrogram(
                mel_spectrogram=eegs_mel_spectrogram_gt,
                pooled_outputs=True,
                return_hidden_states=False,
                expected_shape=(
                    batch_size,
                    self.h_dim,
                ),
            )
            ecgs_latent = self.ecgs_ae.encode_mel_spectrogram(
                mel_spectrogram=ecgs_mel_spectrogram_gt,
                pooled_outputs=True,
                return_hidden_states=False,
                expected_shape=(
                    batch_size,
                    self.h_dim,
                ),
            )
            assert eegs_latent.shape == ecgs_latent.shape

        ##################################
        ##################################
        # REASONER
        ##################################
        ##################################
        with profiler.record_function("reasoner".upper()):
            eegs_latent_transformed = self.reasoner(eegs_latent)

        ##################################
        ##################################
        # DECODER
        ##################################
        ##################################
        with profiler.record_function("decoder".upper()):
            mel_spectrogram_gen = self.ecgs_ae.decode_mel_spectrogram(
                latent=eegs_latent_transformed,
                out_length=1,
            )
        return {
            "eegs_latent": eegs_latent,
            "ecgs_latent": ecgs_latent,
            "eegs_latent_transformed": eegs_latent_transformed,
            "eegs_mel_spectrogram_gt": eegs_mel_spectrogram_gt,
            "ecgs_mel_spectrogram_gt": ecgs_mel_spectrogram_gt,
            "ecgs_mel_spectrogram_gen": mel_spectrogram_gen,
        }

    def reconstruction_loss(self, pred, gt, norm=1):
        if norm == 1:
            loss_fn = F.l1_loss
        elif norm == 2:
            loss_fn = F.mse_loss
        else:
            raise Exception(f"norm must be 1 or 2, got {norm}")
        return loss_fn(input=pred, target=gt)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def get_dataloader_length(self):
        if self.training:
            return len(self.trainer.train_dataloader)
        else:
            return len(self.trainer.val_dataloader)

    def get_phase_name(self):
        return "train" if self.training else "val"

    @staticmethod
    def shape_check(tensor, expected_shape, name=None):
        error_msg = f"expected shape {expected_shape}, got {tuple(tensor.shape)}"
        if name is not None:
            error_msg += f" for tensor {name}"
        assert tensor.shape == expected_shape, error_msg

    @staticmethod
    def get_padding_for_patches(height, width, kernel_size):
        padding_w = kernel_size * ceil(width / kernel_size) - width
        padding_h = kernel_size * ceil(height / kernel_size) - height
        if padding_w % 2 == 0:
            padding_left = padding_right = padding_w // 2
        else:
            padding_left = floor(padding_w / 2)
            padding_right = padding_left + 1
        if padding_h % 2 == 0:
            padding_up = padding_down = padding_h // 2
        else:
            padding_up = floor(padding_h / 2)
            padding_down = padding_up + 1
        return padding_left, padding_right, padding_up, padding_down


if __name__ == "__main__":
    from icecream import ic

    model = AutoEncoder(
        channels=14,
        sampling_rate=128,
        seconds=2,
        layers=4,
        min_frequency=1,
        max_frequency=50,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.randn(
        [8, model.channels, model.seconds * model.sampling_rate],
        device=model.device,
    )
    print(model)
    with profiler.profile(
        with_stack=False,
        profile_memory=True,
        use_cuda=True if torch.cuda.is_available() else False,
    ) as prof:
        # sample_outs = model.shared_step(sample_batch, 0)
        sample_outs = model.shared_step(sample_batch, 0)
    print(
        prof.key_averages().table(
            sort_by="cuda_time",
            row_limit=10,
        )
    )
