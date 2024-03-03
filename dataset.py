from copy import deepcopy
import json
import gc
import os
from os import listdir, makedirs
from os.path import isdir, join, exists, normpath
from typing import Optional, Any, Union, List, Dict
from multiprocessing import Pool
from math import ceil, sqrt

import mne
import nibabel as nib
import numpy as np
import scipy.io as sio
import einops
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class EEG2fMRIPreprocessedDataset(Dataset):
    def __init__(
        self,
        path: str,
        normalize_eegs: bool = True,
        normalize_fmris: bool = True,
    ):
        super().__init__()
        gc.collect()

        assert isdir(path)
        self.path: str = path
        
        self.metas_path = join(self.path, "metas.json")
        with open(self.metas_path, "r") as fp:
            self.metas: Dict[str, Any] = json.load(fp)
        self.metas.pop("samples")
            
        # eegs-related infos
        self.eegs_sampling_rate: int = self.metas["eegs_info"]["sampling_rate"]
        self.eegs_electrodes: List[str] = [f"E{i:02}" for i in range(self.metas["eegs_info"]["shape"][0])]
        assert isinstance(normalize_eegs, bool)
        self.normalize_eegs: bool = normalize_eegs
        self.eegs_seconds = self.metas["eegs_info"]["seconds"]
        self.eegs_samples = ceil(self.eegs_sampling_rate * self.eegs_seconds)
        
        # fMRI-related infos
        assert isinstance(normalize_fmris, bool)
        self.normalize_fmris: bool = normalize_fmris
        self.fmris_shape = self.metas["fmris_info"]["shape"]

        # subjects-related infos
        self.samples: List[Dict[str, Union[str, np.ndarray]]] = []
        for subject_id in listdir(self.path):
            if not isdir(join(self.path, subject_id)):
                continue
            fmris_path = join(self.path, subject_id, "fmris")
            eegs_path = join(self.path, subject_id, "eegs")
            for run_name in listdir(fmris_path):
                self.samples.append({
                    "subject_id": subject_id,
                    "eegs_path": join(eegs_path, run_name),
                    "fmris_path": join(fmris_path, run_name),
                })
        self.subject_ids: List[str] = list({sample["subject_id"] for sample in self.samples})
        gc.collect()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict[str, Union[int, str, np.ndarray]]:
        sample_data = deepcopy(self.samples[i])
        for signal_type in ["eegs", "fmris"]:
            sample_data[signal_type] = np.load(sample_data[f"{signal_type}_path"])
        # normalizes values
        if self.normalize_eegs:
            sample_data["eegs"] = self.normalize(sample_data["eegs"], vmin=-1, vmax=1)
        if self.normalize_fmris:
            sample_data["fmris"] = self.normalize(sample_data["fmris"], vmin=0, vmax=1)
        return sample_data

    @staticmethod
    def get_subject_ids_static(path: str) -> List[str]:
        assert isdir(path)
        subject_ids = sorted(os.listdir(path))
        return subject_ids

    # def parse_samples(self):
    #     samples = []
    #     for subject_id in os.listdir(self.path):
    #         for filename in os.listdir(join(self.path, subject_id, "eegs")):
    #             sample = {
    #                 "subject_id": subject_id,
    #                 "sample_id": filename.split(".")[0],
    #                 "eegs_path": join(self.path, subject_id, "eegs", filename),
    #                 "fmris_path": join(self.path, subject_id, "fmris", filename),
    #             }
    #             assert exists(
    #                 sample["eegs_path"]
    #             ), f"eegs {sample['eegs_path']} does not exists"
    #             assert exists(
    #                 sample["fmris_path"]
    #             ), f"fmris{sample['fmris_path']} does not exists"
    #             samples.append(sample)
    #     return samples

    def normalize(self, x: List[np.ndarray], mode="minmax", vmin=0, vmax=1):
        if mode == "std":
            # mean = x.mean(axis=0)
            # std = x.std(axis=0)
            # waveform_scaled = (x - mean) / std
            # min = waveform_scaled.min(axis=0)
            # max = waveform_scaled.max(axis=0)
            # waveform_scaled = 2 * ((waveform_scaled - min) / (max - min)) - 1
            raise NotImplementedError
        if mode == "minmax":
            if len(x.shape) == 2:  # eegs are normalized by channel
                min = x.min(axis=1, keepdims=True)
                max = x.max(axis=1, keepdims=True)
            elif len(x.shape) == 3:  # fmris are normalized globally
                min = x.min()
                max = x.max()
            # x_normalized = 2 * ((x - min) / (max - min)) - 1
            x_normalized = (vmax - vmin) * ((x - min) / (max - min)) + vmin
            assert (x_normalized <= vmax).all() and (x_normalized >= vmin).all()
        return x_normalized

    @staticmethod
    def plot_fmris(fmris, save_path: str, mode="mip", scale=4):
        assert len(fmris.shape) == 3
        import matplotlib

        matplotlib.use("agg")
        import matplotlib.pyplot as plt

        if mode == "mip":
            fig, ax_pc = plt.subplots(1, 1, figsize=(scale, scale), dpi=300)
            mip = np.max(fmris, axis=0)  # Maximum Intensity Projection along z-axis
            im = ax_pc.imshow(mip, cmap="gray")

            # colorbar
            cbar_ax = fig.add_axes(
                [0.85, 0.15, 0.01, 0.65]
            )  # [left, bottom, width, height]
            fig.colorbar(im, cax=cbar_ax)

            fig.subplots_adjust(right=0.8)  # make room for colorbar
            fig.suptitle(f"Maximum Intensity Projection of fMRI")
            fig.savefig(save_path)
            fig.clf()
        elif mode == "pc":
            fmris = fmris / fmris.max()

            # Generate meshgrid for coordinates
            z, y, x = np.meshgrid(np.arange(fmris.shape[0]), np.arange(fmris.shape[1]), np.arange(fmris.shape[2]), indexing='ij')

            # Flatten the meshgrids to get xs, ys, zs
            xs = x.flatten()
            ys = y.flatten()
            zs = z.flatten()

            # Flatten tensor for coloring and transparency
            t_low = fmris.flatten()

            # 1. Dimensionality and Clutter: Apply threshold 
            threshold = np.percentile(fmris, q=90)
            mask = t_low > threshold

            # 2. Transparency: Adjust transparency mapping (using a simple linear scaling in this example)
            alpha_values = t_low/t_low.max()
            # alpha_values = alpha_values * 0.8 + 0.2  # This ensures values are in the range [0.2, 1]

            # 3. Color: Use a diverging colormap
            colormap = plt.cm.coolwarm

            # 4. Size: Map intensity to point sizes
            point_sizes = (t_low * 50) + 10  # Adjust scaling factor and base size as needed

            xs = xs[mask]
            ys = ys[mask]
            zs = zs[mask]
            t_low = t_low[mask]
            alpha_values = alpha_values[mask]
            point_sizes = point_sizes[mask]

            fig = plt.figure(figsize=(scale, scale*1.25), dpi=300, tight_layout=True)
            ax_pc = fig.add_subplot(projection='3d')
            
            im = ax_pc.scatter(xs, ys, zs, c=t_low, cmap=colormap, alpha=alpha_values, s=point_sizes, vmin=0, vmax=1)
            cbar = fig.colorbar(im, ax=ax_pc, orientation='horizontal', pad=0.1, aspect=2**6)
            cbar.set_label('Intensity')

            ax_pc.set_xlabel('X')
            ax_pc.set_ylabel('Y')
            ax_pc.set_zlabel('Z')

            fig.savefig(save_path)
            fig.clf()


    @staticmethod
    def plot_eegs(eegs, save_path: str, scale=8):
        assert len(eegs.shape) == 2
        import matplotlib

        matplotlib.use("agg")
        import matplotlib.pyplot as plt

        num_channels = eegs.shape[0]
        grid_size = ceil(sqrt(num_channels))

        fig, axs = plt.subplots(
            grid_size, grid_size, figsize=(scale, scale), dpi=300, tight_layout=True
        )
        axs = axs.flatten()

        # Plot each channel
        for i in range(num_channels):
            axs[i].plot(eegs[i, :])

        # Turn off any unused subplots
        for i in range(num_channels, grid_size * grid_size):
            axs[i].axis("off")

        # Save the figure
        fig.suptitle(f"EEGs")
        fig.savefig(save_path)
        plt.close(fig)

if __name__ == "__main__":
    sampling_rate = 512
    dataset_path = f"../../datasets/oddball_preprocessed"

    print("loading oddball preprocessed dataset")
    dataset = EEG2fMRIPreprocessedDataset(
        path=dataset_path,
        normalize_eegs=False,
        normalize_fmris=False,
    )
    print(len(dataset))
    print(len(dataset.subject_ids))
    sample = dataset[12]

    EEG2fMRIPreprocessedDataset.plot_fmris(
        fmris=sample["fmris"], save_path="fmris_mip.png", mode="pc"
    )
    EEG2fMRIPreprocessedDataset.plot_eegs(eegs=sample["eegs"][:4], save_path="eegs.png")
    for sample in tqdm(dataset, desc="validating dataset"):
        pass
    print("eegs shape", dataset[0]["eegs"].shape)
    print("fmris shape", dataset[0]["fmris"].shape)
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [0.8, 0.2])
    print(f"oddball preprocessed loaded")
