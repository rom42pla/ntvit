from copy import deepcopy
import gc
import os
from os import listdir
from os.path import isdir, join, exists
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


class EEGfMRIPreprocessedDataset(Dataset):
    def __init__(
        self,
        path: str,
        eeg_sampling_rate: int = 512,
        normalize_eegs: bool = True,
        normalize_fmris: bool = True,
    ):
        super().__init__()
        gc.collect()

        assert isdir(path)
        self.path: str = path

        # EEG-related infos
        assert (
            isinstance(eeg_sampling_rate, int) and eeg_sampling_rate <= 5000
        ), f"sampling_rate must be below 5000, which is the original value"
        self.eeg_sampling_rate: int = eeg_sampling_rate
        self.eeg_electrodes: List[str] = [
            "Fp1",
            "Fp2",
            "F3",
            "F4",
            "C3",
            "C4",
            "P3",
            "P4",
            "O1",
            "O2",
            "F7",
            "F8",
            "T7",
            "T8",
            "P7",
            "P8",
            "Fz",
            "Cz",
            "Pz",
            "Oz",
            "FC1",
            "FC2",
            "CP1",
            "CP2",
            "FC5",
            "FC6",
            "CP5",
            "CP6",
            "TP9",
            "TP10",
            "POz",
            "ECG",
            "F1",
            "F2",
            "C1",
            "C2",
            "P1",
            "P2",
            "AF3",
            "AF4",
            "FC3",
            "FC4",
            "CP3",
            "CP4",
            "PO3",
            "PO4",
            "F5",
            "F6",
            "C5",
            "C6",
            "P5",
            "P6",
            "AF7",
            "AF8",
            "FT7",
            "FT8",
            "TP7",
            "TP8",
            "PO7",
            "PO8",
            "FT9",
            "FT10",
            "Fpz",
            "CPz",
        ]
        assert isinstance(normalize_eegs, bool)
        self.normalize_eegs: bool = normalize_eegs
        self.eegs_samples = ceil(self.eeg_sampling_rate * 2.16)

        # fMRI-related infos
        assert isinstance(normalize_fmris, bool)
        self.normalize_fmris: bool = normalize_fmris
        self.fmris_shape = (64, 64, 30)

        # subjects-related infos
        self.subject_ids: List[str] = self.get_subject_ids_static(self.path)
        self.samples: List[Dict[str, Union[str, np.ndarray]]] = self.parse_samples()

        # normalizes the data
        # if self.normalize_eegs:
        #     self.eegs_data = self.normalize(self.eegs_data, mode="minmax")
        # if self.normalize_ecgs:
        #     self.ecgs_data = self.normalize(self.ecgs_data, mode="minmax")

        gc.collect()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict[str, Union[int, str, np.ndarray]]:
        sample_data = deepcopy(self.samples[i])
        for signal_type in ["eegs", "fmris"]:
            sample_data[signal_type] = np.load(sample_data[f"{signal_type}_path"])
        if self.normalize_eegs:
            sample_data["eegs"] = self.normalize(sample_data["eegs"])
        if self.normalize_fmris:
            sample_data["fmris"] = self.normalize(sample_data["fmris"])
        return sample_data

    @staticmethod
    def get_subject_ids_static(path: str) -> List[str]:
        assert isdir(path)
        subject_ids = sorted(os.listdir(path))
        return subject_ids

    def parse_samples(self):
        samples = []
        for subject_id in os.listdir(self.path):
            for filename in os.listdir(join(self.path, subject_id, "eegs")):
                sample = {
                    "subject_id": subject_id,
                    "sample_id": filename.split(".")[0],
                    "eegs_path": join(self.path, subject_id, "eegs", filename),
                    "fmris_path": join(self.path, subject_id, "fmris", filename),
                }
                assert exists(
                    sample["eegs_path"]
                ), f"eegs {sample['eegs_path']} does not exists"
                assert exists(
                    sample["fmris_path"]
                ), f"fmris{sample['fmris_path']} does not exists"
                samples.append(sample)
        return samples

    def normalize(self, x: List[np.ndarray], mode="minmax"):
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
            x_normalized = 2 * ((x - min) / (max - min)) - 1
        return x_normalized

    @staticmethod
    def plot_fmris(fmris, save_path: str, mode="mip", scale=4):
        assert len(fmris.shape) == 3
        import matplotlib.pyplot as plt

        if mode == "mip":
            fig, ax = plt.subplots(
                1, 1, figsize=(scale, scale), dpi=300
            )
            mip = np.max(fmris, axis=2) # Maximum Intensity Projection along z-axis
            im = ax.imshow(mip, cmap="gray")
            
            # colorbar
            cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.65])  # [left, bottom, width, height]
            fig.colorbar(im, cax=cbar_ax)
            
            fig.subplots_adjust(right=0.8)  # make room for colorbar
            fig.suptitle(f"Maximum Intensity Projection of fMRI")
            fig.savefig(save_path)
            fig.clf()
            
    @staticmethod
    def plot_eegs(eegs, save_path: str, scale=8):
        assert len(eegs.shape) == 2
        import matplotlib.pyplot as plt
        
        num_channels = eegs.shape[0]
        grid_size = ceil(sqrt(num_channels))
        
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(scale, scale), dpi=300, tight_layout=True)
        axs = axs.flatten()
        
        # Plot each channel
        for i in range(num_channels):
            axs[i].plot(eegs[i, :])
        
        # Turn off any unused subplots
        for i in range(num_channels, grid_size*grid_size):
            axs[i].axis('off')
        
        # Save the figure
        fig.suptitle(f"EEGs")
        fig.savefig(save_path)
        plt.close(fig)


def save_data_to_numpy(dataset_path: str, save_path: str, sampling_rate: int = 256):
    from os import makedirs

    def get_subject_ids_static(path: str) -> List[str]:
        assert isdir(path)
        users_per_signal = {
            "fMRI": set(os.listdir(join(path, "fMRI"))),
            "EEG": set(os.listdir(join(path, "EEG1")) + os.listdir(join(path, "EEG2"))),
        }
        common_users = sorted(list(users_per_signal["fMRI"] & users_per_signal["EEG"]))
        # user 35 is somehow broken
        if "35" in common_users:
            common_users.remove("35")
        return common_users

    subject_ids = get_subject_ids_static(dataset_path)

    global parse_subject_data

    def parse_subject_data(subject_no):
        subject_id: str = subject_ids[subject_no]
        subject_folder = join(save_path, subject_id)
        st = time.time()

        # loads fmri data
        fmri_data_filename = [
            f
            for f in os.listdir(join(dataset_path, "fMRI", subject_id))
            if f.endswith("rest_with_cross.nii.gz")
        ][0]
        fmris = nib.load(join(dataset_path, "fMRI", subject_id, fmri_data_filename))
        fmris = fmris.get_fdata()  # (x z y t)
        # saves to numpy arrays
        fmris_folder = join(subject_folder, "fmris")
        if not isdir(fmris_folder):
            makedirs(fmris_folder)
        for i in range(fmris.shape[-1]):
            np.save(join(fmris_folder, f"{i}.npy"), fmris[:, :, :, i])

        # loads eeg data
        eeg_subject_folder = (
            "EEG1" if subject_id in os.listdir(join(dataset_path, "EEG1")) else "EEG2"
        )
        eeg_data_filename = [
            f
            for f in os.listdir(
                join(dataset_path, eeg_subject_folder, subject_id, "raw")
            )
            if f.endswith("vhdr")
        ][0]
        eegs = mne.io.read_raw_brainvision(
            join(
                dataset_path, eeg_subject_folder, subject_id, "raw", eeg_data_filename
            ),
            preload=True,
            verbose=False,
        )
        sampling_rate = int(eegs.info["sfreq"])
        # Compute the mean across channels for each time point
        mean_amplitude = np.mean(np.abs(eegs._data), axis=0)  # (t)
        mean_amplitude = np.convolve(
            mean_amplitude, np.ones(sampling_rate) / sampling_rate, mode="same"
        )
        # computes the cut-off threshold for cropping preliminary parts
        threshold = np.quantile(
            mean_amplitude,
            (sampling_rate * len(fmris) * 2.16) / len(mean_amplitude),
        )
        # search for when the record must start
        for i in range(len(mean_amplitude)):
            if np.isclose(mean_amplitude[i], threshold, atol=1e-4):
                i_end = ceil(i + sampling_rate + sampling_rate * len(fmris) * 2.16)
                start_time = eegs.times[i]
                end_time = eegs.times[i_end]
                break
        # crops the eegs
        eegs = eegs.crop(
            tmin=start_time, tmax=end_time, include_tmax=True, verbose=False
        )
        # resamples the eegs
        eegs = eegs.resample(sampling_rate, n_jobs=os.cpu_count(), verbose=False)
        # extract the numpy array from the mne raw object
        samples_per_fmri = ceil(sampling_rate * 2.16)
        eegs, _ = eegs[:, :]  # (c t)
        eegs_folder = join(subject_folder, "eegs")
        if not isdir(eegs_folder):
            makedirs(eegs_folder)
        for i in range(fmris.shape[-1]):
            np.save(
                join(eegs_folder, f"{i}.npy"),
                eegs[:, i * samples_per_fmri : i * samples_per_fmri + samples_per_fmri],
            )
        print("loaded subject", subject_id, "in", np.round(time.time() - st, 2))
        return eegs, fmris, subject_id

    st = time.time()
    for i in range(len(subject_ids)):
        parse_subject_data(i)
    print("saved dataset in", np.round(time.time() - st, 2))


if __name__ == "__main__":
    import time

    # dataset_path = "../../datasets/eeg_fmri_preprocessed_512hz"
    dataset_path = "./eeg_fmri_prep_512hz"
    print("loading EEG fMRI preprocessed dataset")
    dataset = EEGfMRIPreprocessedDataset(path=dataset_path, eeg_sampling_rate=512, normalize_eegs=True, normalize_fmris=True)
    sample = dataset[12]

    EEGfMRIPreprocessedDataset.plot_fmris(
        fmris=sample["fmris"], save_path="fmris_mip.png"
    )
    EEGfMRIPreprocessedDataset.plot_eegs(
        eegs=sample["eegs"][:4], save_path="eegs.png"
    )

    print("eegs shape", dataset[0]["eegs"].shape)
    print("fmris shape", dataset[0]["fmris"].shape)
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [0.8, 0.2])
    print(f"EEG fMRI preprocessed loaded")
