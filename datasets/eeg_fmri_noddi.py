import gc
import os
from os.path import isdir, join
from typing import Optional, Any, Union, List, Dict
from multiprocessing import Pool
from math import ceil

import mne
import nibabel as nib
import numpy as np
import scipy.io as sio
import einops
import torch
from torch.utils.data import Dataset


class EEGfMRINODDIDataset(Dataset):
    def __init__(
        self,
        path: str,
        sampling_rate: int = 256,
        window_size: Union[float, int] = 1,
        window_stride: Union[float, int] = 1,
        drop_last: Optional[bool] = False,
        discretize_labels: bool = False,
        normalize_eegs: bool = True,
        normalize_ecgs: bool = True,
    ):
        super().__init__()
        gc.collect()

        assert isdir(path)
        self.path: str = path

        # windows-related infos
        assert window_size > 0
        self.window_size: float = float(window_size)  # s
        assert window_stride > 0
        self.window_stride: float = float(window_stride)  # s
        assert isinstance(drop_last, bool)
        self.drop_last: bool = drop_last

        # EEG-related infos
        assert (
            isinstance(sampling_rate, int) and sampling_rate <= 5000
        ), f"sampling_rate must be below 5000, which is the original value"
        self.eeg_sampling_rate: int = sampling_rate
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
        self.eeg_samples_per_window: int = int(
            np.floor(self.eeg_sampling_rate * self.window_size)
        )
        self.eeg_samples_per_stride: int = int(
            np.floor(self.eeg_sampling_rate * self.window_stride)
        )
        assert isinstance(normalize_eegs, bool)
        self.normalize_eegs: bool = normalize_eegs

        # ECG-related infos
        self.ecg_sampling_rate: int = 256
        self.ecg_electrodes: List[str] = ["ECG1", "ECG2"]
        self.ecg_samples_per_window: int = int(
            np.floor(self.ecg_sampling_rate * self.window_size)
        )
        self.ecg_samples_per_stride: int = int(
            np.floor(self.ecg_sampling_rate * self.window_stride)
        )
        assert isinstance(normalize_ecgs, bool)
        self.normalize_ecgs: bool = normalize_ecgs

        # labels-related infos
        self.labels: List[str] = ["valence", "arousal", "dominance"]
        self.labels_classes = 2
        assert isinstance(discretize_labels, bool)
        self.discretize_labels: bool = discretize_labels

        # subjects-related infos
        self.subject_ids: List[str] = self.get_subject_ids_static(self.path)

        (
            self.eegs_data,
            self.ecgs_data,
            self.labels_data,
            self.subject_ids_data,
        ) = self.load_data()

        # discard corrupted and null experiments
        non_null_indices = {
            i
            for i, eegs in enumerate(self.eegs_data)
            if np.count_nonzero(np.isnan(eegs)) <= eegs.size * 0.9
        }
        self.eegs_data = [
            v for i, v in enumerate(self.eegs_data) if i in non_null_indices
        ]
        self.ecgs_data = [
            v for i, v in enumerate(self.ecgs_data) if i in non_null_indices
        ]
        # normalizes the data
        if self.normalize_eegs:
            self.eegs_data = self.normalize(self.eegs_data, mode="minmax")
        if self.normalize_ecgs:
            self.ecgs_data = self.normalize(self.ecgs_data, mode="minmax")
        self.labels_data = [
            v for i, v in enumerate(self.labels_data) if i in non_null_indices
        ]
        self.subject_ids_data = [
            v for i, v in enumerate(self.subject_ids_data) if i in non_null_indices
        ]
        assert (
            len(self.eegs_data)
            == len(self.ecgs_data)
            == len(self.labels_data)
            == len(self.subject_ids_data)
        )

        # windows the data
        self.windows = self.get_windows()
        # assert all([e.shape[-1] == len(self.electrodes) for e in self.eegs_data])
        gc.collect()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> Dict[str, Union[int, str, np.ndarray]]:
        window = self.windows[i]
        eegs = self.eegs_data[window["experiment"]][
            window["eeg"]["start"] : window["eeg"]["end"]
        ]
        ecgs = self.ecgs_data[window["experiment"]][
            window["ecg"]["start"] : window["ecg"]["end"]
        ]
        # eventually pad the eegs
        # if eegs.shape[0] != self.samples_per_window:
        #     eegs = np.concatenate([eegs,
        #                            np.zeros([self.samples_per_window - eegs.shape[0], eegs.shape[1]])],
        #                           axis=0)
        assert (
            eegs.shape[0] == self.eeg_samples_per_window
        ), f"{eegs.shape[0]} != {self.eeg_samples_per_window}"
        assert (
            ecgs.shape[0] == self.ecg_samples_per_window
        ), f"{ecgs.shape[0]} != {self.ecg_samples_per_window}"
        eegs = einops.rearrange(eegs, "t c -> c t").astype(np.float32)
        ecgs = einops.rearrange(ecgs, "t c -> c t").astype(np.float32)
        return {
            # "sampling_rates": self.sampling_rate,
            "subject_id": window["subject_id"],
            "eegs": eegs,
            "ecgs": ecgs,
            "labels": window["labels"],
        }

    def prepare_data(self) -> None:
        pass

    @staticmethod
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

    def load_data(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        global parse_subject_data

        def parse_subject_data(subject_no):
            subject_id: str = self.subject_ids[subject_no]
            assert subject_id in self.subject_ids
            st = time.time()

            # loads fmri data
            fmri_data_filename = [
                f
                for f in os.listdir(join(self.path, "fMRI", subject_id))
                if f.endswith("rest_with_cross.nii.gz")
            ][0]
            fmris = nib.load(join(self.path, "fMRI", subject_id, fmri_data_filename))
            fmris = fmris.get_fdata()  # (x z y t)
            # splits the fmris into a list
            fmris = [fmris[:, :, :, i] for i in range(fmris.shape[-1])]
            # print(f"fMRIs", len(fmris), fmris[0].shape)

            # loads eeg data
            eeg_subject_folder = (
                "EEG1" if subject_id in os.listdir(join(self.path, "EEG1")) else "EEG2"
            )
            eeg_data_filename = [
                f
                for f in os.listdir(
                    join(self.path, eeg_subject_folder, subject_id, "raw")
                )
                if f.endswith("vhdr")
            ][0]
            eegs = mne.io.read_raw_brainvision(
                join(
                    self.path, eeg_subject_folder, subject_id, "raw", eeg_data_filename
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
            eegs = eegs.resample(self.eeg_sampling_rate, verbose=False)
            # extract the numpy array from the mne raw object
            samples_per_fmri = ceil(self.eeg_sampling_rate * 2.16)
            eegs, _ = eegs[:, :]  # (c t)
            # splits the eegs into a list, one record for each fMRI image
            eegs = [
                eegs[:, i * samples_per_fmri : i * samples_per_fmri + samples_per_fmri]
                for i in range(len(fmris))
            ]
            assert all(
                [e.shape == eegs[0].shape for e in eegs]
            ), f"some eegs are not the same shape"
            print("loaded subject", subject_id, "in", np.round(time.time() - st, 2))
            return eegs, fmris, subject_id

        st = time.time()
        with Pool(processes=os.cpu_count()) as pool:
            data_pool = pool.map(
                parse_subject_data, [i for i in range(len(self.subject_ids))]
            )
            data_pool = [d for d in data_pool if d is not None]
            eegs: List[np.ndarray] = [e for eegs, _, _ in data_pool for e in eegs]
            fmris: List[np.ndarray] = [e for _, fmris, _ in data_pool for e in fmris]
            subject_ids: List[str] = [
                s_id
                for eegs_lists, _, _, subject_id in data_pool
                for s_id in [subject_id] * len(eegs_lists)
            ]
        assert len(eegs) == len(fmris) == len(subject_ids)
        print("read dataset in", np.round(time.time() - st, 2))
        raise
        return eegs, fmris, subject_ids

    def normalize(self, waveforms: List[np.ndarray], mode="minmax"):
        # scales to zero mean and unit variance
        for i_experiment, experiment in enumerate(waveforms):
            #     # scales to zero mean and unit variance
            #     # experiment_scaled = (experiment - experiment.mean(axis=0)) / experiment.std(axis=0)
            #     # scaler = mne.decoding.Scaler(info=mne.create_info(ch_names=self.electrodes, sfreq=self.sampling_rate,
            #     #                                                   verbose=False, ch_types="eeg"),
            #     #                              scalings="mean")
            #     # experiment_scaled = einops.rearrange(
            #     #     scaler.fit_transform(einops.rearrange(experiment, "s c -> () c s")),
            #     #     "b c s -> s (b c)"
            #     # )
            #     # experiment_scaled = np.nan_to_num(experiment_scaled)
            #     # normalizes between -1 and 1
            #     # experiment_scaled = 2 * ((experiment_scaled - experiment_scaled.min(axis=0)) /
            #     #                          (experiment_scaled.max(axis=0) - experiment_scaled.min(axis=0))) - 1
            #
            # experiment_scaled = np.swapaxes(experiment, 0, 1) # c t

            if mode == "std":
                mean = experiment.mean(axis=0)
                std = experiment.std(axis=0)
                waveform_scaled = (experiment - mean) / std
                min = waveform_scaled.min(axis=0)
                max = waveform_scaled.max(axis=0)
                waveform_scaled = 2 * ((waveform_scaled - min) / (max - min)) - 1
            if mode == "minmax":
                min = experiment.min(axis=0)
                max = experiment.max(axis=0)
                waveform_scaled = 2 * ((experiment - min) / (max - min)) - 1
            waveforms[i_experiment] = waveform_scaled
        return waveforms

    def get_windows(self) -> List[Dict[str, Union[int, str]]]:
        windows: List[Dict[str, Union[int, str]]] = []
        for i_experiment in range(len(self.eegs_data)):
            subject_id = self.subject_ids_data[i_experiment]
            labels = np.asarray(self.labels_data[i_experiment])
            eeg_total_samples = len(self.eegs_data[i_experiment])
            eeg_ptr, ecg_ptr = 0, 0
            while eeg_ptr < eeg_total_samples:
                next_eeg_ptr = eeg_ptr + self.eeg_samples_per_stride
                next_ecg_ptr = ecg_ptr + self.ecg_samples_per_stride
                eeg_samples = (
                    min(eeg_total_samples, eeg_ptr + self.eeg_samples_per_window)
                    - eeg_ptr
                )
                if eeg_samples != self.eeg_samples_per_window:
                    break
                window = {
                    "experiment": i_experiment,
                    "subject_id": subject_id,
                    "labels": labels,
                    "eeg": {
                        "start": eeg_ptr,
                        "end": eeg_ptr + self.eeg_samples_per_window,
                    },
                    "ecg": {
                        "start": ecg_ptr,
                        "end": ecg_ptr + self.ecg_samples_per_window,
                    },
                }
                eeg_ptr, ecg_ptr = next_eeg_ptr, next_ecg_ptr
                windows += [window]
        return windows

def save_data_to_numpy(dataset_path:str, save_path: str, new_sampling_rate: int = 256):
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
            np.save(join(fmris_folder, f"{i}.npy"), fmris[:,:,:,i])

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
        original_sampling_rate = int(eegs.info["sfreq"])
        # Compute the mean across channels for each time point
        mean_amplitude = np.mean(np.abs(eegs._data), axis=0)  # (t)
        mean_amplitude = np.convolve(
            mean_amplitude, np.ones(original_sampling_rate) / original_sampling_rate, mode="same"
        )
        # computes the cut-off threshold for cropping preliminary parts
        threshold = np.quantile(
            mean_amplitude,
            (original_sampling_rate * fmris.shape[-1] * 2.16) / len(mean_amplitude),
        )
        # search for when the record must start
        for i in range(len(mean_amplitude)):
            if np.isclose(mean_amplitude[i], threshold, atol=1e-4):
                i_end = ceil(i + original_sampling_rate + original_sampling_rate * fmris.shape[-1] * 2.16)
                start_time = eegs.times[i]
                end_time = eegs.times[i_end]
                break
        # crops the eegs
        eegs = eegs.crop(
            tmin=start_time, tmax=end_time, include_tmax=True, verbose=False
        )
        # resamples the eegs
        eegs = eegs.resample(new_sampling_rate, n_jobs=os.cpu_count(), verbose=False)
        # extract the numpy array from the mne raw object
        eegs, _ = eegs[:, :]  # (c t)
        samples_per_fmri = ceil(new_sampling_rate * 2.16)
        eegs_folder = join(subject_folder, "eegs")
        if not isdir(eegs_folder):
            makedirs(eegs_folder)
        for i in range(fmris.shape[-1]):
            eegs_to_save = eegs[:, i * samples_per_fmri : (i+1) * samples_per_fmri]
            assert eegs_to_save.shape[-1] == samples_per_fmri 
            np.save(join(eegs_folder, f"{i}.npy"), eegs_to_save)
        print("loaded subject", subject_id, "in", np.round(time.time() - st, 2))
        return eegs, fmris, subject_id

    st = time.time()
    for i in range(len(subject_ids)):
        parse_subject_data(i)
    print("saved dataset in", np.round(time.time() - st, 2))
    
if __name__ == "__main__":
    import time

    dataset_path = "../../datasets/eeg_fmri_noddi"
    save_path=join("fmri_eeg_numpy_512Hz_2")
    print(f"saving EEG fMRI NODDI dataset from {dataset_path} to {save_path}")
    st = time.time()
    save_data_to_numpy(dataset_path=dataset_path, save_path=save_path, new_sampling_rate=512)
    # dataset = EEGfMRINODDIDataset(
    #     path=dataset_path,
    #     discretize_labels=True,
    #     normalize_eegs=True,
    #     window_size=2,
    #     window_stride=1,
    # )
    # dataset_train, dataset_val = torch.utils.data.random_split(dataset, [0.8, 0.2])
    # print(f"DREAMER loaded in {int(time.time() - st)}s")
