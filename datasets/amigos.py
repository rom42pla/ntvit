import gc
import os
from os.path import isdir, join, basename, splitext
from typing import Optional, Any, Tuple, Union, List, Dict
from multiprocessing import Pool

import numpy as np
import scipy.io as sio
import einops
import torch
from torch.utils.data import Dataset


class AMIGOSDataset(Dataset):
    def __init__(
        self,
        path: str,
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
        self.eeg_sampling_rate: int = 128
        self.eeg_electrodes: List[str] = [
            "AF3",
            "F7",
            "F3",
            "FC5",
            "T7",
            "P7",
            "O1",
            "O2",
            "P8",
            "T8",
            "FC6",
            "F4",
            "F8",
            "AF4",
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
        self.ecg_sampling_rate: int = 128
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
        self.labels: List[str] = [
            "arousal",
            "valence",
            "dominance",
            "liking",
            "familiarity",
            "neutral",
            "disgust",
            "happiness",
            "surprise",
            "anger",
            "fear",
            "sadness",
        ]
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
            for i in range(len(self.eegs_data))
            if np.count_nonzero(np.isnan(self.eegs_data[i]))
            <= self.eegs_data[i].size * 0.9
            and np.count_nonzero(np.isnan(self.ecgs_data[i]))
            <= self.ecgs_data[i].size * 0.9
        }
        self.eegs_data = [
            v for i, v in enumerate(self.eegs_data) if i in non_null_indices
        ]
        self.ecgs_data = [
            v for i, v in enumerate(self.ecgs_data) if i in non_null_indices
        ]
        # normalizes the data
        if self.normalize_eegs:
            self.eegs_data = self.normalize(self.eegs_data)
        if self.normalize_ecgs:
            self.ecgs_data = self.normalize(self.ecgs_data)
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
        subject_ids = [
            basename(splitext(s)[0]).split("_")[-1]
            for s in os.listdir(join(path, "data_preprocessed"))
        ]
        subject_ids.sort()
        return subject_ids

    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        global parse_eegs

        def parse_eegs(
            subject_id: str,
        ) -> Tuple[List[np.ndarray], List[np.ndarray], str]:
            subject_data = sio.loadmat(
                join(
                    self.path,
                    "data_preprocessed",
                    f"Data_Preprocessed_{subject_id}.mat",
                ),
                simplify_cells=True,
            )
            # some data is corrupted
            valid_indices = {
                i
                for i, (eegs, labels) in enumerate(
                    zip(
                        subject_data["joined_data"],
                        subject_data["labels_selfassessment"],
                    )
                )
                if eegs.any() and labels.any()
            }
            eegs: List[np.ndarray] = [
                e[:, :14].astype(np.float32)
                for i, e in enumerate(subject_data["joined_data"])
                if i in valid_indices
            ]
            ecgs: List[np.ndarray] = [
                e[:, 14:16].astype(np.float32)
                for i, e in enumerate(subject_data["joined_data"])
                if i in valid_indices
            ]
            labels: List[np.ndarray] = [
                e.astype(int)
                for i, e in enumerate(subject_data["labels_selfassessment"])
                if i in valid_indices
            ]
            assert len(eegs) == len(labels)
            for i_trial, labels_trial in enumerate(labels):
                if self.discretize_labels:
                    labels[i_trial][:5] = np.asarray(
                        [1 if label > 5 else 0 for label in labels_trial[:5]]
                    )
                else:
                    labels_trial[labels_trial > 9] = 9
                    labels[i_trial][:5] = (labels_trial[:5] - 1) / 8
                    assert labels[i_trial][:5].min() >= 0
                    assert labels[i_trial][:5].max() <= 9
            return eegs, ecgs, labels, subject_id

        with Pool(processes=len(self.subject_ids)) as pool:
            data_pool = pool.map(parse_eegs, [s_id for s_id in self.subject_ids])
            data_pool = [d for d in data_pool if d is not None]
            eegs: List[np.ndarray] = [
                e for eeg_lists, _, _, _ in data_pool for e in eeg_lists
            ]
            ecgs: List[np.ndarray] = [
                e for _, ecgs_lists, _, _ in data_pool for e in ecgs_lists
            ]
            labels: List[np.ndarray] = [
                l for _, _, labels_lists, _ in data_pool for l in labels_lists
            ]
            subject_ids: List[str] = [
                s_id
                for eegs_lists, _, _, subject_id in data_pool
                for s_id in [subject_id] * len(eegs_lists)
            ]
        assert len(eegs) == len(labels) == len(subject_ids)
        return eegs, ecgs, labels, subject_ids

    def normalize(self, waveforms: List[np.ndarray]):
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
            # mean = experiment.mean(axis=0)
            # std = experiment.std(axis=0)
            # waveform_scaled = (experiment - mean) / std
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


if __name__ == "__main__":
    import time

    dataset_path = "../../datasets/amigos"
    print("loading AMIGOS")
    st = time.time()
    dataset = AMIGOSDataset(
        path=dataset_path,
        discretize_labels=True,
        normalize_eegs=True,
        window_size=2,
        window_stride=1,
    )
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [0.8, 0.2])
    print(f"AMIGOS loaded in {int(time.time() - st)}s")
