import gc
import os
from os.path import isdir, join
from typing import Optional, Any, Union, List, Dict
from multiprocessing import Pool

import numpy as np
import scipy.io as sio
import einops
import torch
from torch.utils.data import Dataset


class DREAMERDataset(Dataset):
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
        data: Dict[str, Any] = sio.loadmat(
            join(path, "DREAMER.mat"), simplify_cells=True
        )["DREAMER"]["Data"]
        subject_ids: List[str] = [f"s{i}" for i in range(len(data))]
        subject_ids.sort()
        return subject_ids

    def load_data(self):
        # loads DREAMER.mat
        data_raw = sio.loadmat(join(self.path, "DREAMER.mat"), simplify_cells=True)[
            "DREAMER"
        ]["Data"]

        global parse_dreamer

        def parse_dreamer(subject_no):
            subject_id: str = self.subject_ids[subject_no]
            assert subject_id in self.subject_ids
            subject_data = data_raw[subject_no]
            eegs: List[np.ndarray] = []
            ecgs: List[np.ndarray] = []
            labels: List[np.ndarray] = []
            experiments_no = len(subject_data["EEG"]["stimuli"])
            assert (
                experiments_no
                == len(subject_data["EEG"]["stimuli"])
                == len(subject_data["ECG"]["stimuli"])
                == len(subject_data["EEG"]["stimuli"])
                == len(subject_data["ScoreArousal"])
                == len(subject_data["ScoreValence"])
                == len(subject_data["ScoreDominance"])
            )
            for i_experiment in range(experiments_no):
                # loads the eeg for the experiment
                eegs += [subject_data["EEG"]["stimuli"][i_experiment]]  # (s c)
                ecgs += [subject_data["ECG"]["stimuli"][i_experiment]]  # (s c)
                # loads the labels for the experiment
                labels += [
                    np.asarray(
                        [
                            subject_data[k][i_experiment]
                            for k in ["ScoreArousal", "ScoreValence", "ScoreDominance"]
                        ]
                    )
                ]  # (l)
            # eventually discretizes the labels
            labels = [
                [1 if label > 3 else 0 for label in w]
                if self.discretize_labels
                else (w - 1) / 4
                for w in labels
            ]
            return eegs, ecgs, labels, subject_id

        with Pool(processes=os.cpu_count()) as pool:
            data_pool = pool.map(
                parse_dreamer, [i for i in range(len(self.subject_ids))]
            )
            data_pool = [d for d in data_pool if d is not None]
            eegs: List[np.ndarray] = [
                e for eeg_lists, _, _, _ in data_pool for e in eeg_lists
            ]
            ecgs: List[np.ndarray] = [
                e for _, ecg_lists, _, _ in data_pool for e in ecg_lists
            ]
            labels: List[np.ndarray] = [
                l for _, _, labels_lists, _ in data_pool for l in labels_lists
            ]
            subject_ids: List[str] = [
                s_id
                for eegs_lists, _, _, subject_id in data_pool
                for s_id in [subject_id] * len(eegs_lists)
            ]
        assert len(eegs) == len(ecgs) == len(labels) == len(subject_ids)
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

    dataset_path = "../../datasets/dreamer"
    print("loading DREAMER")
    st = time.time()
    dataset = DREAMERDataset(
        path=dataset_path,
        discretize_labels=True,
        normalize_eegs=True,
        window_size=2,
        window_stride=1,
    )
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [0.8, 0.2])
    print(f"DREAMER loaded in {int(time.time() - st)}s")
