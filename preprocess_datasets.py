import argparse
import time
import os
from os import listdir, makedirs
from os.path import isdir, exists, join, normpath
import json
from typing import List
from tqdm import tqdm
from math import ceil
import numpy as np
from scipy import io
import einops

import mne
import nibabel as nib

def preprocess_noddi(
        raw_dataset_path: str, output_path: str, 
        sampling_rate: int = 512,
    ):
    if not isdir(output_path):
        makedirs(output_path)
    def get_subject_ids(path: str) -> List[str]:
        assert isdir(path)
        users_per_signal = {
            "fMRI": set(os.listdir(join(path, "fMRI"))),
            "EEG": set(
                os.listdir(join(path, "EEG1")) + os.listdir(join(path, "EEG2"))
            ),
        }
        common_users = sorted(
            list(users_per_signal["fMRI"] & users_per_signal["EEG"])
        )
        # user 35 is somehow broken
        if "35" in common_users:
            common_users.remove("35")
        return common_users
    
    subject_ids = get_subject_ids(raw_dataset_path)
    eegs_seconds = 2.16
    metas = {
        "subject_ids": subject_ids,
        "fmris_info": {},
        "eegs_info": {"seconds": eegs_seconds, "sampling_rate": sampling_rate},
        "samples": [],
    }

    def parse_subject_data(subject_no, sampling_rate):
        subject_id: str = subject_ids[subject_no]
        subject_folder = join(output_path, subject_id)
        st = time.time()

        # loads fmri data
        fmri_data_filename = [
            f
            for f in os.listdir(join(raw_dataset_path, "fMRI", subject_id))
            if f.endswith("rest_with_cross.nii.gz")
        ][0]
        fmris = nib.load(
            join(raw_dataset_path, "fMRI", subject_id, fmri_data_filename)
        )
        fmris = fmris.get_fdata()  # (x z y t)
        fmris = einops.rearrange(fmris, "x z y t -> t y x z")  # (t y x z)
        # checks that the shape is correct
        if not "shape" in metas["fmris_info"]:
            metas["fmris_info"]["shape"] = list(fmris.shape[1:])
        else:
            assert (
                list(fmris.shape[1:]) == metas["fmris_info"]["shape"]
            ), f"found fmris with odd shape {fmris.shape[1:]}, instead of {metas['fmris_info']['shape']}"
        # saves to numpy arrays
        fmris_folder = join(subject_folder, "fmris")
        if not isdir(fmris_folder):
            makedirs(fmris_folder)
        # initialize the metas for this subjects
        metas_samples_subject = []
        for i in range(len(fmris)):
            fmris_sample_path = join(fmris_folder, f"{i}.npy")
            np.save(fmris_sample_path, fmris[i].astype(np.float32))
            fmris_normalized_path = join(
                *(normpath(fmris_sample_path).split(os.sep)[-3:])
            )
            assert exists(join(output_path, fmris_normalized_path))
            metas_samples_subject.append(
                {
                    "subject_id": subject_id,
                    "fmris_path": fmris_normalized_path,
                }
            )
        # loads eeg data
        eeg_subject_folder = (
            "EEG1"
            if subject_id in os.listdir(join(raw_dataset_path, "EEG1"))
            else "EEG2"
        )
        eeg_data_filename = [
            f
            for f in os.listdir(
                join(raw_dataset_path, eeg_subject_folder, subject_id, "raw")
            )
            if f.endswith("vhdr")
        ][0]
        eegs_raw = mne.io.read_raw_brainvision(
            join(
                raw_dataset_path,
                eeg_subject_folder,
                subject_id,
                "raw",
                eeg_data_filename,
            ),
            preload=True,
            verbose=False,
        )
        # saves the original sampling rate
        original_sampling_rate = int(eegs_raw.info["sfreq"])
        if not "original_sampling_rate" in metas["eegs_info"]:
            metas["eegs_info"]["original_sampling_rate"] = original_sampling_rate
        # compute the mean across channels for each time point
        mean_amplitude = np.mean(np.abs(eegs_raw._data), axis=0)  # (t)
        mean_amplitude = np.convolve(
            mean_amplitude,
            np.ones(original_sampling_rate) / original_sampling_rate,
            mode="same",
        )
        # computes the cut-off threshold for cropping preliminary parts
        threshold = np.quantile(
            mean_amplitude,
            (original_sampling_rate * len(fmris) * eegs_seconds)
            / len(mean_amplitude),
        )
        # search for when the record must start
        for i in range(len(mean_amplitude)):
            if np.isclose(mean_amplitude[i], threshold, atol=1e-4):
                i_start = ceil(
                    i 
                )
                i_end = ceil(
                    i
                    + original_sampling_rate
                    + original_sampling_rate * len(fmris) * eegs_seconds
                )
                start_time = eegs_raw.times[i_start]
                end_time = eegs_raw.times[i_end]
                break
        # crops the eegs
        eegs_raw = eegs_raw.crop(
            tmin=start_time, tmax=end_time, include_tmax=True, verbose=False
        )
        # filters the eegs
        eegs_raw = eegs_raw.filter(0.5, 150, n_jobs=os.cpu_count(), verbose=False)
        # sets the common reference
        eegs_raw = eegs_raw.set_eeg_reference(ref_channels="average", verbose=False)
        # resamples the eegs
        if sampling_rate != original_sampling_rate:
            eegs_raw = eegs_raw.resample(sampling_rate, n_jobs=os.cpu_count(), verbose=False)
        else:
            sampling_rate = original_sampling_rate
        # extract the numpy array from the mne raw object
        samples_per_fmri = ceil(sampling_rate * eegs_seconds)
        eegs, _ = eegs_raw[:]  # (c t)
        ecgs_index = eegs_raw.info["ch_names"].index("ECG")
        ecgs = eegs[ecgs_index][np.newaxis, :]
        eegs = np.delete(eegs, ecgs_index, axis=0)
        # saves the name of the channels
        electrodes = list(eegs_raw.info["ch_names"])
        electrodes.remove("ECG")
        if not "electrodes" in metas["eegs_info"]:
            metas["eegs_info"]["electrodes"] = electrodes
        else:
            metas["eegs_info"]["electrodes"] = list(set(metas["eegs_info"]["electrodes"]) | set(electrodes))
        # prepares the folders
        eegs_folder = join(subject_folder, "eegs")
        if not isdir(eegs_folder):
            makedirs(eegs_folder)
        ecgs_folder = join(subject_folder, "ecgs")
        if not isdir(ecgs_folder):
            makedirs(ecgs_folder)
        for i in range(len(fmris)):
            # saves the eeg
            eegs_sample_path = join(eegs_folder, f"{i}.npy")
            eegs_sample = eegs[
                :,
                i * samples_per_fmri : i * samples_per_fmri + samples_per_fmri,
            ]
            # checks that the shape is correct
            if not "shape" in metas["eegs_info"]:
                metas["eegs_info"]["shape"] = list(eegs_sample.shape)
            else:
                assert (
                    list(eegs_sample.shape) == metas["eegs_info"]["shape"]
                ), f"found eegs with odd shape {eegs_sample.shape}, instead of {metas['eegs_info']['shape']}"
            np.save(
                eegs_sample_path,
                eegs_sample.astype(np.float32),
            )
            eegs_normalized_path = join(
                *(normpath(eegs_sample_path).split(os.sep)[-3:])
            )
            assert exists(join(output_path, eegs_normalized_path))
            
            # saves the ecg
            ecgs_sample_path = join(ecgs_folder, f"{i}.npy")
            ecgs_sample = ecgs[
                :,
                i * samples_per_fmri : i * samples_per_fmri + samples_per_fmri,
            ]
            np.save(
                ecgs_sample_path,
                ecgs_sample.astype(np.float32),
            )
            ecgs_normalized_path = join(
                *(normpath(ecgs_sample_path).split(os.sep)[-3:])
            )
            assert exists(join(output_path, ecgs_normalized_path))
            metas_samples_subject[i].update({
                    "eegs_electrodes": electrodes,
                    "eegs_path": eegs_normalized_path,
                    "ecgs_path": ecgs_normalized_path,
                })
        for sample in metas_samples_subject:
            metas["samples"].append(sample)
        print("loaded subject", subject_id, "in", np.round(time.time() - st, 2))
        with open(join(output_path, "metas.json"), 'w', encoding='utf-8') as f:
            json.dump(metas, f, ensure_ascii=True, indent=4)
        return eegs, fmris, subject_id

    st = time.time()
    for i in range(len(subject_ids)):
        parse_subject_data(i, sampling_rate)
    print("saved dataset in", np.round(time.time() - st, 2))
        
def preprocess_oddball(dataset_path: str, output_path: str, sampling_rate: int = 512):
    assert isdir(dataset_path)
    assert isinstance(sampling_rate, int) and sampling_rate >= 1
    dataset_prep_path = join(output_path, "oddball_preprocessed")
    makedirs(dataset_prep_path, exist_ok=True)
    eegs_seconds = 2
    original_sampling_rate = 1000
    metas = {
        # "subject_ids": subject_ids,
        "fmris_info": {},
        "eegs_info": {"seconds": eegs_seconds, "sampling_rate": sampling_rate, "original_sampling_rate": original_sampling_rate},
        "samples": [],
    }
    samples_per_fmri = ceil(sampling_rate * eegs_seconds)
    
    for subject_no in tqdm(range(1, 17+1), desc="subjects parsed"):
        subject_id = f"sub{subject_no:03}"
        subject_folder = join(dataset_path, subject_id)
        for run_name in listdir(join(subject_folder, "BOLD")):
            ##############################
            # fMRIs
            ##############################
            fmris = nib.load(
                    join(subject_folder, "BOLD", run_name, "bold.nii.gz")
                ).get_fdata() # (x z y t)
            fmris = einops.rearrange(fmris, "x z y t -> t y x z")  # (t y x z)
            # checks that the shape is correct
            if not "shape" in metas["fmris_info"]:
                metas["fmris_info"]["shape"] = list(fmris.shape[1:])
            else:
                assert (
                    list(fmris.shape[1:]) == metas["fmris_info"]["shape"]
                ), f"found fmris with odd shape {fmris.shape[1:]}, instead of {metas['fmris_info']['shape']}"
            # saves to numpy arrays
            fmris_folder = join(dataset_prep_path, subject_id, "fmris")
            if not isdir(fmris_folder):
                makedirs(fmris_folder)
            for i in range(len(fmris)):
                fmris_sample_path = join(fmris_folder, f"{run_name}_{i:03}.npy")
                np.save(fmris_sample_path, fmris[i].astype(np.float32))
                fmris_normalized_path = join(
                    *(normpath(fmris_sample_path).split(os.sep)[-3:])
                )
            ##############################
            # EEGs
            ##############################
            # loads the eegs
            eegs_raw = io.loadmat(join(subject_folder, "EEG", run_name, "EEG_rereferenced.mat"), squeeze_me=True, simplify_cells=True)["data_reref"][:34]
            eegs_raw = mne.io.RawArray(eegs_raw, mne.create_info(ch_names=eegs_raw.shape[0], sfreq=original_sampling_rate, ch_types="eeg"), first_samp=0, verbose=False)
            # filters the eegs
            eegs_raw = eegs_raw.filter(0.5, 50, n_jobs=os.cpu_count(), verbose=False)
            # sets the common reference
            eegs_raw = eegs_raw.set_eeg_reference(ref_channels="average", verbose=False)
            # resamples the eegs
            if sampling_rate != original_sampling_rate:
                eegs_raw = eegs_raw.resample(sampling_rate, n_jobs=os.cpu_count(), verbose=False)
            else:
                sampling_rate = original_sampling_rate
            # extracts the eegs array
            eegs, _ = eegs_raw[:]  # (c t)
            # saves to numpy arrays
            eegs_folder = join(dataset_prep_path, subject_id, "eegs")
            if not isdir(eegs_folder):
                makedirs(eegs_folder)
            for i in range(len(fmris)):
                eegs_sample_path = join(eegs_folder, f"{run_name}_{i:03}.npy")
                eegs_sample = eegs[
                        :,
                        i * samples_per_fmri : i * samples_per_fmri + samples_per_fmri,
                    ]
                # checks that the shape is correct
                if not "shape" in metas["eegs_info"]:
                    metas["eegs_info"]["shape"] = list(eegs_sample.shape)
                else:
                    assert (
                        list(eegs_sample.shape) == metas["eegs_info"]["shape"]
                    ), f"found eegs with odd shape {eegs_sample.shape}, instead of {metas['eegs_info']['shape']}"
                np.save(eegs_sample_path, eegs_sample.astype(np.float32))
                eegs_normalized_path = join(
                    *(normpath(eegs_sample_path).split(os.sep)[-3:])
                )
            ##############################
            # metas
            ##############################
            metas["samples"].append(
                    {
                        "subject_id": subject_id,
                        "fmris_path": fmris_normalized_path,
                        "eegs_path": eegs_normalized_path,
                    }
                )
    with open(join(dataset_prep_path, "metas.json"), 'w', encoding='utf-8') as f:
        json.dump(metas, f, ensure_ascii=True, indent=4)
    
    

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
        "--output_path",
        type=str,
        required=True,
        help="Path where the output will be stored",
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=512,
        help="The new sampling rate to use",
    )
    args = parser.parse_args()
    assert isdir(args.dataset_path)
    assert args.sampling_rate >= 1
    if args.dataset_type == "noddi":
        preprocess_noddi(args.dataset_path, args.output_path, sampling_rate=args.sampling_rate)
    elif args.dataset_type == "oddball":
        preprocess_oddball(args.dataset_path, args.output_path, sampling_rate=args.sampling_rate)