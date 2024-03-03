from os import makedirs
from os.path import isdir, join
import random
from typing import List, Dict
import numpy as np
import pandas as pd
import torch


def set_seed(seed):
    assert isinstance(seed, int), f"expected a positive int for seed, got {seed}"
    assert 0 < seed <= 2**32, f"expected a positive int for seed, got {seed}"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_loso_runs(dataset) -> List[Dict[str, List[int]]]:
    runs = []
    data_df = pd.DataFrame(
        [
            {
                "subject_id": sample["subject_id"],
            }
            for sample in dataset
        ]
    )
    for subject_id in sorted(data_df["subject_id"].unique()):
        train_indices = data_df[data_df["subject_id"] != subject_id].index.tolist()
        val_indices = data_df[data_df["subject_id"] == subject_id].index.tolist()
        assert sorted(train_indices + val_indices) == data_df.index.tolist()
        runs.append(
            {
                "train_indices": train_indices,
                "val_indices": val_indices,
                "subject_id": subject_id,
            }
        )
    return runs

def get_kfold_runs(dataset, k:int) -> List[Dict[str, List[int]]]:
    assert isinstance(k, int)
    assert k >= 2
    runs = []
    shuffled_indices = np.arange(len(dataset))
    np.random.shuffle(shuffled_indices)
    folds = np.split(shuffled_indices, k)
    for i_val_fold in range(len(folds)):
        train_indices = [i for i_fold, fold in enumerate(folds) for i in fold if i_fold != i_val_fold]
        val_indices = [i for i_fold, fold in enumerate(folds) for i in fold if i_fold == i_val_fold]
        assert sorted(train_indices + val_indices) == list(range(len(dataset)))
        runs.append(
            {
                "train_indices": train_indices,
                "val_indices": val_indices,
                "subject_id": i_val_fold,
            }
        )
    return runs

def download_from_wandb(user_id: str, project_id: str, run_id: str, output_folder: str):
    import pandas as pd
    import wandb

    api = wandb.Api()
    run = api.run(f"{user_id}/{project_id}/{run_id}")

    config = {k: v for k, v in run.config.items() if not k.startswith("_")}
    name = run.name
    
    if not isdir(output_folder):
        makedirs(join(output_folder))
    
    run_history = run.history(samples=999999)
    
    run_history = run_history.sort_values(by="_step", axis=0)
    run_history = run_history.reindex(sorted(run_history.columns), axis=1)
    run_history = run_history.set_index("_step")
    run_history.to_csv(join(output_folder, "charts.csv"))
    pd.DataFrame.from_dict(config).to_csv(join(output_folder, "config.csv"))
    print(f"logs saved to {output_folder}")
