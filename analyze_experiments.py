import argparse
from pprint import pprint 
from os import listdir
from os.path import isdir, join
import numpy as np
import pandas as pd

def round_float(x, chars=4):
    x_split = str(x).split(".")
    left_part = x_split[0]
    if len(left_part) > chars:
        return np.round(left_part, chars)
    elif len(left_part) == chars:
        return left_part
    else:
        return round(x, chars-len(left_part))
        
if __name__ == "__main__":
    ##############################
    # ARGS
    ##############################
    parser = argparse.ArgumentParser()
    # dataset params
    parser.add_argument(
        "--experiments_path",
        type=str,
        help="Path to the saved experiments folder",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices={"results", "ablation"},
        default="results",
        help="How to print the results",
    )
    args = parser.parse_args()
    

    assert isdir(args.experiments_path)
    for experiment_name in sorted(listdir(args.experiments_path)):
        try:
            config_df = pd.read_csv(join(args.experiments_path, experiment_name, "config.csv"))
            charts_df = pd.read_csv(join(args.experiments_path, experiment_name, "charts.csv"))
            run_ids = {c.split("/")[0] for c in charts_df.columns if c.startswith("run")}
            stats = {}
            for run_id in run_ids:
                run_df = charts_df[[c for c in charts_df.columns if c.startswith(f"{run_id}/val")]]
                run_df = run_df.sort_values(by=f"{run_id}/val/psnr", ascending=False)
                best_row = run_df.iloc[0]
                metrics = ["n_params", "macs", "psnr", "ssim", "rmse", "mae", "cfv"]            
                stats[run_id] = {
                    k: best_row[f"{run_id}/val/{k}"]
                    for k in metrics
                    if f"{run_id}/val/{k}" in best_row.index
                }
            stats_df = pd.DataFrame.from_dict(stats, orient="index")
            
            print(f"row for experiment {experiment_name}:")
            if args.mode == "results":
                keys = ["rmse", "mae", "cfv", "psnr", "ssim"]
            else:
                keys = ["n_params", "macs", "psnr", "ssim"]
            values = []
            for k in keys:
                if k not in stats_df.mean().index:
                    continue
                mean, std = stats_df.mean()[k], stats_df.std().dropna(axis=0)[k]
                if std == 0:
                    values.append(f"${round_float(mean, 4)}$")
                else:
                    values.append(f"${round_float(mean, 4)} \pm {round_float(std, 4)}$")
            print(" & ".join(values))
            # print(" & ".join([f"${round_float(stats_df.mean()[k], 4)} \pm {round_float(stats_df.std()[k], 4)}$" for k in keys if k in stats_df.mean().index]))
            print()
        except Exception as e:
            print(f"experiment {experiment_name} is broken")
            print(e)