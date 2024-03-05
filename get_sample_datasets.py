import os
from os import listdir, makedirs
from os.path import isdir, join, basename
from shutil import copy

if __name__ == "__main__":
    import argparse
    
    ##############################
    # ARGS
    ##############################
    parser = argparse.ArgumentParser()
    # dataset params
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the preprocessed dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=".",
        help="Path where the sample dataset will be saved",
    )
    args = parser.parse_args()
    assert isdir(args.path)
    print(args.path, args.path.split(os.sep))
    output_path = join(args.output_path, f"{[v for v in args.path.split(os.sep) if v][-1]}_sample")
    makedirs(output_path, exist_ok=True)
    for filepath in [join(args.path, filename) for filename in listdir(args.path)]:
        if not isdir(filepath):
            copy(filepath, output_path)
        else:
            subject_id = basename(filepath)
            import random
            random_sample = random.sample(listdir(join(filepath, "eegs")), k=1)[0]
            for signal in ["eegs", "fmris"]:
                makedirs(join(output_path, subject_id, signal), exist_ok=True)
                copy(join(filepath, signal, random_sample), join(output_path, subject_id, signal, random_sample))
            print(random_sample) 
            
    print(listdir(args.path))
    