from pathlib import Path
import sys
# support running without installing as a package
wd = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.append(str(wd))

from utils.smiles_utils import *
import glob
import os
from typing import List
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, cpu_count
import argparse


def prepare_full(
    source_path: Path,
    dest_path: Path,
    filenames_subset: List[str] = None,
) -> None:

    # Use the provided filenames_subset or default to all filenames
    filenames = filenames_subset
    
    if not filenames:
        raise RuntimeError(
            f"No files matching found at {source_path}. \n"
            "Make sure you download the data..."
        )
    for filepath in filenames:
        dest_file = os.path.join(dest_path, os.path.basename(filepath))
        str_set = set()
        original_line_count = 0
        with open(filepath, "r") as f:
            for k, row in tqdm(enumerate(f), desc=f"Processing {filepath}"):
                text = row.split()[0]
                canonicalized_text = canonicalize_smiles(text)
                str_set.add(canonicalized_text)
                original_line_count += 1
        
        new_line_count = 0
        with open(dest_file, "w") as f:
            for text in str_set:
                f.write(text + "\n")
                new_line_count += 1
        
        print(f"Original line count: {original_line_count}, New line count: {new_line_count}")




def prepare(
    source_path,
    dest_path,
):
    import time
    os.makedirs(dest_path, exist_ok=True)

    filenames = glob.glob(os.path.join(source_path, "*.txt"), recursive=True)

    num_processes = min(cpu_count(), len(filenames))
    num_processes = 50 # for a 52-core server, 50 processes is optimal
    chunked_filenames = np.array_split(filenames, num_processes)
    print(f"Number of processes: {num_processes}")

    processes = []
    start_time = time.time()

    for i, subset in enumerate(chunked_filenames):
        p = Process(target=prepare_full, args=(source_path, dest_path, list(subset)))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_path",
        default="../../../../data/raw_data/zinc20/shuffled_data/",
        help="The directory to the preprocessed data",
    )
    parser.add_argument(
        "--canonicalized_data_path",
        default="../../../../data/raw_data/zinc20/canonicalized_data/",
        help="The directory to the preprocessed data",
    )
    return parser.parse_args()

if __name__ == "__main__":
    # parse the arguments
    args = parse_args()
    prepare(args.source_path, args.canonicalized_data_path)
    
