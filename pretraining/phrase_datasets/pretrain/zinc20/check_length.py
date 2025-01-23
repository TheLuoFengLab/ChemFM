from pathlib import Path
import sys
import glob
import os
from typing import List
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, cpu_count
import argparse

# support running without installing as a package
wd = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.tokenizer import Tokenizer

def prepare_full(
    source_path: Path,
    tokenizer_path: Path,
    length_list_dir: Path,
    filenames_subset: List[str] = None,
) -> None:


    tokenizer = Tokenizer(tokenizer_path)
    eos_token = tokenizer.processor.decode([tokenizer.eos_id])

    # Use the provided filenames_subset or default to all filenames
    filenames = filenames_subset
    
    if not filenames:
        raise RuntimeError(
            f"No files matching found at {source_path}. \n"
            "Make sure you download the data..."
        )
    for filepath in filenames:
        length_list_file = os.path.join(length_list_dir, f"{os.path.basename(filepath).split('.')[0]}_length_list.npy")
        length_list = []
        with open(filepath, "r") as f:
            for k, row in tqdm(enumerate(f), desc=f"Processing {filepath}"):
                text = row.split()[0]
                text_ids = tokenizer.encode(text)
                decoded_text = tokenizer.decode(text_ids)
                length = len(text_ids) # length of the tokenized text; this takes into account the <eos> token
                decoded_text = decoded_text.replace(eos_token, "")
                decoded_text = decoded_text.replace(" ", "")
                assert text == decoded_text, f"{text} != {decoded_text}"

                length_list.append(length)
        np.save(length_list_file, np.array(length_list))




def prepare(
    source_path,
    tokenizer_path,
):
    import time
    length_list_dir = os.path.join(source_path, "zinc_length_list")
    os.makedirs(length_list_dir, exist_ok=True)

    filenames = glob.glob(os.path.join(source_path, "*.txt"), recursive=True)

    num_processes = min(cpu_count(), len(filenames))
    num_processes = 50 # for a 52-core server, 50 processes is optimal
    chunked_filenames = np.array_split(filenames, num_processes)
    print(f"Number of processes: {num_processes}")

    processes = []
    start_time = time.time()

    for i, subset in enumerate(chunked_filenames):
        p = Process(target=prepare_full, args=(source_path, tokenizer_path, length_list_dir, list(subset)))
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
        default="../../../data/raw_data/zinc20/shuffled_data/",
        help="The directory to the preprocessed data",
    )
    parser.add_argument(
        "--tokenizer_path",
        default="../../../tokenizers/pretrain_tokenizer",
        help="The directory to the tokenizer",
    )
    return parser.parse_args()

if __name__ == "__main__":
    # parse the arguments
    args = parse_args()
    prepare(args.source_path, Path(args.tokenizer_path))
    
