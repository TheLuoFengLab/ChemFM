from pathlib import Path
import sys
import glob
import os
from typing import List
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, cpu_count
import random
import argparse

# support running without installing as a package
wd = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import packed_dataset
from lit_gpt.tokenizer import Tokenizer

def prepare_full(
    source_path: Path,
    tokenizer_path: Path,
    destination_path: Path,
    max_tokens: int,
    chunk_sequences: int,
    train_filenames: set[str] = None,
    filenames_subset: List[str] = None,
    process_id: int = 0,
) -> None:

    destination_path.mkdir(parents=True, exist_ok=True)

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
        if filepath in train_filenames:
            split = "train"
        else:
            split = "test"
        file_id = int(os.path.basename(filepath).split(".")[0].split("_")[-1])
    
        builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=f"{split}_zinc20_{process_id}_{file_id}",  # Use process_id to differentiate builders
            chunk_size=chunk_sequences*max_tokens,
            sep_token=tokenizer.eos_id,
            dtype=np.int16,
            vocab_size=tokenizer.vocab_size,
        )

        with open(filepath, "r") as f:
            for k, row in tqdm(enumerate(f)):
                text = row.split()[0]
                text_ids = tokenizer.encode(text)
                decoded_text = tokenizer.decode(text_ids)
                decoded_text = decoded_text.replace(eos_token, "")
                decoded_text = decoded_text.replace(" ", "")
                assert text == decoded_text, f"{text} != {decoded_text}"
                if text_ids.shape[0] > max_tokens:
                    #print(f"Skipping {text} because it is too long")
                    continue
                assert text_ids.shape[0] <= max_tokens, f"Text length is {text_ids.shape[0]}"
                text_ids = np.pad(text_ids, (0, max_tokens - text_ids.shape[0]), constant_values=-100) # pad with -100, this will be masked out in the loss function
                builder.add_array(text_ids)
        builder.write_reminder()

def prepare(
    source_path,
    tokenizer_path,
    destination_path,
    max_tokens,
    chunk_sequences,
    percentage,
    random_seed,
):
    import time

    if not destination_path.exists():
        os.makedirs(destination_path, exist_ok=True)

    # it will generate 491575 files in total
    filenames = glob.glob(os.path.join(source_path, "*.txt"), recursive=True)
    random.seed(random_seed)
    filenames = random.sample(filenames, len(filenames))
    train_filenames = filenames[:int(len(filenames) * percentage)]
    
    num_processes = min(cpu_count(), len(filenames))
    num_processes = 50 # for a 54 core machine, this is the optimal number of processes
    chunked_filenames = np.array_split(filenames, num_processes)
    print(f"Number of processes: {num_processes}")

    processes = []
    start_time = time.time()

    for i, subset in enumerate(chunked_filenames):
        p = Process(target=prepare_full, args=(source_path, tokenizer_path, destination_path, max_tokens, chunk_sequences, train_filenames, list(subset), i))
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
    parser.add_argument(
        "--chunks_path",
        default="../../../data/pretrain_data/ZINC20/chunks_SMILES",
        help="The directory to save the chunks",
    )
    parser.add_argument(
        "--max_tokens",
        default=512,
        type=int,
        help="The maximum number of tokens in a sequence",
    )
    parser.add_argument(
        "--chunk_sequences",
        default=2048,
        type=int,
        help="The number of sequences in a chunk",
    )
    parser.add_argument(
        "--percentage",
        default=0.9,
        type=float,
        help="The percentage of the data to use for training",
    )
    parser.add_argument(
        "--random_seed",
        default=533,
        type=int,
        help="The seed to shuffle the data files",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    prepare(Path(args.source_path), 
            Path(args.tokenizer_path), 
            Path(args.chunks_path),
            args.max_tokens, args.chunk_sequences,
            args.percentage, args.random_seed)
