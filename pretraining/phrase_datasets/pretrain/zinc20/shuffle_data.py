import os
import random
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', default='../../../data/raw_data/zinc20', help='The directory to the raw data')
    parser.add_argument('--shuffle_seed', default=533, type=int, help='The seed to shuffle the data')
    return parser.parse_args()

if __name__ == '__main__':
    # parse the arguments
    args = parse_args()

    original_data_path = os.path.join(args.raw_data_dir, 'original_data')
    shuffled_data_path = os.path.join(args.raw_data_dir, 'shuffled_data')

    # this is only used temporarily to shuffle the data
    # should be removed after the all the preprocessed data is saved
    os.makedirs(shuffled_data_path, exist_ok=True)

    all_files = [file for file in os.listdir(original_data_path) if file.endswith(".txt")]

    random.seed(args.shuffle_seed)  # set seed for reproducibility

    for file_index, file in enumerate(all_files):
        print(f"Shuffling file {file_index + 1} of {len(all_files)}")
        with open(os.path.join(original_data_path, file), "r") as input_file:
            lines = input_file.readlines()
        random.shuffle(lines)  # shuffle lines within each file
        with open(os.path.join(shuffled_data_path, f"shuffled_{file_index}.txt"), "w") as output_file:
            output_file.writelines(lines)
    
    print("Shuffling complete!")

    
