import argparse
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../../../data/raw_data/unichem', help='The directory to the raw data')
    return parser.parse_args()

if __name__ == '__main__':
    # parse the arguments
    args = parse_args()

    original_data_path = os.path.join(args.data_path, 'structure.tsv')
    # we hard code the number of files to split into
    num_files = 50
    
    folder = os.path.join(args.data_path, 'inchis')
    os.makedirs(folder, exist_ok=True)
    files = [os.path.join(folder, f'inchis_{i}.txt') for i in range(num_files)]
    fs = [open(file, 'w') for file in files]

    with open(original_data_path, 'r') as f:
        next(f)
        lines = f.readlines()

        for i, line in tqdm(enumerate(lines), total=len(lines)):
            inchi = line.split('\t')[1].strip()
            fs[i % num_files].write(inchi + '\n')

    for f in fs:
        f.close()


    
    