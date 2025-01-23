import os
import argparse
import glob
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../../../data/raw_data/unichem/', help='The directory to the raw data')
    return parser.parse_args()

if __name__ == '__main__':
    # parse the arguments
    args = parse_args()
    
    smiles_path = os.path.join(args.data_path, 'smiles')
    aug_smiles_path = os.path.join(args.data_path, 'aug_smiles')

    num_aug_smiles_file = 100

    total_count = 0
    for i in range(num_aug_smiles_file):
        files = glob.glob(os.path.join(aug_smiles_path, f'smiles_*_{i}.txt'))
        # read each file and count the number of lines
        # also merge and randomly shuffle the lines
        whole_lines = []
        for file in files:
            with open(file, 'r') as f:
                lines = f.readlines()
                whole_lines.extend(lines)
        random.shuffle(whole_lines)
        file_count = len(whole_lines)
        with open(os.path.join(aug_smiles_path, f'smiles_{i}.txt'), 'w') as f:
            f.writelines(whole_lines)
        
        print(f"File {i}: {file_count}")
        total_count += file_count
        
        
    # 1 779 617 164
    print(f"Total number of lines: {total_count}")