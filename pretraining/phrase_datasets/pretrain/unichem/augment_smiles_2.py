import argparse
import os
from tqdm import tqdm
import random
from multiprocessing import Process

from rdkit import RDLogger

# Suppress RDKit INFO messages
RDLogger.DisableLog('rdApp.*')



from pathlib import Path
import sys
# support running without installing as a package
wd = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.append(str(wd))

from utils.smiles_utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../../../data/raw_data/unichem/', help='The directory to the raw data')
    return parser.parse_args()

def augment_smiles(smiles_path, aug_smiles_path, process, num_aug_smiles_file, num_augs):
    print("start process: ", process)

    aug_smiles_file_paths = [os.path.join(aug_smiles_path, f'smiles_{process}_{i}.txt') for i in range(num_aug_smiles_file)]
    fs = [open(f, 'w') for f in aug_smiles_file_paths]

    random.seed(533)

    sme = SmilesEnumerator()

    smiles_path = os.path.join(smiles_path, f"smiles_{process}.txt")
    with open(smiles_path, 'r') as f:
        smiles = f.readlines()
    
    for smile in tqdm(smiles, total=len(smiles), leave=False, desc='Processing smiles'):
        smile = smile.strip()

        for i in range(num_augs):
            if i == 0:
                aug_smile = smile
            else:
                aug_smile = sme.randomize_smiles(smile)
            if aug_smile != "":
                # random choose a file to write
                f = random.choice(fs)
                f.write(aug_smile + '\n')

if __name__ == '__main__':
    # parse the arguments
    args = parse_args()
    
    smiles_path = os.path.join(args.data_path, 'smiles')
    aug_smiles_path = os.path.join(args.data_path, 'aug_smiles')
    os.makedirs(aug_smiles_path, exist_ok=True)

    num_processes = 50
    num_aug_smiles_file = 100
    num_augs = 10

    processes = []
    for i in range(num_processes):
        p = Process(target=augment_smiles, args=(smiles_path, aug_smiles_path, i, num_aug_smiles_file, num_augs))
        p.start()
    
    for p in processes:
        p.join()