from rdkit import Chem
import argparse
import os
from tqdm import tqdm
from multiprocessing import Process, cpu_count
import pandas as pd

from rdkit import RDLogger

# Suppress RDKit INFO messages
RDLogger.DisableLog('rdApp.*')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../../../data/raw_data/unichem/', help='The directory to the raw data')
    return parser.parse_args()

def convert_to_smiles(inchi_file, smiles_file):
    with open(inchi_file, 'r') as f:
        inchis = f.readlines()
    
    with open(smiles_file, 'w') as f:
        for inchi in tqdm(inchis, total=len(inchis)):
            try:
                inchi = inchi.strip()
                mol = Chem.MolFromInchi(inchi)
                smiles = Chem.MolToSmiles(mol)
                f.write(smiles + '\n')
            except:
                pass

if __name__ == '__main__':
    # parse the arguments
    args = parse_args()
    
    inchis_path = os.path.join(args.data_path, 'inchis')
    smiles_path = os.path.join(args.data_path, 'smiles')
    os.makedirs(smiles_path, exist_ok=True)

    processes = []
    # we also hard code the number of processes to use
    num_processes = 50
    
    for i in range(num_processes):
        inchi_file = os.path.join(inchis_path, f'inchis_{i}.txt')
        smiles_file = os.path.join(smiles_path, f'smiles_{i}.txt')
        
        p = Process(target=convert_to_smiles, args=(inchi_file, smiles_file))
        p.start()
        processes.append(p)

