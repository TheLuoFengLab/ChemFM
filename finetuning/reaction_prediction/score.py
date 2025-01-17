from rdkit import Chem
import os
import argparse
from tqdm import tqdm
import multiprocessing
import pandas as pd
from rdkit import RDLogger
import re
import ast
import numpy as np

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

def canonicalize_smiles_clear_map(smiles,return_max_frag=False):
    mol = Chem.MolFromSmiles(smiles,sanitize=True)
    if mol is not None:
        [atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]
        try:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        except:
            if return_max_frag:
                return '',''
            else:
                return ''
        if return_max_frag:
            sub_smi = smi.split(".")
            sub_mol = [Chem.MolFromSmiles(smiles,sanitize=True) for smiles in sub_smi]
            sub_mol_size = [(sub_smi[i], len(m.GetAtoms())) for i, m in enumerate(sub_mol) if m is not None]
            if len(sub_mol_size) > 0:
                return smi, canonicalize_smiles_clear_map(sorted(sub_mol_size,key=lambda x:x[1],reverse=True)[0][0],return_max_frag=False)
            else:
                return smi, ''
        else:
            return smi
    else:
        if return_max_frag:
            return '',''
        else:
            return ''


def compute_rank(prediction,raw=False,alpha=1.0):
    valid_score = [[k for k in range(len(prediction[j]))] for j in range(len(prediction))]
    invalid_rates = [0 for k in range(len(prediction[0]))]
    rank = {}
    highest = {}

    for j in range(len(prediction)):
        for k in range(len(prediction[j])):
            if prediction[j][k] == "":
                valid_score[j][k] = opt.beam_size + 1
                invalid_rates[k] += 1
        de_error = [i[0] for i in sorted(list(zip(prediction[j], valid_score[j])), key=lambda x: x[1]) if i[0] != ""]
        prediction[j] = list(set(de_error))
        prediction[j].sort(key=de_error.index)
        for k, data in enumerate(prediction[j]):
            if data in rank:
                rank[data] += 1 / (alpha * k + 1)
            else:
                rank[data] = 1 / (alpha * k + 1)
            if data in highest:
                highest[data] = min(k,highest[data])
            else:
                highest[data] = k
    return rank,invalid_rates


def main(opt):
    print('Reading predictions from file ...')
    data = pd.read_csv(opt.data_path)
    # get the targets
    targets = data['target_smiles'].tolist()
    pool = multiprocessing.Pool(processes=opt.process_number)
    targets = pool.map(func=canonicalize_smiles_clear_map, iterable=targets)
    pool.close()
    pool.join()
    # get the predictions
    predictions = data['predictions'].apply(ast.literal_eval) # data_len x augmentation x beam_size

    original_augmentation = len(predictions[0])
    assert opt.augmentation <= original_augmentation, f"The set augmentation {opt.augmentation} is larger than the original augmentation {original_augmentation}"
    predictions = predictions.apply(lambda x: x[:opt.augmentation])
    
    
    data_size = len(predictions)
    augmentation = len(predictions[0])
    beam_size = len(predictions[0][0])
    assert data_size == len(targets), "the prediction len does not equal to the data_len"
    assert augmentation == opt.augmentation, "the inferenced augmentation does not " + \
                                                "equal to the set augmentation"
    assert beam_size == opt.beam_size, "the inferenced beam size does not" + \
                                            "equal to the set beam size"

    ground_truth = targets
    print("Origin Target Lentgh, ", len(ground_truth))

    
    print("Cutted Length, ",data_size)
    accuracy = [0 for j in range(opt.n_best)]
    accurate_indices = [[] for j in range(opt.n_best)]
    invalid_rates = [0 for j in range(opt.beam_size)]
    sorted_invalid_rates = [0 for j in range(opt.beam_size)]
    unique_rates = 0
    ranked_results = []

    for i in tqdm(range(len(predictions))):
        
        accurate_flag = False

        rank, invalid_rate = compute_rank(predictions[i], alpha=opt.score_alpha)
        for j in range(opt.beam_size):
            invalid_rates[j] += invalid_rate[j]
        rank = list(zip(rank.keys(),rank.values()))
        rank.sort(key=lambda x:x[1],reverse=True)
        rank = rank[:opt.n_best]
        ranked_results.append([item[0] for item in rank])

        for j, item in enumerate(rank):
            if item[0] == ground_truth[i]:
                if not accurate_flag:
                    accurate_flag = True
                    accurate_indices[j].append(i)
                    for k in range(j, opt.n_best):
                        accuracy[k] += 1

    for i in range(opt.n_best):
        if i in [0,1,2,3,4,5,6,7,8,9,19,49]:
            print("Top-{} Acc:{:.3f}%, ".format(i+1,accuracy[i] / data_size * 100),
                  " Invalid SMILES:{:.3f}% Sorted Invalid SMILES:{:.3f}%".format(invalid_rates[i] / data_size / opt.augmentation * 100,sorted_invalid_rates[i] / data_size / opt.augmentation * 100))

    print("Unique Rates:{:.3f}%".format(unique_rates / data_size / opt.beam_size * 100))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='score.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-beam_size', type=int, default=10,help='Beam size')
    parser.add_argument('-n_best', type=int, default=10,help='n best')
    parser.add_argument('-data_path', type=str, required=True,
                        help="Path to file containing the data")
    parser.add_argument('-augmentation', type=int, default=20)
    parser.add_argument('-score_alpha', type=float, default=1.0)
    parser.add_argument('-length', type=int, default=-1)
    parser.add_argument('-process_number', type=int, default=multiprocessing.cpu_count())

    opt = parser.parse_args()
    print(opt)
    main(opt)