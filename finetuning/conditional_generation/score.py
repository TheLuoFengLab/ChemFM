import argparse
import multiprocessing
import pandas as pd
from rdkit import Chem

def canonicalize_smiles(smiles):
    try:
        return Chem.CanonSmiles(smiles)
    except:
        return ""

def compute_metrics(df, train_data, opt, scaffold_list, property_names):
    for scaffold in scaffold_list:
        if scaffold is not None:
            df_scaffold = df[df['scaffold'] == scaffold]
        else:
            df_scaffold = df
        
        # check how many sample points
        num_sample_points = len(df_scaffold['condition'].unique())

        # get the valid smiles
        valid_smiles = df_scaffold[df_scaffold['smiles'] != ""]
        if scaffold is not None:
            valid_smiles = valid_smiles[valid_smiles['similarity'] > opt.similarity_threshold]
        num_valid_smiles = len(valid_smiles)

        # get the unique smiles
        unique_smiles = valid_smiles.drop_duplicates(subset='smiles')
        #print(unique_smiles)
        #unique_smiles = valid_smiles['smiles'].unique()
        #print(unique_smiles)
        #exit()
        num_unique_smiles = len(unique_smiles)

        # get the novelty
        #unique_smiles = set(unique_smiles)
        #novel_smiles = unique_smiles - train_data#unique_smiles[unique_smiles['in_train_data'] == False]
        novel_smiles = unique_smiles[unique_smiles['in_train_data'] == False]
        num_novel_smiles = len(novel_smiles)



        print(f"-------------------{scaffold}-------------------")

        print("validity:", num_valid_smiles, num_valid_smiles / (opt.generation_samples*num_sample_points))
        print("unique:", num_unique_smiles, num_unique_smiles / len(valid_smiles))
        print("novelty:", num_novel_smiles, num_novel_smiles / len(unique_smiles))
        # print the same scaffold similarity 
        if scaffold is not None:
            same_scaffold_smiles = novel_smiles[novel_smiles['similarity'] == 1]
            print("same scaffold:", len(same_scaffold_smiles), len(same_scaffold_smiles) / len(novel_smiles))
            

        for property_name in property_names:
            df_scaffold[f'{property_name}_diff'] = abs(valid_smiles[f'{property_name}_condition'] - valid_smiles[f'{property_name}_measured'])
            diff_mean = df_scaffold[f'{property_name}_diff'].mean()
            diff_std = df_scaffold[f'{property_name}_diff'].std()
            print(f"{property_name} diff mean:", diff_mean)
            print(f"{property_name} diff std:", diff_std)

        print("--------------------------------")


def main(opt):
    # load the training data
    train_data = pd.read_csv(opt.train_data_path)['smiles'].tolist()
    pool = multiprocessing.Pool(processes=opt.process_number)
    train_data = pool.map(func=canonicalize_smiles, iterable=train_data)
    train_data = set(train_data)

    # load the generation data
    df = pd.read_csv(opt.data_path)
    df.columns = df.columns.str.lower()
    df['smiles'] = pool.map(func=canonicalize_smiles, iterable=df['smiles'].tolist())

    # get the valid smiles
    #df['in_train_data'] = pool.map(func=lambda x: x in train_data, iterable=df['smiles'].tolist())
    df['in_train_data'] = df['smiles'].apply(lambda x: x in train_data)

    # get the scaffold list
    if 'scaffold' in df.columns:
        # get the scaffold list
        scaffold_list = df['scaffold'].unique()
    else:
        scaffold_list = [None]
    
    
    # get the property names
    property_names = [i.split("_")[0] for i in [col for col in df.columns if col.endswith("_condition")]]

    df['condition'] = df.apply(lambda row: tuple(row[f'{prop}_condition'] for prop in property_names), axis=1)

    # comput the diff
    for property_name in property_names:
        df[f'{property_name}_diff'] = abs(df[f'{property_name}_condition'] - df[f'{property_name}_measured'])

    

    compute_metrics(df, train_data, opt, scaffold_list, property_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='score.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-data_path', type=str, required=True,
                        help="Path to file containing the data")
    parser.add_argument('-train_data_path', type=str, required=True,
                        help="Path to file containing the training data, used to calculate the novelty")
    parser.add_argument('-process_number', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('-generation_samples', type=int, default=10000)
    parser.add_argument('-similarity_threshold', type=float, default=0.8, 
                        help="The threshold for the similarity between the scaffold and the SMILES")

    opt = parser.parse_args()
    main(opt)