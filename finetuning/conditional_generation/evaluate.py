import argparse
import transformers
from utils import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    smart_tokenizer_and_embedding_resize,
    make_data_module,
    make_test_data_module
)
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    )
import torch
import numpy as np
import random
from accelerate import Accelerator
from peft import PeftModel
from torch.utils.data.dataloader import DataLoader
import pandas as pd
import json
from torch.nn import functional as F
import importlib
from metric_calculator import get_similarity, get_scaffold

if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from llama_customized_models import LlamaForCausalLMWithNumericalEmbedding

from tqdm import tqdm
import os

from rdkit import RDLogger, Chem
# Suppress RDKit INFO messages
RDLogger.DisableLog('rdApp.*')

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

def load_model_and_tokenizer(model_args, args):
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    # load tokenizer if it is provided
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_path,
        padding_side="right",
        use_fast=True,
        trust_remote_code=args.trust_remote_code
        )
    
    model = LlamaForCausalLMWithNumericalEmbedding.from_pretrained(
        model_args.model_path,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code)

    special_tokens_dict = dict(pad_token=DEFAULT_PAD_TOKEN)
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    try: 
        model.print_trainable_parameters()
    except:
        dtypes = {}
        dtypes_trainable = {}
        for _, p in model.named_parameters():
            dtype = p.dtype
            if dtype not in dtypes: 
                dtypes[dtype] = 0
                dtypes_trainable[dtype] = 0
            dtypes[dtype] += p.numel()
            if p.requires_grad:
                dtypes_trainable[dtype] += p.numel()

        total = 0
        trainable = 0
        for k, v in dtypes.items(): total+= v
        for k, v in dtypes_trainable.items(): trainable += v
        for k, v in dtypes.items():
            print(f"dtypes: {k} || trainable params: {dtypes_trainable[k]} || {k} params: {v} || all params: {total} || trainable%: {dtypes_trainable[k]/v:.3f}")
    
    model.config.use_cache = False
    model.eval()

    return model, tokenizer

def generate(model, loader, accelerator, tokenizer, max_length):
    model.eval()
    
    df = []
    pbar = tqdm(loader, desc=f"Evaluating...", leave=False)
    for it, batch in enumerate(pbar):
        sub_df = dict()
        
        batch_size = batch['input_ids'].shape[0]
        assert batch_size == 1, "The batch size should be 1"

        temperature = batch['temperature'][0]
        property_names = batch['property_names'][0]
        non_normalized_properties = batch['non_normalized_properties'][0]
        scaffold = batch['scaffold'][0]

        num_generations = 1
        del batch['temperature']
        del batch['property_names']
        del batch['non_normalized_properties']
        del batch['scaffold']

        input_length = batch['input_ids'].shape[1]
        steps = max_length - input_length

        with torch.set_grad_enabled(False):
            early_stop_flags = torch.zeros(num_generations, dtype=torch.bool).to(model.device)
            for k in range(steps):
                logits = model(**batch)['logits']
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                ix = torch.multinomial(probs, num_samples=num_generations)

                ix[early_stop_flags] = tokenizer.eos_token_id

                batch['input_ids'] = torch.cat([batch['input_ids'], ix], dim=-1)
                early_stop_flags |= (ix.squeeze() == tokenizer.eos_token_id)

                if torch.all(early_stop_flags):
                    break
                
        generations = tokenizer.batch_decode(batch['input_ids'][:, input_length:], skip_special_tokens=True)
        generations = map(lambda x: x.replace(" ", ""), generations)

        predictions = []
        for generation in generations:
            try:
                predictions.append(Chem.MolToSmiles(Chem.MolFromSmiles(generation)))
            except:
                predictions.append("")
        
        sub_df['SMILES'] = predictions[0]
        sub_df['property_names'] = property_names
        sub_df['property'] = batch['properties'][0]
        sub_df['non_normalized_properties'] = non_normalized_properties
        if scaffold is not None:
            sub_df['scaffold'] = scaffold

        gathered_sub_df = accelerator.gather_for_metrics([sub_df])
        df.extend(gathered_sub_df)
            
    df = pd.DataFrame(df) 
    return df
    
def phrase_df(df):
    metric_calculator = importlib.import_module("metric_calculator")

    new_df = []
    # iterate over the dataframe
    for i in range(len(df)):
        sub_df = dict()
        
        # get the SMILES
        smiles = df.iloc[i]['SMILES']
        # get the property names
        property_names = df.iloc[i]['property_names']
        # get the non normalized properties
        non_normalized_properties = df.iloc[i]['non_normalized_properties']
        
        

        sub_df['SMILES'] = smiles

        if 'scaffold' in df.columns:
            scaffold = df.iloc[i]['scaffold']
            sub_df['scaffold'] = scaffold
            if smiles == "":
                sub_df['Similarity'] = np.nan
            else:
                sub_df['Similarity'] = get_similarity(get_scaffold(smiles), scaffold)
        
        # compute the similarity between the scaffold and the SMILES

        for j in range(len(property_names)):
            # get the property name
            property_name = property_names[j]
            # get the non normalized property
            non_normalized_property = non_normalized_properties[j]

            sub_df[f'{property_name}_condition'] = non_normalized_property

            if smiles == "":
                sub_df[f'{property_name}_measured'] = np.nan
            else:
                property_eval_func_name = f"compute_{property_name}"
                property_eval_func = getattr(metric_calculator, property_eval_func_name)
                sub_df[f'{property_name}_measured'] = property_eval_func(Chem.MolFromSmiles(smiles))
            
        new_df.append(sub_df)
    
    new_df = pd.DataFrame(new_df)
    return new_df
        

def evaluate():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments
    ))
    model_args, data_args, training_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)

    # if training_args_file is provided, overwrite the arguments with the arguments in the file
    if training_args.training_args_file is not None:
        model_args, data_args, training_args = hfparser.parse_yaml_file(training_args.training_args_file)
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    # set seed for transformers, torch, and numpy, and random
    ## we cannot set the seed here, if so, the model will generate the same sequence for each batch
    #set_seed(args.seed)
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #random.seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(model_args, args)

    # set up the accelerator
    accelerator = Accelerator()

    data_module = make_test_data_module(tokenizer=tokenizer, ignore_index=IGNORE_INDEX, args=args)
    data_collator = data_module['data_collator']
    test_loader = DataLoader(data_module['test_dataset'], batch_size=1, shuffle=False, collate_fn=data_collator)

    model, test_loader = accelerator.prepare(model, test_loader)

    df = generate(model, test_loader, accelerator, tokenizer, args.source_max_len + args.target_max_len)
    if accelerator.is_main_process:
        df = phrase_df(df)
        # save the dataframe to the output file
        output_path = args.generation_output_path
        # create the output directory if it does not exist
        # get the folder name from the output path
        folder_name = os.path.dirname(output_path)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        df.to_csv(output_path, index=False)
        

if __name__ == '__main__':
    evaluate()