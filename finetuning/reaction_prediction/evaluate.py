import argparse
import transformers
from utils import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    smart_tokenizer_and_embedding_resize,
    make_data_module
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

if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
    
    #TODO: support the model from the remote
    if not args.lora:
        config = AutoConfig.from_pretrained(
            model_args.model_path,
            device_map=device_map,
            trust_remote_code=args.trust_remote_code,
        )
        config.attention_dropout = args.attention_dropout

        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_path,
            config=config,
            device_map=device_map,
            trust_remote_code=args.trust_remote_code,
        )
    
        # the finetune tokenizer could be in different size with pretrain tokenizer, and also, we need to add PAD_TOKEN
        special_tokens_dict = dict(pad_token=DEFAULT_PAD_TOKEN)
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model
        )
        model.config.pad_token_id = tokenizer.pad_token_id

    else:
        config = AutoConfig.from_pretrained(
            args.pretrain_model,
            trust_remote_code=args.trust_remote_code,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            args.pretrain_model,
            config=config,
            trust_remote_code=args.trust_remote_code,
            device_map=device_map
        )

        # we should resize the embedding layer of the base model to match the adapter's tokenizer
        special_tokens_dict = dict(pad_token=DEFAULT_PAD_TOKEN)
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=base_model
        )
        base_model.config.pad_token_id = tokenizer.pad_token_id

        # load the adapter model
        model = PeftModel.from_pretrained(
            base_model,
            model_args.model_path,
        )

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

def generate(model, loader, accelerator, tokenizer, max_length, beam_size):
    model.eval()
    # if only_loss is False, we will also evaluate the accuracy, 
    # but this requires the batch size to be 1
    
    df = []

    pbar = tqdm(loader, desc=f"Evaluating...", leave=False)
    for it, batch in enumerate(pbar):
        target_smiles = batch['tgt_smiles']
        generation_prompts = batch['generation_prompts']

        sub_df = dict()

        predictions = [] # [num_augmentations, beam_size]
        non_canonized_predictions = [] # [num_augmentations, beam_size]
        for generation_prompt in generation_prompts:
            inputs = tokenizer(generation_prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
            del inputs['token_type_ids']

            # generate
            with torch.set_grad_enabled(False):
                outputs = model.module.generate(**inputs, max_length=max_length, num_return_sequences=beam_size,
                               do_sample=False, num_beams=beam_size,
                               eos_token_id=tokenizer.eos_token_id,
                               early_stopping='never',
                               pad_token_id=tokenizer.pad_token_id,
                               length_penalty=0.0,
                               )
            
            original_smiles_list = tokenizer.batch_decode(outputs[:, len(inputs['input_ids'][0]):],
                                                          skip_special_tokens=True) 
            original_smiles_list = map(lambda x: x.replace(" ", ""), original_smiles_list)
            # canonize the SMILES
            canonized_smiles_list = []
            temp = []
            for original_smiles in original_smiles_list:
                temp.append(original_smiles)
                try:
                    canonized_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(original_smiles)))
                except:
                    canonized_smiles_list.append("")
            
            predictions.append(canonized_smiles_list)
            non_canonized_predictions.append(temp)
        
        # save the predictions
        sub_df['target_smiles'] = target_smiles
        sub_df['predictions'] = predictions
        sub_df['generation_prompts'] = generation_prompts
        sub_df['non_canonized_predictions'] = non_canonized_predictions

        gathered_sub_df = accelerator.gather_for_metrics([sub_df])
        df.extend(gathered_sub_df)

    df = pd.DataFrame(df)
    return df
    

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

    assert args.task_type in ['retrosynthesis', 'synthesis']

    # set seed for transformers, torch, and numpy, and random
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, args)

    # set up the accelerator
    accelerator = Accelerator()
    
    # set up the dataset
    data_module = make_data_module(tokenizer=tokenizer, dataset=args.dataset, ignore_index=IGNORE_INDEX,
                                   args=args, evaluation=True)
    data_collator = data_module['data_collator']
    test_loader = DataLoader(data_module['test_dataset'], batch_size=1, shuffle=False, collate_fn=data_collator)

    model, test_loader = accelerator.prepare(
        model, test_loader
    )

    df = generate(model, test_loader, accelerator, tokenizer, args.source_max_len + args.target_max_len, args.beam_size)

    output_path = os.path.join(args.model_path, f"predictions.csv")
    
    if accelerator.is_main_process:
        df.to_csv(output_path, index=False)

if __name__ == '__main__':
    evaluate()