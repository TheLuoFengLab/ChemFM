import argparse
import transformers
import torch
import math

from utils import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    smart_tokenizer_and_embedding_resize,
    make_data_module,
)
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoConfig,
    LlamaForCausalLM,
)
from peft import (
    get_peft_model,
    LoraConfig
)
import os
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader

import numpy as np
import pandas as pd
import random

if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from tqdm import tqdm

from rdkit import RDLogger, Chem
# Suppress RDKit INFO messages
RDLogger.DisableLog('rdApp.*')

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

def load_model_and_tokenizer(model_args, args, adapter_name):

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
    
    # load model
    config = AutoConfig.from_pretrained(
        model_args.pretrain_model,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
    )
    # apply dropout for the base model. TODO: It looks interesting that applying dropout for the base model can affect the performance.
    config.attention_dropout = args.attention_dropout

    model = LlamaForCausalLM.from_pretrained(
        model_args.pretrain_model,
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
    
    # apply lora if specified
    if args.lora:
        lora_alpha = int(args.lora_rank * args.lora_alpha_ratio)
        print(f"Using LORA with rank {args.lora_rank}, alpha {lora_alpha}, and dropout {args.lora_dropout}.")
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_rank,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj",
                            "up_proj", "down_proj"],
            modules_to_save=["embed_tokens", "lm_head"],
            inference_mode=False,
            lora_alpha=lora_alpha, lora_dropout=args.lora_dropout,
            use_rslora=False
        )
        model = get_peft_model(model, lora_config, adapter_name=adapter_name)
    
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

    return model, tokenizer

def train_epoch(model, optimizer, scheduler, train_loader, accelerator, training_args, epoch, step_count):
    model.train()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for it, batch in enumerate(pbar):
        if it > 10:
            break
        del batch['tgt_smiles']
        del batch['generation_prompts']

        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
    
            lr = scheduler.get_last_lr()[0]
            pbar.set_description(f"epoch {epoch} iter {it}/{len(train_loader)}: train loss {loss.item():.5f}; lr {lr:e}; grad norm {grad_norm:.5f}")

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        step_count += 1
    
    return step_count

def evaluate(model, loader, accelerator, split, tokenizer, max_length, beam_size, only_loss=False):
    model.eval()
    # if only_loss is False, we will also evaluate the accuracy, 
    # but this requires the batch size to be 1
    
    evaluation_losses = []
    hit_lists = []

    pbar = tqdm(loader, desc=f"Evaluating {split}")
    for it, batch in enumerate(pbar):
        batch_size = batch['input_ids'].shape[0]
        target_smiles = batch['tgt_smiles']
        del batch['tgt_smiles']
        generation_prompts = batch['generation_prompts']
        del batch['generation_prompts']
        
        if not only_loss:
            assert batch_size == 1, f"The batch size must be 1 rather than {batch_size} for evaluating the accuracy"

        with torch.set_grad_enabled(False):
            outputs = model(**batch)
            loss = outputs.loss
            gathered_loss = accelerator.gather_for_metrics(loss)
            gathered_loss = gathered_loss.cpu().numpy().reshape(-1, 1)
            evaluation_losses.append(gathered_loss)

        if not only_loss:
            # generate
            inputs = tokenizer(generation_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            del inputs['token_type_ids']
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
            for original_smiles in original_smiles_list:
                try:
                    canonized_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(original_smiles)))
                except:
                    canonized_smiles_list.append("")
            # check the accuracy
            hit_list = [0 for _ in range(beam_size)]
            # canonize the targets
            target_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(target_smiles[0]))
            for i, canonized_smiles in enumerate(canonized_smiles_list):
                if canonized_smiles == target_smiles:
                    hit_list[i] = 1
                    break

            gathered_hit_list = accelerator.gather_for_metrics([hit_list])
            gathered_hit_list = np.array(gathered_hit_list).reshape(-1, beam_size)
            hit_lists.append(gathered_hit_list)
    

    evaluation_losses = np.concatenate(evaluation_losses, axis=0)
    loss = np.mean(evaluation_losses)

    if only_loss:
        return loss
    
    
    hit_lists = np.concatenate(hit_lists, axis=0)
    total_sample = len(evaluation_losses)
    # calculate the top-k accuracy
    hit_lists = hit_lists.sum(axis=0)
    accuracy = [np.sum(hit_lists[:i+1]) / total_sample for i in range(beam_size)]

    return loss, accuracy

def main():
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

    # TODO: check what happens if we set the full parameters
    if args.adapter_name is None:
        adapter_name = f"{args.dataset}"
    else:
        adapter_name = args.adapter_name
    
    # set seed for transformers, torch, and numpy, and random
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, args, adapter_name)

    # set up the accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    )

    # set up the dataset
    data_module = make_data_module(tokenizer=tokenizer, dataset=args.dataset, ignore_index=IGNORE_INDEX,
                                   args=args)
    data_collator = data_module['data_collator']

    train_loader = DataLoader(data_module['train_dataset'], batch_size=training_args.per_device_train_batch_size, shuffle=True,
                              collate_fn=data_collator, num_workers=args.dataloader_num_workers)
    val_loader = DataLoader(data_module['val_dataset'], batch_size=1, shuffle=False, collate_fn=data_collator)
    test_loader = DataLoader(data_module['test_dataset'], batch_size=1, shuffle=False, collate_fn=data_collator)
    print("Data loaders set up")

    # get the number of steps
    len_dataloader = len(train_loader)
    num_update_steps_per_epoch = len_dataloader / training_args.gradient_accumulation_steps
    num_update_steps_per_epoch = max(1, num_update_steps_per_epoch)
    max_steps = math.ceil(num_update_steps_per_epoch * training_args.num_train_epochs)
    num_train_epochs = math.ceil(training_args.num_train_epochs)

    # set up the optimizer and scheduler
    if training_args.optim == 'adamw_torch':
        optimizer = transformers.AdamW(
                params=model.parameters(),
                lr=training_args.learning_rate,
                betas=(training_args.adam_beta1, training_args.adam_beta2),
                eps=training_args.adam_epsilon,
                weight_decay=training_args.weight_decay
                )
    else:
        raise ValueError(f"Optimizer {training_args.optim} not supported.")
    
    scheduler = transformers.get_scheduler(
        training_args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=int(max_steps * training_args.warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": training_args.min_learning_rate} if training_args.lr_scheduler_type == "cosine_with_min_lr" else {}
    )
    print("scheduler and optimizer set up")

    model, optimizer, train_loader, val_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, scheduler)

    # start training

    step_count = 0

    best_val_accuracy = float('-inf')
    best_val_accuracy_epoch = 0
    test_accuracy_at_best_val_accuracy = None
    best_test_accuracy = [0.0] * args.beam_size

    accuracy_list = []

    for epoch in range(1, num_train_epochs+1):
        step_count = train_epoch(model, optimizer, scheduler, train_loader,
                    accelerator, training_args,
                    epoch, step_count)

        val_loss, val_accuracy = evaluate(model, val_loader, accelerator, "val", tokenizer, args.source_max_len + args.target_max_len, args.beam_size)
        test_loss, test_accuracy = evaluate(model, test_loader, accelerator, "test", tokenizer, args.source_max_len + args.target_max_len, args.beam_size)

        if val_accuracy[0] > best_val_accuracy:
            best_val_accuracy = val_accuracy[0]
            test_accuracy_at_best_val_accuracy = test_accuracy
        
        if test_accuracy[0] > best_test_accuracy[0]:
            best_test_accuracy = test_accuracy

        if accelerator.is_main_process:
            # log the results
            print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy[0]:.4f}, "+
                  f"Test loss: {test_loss:.4f}, Test Accuracy: {test_accuracy[0]:.4f}, "  +
                  f"Test Accuracy at Best Val Accuracy: {test_accuracy_at_best_val_accuracy[0]:.4f}, Best Test Accuracy: {best_test_accuracy[0]:.4f}" 
                )

            accuracy_list.append([test_accuracy])

            # save the model
            model_path = os.path.join(training_args.output_dir, f"checkpoint_{epoch}")
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(model_path)
    
    accelerator.end_training()

    # save the accuracy list
    df = pd.DataFrame(accuracy_list, columns=[f"Top-{i+1}" for i in range(len(accuracy_list[0]))])
    df.insert(0, "Epoch", range(1, len(accuracy_list) + 1))  # Add epoch column
    df.to_csv(os.path.join(training_args.output_dir, "accuracy.csv"), index=False)

if __name__ == "__main__":
    main()