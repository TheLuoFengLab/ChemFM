import argparse
import transformers
from utils import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    smart_tokenizer_and_embedding_resize,
    make_data_module,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    LlamaForSequenceClassification,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)
import os
from accelerate import Accelerator

from torch.utils.data.dataloader import DataLoader
import math
import torch

from sklearn.metrics import mean_absolute_error, roc_auc_score, average_precision_score, root_mean_squared_error
from scipy.stats import spearmanr

from tqdm import tqdm
import numpy as np

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

def load_model_and_tokenizer(model_args, args):

    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    # load tokenizer if it is provided
    if model_args.tokenizer_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_path,
            padding_side="right",
            use_fast=True,
            trust_remote_code=args.trust_remote_code
            )
    # otherwise, load tokenizer from pretrain_model
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.pretrain_model,
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

    config.num_labels = args.num_tasks
    model = LlamaForSequenceClassification.from_pretrained(
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
            task_type="SEQ_CLS",
            r=args.lora_rank,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj",
                            "up_proj", "down_proj"],
            modules_to_save=[],
            inference_mode=False,
            lora_alpha=lora_alpha, lora_dropout=args.lora_dropout,
            use_rslora=False
        )
        model = get_peft_model(model, lora_config, adapter_name=f"{args.dataset}")
    
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

def train_epoch(model, optimizer, scheduler, train_loader, loss_fcns, accelerator, args, training_args, epoch, step_count):
    model.train()
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    for it, batch in pbar:
        with accelerator.accumulate(model):
            labels = batch['labels']
            outputs = model(**batch)

            if loss_fcns != []:
                # calculate the loss
                losses = []
                for i, loss_fcn in enumerate(loss_fcns):
                    if args.has_nan_in_dataset:
                        # we need to mask the loss
                        mask = ~torch.isnan(labels[:, i])
                        sum_valid_labels = mask.sum()
                        if sum_valid_labels>0:
                            sum_loss = loss_fcn(outputs.logits[:, i][mask], labels[:, i][mask]).sum()
                            losses.append(sum_loss / sum_valid_labels)
                    else:
                        losses.append(loss_fcn(outputs.logits[:, i], labels[:, i]))
                loss = torch.stack(losses).mean()
                
            else:
                loss = outputs.loss

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        lr = scheduler.get_last_lr()[0]
        pbar.set_description(f"epoch {epoch} iter/total_iter {it}/{step_count}: train loss {loss.item():.5f}. lr {lr:e}")

        step_count += 1
    
    return step_count

def evaluate(model, data_loader, loss_fcns, accelerator, args, split, metric_evaluator, scaler):
    model.eval()

    return_results = dict()
    evaluation_losses = []

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"{split} evaluation")
    if metric_evaluator != None:
        labels_list = []
        preds_list = []
    
    for it, batch in pbar:
        labels = batch['labels']
        with torch.set_grad_enabled(False):
            outputs = model(**batch)

            if loss_fcns != []:
                # calculate the loss
                losses = []
                for i, loss_fcn in enumerate(loss_fcns):
                    if args.has_nan_in_dataset:
                        # we need to mask the loss
                        mask = ~torch.isnan(labels[:, i])
                        sum_valid_labels = mask.sum()
                        if sum_valid_labels>0:
                            sum_loss = loss_fcn(outputs.logits[:, i][mask], labels[:, i][mask]).sum()
                            losses.append(sum_loss / sum_valid_labels)
                    else:
                        losses.append(loss_fcn(outputs.logits[:, i], labels[:, i]))
                loss = torch.stack(losses).mean()

            else:
                loss = outputs.loss

        evaluation_losses.append((loss.item(), len(labels)))

        # add the predictions and labels to the list if we have a metric evaluator
        if metric_evaluator != None:
            predictions = outputs.logits
            preds, refs = accelerator.gather_for_metrics(
                (predictions, labels)
            )
            preds_list.append(preds.cpu().detach().numpy())
            labels_list.append(refs.cpu().detach().numpy())
        
    # concatenate the predictions and labels to calculate the metrics if we have a metric evaluator
    if metric_evaluator != None:
        preds = np.concatenate(preds_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        # if we have a scaler, we need to inverse transform the predictions
        if scaler is not None:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        # calculate the metrics
        # TODO: I only know the auroc will work for averaging across tasks, need to check the others
        #       Let's just use auroc for now
        metrics = []
        for i in range(args.num_tasks):
            task_pred = preds[:, i]
            task_label = labels[:, i]
            if args.has_nan_in_dataset:
                # we need to mask the labels and predictions
                mask = ~np.isnan(task_label)
                task_pred = task_pred[mask]
                task_label = task_label[mask]
            if args.metric == "auroc":
                # if metric is auroc, we should check if there are at least two classes in the label
                if len(set(task_label)) <=1:
                    continue
            metrics.append( metric_evaluator(task_label, task_pred) if args.metric != "spearman" else metric_evaluator(task_label, task_pred)[0] )
        metric_value = np.mean(metrics)
        return_results["metric"] = metric_value

    evaluation_loss = np.sum([loss * n for loss, n in evaluation_losses]) / np.sum([n for _, n in evaluation_losses])
    return_results["loss"] = evaluation_loss

    return return_results

def train_with_one_seed(model_args, data_args, training_args, args, data_seed):

    # load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, args)

    # set up the accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    )

    # set up the dataset
    data_module = make_data_module(tokenizer=tokenizer, dataset=args.dataset,
                                   seed=data_seed, args=args)
    data_collator = data_module['data_collator']

    train_loader = DataLoader(data_module['train_dataset'], batch_size=training_args.per_device_train_batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(data_module['val_dataset'], batch_size=training_args.per_device_eval_batch_size, shuffle=False, collate_fn=data_collator)
    test_loader = DataLoader(data_module['test_dataset'], batch_size=training_args.per_device_eval_batch_size, shuffle=False, collate_fn=data_collator)
    print("Data loaders set up")

    # get the number of steps
    len_dataloader = len(train_loader)
    num_update_steps_per_epoch = len_dataloader / training_args.gradient_accumulation_steps
    num_update_steps_per_epoch = max(1, num_update_steps_per_epoch)
    max_steps = math.ceil(num_update_steps_per_epoch * training_args.num_train_epochs)
    num_train_epochs = math.ceil(training_args.num_train_epochs)

    # set up the loss functions
    loss_fcns = []
    if args.task_type == "regression":
        # we do not need to set up the loss functions for regression tasks, and we will use the default loss function defined by Hugging Face
        pass
    else:
        reduction = 'none' if args.has_nan_in_dataset else 'mean' #'sum'
        
        for i in range(args.num_tasks):
            assert len(data_module['loss_weights']) == args.num_tasks, "The number of labels does not match the number of loss weights."
            loss_weight = data_module['loss_weights'][i]

            if loss_weight == (1.0, 1.0):
                loss_fcn = torch.nn.BCEWithLogitsLoss(reduction=reduction)
            else:
                print(f"Loss weight: {loss_weight[0] / loss_weight[1]}")
                loss_fcn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(loss_weight[0] / loss_weight[1])]).to(accelerator.device), reduction=reduction)
            loss_fcns.append(loss_fcn)
    
    if args.task_type == "regression":
        assert len(loss_fcns) == 0, "Loss functions should not be set up for regression tasks."
    else:
        assert len(loss_fcns) == args.num_tasks, "The number of loss functions does not match the number of tasks."
    
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

    model, loss_fcns, optimizer, train_loader, val_loader, test_loader, scheduler = accelerator.prepare(
        model, loss_fcns, optimizer, train_loader, val_loader, test_loader, scheduler)
    
    # set the metric
    metric_map = {"regression": {"mae": (mean_absolute_error, float("inf")), "spearman": (spearmanr, -float("inf")), "rmse": (root_mean_squared_error, float("inf"))},
                  "classification": {"auroc": (roc_auc_score, -float("inf")), "auprc": (average_precision_score, -float("inf"))}}
    
    assert args.metric in metric_map[args.task_type], f"Metric {args.metric} not supported for task type {args.task_type}."
    metric_evaluator = metric_map[args.task_type][args.metric][0]

    # set the metric logging
    best_val_loss = float('inf')
    best_val_metric = metric_map[args.task_type][args.metric][1]

    test_metric_on_best_val_loss = metric_map[args.task_type][args.metric][1]
    test_metric_on_best_val_metric = metric_map[args.task_type][args.metric][1]
    best_val_loss_epoch = 0
    best_val_metric_epoch = 0

    # start training now
    step_count = 0
    for epoch in range(1, num_train_epochs+1):
        model.train()

        step_count = train_epoch(model, optimizer, scheduler, train_loader, loss_fcns, accelerator, args, training_args, epoch, step_count)

        # evaluate on validation set
        val_result = evaluate(model, val_loader, loss_fcns, accelerator, args, 'val', metric_evaluator, data_module['scaler'])
        val_loss = val_result["loss"]
        val_metric = val_result["metric"]

        # evaluate on test set
        test_result = evaluate(model, test_loader, loss_fcns, accelerator, args, 'test', metric_evaluator, data_module['scaler'])
        test_loss = test_result["loss"]
        test_metric = test_result["metric"]

        # check if the validation loss is the best
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_val_loss_epoch = epoch
            test_metric_on_best_val_loss = test_metric

        # check if the validation metric is the best
        if metric_map[args.task_type][args.metric][1] == float("inf") and val_metric <= best_val_metric or \
            metric_map[args.task_type][args.metric][1] == -float("inf") and val_metric >= best_val_metric:
                best_val_metric = val_metric
                test_metric_on_best_val_metric = test_metric
                best_val_metric_epoch = epoch

        # check if the test metric is the best
        if metric_map[args.task_type][args.metric][1] == float("inf") and test_metric <= best_test_metric or \
            metric_map[args.task_type][args.metric][1] == -float("inf") and test_metric >= best_test_metric:
                best_test_metric = test_metric
                best_test_metric_epoch = epoch
        
        # log the results
        print(f"Epoch {epoch} Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f},  " +
              f"Test Loss: {test_loss:.4f}, Test Metric: {test_metric:.4f}, " +
              f"Test Metric on Best Val Metric: {test_metric_on_best_val_metric:.4f}, Best Test Metric: {best_test_metric:.4f}")
    
        




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

    assert not (args.num_data_seeds>1 and args.num_run_seeds>1), "num_data_seeds and num_run_seeds cannot be set >1 at the same time."

    if args.do_train:
        if args.task_type == "regression":
            assert args.weight_loss == False, "Weight loss not supported for regression tasks."
        if args.num_data_seeds > 1:
            for data_seed in range(args.num_data_seeds):
                # the run seed is set to the argument seed: args.seed
                results_one_seed = train_with_one_seed(model_args, data_args, training_args, args, data_seed)
                assert 1==2
                #results.append(results_one_seed)
        else:
            print("run seed is ignored since num_data_seeds is set greater than 1")
            data_seed = 0
            for run_seed in range(args.num_run_seeds):
                args.seed = run_seed
                results_one_seed, wandb_step = train_with_one_seed(model_args, data_args, training_args, args, data_seed)
                #results.append(results_one_seed)
    else:
        if args.do_eval:
            pass


    

if __name__ == "__main__":
    main()