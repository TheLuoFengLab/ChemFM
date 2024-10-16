import argparse
from peft import PeftModel
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from utils import smart_tokenizer_and_embedding_resize, make_evaluation_data_module
import torch
import pickle
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_absolute_error, roc_auc_score, average_precision_score, root_mean_squared_error
from scipy.stats import spearmanr
import os

if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

DEFAULT_PAD_TOKEN = "[PAD]"
device_map = "auto"

def main():
    parser = argparse.ArgumentParser(description='Evaluate model using Hugging Face Adapter')
    parser.add_argument('--base_model_id', type=str, help='base model path or id', required=True)
    parser.add_argument('--adapter_id', type=str, help='adapter path or id', required=True)
    parser.add_argument('--remote_adapter', action='store_true', help='whether the adapter is remote')
    parser.add_argument('--dataset_group', type=str, help='dataset group', required=True)
    parser.add_argument('--dataset', type=str, help='dataset name', required=True)
    parser.add_argument('--metric', type=str, help='metric to evaluate', required=True)
    parser.add_argument('--data_seed', type=int, default=0, help='data seed')
    parser.add_argument('--task_type', type=str, help='task type', choices=['regression', 'classification'], required=True)
    parser.add_argument('--num_tasks', type=int, default=1, help='number of tasks')
    parser.add_argument('--source_max_len', type=int, default=512, help='Maximum length of the source sequence')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    args = parser.parse_args()

    # load the base model
    config = AutoConfig.from_pretrained(
        args.base_model_id,
        num_labels=args.num_tasks,
        finetuning_task="classification", # this is not about our task type
        trust_remote_code=True,
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_id,
        config=config,
        device_map=device_map,
        trust_remote_code=True,
    )
    
    # load the tokenizer from the adapter since the adapter has the special tokens
    assert args.remote_adapter or os.path.exists("./tokenizer"), "Local tokenizer not found."
    tokenizer_path = args.adapter_id if args.remote_adapter else "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        #args.adapter_id,
        tokenizer_path,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
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
    lora_model = PeftModel.from_pretrained(base_model, args.adapter_id)
    lora_model.eval()

    if args.remote_adapter:
        # we should download the scaler and string_template if it exists
        ## TODO: we should check if this works when the scaler is not available
        scaler_path = hf_hub_download(args.adapter_id, "scaler.pkl")
        if scaler_path:
            scaler = pickle.load(open(scaler_path, "rb"))
    
        args.string_template_path = hf_hub_download(args.adapter_id, "string_template.json")
    else:
        # otherwise, we should load the scaler and string_template from the local path
        scaler_path = os.path.join(args.adapter_id, "scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = pickle.load(open(scaler_path, "rb"))
        
        args.string_template_path = "string_template.json" # this is the default path for local adapter
        assert os.path.exists(args.string_template_path), f"String template file not found at {args.string_template_path}"
    
    # create the datasets
    data_module = make_evaluation_data_module(tokenizer=tokenizer, dataset=args.dataset,
                                   seed=args.data_seed, args=args)
    # create the data loaders
    test_loader = DataLoader(data_module['test_dataset'], 
                             batch_size=args.batch_size, 
                             shuffle=False, 
                             collate_fn=data_module['data_collator'])

    y_pred_test = []
    labels_test = []
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating"):
        labels_test.append(batch.pop("labels").cpu().detach().numpy())
        with torch.no_grad():
            batch = {k: v.to(lora_model.device) for k, v in batch.items()}
            outputs = lora_model(**batch)
        if args.task_type == 'regression':
            y_pred_test.append(outputs.logits.cpu().detach().numpy())
        else:
            y_pred_test.append((torch.sigmoid(outputs.logits) > 0.5).cpu().detach().numpy())

    y_pred_test = np.concatenate(y_pred_test, axis=0)
    labels_test = np.concatenate(labels_test, axis=0)
    if scaler and args.task_type == 'regression':
        y_pred_test = scaler.inverse_transform(y_pred_test)
        # for the labels, we don't need to inverse transform since we didn't scale them

    y_pred_test = y_pred_test.flatten()
    labels_test = labels_test.flatten()

    # evaluate the predictions
    metric_map = {"regression": {"mae": (mean_absolute_error, float("inf")), "spearman": (spearmanr, -float("inf")), "rmse": (root_mean_squared_error, float("inf"))},
                  "classification": {"auroc": (roc_auc_score, -float("inf")), "auprc": (average_precision_score, -float("inf"))}}
    
    assert args.metric in metric_map[args.task_type], f"Metric {args.metric} not supported for task type {args.task_type}."
    metric_evaluator = metric_map[args.task_type][args.metric][0]

    results = metric_evaluator(labels_test, y_pred_test) if args.metric != "spearman" else \
        metric_evaluator(labels_test, y_pred_test)[0]
    
    print(f"{args.metric}: {results}")

if __name__ == "__main__":
    main()