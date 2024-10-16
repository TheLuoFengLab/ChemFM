from tdc.benchmark_group import admet_group
import transformers
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, set_seed, LlamaForSequenceClassification
from utils import smart_tokenizer_and_embedding_resize, Scaler, DataCollator
from peft import LoraConfig, PeftModel
import os
import pickle
import json
from datasets import Dataset
import json
import math
from pysmilesutils.augment import SMILESAugmenter
from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np
import random
from sklearn.metrics import mean_absolute_error, roc_auc_score, average_precision_score, root_mean_squared_error
from peft import get_peft_model
import argparse

if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser(description='Submit results to TDC benchmark')
parser.add_argument('--model_path', type=str, help='Path to model', required=True)
parser.add_argument('--dataset', type=str, help='Dataset name', required=True)
parser.add_argument('--task_type', type=str, help='Task type', choices=['regression', 'classification'], required=True)
parser.add_argument('--tokenizer_path', type=str, default="./tokenizer", help='Path to tokenizer')
parser.add_argument('--string_template_path', type=str, default="./string_template.json", help='Path to string template')
parser.add_argument('--source_max_len', type=int, default=512, help='Maximum length of the source sequence')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

args = parser.parse_args()

# assert there are five folders in the model_path since we have 5 seeds
subfolders = sorted([f.path for f in os.scandir(args.model_path) if f.is_dir()])
assert len(subfolders) == 5, "There should be 5 subfolders in the model_path"

# load any of the folders to get the config
model_path = subfolders[0]
config = LoraConfig.from_pretrained(model_path)
base_model_path = config.base_model_name_or_path

tokenizer_path = args.tokenizer_path
string_template_path = args.string_template_path
source_max_len = args.source_max_len
batch_size = args.batch_size
task = args.task_type
dataset = args.dataset

device_map = "auto"
DEFAULT_PAD_TOKEN = "[PAD]"

# load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path,
    padding_side="right",
    use_fast=True, # Fast tokenizer giving issues.
    trust_remote_code=True,
)
config = AutoConfig.from_pretrained(
    base_model_path,
    num_labels=1,
    finetuning_task="classification",
    trust_remote_code=True,
)
base_model = LlamaForSequenceClassification.from_pretrained(
    base_model_path,
    config=config,
    device_map=device_map,
    trust_remote_code=True,
)
special_tokens_dict = dict(pad_token=DEFAULT_PAD_TOKEN)
smart_tokenizer_and_embedding_resize(
    special_tokens_dict=special_tokens_dict,
    tokenizer=tokenizer,
    model=base_model
)
base_model.config.pad_token_id = tokenizer.pad_token_id

def load_adapter(base_model, adapter_path):
    lora_model = PeftModel.from_pretrained(base_model, adapter_path)
    return lora_model



group = admet_group(path = 'data_temp/')
predictions_list = []

for seed in [1, 2, 3, 4, 5]:
    benchmark = group.get(dataset)
    # all benchmark names in a benchmark group are stored in group.dataset_names
    predictions = {}
    name = benchmark['name']
    train_val, test = benchmark['train_val'], benchmark['test']

    # we don't need to train the model here, just predict the test set
    #train, valid = group.get_train_valid_split(benchmark = name, split_type = 'default', seed = seed)
    
    # --------------------------------------------- # 
    #  Train your model using train, valid, test    #
    #  Save test prediction in y_pred_test variable #
    # --------------------------------------------- #

    # load the adapter
    
    ## the seed-1 is because our training script saves the model in the seed-1 folder
    ## this is aligned with the seed in the benchmark
    adapter_path = subfolders[seed-1]
    print(adapter_path)
    lora_model = PeftModel.from_pretrained(base_model, adapter_path)
    lora_model.eval()

    # if the scaler is saved, load it
    scaler_path = os.path.join(adapter_path, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = pickle.load(open(scaler_path, "rb"))
        print("scaler loaded")

    # rename the Drug column to smiles
    test_dataset = test.drop(columns=["Drug_ID"])
    test_dataset = test_dataset.rename(columns={"Drug": "smiles"})
    test_dataset = Dataset.from_pandas(test_dataset)
    test_dataset = test_dataset.map(lambda x: {"is_aug": False})

    # load the string template
    ## TODO: the start token for the molecule could be not necessary
    string_template = json.load(open(string_template_path, 'r'))
    molecule_start_str = string_template['MOLECULE_START_STRING']
    end_str = string_template['END_STRING']

    data_collator = DataCollator(
        tokenizer=tokenizer,
        source_max_len=source_max_len,
        molecule_source_aug_prob=0.0,
        molecule_start_str=molecule_start_str,
        end_str=end_str,
        sme=SMILESAugmenter()
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    y_pred_test = []
    for i, batch in enumerate(test_loader):
        with torch.no_grad():
            batch = {k: v.to(lora_model.device) for k, v in batch.items()}
            outputs = lora_model(**batch)
        if task == 'regression':
            y_pred_test.append(outputs.logits.cpu().detach().numpy())
        else:
            y_pred_test.append((torch.sigmoid(outputs.logits) > 0.5).cpu().detach().numpy())
    
    y_pred_test = np.concatenate(y_pred_test, axis=0)
    if scaler and task == 'regression':
        y_pred_test = scaler.inverse_transform(y_pred_test)
    y_pred_test = y_pred_test.flatten()

    predictions[name] = y_pred_test
    predictions_list.append(predictions)

results = group.evaluate_many(predictions_list)
print(results)
