import argparse
from huggingface_hub import HfApi
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, set_seed, LlamaForSequenceClassification
from peft import LoraConfig, PeftModel
from utils import smart_tokenizer_and_embedding_resize, Scaler, DataCollator
from huggingface_hub.repocard import RepoCard
import os
import json

parser = argparse.ArgumentParser(description='Submit model to Hugging Face Model Hub')
parser.add_argument('--model_path', type=str, help='Path to model')
parser.add_argument('--model_id', type=str, help='Model ID')
parser.add_argument('--tokenizer_path', type=str, default="./tokenizer", help='Path to tokenizer')
parser.add_argument('--string_template_path', type=str, default="./string_template.json", help='Path to string template')
parser.add_argument('--dataset_group', type=str, help='Dataset group', required=True)
parser.add_argument('--dataset', type=str, help='Dataset name', required=True)
args = parser.parse_args()

# Create the markdown template with placeholders
admet_markdown_template = """
### Model Card: ChemFM Adapters for {dataset} Prediction on ADMET Group

**Model Overview**:  
This adapter is fine-tuned on the `{dataset}` dataset. It uses ChemFM as the base model and is trained on SMILES representations.

**Task Description**:  
The task a {task_type} task, and the objective is to {task_description}.

**Dataset**:  
The whole dataset contains {num_samples} samples. For more details about the dataset, visit [ADMET benchmark](https://tdcommons.ai/benchmark/admet_group/{dataset}).

**How to Use**:  
For more details on how to use this model, visit [ChemFM GitHub](https://github.com/TheLuoFengLab/ChemFM/tree/master/finetuning/property_prediction).
"""

if args.dataset_group == "ADMET":
    property = args.dataset
    dataset = args.dataset
    # read the dataset description from the json
    with open("./dataset_descriptions/admet.json") as f:
        admet_datasets_description = json.load(f)
    dataset_description = admet_datasets_description[args.dataset]
    markdown = admet_markdown_template.format(property=property,
                                              dataset=dataset,
                                              task_type=dataset_description["task_type"],
                                              task_description=dataset_description["description"],
                                              num_samples=dataset_description["num_molecules"])
    card = RepoCard(markdown)
    card.save(os.path.join(args.model_path, "README.md"))
else:
    raise ValueError("Invalid dataset group")


# load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_path,
    padding_side="right",
    use_fast=True, # Fast tokenizer giving issues.
    trust_remote_code=True,
)
config = LoraConfig.from_pretrained(args.model_path)
base_model_path = config.base_model_name_or_path
#print(base_model_path)

DEFAULT_PAD_TOKEN = "[PAD]"

config = AutoConfig.from_pretrained(
    base_model_path,
    num_labels=1,
    finetuning_task="classification",
    trust_remote_code=True,
)
base_model = LlamaForSequenceClassification.from_pretrained(
    base_model_path,
    config=config
)

special_tokens_dict = dict(pad_token=DEFAULT_PAD_TOKEN)
smart_tokenizer_and_embedding_resize(
    special_tokens_dict=special_tokens_dict,
    tokenizer=tokenizer,
    model=base_model
)
base_model.config.pad_token_id = tokenizer.pad_token_id

lora_model = PeftModel.from_pretrained(base_model, args.model_path)

# upload model and tokenizer to Hugging Face Model Hub
lora_model.push_to_hub(args.model_id, private=True)
tokenizer.push_to_hub(args.model_id, private=True)
card.push_to_hub(args.model_id)



## we should also upload the scaler to the model hub if it exists
api = HfApi()
scaler_path = os.path.join(args.model_path, "scaler.pkl")
if os.path.exists(scaler_path):
    api.upload_file(
        path_or_fileobj=scaler_path,
        path_in_repo = "scaler.pkl",
        repo_id = args.model_id,
    )

# upload the string template if it exists
api.upload_file(
    path_or_fileobj=args.string_template_path,
    path_in_repo = "string_template.json",
    repo_id = args.model_id,
)
