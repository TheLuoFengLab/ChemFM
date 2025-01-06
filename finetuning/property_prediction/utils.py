from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import transformers
import os
import numpy as np
from sklearn import preprocessing
from datasets import Dataset
import json
import math
from pysmilesutils.augment import SMILESAugmenter
import pandas as pd
import random
from torch.nn.utils.rnn import pad_sequence
import torch

from rdkit import RDLogger, Chem
# Suppress RDKit INFO messages
RDLogger.DisableLog('rdApp.*')

@dataclass
class ModelArguments:
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the tokenizer."}
    )
    pretrain_model: Optional[str] = field(
        default="feiyang-cai/ChemFM-1b",
        metadata={"help": "The model name."}
    )
    attention_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "The dropout rate for attention."}
    )
    lora: Optional[bool] = field(
        default=True,
        metadata={"help": "Use LORA for training."}
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "The number of LORA ranks."}
    )
    lora_alpha_ratio: Optional[float] = field(
        default=1,
        metadata={"help": "The alpha ratio for LORA."}
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "The dropout rate for LORA."}
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    lora_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the LORA model."}
    )

@dataclass
class DataArguments:
    dataset_group: Optional[str] = field(
        default="MoleculeNet",
        metadata={"help": "The dataset group."}
    )
    custom_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the custom dataset."}
    )
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the supervised fine-tuning dataset to use."}
    )
    combine_train_val_test: Optional[bool] = field(
        default=False,
        metadata={"help": "Combine the training and validation sets."}
    )
    num_data_seeds: Optional[int] = field(
        default=3,
        metadata={"help": "The number of data seeds to use."}
    )
    num_run_seeds: Optional[int] = field(
        default=1,
        metadata={"help": "The number of run seeds to use."}
    )
    task_type: Optional[str] = field(
        default="regression",
        metadata={"help": "The type of the task."}
    )
    num_tasks: Optional[int] = field(
        default=1,
        metadata={"help": "The number of tasks."}
    )
    scaler: Optional[bool] = field(
        default=False,
        metadata={"help": "Scale the labels."}
    )
    log_scaler: Optional[bool] = field(
        default=False,
        metadata={"help": "Log the scaler values."}
    )
    weight_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "Weight the loss."}
    )
    metric: Optional[str] = field(
        default="mae",
        metadata={"help": "The metric to use."}
    )
    string_template_path: Optional[str] = field(
        default="./string_template.json",
        metadata={"help": "The path to the string template."}
    )
    source_max_len: int = field(
        default=512,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    molecule_source_aug_prob: float = field(
        default=0.0,
        metadata={"help": "The probability of augmenting the molecule in the source."}
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    training_args_file: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the training arguments file."}
    )
    output_dir: str = field(
        default=None,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    adapter_name: Optional[str] = field(
        default=None,
        metadata={"help": "The adapter name."}
    )
    optim: str = field(default='adamw_torch', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=16, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    per_device_eval_batch_size: int = field(default=32, metadata={"help": 'The evaluation batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=0, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'})
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    min_learning_rate: float = field(default=2e-5, metadata={"help": 'The minimum learning rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=1.0, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=False, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    early_stopping_patience: int = field(default=10, metadata={"help": 'The number of epochs to wait for improvement before stopping training'})
    early_stopping_start_epoch: int = field(default=20, metadata={"help": 'The epoch to start early stopping'})
    #wandb_run_name: str = field(default=None, metadata={"help": 'The wandb run name'})

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    non_special_tokens = None,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict) + tokenizer.add_tokens(non_special_tokens)
    num_old_tokens = model.get_input_embeddings().weight.shape[0]
    num_new_tokens = len(tokenizer) - num_old_tokens
    if num_new_tokens == 0:
        return
    
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
    print(f"Resized tokenizer and embedding from {num_old_tokens} to {len(tokenizer)} tokens.")

class Scaler:
    def __init__(self, log=False):
        self.log = log
        self.offset = None
        self.scaler = None

    def fit(self, y):
        # make the values non-negative
        self.offset = np.min([np.min(y), 0.0])
        y = y.reshape(-1, 1) - self.offset

        # scale the input data
        if self.log:
            y = np.log10(y + 1.0)

        self.scaler = preprocessing.StandardScaler().fit(y)

    def transform(self, y):
        y = y.reshape(-1, 1) - self.offset

        # scale the input data
        if self.log:
            y = np.log10(y + 1.0)

        y_scale = self.scaler.transform(y)

        return y_scale

    def inverse_transform(self, y_scale):
        y = self.scaler.inverse_transform(y_scale.reshape(-1, 1))

        if self.log:
            y = 10.0**y - 1.0

        y = y + self.offset

        return y

def make_data_module(tokenizer: transformers.PreTrainedTokenizer, dataset, seed, args) -> Dict:
    
    if args.dataset_group == "MoleculeNet":
        from chembench import load_data
        df, induces = load_data(dataset)

        train_idx, valid_idx, test_idx = induces[seed]
        train_dataset = df.iloc[train_idx]
        val_dataset = df.iloc[valid_idx]
        test_dataset = df.iloc[test_idx]

        # drop the index
        train_dataset = train_dataset.reset_index(drop=True)
        val_dataset = val_dataset.reset_index(drop=True)
        test_dataset = test_dataset.reset_index(drop=True)

        # if the index col is present, drop it
        if "index" in train_dataset.columns:
            train_dataset = train_dataset.drop(columns=["index"])
            val_dataset = val_dataset.drop(columns=["index"])
            test_dataset = test_dataset.drop(columns=["index"])

    elif args.dataset_group == "ADMET":
        from tdc.benchmark_group import admet_group
        group = admet_group(path = "./data_temp")
        benchmark = group.get(dataset)
        name = benchmark['name']
        train_val, test_dataset = benchmark['train_val'], benchmark['test']
        # seed should be add 1. This is from the default setting in Admet dataset
        train_dataset, val_dataset = group.get_train_valid_split(benchmark = name, split_type = 'default', seed = seed+1)

        # drop the "Drug_ID"
        if args.combine_train_val_test:
            train_dataset = pd.concat([train_val, test_dataset], ignore_index=True)
            train_dataset = train_dataset.drop(columns=["Drug_ID"])
        else:
            train_dataset = train_dataset.drop(columns=["Drug_ID"])
        val_dataset = val_dataset.drop(columns=["Drug_ID"])
        test_dataset = test_dataset.drop(columns=["Drug_ID"])

        # rename the Drug column to smiles
        train_dataset = train_dataset.rename(columns={"Drug": "smiles"})
        val_dataset = val_dataset.rename(columns={"Drug": "smiles"})
        test_dataset = test_dataset.rename(columns={"Drug": "smiles"})

    elif args.dataset_group == "AttentiveFP":
        data_path = os.path.join("./data_temp/attentive_fp", dataset)
        train_dataset = pd.read_csv(os.path.join(data_path, str(seed), "train.csv"))
        val_dataset = pd.read_csv(os.path.join(data_path, str(seed), "val.csv"))
        test_dataset = pd.read_csv(os.path.join(data_path, str(seed), "test.csv"))

    elif args.dataset_group == "MoleBert":
        data_path = os.path.join("./data_temp/molebert", dataset)
        train_dataset = pd.read_csv(os.path.join(data_path, "train.csv"))
        val_dataset = pd.read_csv(os.path.join(data_path, "val.csv"))
        test_dataset = pd.read_csv(os.path.join(data_path, "test.csv"))

    elif args.dataset_group == "CustomDataset":
        assert args.custom_dataset_path is not None, "Custom dataset path is not provided."
        data_path = os.path.join(args.custom_dataset_path, f"seed_{seed}")
        train_dataset = pd.read_csv(os.path.join(data_path, "train.csv"))
        val_dataset = pd.read_csv(os.path.join(data_path, "val.csv"))
        test_dataset = pd.read_csv(os.path.join(data_path, "test.csv"))
    else:
        raise ValueError(f"Dataset group {args.dataset_group} not supported.")
    
    args.has_nan_in_dataset = False

    if args.num_tasks == 1:
        # we can directly drop the rows with NaN values
        train_dataset = train_dataset.dropna(subset=[train_dataset.columns[1]])
        val_dataset = val_dataset.dropna(subset=[val_dataset.columns[1]])
        test_dataset = test_dataset.dropna(subset=[test_dataset.columns[1]])
    
    else:
        # we need to check if there are NaN values in the dataset
        for col in train_dataset.columns:
            if col == "smiles":
                continue
            if train_dataset[col].isnull().values.any():
                args.has_nan_in_dataset = True
                print(f"warning: NaN values in the dataset.")
                break

    # standarlize the labels using the mean and std of the training dataset
    if args.task_type == "regression" and args.scaler:
        assert args.num_tasks == 1, "Scaler only supported for single label regression tasks."

        # we assume the first column is the smiles column, and the second column is the label column
        scaler = Scaler(log=args.log_scaler)
        label_col_name = train_dataset.columns[1]
        scaler.fit(train_dataset[label_col_name].values)

        train_dataset[label_col_name] = scaler.transform(train_dataset[label_col_name].values)
        val_dataset[label_col_name] = scaler.transform(val_dataset[label_col_name].values)
        test_dataset[label_col_name] = scaler.transform(test_dataset[label_col_name].values)

    else:
        scaler = None

    # get the weights for the tasks
    loss_weights = []
    if args.weight_loss:
        assert args.task_type == "classification", "Weight loss only supported for classification tasks."
        # get the columns 
        for col in train_dataset.columns:
            if col == "smiles":
                continue
            # check if the column has only 0, 1, and NaN values
            if set(train_dataset[col].unique()) != set([0, 1]):
                for i, val in enumerate(train_dataset[col].unique()):
                    if val not in [0, 1]:
                        assert math.isnan(val), f"Column {col} has values other than 0, 1, and NaN."
            

            num_negatives = train_dataset[train_dataset[col] == 0].shape[0]
            num_positives = train_dataset[train_dataset[col] == 1].shape[0]
            loss_weights.append((num_negatives / (num_negatives + num_positives), 
                                 num_positives / (num_negatives + num_positives)))
        assert len(loss_weights) == args.num_tasks, "The number of tasks does not match the number of columns."
    else:
        loss_weights = [(1.0, 1.0) for _ in range(args.num_tasks)]

    # Load dataset.
    train_dataset = Dataset.from_pandas(train_dataset)
    val_dataset = Dataset.from_pandas(val_dataset)
    test_dataset = Dataset.from_pandas(test_dataset)

    # add the is_aug column, we will have the augmented and non-augmented validation sets
    train_dataset = train_dataset.map(lambda x: {"is_aug": True})
    val_dataset = val_dataset.map(lambda x: {"is_aug": False})
    aug_val_dataset = val_dataset.map(lambda x: {"is_aug": True})
    test_dataset = test_dataset.map(lambda x: {"is_aug": False})

    # load the string template
    ## TODO: the start token for the molecule could be not necessary
    ## TODO: this could be also in the tokenizer, now this is not elegant
    string_template = json.load(open(args.string_template_path, 'r'))
    molecule_start_str = string_template['MOLECULE_START_STRING']
    end_str = string_template['END_STRING']

    data_collator = DataCollator(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        molecule_source_aug_prob=args.molecule_source_aug_prob,
        molecule_start_str=molecule_start_str,
        end_str=end_str,
        sme=SMILESAugmenter()
    )

    return dict(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        aug_val_dataset=aug_val_dataset,
        test_dataset=test_dataset,
        data_collator=data_collator,
        scaler=scaler,
        loss_weights=loss_weights,
    )

def make_evaluation_data_module(tokenizer: transformers.PreTrainedTokenizer, dataset, seed, args) -> Dict:
    
    if args.dataset_group == "MoleculeNet":
        from chembench import load_data
        df, induces = load_data(dataset)

        train_idx, valid_idx, test_idx = induces[seed]
        test_dataset = df.iloc[test_idx]

        # drop the index
        test_dataset = test_dataset.reset_index(drop=True)

        # if the index col is present, drop it
        if "index" in train_dataset.columns:
            test_dataset = test_dataset.drop(columns=["index"])

    elif args.dataset_group == "ADMET":
        from tdc.benchmark_group import admet_group
        group = admet_group(path = "./data_temp")
        benchmark = group.get(dataset)
        name = benchmark['name']
        train_val, test_dataset = benchmark['train_val'], benchmark['test']
        # seed should be add 1. This is from the default setting in Admet dataset
        train_dataset, val_dataset = group.get_train_valid_split(benchmark = name, split_type = 'default', seed = seed+1)

        # drop the "Drug_ID"
        test_dataset = test_dataset.drop(columns=["Drug_ID"])

        # rename the Drug column to smiles
        test_dataset = test_dataset.rename(columns={"Drug": "smiles"})

    elif args.dataset_group == "AttentiveFP":
        data_path = os.path.join("./data_temp/attentive_fp", dataset)
        test_dataset = pd.read_csv(os.path.join(data_path, str(seed), "test.csv"))

    elif args.dataset_group == "MoleBert":
        data_path = os.path.join("./data_temp/molebert", dataset)
        test_dataset = pd.read_csv(os.path.join(data_path, "test.csv"))

    else:
        raise ValueError(f"Dataset group {args.dataset_group} not supported.")

    # Load dataset.
    test_dataset = Dataset.from_pandas(test_dataset)

    test_dataset = test_dataset.map(lambda x: {"is_aug": False})

    # load the string template
    ## TODO: the start token for the molecule could be not necessary
    ## TODO: this could be also in the tokenizer, now this is not elegant
    string_template = json.load(open(args.string_template_path, 'r'))
    molecule_start_str = string_template['MOLECULE_START_STRING']
    end_str = string_template['END_STRING']

    data_collator = DataCollator(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        molecule_source_aug_prob=0.0,
        molecule_start_str=molecule_start_str,
        end_str=end_str,
        sme=SMILESAugmenter()
    )

    return dict(
        test_dataset=test_dataset,
        data_collator=data_collator,
    )

@dataclass
class DataCollator(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    molecule_source_aug_prob: float
    molecule_start_str: str
    end_str: str
    sme: SMILESAugmenter

    def augment_molecule(self, molecule: str) -> str:
        return self.sme.augment([molecule])[0]

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        sources = []
        targets = []
        is_aug = instances[0]['is_aug']
        
        for example in instances:
            smiles = example['smiles'].strip()
            if self.molecule_source_aug_prob > 0.0 and is_aug:
                if random.random() < self.molecule_source_aug_prob:
                    try:
                        smiles = self.augment_molecule(smiles)
                    except:
                        smiles = smiles
            else:
                try:
                    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
                except:
                    # we found in some datasets the smiles are not valid
                    # so we directly use the smiles 
                    #raise ValueError(f"Invalid SMILES: {smiles}")
                    smiles = smiles

            # get the properties except the smiles and mol_id cols
            props = [example[col] if example[col] is not None else np.nan for col in example.keys() if col not in ['smiles', 'is_aug']]
            #props = [example[col] if example[col] is not None else np.nan for col in sorted(example.keys()) if col not in ['smiles', 'is_aug']]
            source = f"{self.molecule_start_str}{smiles}{self.end_str}"
            sources.append(source)
            targets.append(props)
        
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        input_ids = [torch.tensor(tokenized_source) for tokenized_source in tokenized_sources_with_prompt['input_ids']]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        targets = torch.tensor(targets, dtype=torch.float32)

        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
            'labels': targets,
        }
        
        return data_dict