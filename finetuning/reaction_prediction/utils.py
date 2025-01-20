from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import transformers
import pandas as pd
import os
import json
from pysmilesutils.augment import SMILESAugmenter
import torch
from torch.nn.utils.rnn import pad_sequence
import copy
from datasets import Dataset

from rdkit import RDLogger, Chem
# Suppress RDKit INFO messages
RDLogger.DisableLog('rdApp.*')

@dataclass
class ModelArguments:
    tokenizer_path: Optional[str] = field(
        default="./tokenizer",
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
    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the model."}
    )

@dataclass
class DataArguments:
    dataset: Optional[str] = field(
        default="USPTO-MIT",
        metadata={"help": "The dataset name."}
    )
    string_template_path: Optional[str] = field(
        default="./string_template.json",
        metadata={"help": "The path to the string template."}
    )
    source_max_len: int = field(
        default=512,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=512,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    task_type: str = field(
        default="retrosynthesis",
        metadata={"help": "The task type: retrosynthesis or synthesis."}
    )
    beam_size: int = field(
        default=10,
        metadata={"help": "The beam size for evaluation."}
    )
    original_data_augmentations: int = field(
        default=20,
        metadata={"help": "The number of augmentations for the original data. This is used for evaluation."}
    )
    max_test_samples: int = field(
        default=-1,
        metadata={"help": "The maximum number of test samples to use. Set to -1 to use all samples."}
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
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'})
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    min_learning_rate: float = field(default=2e-5, metadata={"help": 'The minimum learning rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=1.0, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=False, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    early_stopping_patience: int = field(default=10, metadata={"help": 'The number of epochs to wait for improvement before stopping training'})
    early_stopping_start_epoch: int = field(default=20, metadata={"help": 'The epoch to start early stopping'})

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

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    reactant_start_str: str
    product_start_str: str
    end_str: str
    task_type: str
    sme: SMILESAugmenter
    ignore_index: int

    def augment_molecule(self, molecule: str) -> str:
        return self.sme.augment([molecule])[0]

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        sources = []
        targets = []
        target_smiles = []
        generation_prompts = []
        if self.task_type == 'retrosynthesis':
            src_start_str = self.product_start_str
            tgt_start_str = self.reactant_start_str
        else:
            src_start_str = self.reactant_start_str
            tgt_start_str = self.product_start_str
        
        for example in instances:
            src = example['src']
            tgt = example['tgt']

            source = f"{src_start_str}{src}{self.end_str}"
            target = f"{tgt_start_str}{tgt}{self.end_str}"
            generation_prompt = f"{src_start_str}{src}{self.end_str}{tgt_start_str}"

            sources.append(source)
            targets.append(target)
            target_smiles.append(tgt)
            generation_prompts.append(generation_prompt)
        
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )

        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            input_ids.append(torch.tensor(tokenized_source + tokenized_target))
            labels.append(
                torch.tensor([self.ignore_index for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
            )
        
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=self.ignore_index)

        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
            'labels': labels,
            'tgt_smiles': target_smiles,
            'generation_prompts': generation_prompts
        }
        
        return data_dict

@dataclass
class DataCollatorForCausalLMEval(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    reactant_start_str: str
    product_start_str: str
    end_str: str
    task_type: str
    sme: SMILESAugmenter

    def augment_molecule(self, molecule: str) -> str:
        return self.sme.augment([molecule])[0]

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        srcs = instances[0]['src']
        original_tgt = instances[0]['tgt']

        tgt = Chem.MolToSmiles(Chem.MolFromSmiles(original_tgt))

        if self.task_type == 'retrosynthesis':
            src_start_str = self.product_start_str
            tgt_start_str = self.reactant_start_str
        else:
            src_start_str = self.reactant_start_str
            tgt_start_str = self.product_start_str

        generation_prompts = []
        for src in srcs:
            generation_prompt = f"{src_start_str}{src}{self.end_str}{tgt_start_str}"
            generation_prompts.append(generation_prompt)

        data_dict = {
            'tgt_smiles': tgt,
            'generation_prompts': generation_prompts
        }

        return data_dict

def make_data_module(tokenizer: transformers.PreTrainedTokenizer, dataset, ignore_index, args, evaluation=False) -> Dict:

    data_dir = f"./data_temp/{dataset}"

    # we should group the data by the original augmentation
    # e.g., if the original data has 20 augmentations, we should group the data every 20 rows
    def group_rows(df, n):
        grouped_data = {
            'src': [],
            'tgt': []
        }
    
        for i in range(0, len(df), n):
            chunk = df.iloc[i:i+n]
            grouped_data['src'].append(chunk['src'].tolist())
            grouped_data['tgt'].append(chunk['tgt'].iloc[0])
    
        return pd.DataFrame(grouped_data)

    if evaluation:
        test_dataset = pd.read_csv(os.path.join(data_dir, "test.csv"))
        grouped_test_dataset = group_rows(test_dataset, args.original_data_augmentations)
        test_dataset = Dataset.from_pandas(grouped_test_dataset)

    else:
        train_dataset = pd.read_csv(os.path.join(data_dir, "train.csv"))
        val_dataset = pd.read_csv(os.path.join(data_dir, "val_single.csv"))
        test_dataset = pd.read_csv(os.path.join(data_dir, "test_single.csv"))

        train_dataset = Dataset.from_pandas(train_dataset)
        val_dataset = Dataset.from_pandas(val_dataset)

        test_dataset = Dataset.from_pandas(test_dataset)

    if args.max_test_samples > 0:
        # randomize the val and test datasets
        test_dataset = test_dataset.shuffle(seed=args.seed)
        test_dataset = test_dataset.select(range(args.max_test_samples))

        if not evaluation:
            val_dataset = val_dataset.shuffle(seed=args.seed)
            val_dataset = val_dataset.select(range(args.max_test_samples))

    # load the string template
    ## TODO: the start token for the molecule could be not necessary
    ## TODO: this could be also in the tokenizer, now this is not elegant
    string_template = json.load(open(args.string_template_path, 'r'))
    reactant_start_str = string_template['REACTANTS_START_STRING']
    product_start_str = string_template['PRODUCTS_START_STRING']
    end_str = string_template['END_STRING']

    
    if evaluation:
        data_collator = DataCollatorForCausalLMEval(
            tokenizer=tokenizer,
            source_max_len=args.source_max_len,
            target_max_len=args.target_max_len,
            reactant_start_str=reactant_start_str,
            product_start_str=product_start_str,
            end_str=end_str,
            task_type=args.task_type,
            sme=SMILESAugmenter(),
        )

        return dict(
            test_dataset=test_dataset,
            data_collator=data_collator
        )
    else:
        data_collator = DataCollatorForCausalLM(
            tokenizer=tokenizer,
            source_max_len=args.source_max_len,
            target_max_len=args.target_max_len,
            reactant_start_str=reactant_start_str,
            product_start_str=product_start_str,
            end_str=end_str,
            task_type=args.task_type,
            sme=SMILESAugmenter(),
            ignore_index=ignore_index
        )
        return dict(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            data_collator=data_collator
        )