from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import transformers
import torch
import random
import numpy as np
from pysmilesutils.augment import SMILESAugmenter
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
import os
import pandas as pd
import json
import copy
import ast

@dataclass
class ModelArguments:
    tokenizer_path: Optional[str] = field(
        default="./tokenizer",
        metadata={"help": "The path to the tokenizer."}
    )
    pretrain_model: Optional[str] = field(
        default="ChemFM/ChemFM-1B",
        metadata={"help": "The model name."}
    )
    lora: Optional[bool] = field(
        default=True,
        metadata={"help": "Use LORA for training."}
    )
    lora_rank: Optional[int] = field(
        default=4,
        metadata={"help": "The number of LORA ranks."}
    )
    lora_alpha_ratio: Optional[float] = field(
        default=1.0,
        metadata={"help": "The alpha parameter for LORA."}
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
        default="MOSES",
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
    molecule_target_aug_prob: float = field(
        default=0.0,
        metadata={"help": "The probability of augmenting the molecule in the target."}
    )
    scaffold_aug_prob: float = field(
        default=0.0,
        metadata={"help": "The probability of augmenting the scaffold."}
    )
    has_scaffold: bool = field(
        default=False,
        metadata={"help": "Whether the dataset has scaffold."}
    )
    generation_config_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the generation config file. This is only used in the generation mode."}
    )
    generation_output_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the generation output file. This is only used in the generation mode."}
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
    gradient_accumulation_steps: int = field(default=1, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'})
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    min_learning_rate: float = field(default=None, metadata={"help": 'The minimum learning rate for the schedule'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=1.0, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=False, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    ddp_find_unused_parameters: bool = field(default=True, metadata={"help": 'Find unused parameters in DDP'})

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


def make_data_module(tokenizer: transformers.PreTrainedTokenizer, ignore_index, args) -> Dict:

    # Load dataset.
    dataset_path = os.path.join("data_temp", args.dataset)
    train_dataset = Dataset.from_pandas(pd.read_csv(os.path.join(dataset_path, "train_data.csv")))
    eval_dataset = Dataset.from_pandas(pd.read_csv(os.path.join(dataset_path, "val_data.csv")))

    # load the string template
    string_template = json.load(open(args.string_template_path, 'r'))
    molecule_start_str = string_template['MOLECULE_START_STRING']
    scaffold_start_str = string_template['SCAFFOLD_MOLECULE_START_STRING']
    property_start_str = string_template['PROPERTY_START_STRING']
    property_inner_sep = string_template['PROPERTY_INNER_SEP']
    property_inter_sep = string_template['PROPERTY_INTER_SEP']
    end_str = string_template['END_STRING']

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        molecule_target_aug_prob=args.molecule_target_aug_prob,
        scaffold_aug_prob=args.scaffold_aug_prob,
        molecule_start_str=molecule_start_str,
        scaffold_start_str=scaffold_start_str,
        property_start_str=property_start_str,
        property_inner_sep=property_inner_sep,
        property_inter_sep=property_inter_sep,
        end_str=end_str,
        sme=SMILESAugmenter(),
        ignore_index=ignore_index,
        has_scaffold=args.has_scaffold
    )

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

def make_test_data_module(tokenizer: transformers.PreTrainedTokenizer, ignore_index, args) -> Dict:
    generation_config = json.load(open(args.generation_config_path, 'r'))

    # load the string template
    string_template = json.load(open(args.string_template_path, 'r'))
    molecule_start_str = string_template['MOLECULE_START_STRING']
    scaffold_start_str = string_template['SCAFFOLD_MOLECULE_START_STRING']
    property_start_str = string_template['PROPERTY_START_STRING']
    property_inner_sep = string_template['PROPERTY_INNER_SEP']
    property_inter_sep = string_template['PROPERTY_INTER_SEP']
    end_str = string_template['END_STRING']

    # generate the dataset from the generation config
    if args.has_scaffold:
        assert "scaffold_list" in generation_config, "The scaffold list is not provided in the generation config"
        scaffold_list = generation_config["scaffold_list"]
    else:
        scaffold_list = [None]
    
    # we define a data frame with three columns: property_name, property_value, scaffold_smiles, temperature
    # we will use this data frame to generate the test dataset
    test_dataset = []
    
    for scaffold in scaffold_list: 
        for i, properties in enumerate(generation_config['properties']):
            properties_string = properties
            num_samples = generation_config['properties'][properties]['num_samples']
            property_means = generation_config['properties'][properties]['means']
            property_stds = generation_config['properties'][properties]['stds']
            sample_points = generation_config['properties'][properties]['sample_points']
            temperature = generation_config['properties'][properties]['temperature']
            properties = properties.split('+')
            for j, sample_point in enumerate(sample_points):
                # compute the normalized values
                # sample_point: [n]
                # property_means: [n]
                # property_stds: [n]
                non_normalized_sample_point = np.array(sample_point).reshape(-1)
                sample_point = (np.array(sample_point) - np.array(property_means)) / np.array(property_stds)
                sub_df = {
                    "property_name": properties,
                    "property_value": sample_point.tolist(),
                    "temperature": temperature,
                    "non_normalized_property_value": non_normalized_sample_point.tolist()
                }
                if scaffold is not None:
                    sub_df["scaffold_smiles"] = scaffold
                
                # we need to duplicate the sub_df num_samples times and append it to the test_dataset
                test_dataset.extend([sub_df] * num_samples)
    
    # construct the dataset as a pandas dataframe
    test_dataset = pd.DataFrame(test_dataset)
    test_dataset = Dataset.from_pandas(test_dataset)

    data_collator = DataCollatorForCausalLMEval(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        molecule_target_aug_prob=args.molecule_target_aug_prob,
        scaffold_aug_prob=args.scaffold_aug_prob,
        molecule_start_str=molecule_start_str,
        scaffold_start_str=scaffold_start_str,
        property_start_str=property_start_str,
        property_inner_sep=property_inner_sep,
        property_inter_sep=property_inter_sep,
        end_str=end_str,
        sme=SMILESAugmenter(),
        ignore_index=ignore_index,
        has_scaffold=args.has_scaffold
    )

    return dict(
        test_dataset=test_dataset,
        data_collator=data_collator
    )

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    molecule_target_aug_prob: float
    molecule_start_str: str
    scaffold_aug_prob: float
    scaffold_start_str: str
    property_start_str: str
    property_inner_sep: str
    property_inter_sep: str
    end_str: str
    sme: SMILESAugmenter
    ignore_index: int
    has_scaffold: bool

    def augment_molecule(self, molecule: str) -> str:
        """
        Augment the molecule.
        """
        return self.sme.augment([molecule])[0]
        #return self.sme.randomize_smiles(molecule)

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        original_property_list = ['qed', 'logp', 'sas', 'tpsa']
        prop_token_map = {
            'qed': '<qed>',
            'logp': '<logp>',
            'sas': '<SAS>',
            'tpsa': '<TPSA>'
        }

        sources = []
        targets = []
        props_list = []
        props_index_list = []
        for example in instances:
            smiles = example['smiles'].strip()
            if self.molecule_target_aug_prob > 0.0:
                smiles = self.augment_molecule(smiles)
            target = f"{self.molecule_start_str}{smiles}{self.end_str}"

            # randomly choose the property and the scaffold combinations:
            props_str = ""
            scaffold_str = ""
            props = []
            props_index = []

            if self.has_scaffold:
                scaffold = example['scaffold_smiles'].strip()
                if self.scaffold_aug_prob > 0.0:
                    scaffold = self.augment_molecule(scaffold)
                scaffold_str = f"{self.scaffold_start_str}{scaffold}{self.end_str}"

            # randomly sample 1, 2, 3, or 4 properties
            num_props = np.random.choice([1, 2, 3, 4], p=[0.1, 0.2, 0.3, 0.4])
            property_list = np.random.choice(original_property_list, num_props, replace=False).tolist()
            random.shuffle(property_list)

            props_str = f"{self.property_start_str}"
            for i, prop in enumerate(property_list):
                props_str += f"{prop_token_map[prop]}{self.property_inner_sep}{self.molecule_start_str}{self.property_inter_sep}"
                props.append(example[prop])
                props_index.append(3 + 4 * i) # this is hard coded for the current template
            props_str += f"{self.end_str}"
            
            source = props_str + scaffold_str + "<->>"

            sources.append(source)
            targets.append(target)
            props_list.append(props)
            props_index_list.append(props_index)
        
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
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=self.ignore_index)

        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
            'properties': props_list,
            'properties_index': props_index_list
        }
        
        if labels is not None:
            data_dict['labels'] = labels

        return data_dict

@dataclass
class DataCollatorForCausalLMEval(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    molecule_target_aug_prob: float
    molecule_start_str: str
    scaffold_aug_prob: float
    scaffold_start_str: str
    property_start_str: str
    property_inner_sep: str
    property_inter_sep: str
    end_str: str
    sme: SMILESAugmenter
    ignore_index: int
    has_scaffold: bool

    def augment_molecule(self, molecule: str) -> str:
        """
        Augment the molecule.
        """
        return self.sme.augment([molecule])[0]
        #return self.sme.randomize_smiles(molecule)

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        prop_token_map = {
            'qed': '<qed>',
            'logp': '<logp>',
            'sas': '<SAS>',
            'tpsa': '<TPSA>'
        }

        sources = []
        props_list = []
        non_normalized_props_list = []
        prop_names_list = []
        props_index_list = []
        temperature_list = []
        scaffold_list = []
        for example in instances:
            prop_names = example['property_name']
            prop_values = example['property_value']
            non_normalized_prop_values = example['non_normalized_property_value']
            temperature = example['temperature']
            # we need to convert the string to a list
            
            # randomly choose the property and the scaffold combinations:
            props_str = ""
            scaffold_str = ""
            props = []
            non_nornalized_props = []   
            props_index = []


            if self.has_scaffold:
                scaffold = example['scaffold_smiles'].strip()
                if self.scaffold_aug_prob > 0.0:
                    scaffold = self.augment_molecule(scaffold)
                scaffold_str = f"{self.scaffold_start_str}{scaffold}{self.end_str}"

            props_str = f"{self.property_start_str}"
            for i, prop in enumerate(prop_names):
                prop = prop.lower()
                props_str += f"{prop_token_map[prop]}{self.property_inner_sep}{self.molecule_start_str}{self.property_inter_sep}"
                props.append(prop_values[i])
                non_nornalized_props.append(non_normalized_prop_values[i])
                props_index.append(3 + 4 * i) # this is hard coded for the current template
            props_str += f"{self.end_str}"
            
            source = props_str + scaffold_str + "<->>" + self.molecule_start_str

            sources.append(source)
            props_list.append(props)
            non_normalized_props_list.append(non_nornalized_props)
            props_index_list.append(props_index)
            prop_names_list.append(prop_names)
            temperature_list.append(temperature)
            scaffold_list.append(scaffold) if self.has_scaffold else scaffold_list.append(None)
        
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )

        # Build the input and labels for causal LM
        input_ids = []
        for tokenized_source in tokenized_sources_with_prompt['input_ids']:
            input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
            'properties': props_list,
            'non_normalized_properties': non_normalized_props_list,
            'property_names': prop_names_list,
            'properties_index': props_index_list,
            'temperature': temperature_list,
            'scaffold': scaffold_list
        }

        return data_dict