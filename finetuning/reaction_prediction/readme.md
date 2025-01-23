<a id="readme-top"></a>

# Reaction Prediction

This section of the repository provides all necessary scripts and details for fine-tuning **ChemFM** on **reaction prediction** tasks.

## Overview

Reaction prediction is a fundamental task in computational chemistry, focused on predicting chemical transformations. It includes:

- **Synthesis prediction**: Predicting the reaction product given the reactants (which may include reagents).
- **Retrosynthesis prediction**: Predicting the reactants given a target product.

In our paper, we fine-tune ChemFM using the [Root-aligned SMILES](https://pubs.rsc.org/en/content/articlelanding/2022/sc/d2sc02763a) technique. However, ChemFM can seamlessly integrate with SMILES sequence editing-based methods for reaction prediction by providing a superior pre-trained model.

We provide **comparisons on benchmark datasets**, including the **USPTO benchmark datasets** (USPTO-Full, USPTO-MIT, and USPTO-50K). 
Additionally, we include details to **replicate the results** reported in our paper, along with **model checkpoints and configurations** for each dataset.

Our repository supports both synthesis and retrosynthesis prediction and follows standard evaluation practices:

- **Synthesis prediction**
  - USPTO-MIT
- **Retrosynthesis prediction**
  - USPTO-Full
  - USPTO-MIT
  - USPTO-50K


<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Fine-tuning ChemFM

Fine-tuning reaction prediction tasks can be time-consuming.  
If you prefer to evaluate our model directly, you can download the trained [model checkpoints](https://clemson.box.com/s/mw5rl7qsis7l87viq1nyyqz7o82lwnh2) for each dataset.

### 1. Prepare the Dataset

We use the processed data from the official [Root-aligned SMILES](https://github.com/otori-bird/retrosynthesis) repository. The detailed preprocessing steps (converting to Root-aligned SMILES) can be found in their repository.

Alternatively, you can download the [processed data](https://clemson.box.com/s/kct8hy0pc0i7iyjlpmrxng8cyoj12i9v) for each dataset without accessing the original repository.

### 2. Configure Training Parameters

You can configure the training parameters in two ways:

- **Pass arguments directly to the Python script**: Supply command-line arguments when running the training script.
- **Use a YAML configuration file**: Define all settings in a `.yml` file and pass the file path as an argument.

Predefined configuration files for all experiments are available in [`configs/`](./configs/).

### 3. Fine-tuning Script

To fine-tune ChemFM, run:

```bash
python -m accelerate.commands.launch --config_file accelerate_config.yaml main.py --training_args_file <config_yml_file>
```

Our code is built on the [Accelerate](https://huggingface.co/docs/accelerate/main/en/index) package.  The [`accelerate_config.yaml`](./accelerate_config.yaml) file configures the distribution settings for multi-GPU training.

By default, we use **8× H100 GPUs** on a single node for training. If using a different setup, ensure the distribution settings are adjusted accordingly.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Evaluating Fine-tuned ChemFM Models

### 1. Configure Generation Parameters

The generation parameters define:
- The model to be evaluated.
- Data augmentation settings (used in reaction prediction to improve robustness, following the Root-aligned SMILES setup).

You can configure the generation parameters in two ways:

- **Pass arguments directly to the Python script**.
- **Use a YAML configuration file** (recommended, available in [`configs/`](./configs/)).

### 2. Generate Predictions

Run the following command to generate predictions:

```bash
python -m accelerate.commands.launch --config_file accelerate_config.yaml evaluate.py --training_args_file <config_yml_file>
```

Predictions will be stored in the model directory specified in the configuration. 

By default, we use **8× H100 GPUs** on a single node for inference. Adjust distribution settings if using a different setup.

### 3. Score the Predictions

To evaluate model accuracy, run:

```bash
python ./score.py -data_path <prediction_file> -augmentation <num_augmentation>
```
**We also provide the predictions for each model in the [checkpoint folder](https://clemson.box.com/s/mw5rl7qsis7l87viq1nyyqz7o82lwnh2),** and you can directly use it the check the results.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---




## Benchmark Results

> [!IMPORTANT]
> However, we exclude methods that are not open-sourced or cannot be replicated based on our extensive replication.


| Task Category     | Dataset     | Model                                 | Top-1  | Top-3  | Top-5  |
|------------------|------------|--------------------------------------|--------|--------|--------|
| **Synthesis**    | USPTO-MIT   | Prev. best: AT                  | 90.4   | -      | 96.5   |
|                  |            | Prev. second-best: R-SMILES       | 90.0   | 95.6   | 96.4   |
|                  |            | ChemFM                           | **90.5** | **95.7** | **96.6** |
| **Retro-synthesis** | USPTO-50K | Prev. best: R-SMILES            | 56.0   | 79.0   | 86.1   |
|                  |            | Prev. second-best: Graph2Edits   | 55.1   | 77.3   | 83.4   |
|                  |            | ChemFM                               | 58.0   | **80.0** | **86.3** |
|                  |            | ChemFM<sup>*</sup>                             | **59.7** | 79.2   | 84.2   |
|                  | USPTO-MIT   | Prev. best: R-SMILES            | 60.3   | 77.9   | 82.8   |
|                  |            | Prev. second-best: RetroTRAE     | 60.3   | 77.9   | 82.8   |
|                  |            | ChemFM                               | 61.6   | **78.7** | **83.0** |
|                  |            | ChemFM<sup>*</sup>                            | **62.4** | 78.5   | 82.5   |
|                  | USPTO-Full  | Prev. best: RetroXpert          | 49.4   | 63.6   | 67.6   |
|                  |            | Prev. second-best: R-SMILES      | 48.9   | 66.6   | 72.0   |
|                  |            | ChemFM                           | **51.7** | **68.0** | **72.5** |

ChemFM<sup>*</sup> indicates the model that was trained with more steps, which generally results in better Top-1 accuracy, but Top-3 and Top-5 scores may decrease.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



