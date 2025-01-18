<a id="readme-top"></a>

Warning: still under construction, the checkpoint links are not correct!!!!

# Reaction Prediction

This section of the repository provides all necessary scripts and details for fine-tuning **ChemFM** on **reaction prediction** tasks.

## Overview

Reaction prediction is a fundamental task in computational chemistry, aimed at predicting chemical transformations. It includes:

- **Synthesis prediction**: Predicting the reaction product given the reactants (which may include reagents).
- **Retrosynthesis prediction**: Predicting the reactants based on a given product.

In our paper, we fine-tune ChemFM based on the [Root-aligned SMILES](https://pubs.rsc.org/en/content/articlelanding/2022/sc/d2sc02763a) technique.
But we should note that ChemFM can integrate seamlessly with SMILES sequence editing-based methods designed for reaction prediction by providing a better pre-trained model.

We provide **comparisons on benchmark datasets**, including the **USPTO benchmark datasets** (USPTO-Full, USPTO-MIT, and USPTO-50K). These datasets, consisting of organic chemical reactions extracted from U.S. patents and applications, are widely used to evaluate reaction prediction models.

Additionally, we include details to **replicate the results** reported in the paper, along with **model checkpoints and configurations** for each dataset.

Our repository provides both synthesis and retrosynthesis prediction and follow the normal practice and evaluate on:
- **Synthesis prediction**
-- USPTO-MIT
- **Retrosynthesis prediction**
-- USPTO-FULL
-- USPTO-MIT
-- USPTO-50K

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Steps to Fine-tune ChemFM

Fine-tuning the reaction tasks could take quite long time. 
If you want to evaluate our model, you can download the trained [model checkpoints](https://clemson.box.com/s/9t6399l8fk4n1uvkvhubssmtldwmrzbb) for each dataset. 

<summary> <strong><font size="+1"> 1. Prepare the Dataset </font></strong> </summary>
  
We directly use the phrased data from official repo of [Root-aligned SMILES](https://github.com/otori-bird/retrosynthesis). The detailed pre-processing (converting to Root-aligned SMILES) can found in their official repo [Root-aligned SMILES](https://github.com/otori-bird/retrosynthesis).
You can also download the [phrased data](https://clemson.box.com/s/9t6399l8fk4n1uvkvhubssmtldwmrzbb) for each dataset for not direct to the original repo.

<summary> <strong><font size="+1"> 2. Configure the Parameters for Training </font></strong></summary>
You can configure the parameters for training in two ways:

- **Feed arguments directly to the Python file**: Pass the arguments as command-line parameters when running the training script.
- **Specify the parameters in a YAML file**: Define all configurations in a `.yml` file and pass the file path to the Python script.

For all experiments, you can directly use the configuration files stored in [`configs/`](./configs/).

<summary> <strong><font size="+1"> 3. Fine-tuning Script </font></strong></summary>

To fine-tune ChemFM, you can use the following command:

```bash
python -m accelerate.commands.launch --config_file accelerate_config.yaml main.py --training_args_file <config_yml_file>
```
Our code is based on the [accelerate](https://huggingface.co/docs/accelerate/main/en/index) package, and the [accelerate_config.yaml](./accelerate_config.yaml) file is used to configure the distribution settings for training across multiple devices.
By default, we use x8 H100 GPUs in a single nodel to train our model, if you are use different configuration, make sure to configure the distribution settings.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Steps to Evaluate Fine-tuned ChemFM Model

<summary> <strong><font size="+1"> 1. Configure the Parameters for Generation </font></strong></summary>
The generation parameters will define the model to be evluated, the data augmentation for generation (this is normally used in reaction prediction task to make better and robust results; we use the same setting as Root-aligned SMILES).

You can configure the parameters for generating in two ways:

- **Feed arguments directly to the Python file**: Pass the arguments as command-line parameters when running the evaluation script.
- **Specify the parameters in a YAML file**: Define all configurations in a `.yml` file and pass the file path to the Python script.

You can directly use the generation configuration files stored in [`configs/`](./configs/).

<summary> <strong><font size="+1"> 2. Generate the Predictions </font></strong> </summary>
You should generate the predictions for each sample by using:

```bash
python -m accelerate.commands.launch --config_file accelerate_config.yaml evaluate.py --training_args_file <config_yml_file>
```
The predictions will be stored in the model path defined in the configuration.

By default, we use x8 H100 GPUs in a single nodel to generate the predictions, if you are use different configuration, make sure to configure the distribution settings.


<summary> <strong><font size="+1"> 3. Score the Predictions </font></strong> </summary>
You can get the accuracy of the model by simply running:

```bash
python ./score.py -data_path <prediction_file> -augmentation <num_augmentation>
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Benchmark Results

| Task Category     | Dataset     | Model                                 | Top-1  | Top-3  | Top-5  |
|------------------|------------|--------------------------------------|--------|--------|--------|
| **Synthesis**    | USPTO-MIT   | Prev. best: AT                  | 90.4   | -      | 96.5   |
|                  |            | Prev. second-best: R-SMILES       | 90.0   | 95.6   | 96.4   |
|                  |            | ChemFM                           | **90.5** | **95.7** | **96.6** |
| **Retro-synthesis** | USPTO-50K | Prev. best: R-SMILES            | 56.0   | 79.0   | 86.1   |
|                  |            | Prev. second-best: Graph2Edits   | 55.1   | 77.3   | 83.4   |
|                  |            | ChemFM                               | 58.0   | **80.0** | **86.3** |
|                  |            | ChemFM★                              | **59.7** | 79.2   | 84.2   |
|                  | USPTO-MIT   | Prev. best: R-SMILES            | 60.3   | 77.9   | 82.8   |
|                  |            | Prev. second-best: RetroTRAE     | 60.3   | 77.9   | 82.8   |
|                  |            | ChemFM                               | 61.6   | **78.7** | **83.0** |
|                  |            | ChemFM★                              | **62.4** | 78.5   | 82.5   |
|                  | USPTO-Full  | Prev. best: RetroXpert          | 49.4   | 63.6   | 67.6   |
|                  |            | Prev. second-best: R-SMILES      | 48.9   | 66.6   | 72.0   |
|                  |            | ChemFM                           | **51.7** | **68.0** | **72.5** |
<p align="right">(<a href="#readme-top">back to top</a>)</p>



