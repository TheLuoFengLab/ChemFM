<a id="readme-top"></a>

# Molecular Conditional Generation

This section of the repository provides all necessary scripts and details for fine-tuning **ChemFM** on **molecular conditional generation** tasks.

## Overview

Conditional molecular generation involves designing molecules that satisfy specific conditions, such as desired properties or scaffolds. 
ChemFM can handle different condition combinations, where properties can be discrete classes, continuous values, scaffold SMILES, or any combination of these. The order of the conditions can be arbitrary.

In this repository, we follow the settings of [MolGPT](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00600) and provide code for fine-tuning ChemFM on:
- **Property-conditioned generation**: Trained on the [GuacaMol](https://github.com/BenevolentAI/guacamol) benchmark, the task is to generate molecules satisfying specified properties. Instead of training separate models for different condition combinations, we train a unified model that handles multiple combinations.
- **Scaffold-property-conditioned generation**: Trained on the [MOSES](https://github.com/molecularsets/moses) benchmark, the task is to generate molecules that satisfy both a given scaffold and specific properties. We also train a unified model to accommodate different condition combinations.

We include details to **replicate the results** reported in our paper, along with **model checkpoints and configurations** for each dataset.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Fine-tuning ChemFM

Fine-tuning conditional molecular generation tasks can be computationally intensive.  
If you prefer to evaluate our model directly, you can download the trained [model checkpoints](https://clemson.box.com/s/ajrx5x37lyl0sm7reqiztmhi39hm07m7) for each dataset.

### 1. Prepare the Dataset

We use the **MOSES** and **GuacaMol** benchmark datasets, including physicochemical descriptors such as logP, SAS, TPSA, and QED as conditions. 
You can download the datasets from [GuacaMol](https://github.com/BenevolentAI/guacamol) and [MOSES](https://github.com/molecularsets/moses),
but each property should be normalized before being fed to the model.
Alternatively, you can download our [processed data](https://clemson.box.com/s/pnzawue6f15zxloub04jd0fdp3pnqvht) for each dataset without additional preprocessing.

### 2. Configure Training Parameters

You can configure the training parameters in two ways:

- **Pass arguments directly to the Python script**: Supply command-line arguments when running the training script.
- **Use a YAML configuration file**: Define all settings in a `.yml` file and pass the file path as an argument.

Predefined configuration files for two models are available in [`configs/`](./configs/).

### 3. Fine-tuning Script

To fine-tune ChemFM, run:

```bash
python -m accelerate.commands.launch --config_file accelerate_config.yaml main.py --training_args_file <config_yml_file>
```

Our code is built on the [Accelerate](https://huggingface.co/docs/accelerate/main/en/index) package. The [`accelerate_config.yaml`](./accelerate_config.yaml) file configures the distribution settings for multi-GPU training.

By default, we use **8× H100 GPUs** on a single node for training. If using a different setup, ensure the distribution settings are adjusted accordingly.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Evaluating Fine-tuned ChemFM Models

### 1. Configure Generation Parameters

The generation configuration file (an example [`configs/moses/evaluation.yml`](configs/moses/evaluation.yml)) defines:
- The model to be evaluated.
- The condition generation configuration file path (an example [`configs/moses/generation_configs/TPSA_logp_SAS.json`](configs/moses/generation_configs/TPSA_logp_SAS.json)). This details the number of generations, sample points, and other settings.

You can configure the generation parameters in two ways:
- **Pass arguments directly to the Python script**.
- **Use a YAML configuration file** (recommended, available in [`configs/`](./configs/)).

### 2. Generate Molecules

Run the following command to generate molecules:

```bash
python -m accelerate.commands.launch --config_file accelerate_config.yaml evaluate.py --training_args_file <config_yml_file>
```

Generated molecules will be stored in the `generation_output_path` defined in the configuration file (an example [`configs/moses/evaluation.yml`](configs/moses/evaluation.yml)).

By default, we use **8× H100 GPUs** on a single node for inference. Adjust distribution settings if using a different setup.

### 3. Score the Generated Molecules

To evaluate the generation metrics, run:

```bash
python ./score.py -data_path <generated_molecules_file> -train_data_path <train_data_path> 
```

More arguments can be found in the [`score.py`](score.py) file.

An example script for evaluating scaffold and property-conditioned (scaffold + TPSA + logP + SAS) generation is:

```bash
python ./score.py -train_data_path ./data_temp/moses/train_data.csv -data_path ./outputs/moses/checkpoint/generations/TPSA_logp_SAS.csv
```

**We also provide the generated molecules for each model in the [checkpoint folder](https://clemson.box.com/s/ajrx5x37lyl0sm7reqiztmhi39hm07m7),** and you can directly use it to check the results.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Benchmark Results
<details>
  <summary>Click to expand the GuacaMol results</summary>


| Property              | Model       | Validity ↑ | Uniqueness ↑ | Novelty ↑ | Mean Average Deviation (MAD) ↓ |
|-----------------------|------------|------------|--------------|------------|--------------------------------|
| **logP**             | MolGPT     | 0.971      | 0.998        | 0.977      | 0.230                          |
|                      | ChemFM-3B  | **0.981**  | **1.000**    | **0.985**  | **0.182**                      |
| **TPSA**             | MolGPT     | 0.971      | 0.997        | 0.975      | 3.562                          |
|                      | ChemFM-3B  | **0.979**  | **0.999**    | **0.984**  | **2.466**                      |
| **SAS**              | MolGPT     | 0.978      | 0.996        | 0.966      | 0.133                          |
|                      | ChemFM-3B  | **0.986**  | **0.999**    | **0.971**  | **0.126**                      |
| **QED**              | MolGPT     | 0.974      | 0.997        | 0.968      | 0.056                          |
|                      | ChemFM-3B  | **0.982**  | **1.000**    | **0.980**  | **0.045**                      |
| **SAS + logP**       | MolGPT     | 0.972      | 0.991        | 0.983      | 0.147 / 0.253                  |
|                      | ChemFM-3B  | **0.980**  | **0.995**    | **0.985**  | **0.137 / 0.195**              |
| **SAS + TPSA**       | MolGPT     | 0.971      | 0.988        | 0.984      | 0.155 / 3.785                  |
|                      | ChemFM-3B  | **0.980**  | **0.991**    | **0.985**  | **0.138 / 2.659**              |
| **TPSA + logP**      | MolGPT     | 0.964      | 0.994        | 0.989      | 3.715 / 0.243                  |
|                      | ChemFM-3B  | **0.973**  | **0.997**    | **0.992**  | **2.415 / 0.184**              |
| **TPSA + logP + SAS**| MolGPT     | 0.972      | 0.969        | 0.988      | 3.797 / 0.268 / 0.180          |
|                      | ChemFM-3B  | **0.975**  | **0.971**    | **0.989**  | **2.289 / 0.191 / 0.166**      |
</details>


<details>
  <summary>Click to expand the MOSES results</summary>
# Performance Comparison on Standard Benchmarks for Conditional Molecule Generation on the MOSES Dataset

| Property            | Model  | Generation Count | Valid Molecules ↑ | Unique Molecules ↑ | Novel Molecules ↑ | Same Scaffold Molecules ↑ | MAD ↓ |
|---------------------|--------|------------------|-------------------|--------------------|-------------------|--------------------------|-------|
| **logP**           | MolGPT | 150,000         | 144,404           | 44,558            | 44,558            | 44,545                    | 0.125 |
|                     | ChemFM-3B | 150,000         | **145,682**       | **56,606**        | **56,606**        | **56,107**                | **0.087** |
| **SAS**            | MolGPT | 150,000         | 138,792           | 52,652            | 52,652            | 52,615                    | 0.129 |
|                     | ChemFM-3B | 150,000         | **140,580**       | **68,163**        | **68,162**        | **67,654**                | **0.123** |
| **TPSA**           | MolGPT | 150,000         | 144,211           | 45,030            | 45,030            | 45,020                    | 2.651 |
|                     | ChemFM-3B | 150,000         | **145,367**       | **54,165**        | **54,162**        | **53,586**                | **2.114** |
| **QED**            | MolGPT | 150,000         | 141,458           | 57,594            | 57,594            | 57,569                    | 0.051 |
|                     | ChemFM-3B | 150,000         | **144,794**       | **72,458**        | **72,458**        | **71,836**                | **0.050** |
| **TPSA + logP**    | MolGPT | 200,000         | 181,934           | 54,499            | 54,498            | 54,422                    | 3.771 / 0.186 |
|                     | ChemFM-3B | 200,000         | **187,953**       | **66,648**        | **66,648**        | **65,898**                | **3.266 / 0.159** |
| **SAS + logP**     | MolGPT | 200,000         | 180,063           | 51,550            | 51,550            | 51,426                    | 0.145 / 0.184 |
|                     | ChemFM-3B | 200,000         | **180,804**       | **66,465**        | **66,465**        | **65,983**                | **0.137 / 0.166** |
| **TPSA + SAS**     | MolGPT | 200,000         | 177,118           | 61,510            | 61,510            | 61,383                    | 3.840 / 0.171 |
|                     | ChemFM-3B | 200,000         | **183,209**       | **70,905**        | **70,904**        | **70,338**                | **3.504 / 0.148** |
| **TPSA + logP + SAS** | MolGPT | 400,000         | 313,787           | 67,373            | 67,373            | 67,215                    | 5.370 / 0.352 / 0.255 |
|                     | ChemFM-3B | 400,000         | **323,043**       | **97,314**        | **97,314**        | **96,301**                | **4.780 / 0.329 / 0.217** |
</details>

<p align="right">(<a href="#readme-top">back to top</a>)</p>
