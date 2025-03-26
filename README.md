# EuroBERT: Scaling Multilingual Encoders for European Languages - Optimus Training Library

![EuroBERT](docs/images/bg.png)
[![arXiv](https://img.shields.io/badge/arXiv-2503.05500-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2503.05500)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-EuroBERT_Models-F8D44E?style=for-the-badge&logo=huggingface)](https://huggingface.co/EuroBERT)
[![Blog Post](https://img.shields.io/badge/Blog_Post-018EF5?logo=readme&logoColor=fff&style=for-the-badge)](https://huggingface.co/blog/EuroBERT/release)

## Introduction

[*EuroBERT*](https://arxiv.org/abs/2503.05500) is a multilingual encoder model designed for European languages, trained using the **Optimus** training library. Optimus is a flexible and scalable framework built to train language models efficiently across diverse hardware configurations, including **CPU, AMD, and NVIDIA GPUs**.

### Key Features

- **Hardware Agnostic**: Seamlessly train on CPU, AMD, or NVIDIA hardware.
- **Resumable Training**: Continue training regardless of hardware or environment changes.
- **Scalable Distributed Training**: Supports **Fully Sharded Data Parallel (FSDP)**, **Distributed Data Parallel (DDP)**, and other parallelism strategies.
- **Comprehensive Data Processing**: Includes utilities for **tokenization, packing, subsampling, and dataset inspection**.
- **Highly Customizable**: Fine-tune model architecture, training, and data processing with extensive configuration options.
- **Performance Optimizations**: Implements advanced techniques like **mixed precision training**, **fused operations**, and optimizations such as [Liger Kernel](https://github.com/linkedin/Liger-Kernel) and [Flash Attention](https://github.com/Dao-AILab/flash-attention).


## Installation
You can install EuroBERT using one of the following methods:

Run the following command to install the package directly:  
```bash
pip install git+https://github.com/Nicolas-BZRD/EuroBERT.git
```
or, for development purposes, clone the repository and install it in editable mode:
```bash
git clone https://github.com/Nicolas-BZRD/EuroBERT.git
cd EuroBERT
pip install -e .
```

## Tutorial
Before diving further into the Optimus Library, we encourage you to follow the notebook ‘Continuous Pre-Training of EuroBERT-210M with the Optimus Library’ (compatible with Google Scholar), which covers data processing and training setup.

[Go to Tutorial](https://github.com/Nicolas-BZRD/EuroBERT/tree/main/examples/continuous_pretraining.ipynb)

## Data Processing

Optimus provides an efficient pre-processing pipeline with tokenization, packing, subsampling, and inspection utilities. Full details are available in the [data processing documentation](https://github.com/Nicolas-BZRD/EuroBERT/tree/main/optimus/dataprocess). The [`tokenize_dataset.py`](https://github.com/Nicolas-BZRD/EuroBERT/blob/main/optimus/dataprocess/tokenize_dataset.py) script tokenizes a dataset and saves it in an optimized format. The tokenized data can be **sharded** and processed in parallel using multiple workers.

### Usage

```bash
python -m optimus.dataprocess.tokenize_dataset --input_dir <path> --tokenizer <path_or_name> --dataset <name> [--output_dir <path>] [--num_workers <num>]
```

### Parameters

- `input_dir` (*str*): Path to the input dataset directory.
- `tokenizer` (*str*): Path or name of the tokenizer.
- `dataset` (*str*): Name of the dataset to process.
- `output_dir` (*str*, optional): Directory to save the tokenized dataset.
- `num_workers` (*int* or `max`): Number of worker processes (use `max` for all available CPUs).

## Training

Optimus supports a wide range of configurations for different training scenarios. Detailed configuration options are documented in the [training guide](https://github.com/Nicolas-BZRD/EuroBERT/blob/main/docs/trainer.md).

```bash
python -m optimus.train --huggingface_id EuroBERT/EuroBERT-210m --data_mix_path <path> --batch_size <int> --mlm_probability <flaot> --mask_probability <int>
```

### Parameters

- `model_name` (*str*): Model type (`bert` or `eurobert`). Available models are listed [here](https://github.com/Nicolas-BZRD/EuroBERT/tree/main/optimus/trainer/model/encoder).
- `model_size` (*str*): Model size (e.g., `210m`, `310m`, `2b`).
- `data_mix_path` (*str*): Path to the data mix folder containing train.json and optionally eval.json. These JSON files define parameter configurations for dataset creation, offering similar configuration options to [MosaicML](https://docs.mosaicml.com/projects/streaming/en/stable/dataset_configuration/mixing_data_sources.html).
- `batch_size` (*int*): Number of samples per batch.
- `mlm_probability` (*float*): Probability of applying masked language modeling.
- `mask_probability` (*float*): Probability of replacing a masked token.


## Evaluation

If you're interested in evaluating pre-trained encoder models, we recommend using the [EncodEval](https://github.com/hgissbkh/EncodEval) library. Developed alongside the Optimus library, it provides a fair and consistent framework for evaluating and comparing encoder models.


## Citation

If you use EuroBERT in your research, please cite our paper:

```bibtex
@misc{boizard2025eurobertscalingmultilingualencoders,
      title={EuroBERT: Scaling Multilingual Encoders for European Languages}, 
      author={Nicolas Boizard and Hippolyte Gisserot-Boukhlef and Duarte M. Alves and André Martins and Ayoub Hammal and Caio Corro and Céline Hudelot and Emmanuel Malherbe and Etienne Malaboeuf and Fanny Jourdan and Gabriel Hautreux and João Alves and Kevin El-Haddad and Manuel Faysse and Maxime Peyrard and Nuno M. Guerreiro and Patrick Fernandes and Ricardo Rei and Pierre Colombo},
      year={2025},
      eprint={2503.05500},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.05500}, 
}
```
