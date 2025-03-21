# EuroBERT Data Processing Documentation

This documentation covers the data processing tools provided by the EuroBERT training library (optimus). These tools enable efficient preprocessing for large-scale language model training, including tokenizing, packing, subsampling, and inspecting datasets.

## Table of Contents

- [1. Tokenizing a Dataset](#1-tokenizing-a-dataset)
- [2. Packing a Dataset (Optional)](#2-packing-a-dataset-optional)
- [3. Subsampling a Dataset (Optional)](#3-subsampling-a-dataset-optional)
- [4. Inspecting a Dataset (Optional)](#4-inspecting-a-dataset-optional)

## 1. Tokenizing a Dataset

The `tokenize_dataset.py` script tokenizes a dataset using a specified tokenizer and saves the output in an optimized format. The tokenized data can be split into shards based on a size limit and processed in parallel using multiple workers.

### Usage

```bash
python -m optimus.dataprocess.tokenize_dataset --input_dir <path> --tokenizer <path_or_name> --dataset <name> [--output_dir <path>] [--size_limit <value>] [--num_workers <num>] [--head <num>] [--read_files_kwargs <json>] [--timeout <seconds>] [--tiktoken]
```

### Parameters

* `input_dir` (*str*): Path to the directory containing the input dataset.
* `tokenizer` (*str*): Name or path of the tokenizer to be used.
* `dataset` (*str*): Name of the dataset to process.
* `output_dir` (*str*, optional): Directory to save the tokenized dataset. Defaults to `./output`.
* `size_limit` (*int | str*, optional): Maximum shard size before creating a new shard. Supports human-readable formats (e.g., `100kb`, `1mb`). Defaults to `64MB` (`1 << 26`).
* `num_workers` (*int | str*, optional): Number of worker processes. Can be set to `max` to use all available CPUs. Defaults to `1`.
* `head` (*int*, optional): Number of batches to process. If not specified, processes the entire dataset.
* `read_files_kwargs` (*dict[str, Any]*, optional): Additional parameters for dataset file reading. Defaults to `None`.
* `timeout` (*int*, optional): Maximum time (in seconds) before termination. Defaults to `None` (no timeout).
* `tiktoken` (*bool*, optional): Whether to use `tiktoken` for faster tokenization. Defaults to `False`.

### Example

```bash
python -m optimus.dataprocess.tokenize_dataset --input_dir ./codebagel --tokenizer "meta-llama/Llama-3.1-8B-Instruct" --dataset codeBagel --output_dir ./output --size_limit 100mb --num_workers 4
```

This command tokenizes the `codeBagel` dataset stored in `./codebagel` using the `meta-llama/Llama-3.1-8B-Instruct` tokenizer, saves the output in `./output`, splits shards at 100MB, and runs with 4 workers.

#### Fastest Tokenization with `tiktoken`

For maximum efficiency (up to 2x speedup), use `tiktoken` with a compatible tokenizer and `tokenizer.model` [file](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/original/tokenizer.model):

```bash
python -m optimus.dataprocess.tokenize_dataset --input_dir ./codebagel --tokenizer ./llama_tokenizer.model --dataset codeBagel --output_dir ./tokenized --size_limit 100mb --num_workers 4 --tiktoken
```

---

## 2. Packing a Dataset (Optional)

The `pack_dataset.py` script packs a dataset into blocks (sentences) of a fixed size or uniformly distributed sentence length.

### Usage

```bash
python -m optimus.dataprocess.pack_dataset --local_dir <path> --output_dir <path> [--block_size <int>] [--val_size <int>] [--size_limit <int|str>] [--head <int>] [--num_workers <int>] [--random_size]
```

### Parameters

* `local_dir` (*str*): Local directory path containing the tokenized dataset.
* `output_dir` (*str*): Directory to save the packed dataset.
* `block_size` (*int*, optional): Block size for packing. Defaults to `None`.
* `val_size` (*int*, optional): Validation set size. Defaults to `None`.
* `size_limit` (*int | str*, optional): Size limit for the output files. Defaults to `"64MB"`.
* `head` (*int*, optional): Number of records to process. Defaults to `None`.
* `num_workers` (*int*, optional): Number of workers. Defaults to `1`.
* `random_size` (*bool*, optional): If `True`, selects a random block size between `12` and `block_size`, resulting in a uniformly distributed sentence length. Defaults to `False`.

### Example

```bash
python -m optimus.dataprocess.pack_dataset --local_dir './tokenized' --output_dir './output_pack' --block_size 2048 --random_size
```

---

## 3. Subsampling a Dataset (Optional)

The `subsample_dataset.py` script processes a dataset to split it into subdirectories and merges the indexes. This is particularly useful for optimizing dataset loading, especially in environments like MosaicML, to avoid GPU synchronization timeout errors.

### Usage

```bash
python -m optimus.dataprocess.subsample_dataset --dataset_path <path> --num_shards <int>
```

### Parameters

* `dataset_path` (*str*): The base path where the datasets are located.
* `num_shards` (*int*): The number of shards to split the dataset into.

### Example

```bash
python -m optimus.dataprocess.subsample_dataset --dataset_path "./tokenized" --num_shards 2
```

---

## 4. Inspecting a Dataset (Optional)

The `inspect_dataset.py` script inspects a processed dataset by printing a few samples.

### Usage

```bash
python -m optimus.dataprocess.inspect_dataset --local_dir <path> --tokenizer <path_or_name> [--num_samples <int>]
```

### Parameters

* `local_dir` (*str*): Dataset directory path.
* `tokenizer_name` (*str*): Tokenizer name or path.
* `num_samples` (*int*, optional): Number of samples to print. Defaults to `5`.

### Example

```bash
python -m optimus.dataprocess.inspect_dataset --local_dir './output' --tokenizer "meta-llama/Llama-3.1-8B-Instruct" --num_samples 5
```