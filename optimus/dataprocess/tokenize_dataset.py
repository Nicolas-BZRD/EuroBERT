import gc
import importlib
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Iterable, Iterator, Optional, Protocol, TypedDict

import fire
import numpy as np
import tiktoken
import tiktoken.load
from dateutil import parser
from streaming import MDSWriter
from streaming.base.util import merge_index
from transformers import AutoTokenizer

Metadata = dict[str, Any] | list[Any]


class Record(TypedDict):
    text: str
    metadata: Metadata


class TokRecord(TypedDict):
    tokens: list[int]
    metadata: Metadata


class GetRecordsFunc(Protocol):
    def __call__(self, file: str) -> Iterable[list[Record]]: ...


class Llama3TiktokenTokenizer:
    def __init__(self, path: str):
        llama_3_special_tokens = _llama_special_tokens()
        mergeable_ranks = tiktoken.load.load_tiktoken_bpe(path)

        pat_str = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

        self.encoding = tiktoken.Encoding(
            name="llama3-tiktoken",
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=llama_3_special_tokens,
        )
        self._pat_str = pat_str
        self._special_tokens = llama_3_special_tokens
        self._len = len(mergeable_ranks) + len(llama_3_special_tokens)
        self._eos_id = llama_3_special_tokens["<|end_of_text|>"]

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def pat_str(self) -> str:
        return self._pat_str

    @property
    def special_tokens(self) -> dict[str, int]:
        return self._special_tokens

    @property
    def vocab_size(self) -> int:
        return self._len


def _llama_special_tokens():
    # Special tokens for Llama models, with reserved placeholders for future use.
    special_tokens = {
        "<|begin_of_text|>": 128_000,
        "<|end_of_text|>": 128_001,
        "<|mask|>": 128_002,
        "<|parallel_sep|>": 128_003,
        "<|fim_suffix|>": 128_004,
        "<|step_id|>": 128_005,
        "<|start_header_id|>": 128_006,
        "<|end_header_id|>": 128_007,
        "<|eom_id|>": 128_008,
        "<|eot_id|>": 128_009,
    }
    for i in range(245):
        special_tokens[f"<|reserved_special_token_{i + 9}|>"] = 128_010 + i
    special_tokens["<|python_tag|>"] = 128_255
    return special_tokens


def time_str_to_seconds(time_str: Optional[str]) -> Optional[int]:
    if time_str is None:
        return None
    try:
        time_obj = parser.parse(time_str)
    except parser.ParserError:
        raise ValueError("Time format must be HH:MM:SS")
    return int(time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second)


def futures_timeout(executor, futures, timeout: Optional[int]):
    if timeout:
        timeout = time_str_to_seconds(timeout)
        start_time = time.time()
        while not all(f.done() for f in futures):
            if time.time() - start_time > timeout:
                print("Timeout reached, canceling remaining tasks.")
                executor.shutdown(wait=False, cancel_futures=True)
                break
            time.sleep(10)


def _worker(
    input_subset,
    tokenizer_path,
    output_dir,
    size_limit,
    get_records_fn: GetRecordsFunc,
    head: Optional[int] = None,
    tiktoken: bool = False,
):
    tokenizer = (
        Llama3TiktokenTokenizer(tokenizer_path)
        if tiktoken
        else AutoTokenizer.from_pretrained(tokenizer_path)
    )

    def encode_fn(texts: list[str]) -> list[list[int]]:
        input_ids = (
            tokenizer.encoding.encode_ordinary_batch(texts, num_threads=16)
            if tiktoken
            else tokenizer(texts)["input_ids"]
        )
        eos_id = tokenizer.eos_id if tiktoken else tokenizer.eos_token_id
        return [ids + [eos_id] for ids in input_ids]

    worker_id = output_dir.rstrip("/").split("/")[-1]
    print(f"({worker_id}): Worker started.", flush=True)

    data_per_file = map(get_records_fn, input_subset)
    batches = itertools.chain.from_iterable(data_per_file)

    if head is not None:
        batches = itertools.islice(batches, head)

    def _tokenize(batches: list[Record]) -> Iterator[TokRecord]:
        texts = [r["text"] for r in batches]
        metadata = [r.get("metadata", {}) for r in batches]
        tokens = encode_fn(texts)
        return ({"tokens": t, "metadata": m} for t, m in zip(tokens, metadata))

    tokenized_batches = map(_tokenize, batches)
    tokenized = itertools.chain.from_iterable(tokenized_batches)

    def to_numpy(record: TokRecord) -> dict[str, Any]:
        record["tokens"] = np.array(record["tokens"], dtype=np.int32)
        return record

    tokenized = map(to_numpy, tokenized)

    columns = {
        "tokens": "ndarray:int32",
        "metadata": "json",
    }

    start = time.time()
    number_tokens = 0
    with MDSWriter(out=output_dir, columns=columns, size_limit=size_limit) as out:
        for i, record in enumerate(tokenized, start=1):
            if record["tokens"].size:
                number_tokens += record["tokens"].size
                out.write(record)
            if i % 10000 == 0:
                elapsed = time.time() - start
                records_per_second = i / max(int(elapsed), 1)
                print(
                    f"({worker_id}): Processed {i} records, {records_per_second:.2f} Records/s, {number_tokens} tokens.",
                    flush=True,
                )
                gc.collect()

    return number_tokens


def tokenize_dataset(
    input_dir: str,
    tokenizer: str,
    dataset: str,
    output_dir: str = "output",
    size_limit: Optional[int | str] = "64MB",
    num_workers: int | str = 1,
    head: Optional[int] = None,
    timeout: Optional[int] = None,
    tiktoken: bool = False,
    read_files_kwargs: Optional[dict[str, Any]] = None,
):
    start = time.time()
    file_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_dir = os.path.join(file_dir, "dataset")
    read_files_kwargs = read_files_kwargs or {}

    assert dataset + ".py" in os.listdir(dataset_dir), f"{dataset}.py module not found."
    assert (
        num_workers != "max" or num_workers > 0
    ), "num_workers must be greater than 0."
    assert not os.path.exists(
        output_dir
    ), f"Output directory '{output_dir}' already exists."

    num_workers = os.cpu_count() - 1 if num_workers == "max" else num_workers

    print(
        f"Input directory: {input_dir}\nTokenizer: {tokenizer}\nDataset: {dataset}\nSize limit: {size_limit}\nNum workers: {num_workers}\nTimeout: {timeout}"
    )

    dataset_module = importlib.import_module(f"dataset.{dataset}")
    inputs = dataset_module.get_files(input_dir, **read_files_kwargs)
    assert inputs, "No data files found in the input directory."

    if num_workers > len(inputs):
        print(f"Reduced num_workers to {len(inputs)} to match the number of inputs.")
        num_workers = len(inputs)

    output_dirs = (
        [os.path.join(output_dir, str(i)) for i in range(num_workers)]
        if num_workers > 1
        else [output_dir]
    )
    for dir in output_dirs:
        os.makedirs(dir, exist_ok=True)

    input_subset = np.array_split(inputs, num_workers)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _worker,
                subset,
                tokenizer,
                output_dirs[i],
                size_limit,
                dataset_module.get_text,
                head,
                tiktoken,
            )
            for i, subset in enumerate(input_subset)
        ]
        if timeout:
            futures_timeout(executor, futures, timeout)

    total_tokens = sum(future.result() for future in futures if future.done())
    if num_workers > 1:
        merge_index(output_dir, keep_local=True)

    end = time.time()
    metadata = {
        "tokenizer": tokenizer,
        "input_dir": input_dir,
        "dataset": dataset,
        "size_limit": size_limit,
        "num_workers": num_workers,
        "total_tokens": total_tokens,
        "Runtime": time.strftime("%H:%M:%S", time.gmtime(end - start)),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print("Tokenization completed.")


if __name__ == "__main__":
    fire.Fire(tokenize_dataset)
