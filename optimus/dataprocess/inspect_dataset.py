import json
from typing import Any, Dict

import fire
import streaming
from transformers import AutoTokenizer

from optimus.dataprocess.tokenize_dataset import Llama3TiktokenTokenizer


def print_sample(index: int, sample: Dict[str, Any], tokenizer, tiktoken: bool) -> None:
    """Prints a sample from the dataset with its tokens and metadata."""
    tokens = sample["tokens"]

    text = (
        tokenizer.encoding.decode(tokens)
        if tiktoken
        else tokenizer.batch_decode([tokens], skip_special_tokens=False)[0]
    )

    metadata = sample["metadata"]

    print(f"Sample {index} {'-' * 80}")
    print(f'Text:\n"{text}"')
    print("Tokens:", tokens)
    print("Size:", len(tokens))
    print("Metadata:")
    print(json.dumps(metadata, indent=2))


def inspect_dataset(
    input_dir: str, tokenizer: str, num_samples: int = 5, tiktoken: bool = False
) -> None:
    """
    Inspect a streaming dataset by printing a few samples.

    Args:
        input_dir (str): Dataset directory path.
        tokenizer (str): Tokenizer name or path.
        num_samples (int, optional): Number of samples to print. Defaults to 5.
        tiktoken (bool, optional): Whether to use the Tiktoken tokenizer. Defaults to False.
    """
    dataset = streaming.StreamingDataset(local=input_dir, shuffle=False, batch_size=1)

    tokenizer = (
        Llama3TiktokenTokenizer(tokenizer)
        if tiktoken
        else AutoTokenizer.from_pretrained(tokenizer)
    )

    for i, sample in enumerate(dataset):
        print_sample(i, sample, tokenizer, tiktoken)
        if i >= num_samples - 1:
            break


if __name__ == "__main__":
    fire.Fire(inspect_dataset)
