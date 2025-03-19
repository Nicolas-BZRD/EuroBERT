import fire
import json
import streaming
from typing import Dict, Any
from transformers import AutoTokenizer


def print_sample(index: int, sample: Dict[str, Any], tokenizer: AutoTokenizer) -> None:
    tokens = sample["tokens"]
    text = tokenizer.batch_decode([tokens], skip_special_tokens=False)[0]
    metadata = sample["metadata"]

    print(f"Sample {index} {'-' * 80}")
    print(f'Text:\n"{text}"')
    print("Tokens:", tokens)
    print("Size:", len(tokens))
    print("Metadata:")
    print(json.dumps(metadata, indent=2))


def inspect_dataset(local_dir: str, tokenizer_name: str, num_samples: int = 5) -> None:
    """
    Inspect a streaming dataset by printing a few samples.

    Args:
        local_dir (str): Dataset directory path.
        tokenizer_name (str): Tokenizer name or path.
        num_samples (int, optional): Number of samples to print. Defaults to 5.
    """
    dataset = streaming.StreamingDataset(
        local=local_dir,
        shuffle=False,
        batch_size=1,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    for i, sample in enumerate(dataset):
        print_sample(i, sample, tokenizer)
        if i >= num_samples - 1:
            break


if __name__ == "__main__":
    fire.Fire(inspect_dataset)
