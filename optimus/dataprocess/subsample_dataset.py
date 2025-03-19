import argparse
import logging
import os
import shutil
import numpy as np
from streaming.base.util import merge_index
import json

logging.basicConfig(level=logging.ERROR)
logging.getLogger("streaming.base.storage.upload").setLevel(logging.ERROR)


def manually_create_index(shard_group: list[str], sub_dir: str):
    """
    Manually create index files for shards.

    Args:
        shard_group (list[str]): List of shard names.
        sub_dir (str): Subdirectory where the new index will be saved.
    """
    parent_dir = os.path.dirname(sub_dir)
    index_path = os.path.join(parent_dir, "index.json")

    with open(index_path, "r") as f:
        index = json.load(f)

    index["shards"] = [
        shard
        for shard in index["shards"]
        if shard["raw_data"]["basename"] in shard_group
    ]

    new_index_path = os.path.join(sub_dir, "index.json")
    with open(new_index_path, "w") as f:
        json.dump(index, f, indent=4)


def subsample_dataset(dataset_path: str, num_shards: int):
    """
    Process dataset to split shards into subdirectories and merge indexes.

    Args:
        dataset_path (str): Base path where datasets are located.
        num_shards (int): Number of shards to split into.
    """
    train_path = (
        os.path.join(dataset_path, "train")
        if os.path.exists(os.path.join(dataset_path, "train"))
        else dataset_path
    )

    if not os.path.isdir(train_path):
        logging.error(
            f"Dataset path {train_path} does not exist or is not a directory."
        )
        return

    shards = [
        name
        for name in os.listdir(train_path)
        if os.path.isdir(os.path.join(train_path, name)) or name.endswith(".mds")
    ]

    if not shards:
        logging.warning(f"No shards found in {train_path}. Skipping...")
        return

    split_shards = np.array_split(shards, num_shards)

    for i, shard_group in enumerate(split_shards):
        sub_dir = os.path.join(train_path, f"sub_{i}")
        os.makedirs(sub_dir, exist_ok=True)

        for shard_name in shard_group:
            src_path = os.path.join(train_path, shard_name)
            dst_path = os.path.join(sub_dir, shard_name)
            shutil.move(src_path, dst_path)

        logging.info(f"Shards moved to {sub_dir}")

        if shard_name.endswith(".mds"):
            manually_create_index(shard_group, sub_dir)
        else:
            merge_index(sub_dir, keep_local=True)

    # Remove old index and merge final index in the root train folder
    index_path = os.path.join(train_path, "index.json")
    if os.path.exists(index_path):
        os.remove(index_path)

    merge_index(train_path, keep_local=True)
    logging.info("Dataset processing completed successfully.")


def main():
    """
    Main function to parse arguments and process datasets.
    """
    parser = argparse.ArgumentParser(
        description="Process dataset stored in a folder by splitting it into --num_shards sub-datasets."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path where the dataset is located.",
    )
    parser.add_argument(
        "--num_shards", type=int, default=5, help="Number of shards to split into."
    )

    args = parser.parse_args()
    subsample_dataset(args.dataset_path, args.num_shards)


if __name__ == "__main__":
    main()
