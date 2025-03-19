import concurrent.futures
import gc
import itertools
import json
import pathlib
import time
from typing import Any, Iterable, Optional, TypedDict

import fire
import numpy as np
import streaming
import streaming.base.format.mds
import streaming.base.util

Metadata = dict[str, Any] | list[Any]


class TokRecord(TypedDict):
    tokens: list[int]
    metadata: Any


class PackTokRecord(TypedDict):
    tokens: list[int]
    metadata: list[Any]


def pack_dataset(
    local_dir: str,
    output_dir: str,
    block_size: Optional[int] = None,
    val_size: Optional[int] = None,
    size_limit: Optional[int | str] = "64MB",
    head: Optional[int] = None,
    num_workers: int = 1,
    random_size: bool = False,
):
    """Function to pack a dataset into blocks of a fixed size or uniformly distributed sentence length."""
    print("Number of workers:", num_workers)
    directories = [d.name for d in pathlib.Path(local_dir).iterdir() if d.is_dir()]
    directories = sorted(directories) or [""]

    vals = _get_val_sizes(directories, val_size)

    jobs_args = [
        {
            "local_dir": f"{local_dir}/{d}",
            "train_dir": f"{output_dir}/train/{d}",
            "val_dir": f"{output_dir}/val/{d}",
            "block_size": block_size,
            "val_size": val,
            "size_limit": size_limit,
            "head": head,
            "random_size": random_size,
        }
        for d, val in zip(directories, vals)
    ]

    print("Jobs:")
    for arg in jobs_args:
        print(json.dumps(arg, indent=2))

    start = time.time()
    print(f"Starting {num_workers} workers.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_worker, args) for args in jobs_args]
        for future in futures:
            future.result()

    print("All workers finished.")
    streaming.base.util.merge_index(f"{output_dir}/train", keep_local=True)
    if val_size is not None:
        streaming.base.util.merge_index(f"{output_dir}/val", keep_local=True)

    ellapsed = time.time() - start
    fmt_ellapsed = time.strftime("%H:%M:%S", time.gmtime(ellapsed))
    print(f"Finished in {fmt_ellapsed}.")


def _get_val_sizes(
    directories: list[str], val_size: Optional[int]
) -> list[Optional[int]]:
    if val_size is None:
        return [None] * len(directories)

    val_size_per_dir = val_size // len(directories)
    vals = [val_size_per_dir] * len(directories)
    remainder = val_size % len(directories)

    for i in range(remainder):
        vals[i] += 1

    return vals


def _worker(args: dict[str, Any]):
    local_dir = args["local_dir"]
    train_dir = args["train_dir"]
    val_dir = args["val_dir"]
    block_size = args["block_size"]
    val_size = args["val_size"]
    size_limit = args["size_limit"]
    random_size = args["random_size"]
    head = args["head"]

    directory = pathlib.Path(local_dir).name

    index = _load_index(local_dir)
    shards = index["shards"]

    readers = (
        streaming.base.format.mds.MDSReader.from_json(local_dir, None, s)
        for s in shards
    )
    records = itertools.chain.from_iterable(readers)

    if head is not None:
        print(f"Taking only the first {head} records.", flush=True)
        records = itertools.islice(records, head)

    records = map(_from_numpy, records)

    if block_size is not None:
        records = _pack(records, block_size, random_size, f"{directory} pack")

    records = map(_to_numpy, records)

    val, train = _split_records(records, val_size)

    start = time.time()
    print(f"(Dir {directory}) Started.", flush=True)

    if val is not None:
        print(f"(Dir {directory}) Writing validation set to {val_dir}", flush=True)
        _write_iterable(val_dir, val, size_limit, f"(Dir {directory}) val")

    print(f"(Dir {directory}) Writing training set to {train_dir}", flush=True)
    _write_iterable(train_dir, train, size_limit, f"(Dir {directory}) train")

    ellapsed = time.time() - start
    fmt_ellapsed = time.strftime("%H:%M:%S", time.gmtime(ellapsed))
    print(f"(Dir {directory}) Finished in {fmt_ellapsed}.", flush=True)


def _load_index(local_dir: str) -> dict[str, Any]:
    index_path = pathlib.Path(local_dir) / "index.json"
    with open(index_path, "r") as f:
        index = json.load(f)

    if index["version"] != 2:
        raise ValueError(f"Invalid index version: {index['version']}, expected 2.")

    return index


def _from_numpy(record: dict[str, Any]) -> TokRecord:
    return {
        "tokens": record["tokens"].tolist(),
        "metadata": record["metadata"],
    }


def _to_numpy(record: TokRecord) -> dict[str, Any]:
    return {
        "tokens": np.array(record["tokens"], dtype=np.int32),
        "metadata": record["metadata"],
    }


def _split_records(
    records: Iterable[TokRecord], val_size: Optional[int]
) -> tuple[Optional[Iterable[TokRecord]], Iterable[TokRecord]]:
    if val_size is not None:
        val = itertools.islice(records, val_size)
        train = records
    else:
        val = None
        train = records

    return val, train


def _pack(
    records: Iterable[TokRecord],
    block_size: int,
    random_size: bool = False,
    desc: Optional[str] = None,
) -> Iterable[TokRecord]:
    def _empty() -> TokRecord:
        return {"tokens": [], "metadata": []}

    def _append(
        curr: PackTokRecord, doc: TokRecord, size: int
    ) -> tuple[PackTokRecord, TokRecord]:
        curr["metadata"].append(doc["metadata"])

        leftover_size = size - len(curr["tokens"])
        doc_size = len(doc["tokens"])

        add_size = min(leftover_size, doc_size)
        tokens_to_add = doc["tokens"][:add_size]
        curr["tokens"].extend(tokens_to_add)

        doc["tokens"] = doc["tokens"][add_size:]

        return curr, doc

    curr = _empty()
    current_random_size, remaining_size = 0, 0
    min_doc_size = 12

    for doc in records:
        while len(doc["tokens"]) > 0:
            if random_size:
                if current_random_size == 0 and remaining_size == 0:
                    current_random_size = np.random.randint(
                        min_doc_size, block_size - (min_doc_size - 1)
                    )
                    remaining_size = block_size - current_random_size

                if current_random_size > 0:
                    curr, doc = _append(curr, doc, size=current_random_size)
                    if len(curr["tokens"]) >= current_random_size - min_doc_size:
                        yield curr
                        current_random_size = 0
                        curr = _empty()
                else:
                    curr, doc = _append(curr, doc, size=remaining_size)
                    if len(curr["tokens"]) >= remaining_size - min_doc_size:
                        yield curr
                        remaining_size = 0
                        curr = _empty()
            else:
                curr, doc = _append(curr, doc, size=block_size)
                if len(curr["tokens"]) == block_size:
                    yield curr
                    curr = _empty()

    if desc is not None:
        print(f"({desc}): leftovers:", len(curr["tokens"]))


def _write_iterable(
    output_dir: str,
    records: Iterable[dict[str, Any]],
    size_limit: Optional[int | str],
    desc: Optional[str] = None,
):
    columns = {
        "tokens": "ndarray:int32",
        "metadata": "json",
    }
    start = time.time()
    prefix = f"({desc}): " if desc else ""

    def print_status(i: int, start: float):
        ellapsed = time.time() - start
        total_seconds = int(ellapsed)
        records_per_second = "Inf" if total_seconds == 0 else i / total_seconds
        msg = (
            f"{prefix}Processed {i} records in "
            f"{ellapsed // 3600}H:{(ellapsed % 3600) // 60}m:{ellapsed % 60}s ({records_per_second} Records/s)."
        )
        print(msg, flush=True)

    with streaming.MDSWriter(
        out=output_dir,
        columns=columns,
        size_limit=size_limit,
    ) as out:
        for i, record in enumerate(records, start=1):
            out.write(record)
            if i % 10000 == 0:
                print_status(i, start)
                gc.collect()
        print_status(i, start)
    print(f"{prefix}Finished writing {i} records.", flush=True)


if __name__ == "__main__":
    fire.Fire(pack_dataset)
