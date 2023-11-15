from typing import Optional, Union, List

import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, DistributedSampler
from trl.trainer.utils import ConstantLengthDataset
from transformers import PreTrainedTokenizerBase
import multiprocessing as mp
from concat_dataset import ConcatTokensDataset
from tqdm import tqdm


def get_dataset(
    dataset_name: str,
    subset: Optional[str] = None,
    split: Optional[str] = None,
    **kwargs,
) -> Dataset:
    data_path = dataset_name
    if subset:
        data_path += f"/{subset}"
    return load_dataset(data_path, split=split, **kwargs)


# This function will be run in parallel processes to process dataset partitions
def process_partition(args):
    (
        hf_dataset_partition,
        dataset_text_field,
        tokenizer,
        max_length,
        bos_text,
        eos_text,
        no_wrap,
        num_proc,
        is_tokenized,
    ) = args
    return process_dataset(
        hf_dataset_partition,
        dataset_text_field,
        tokenizer,
        max_length,
        bos_text,
        eos_text,
        no_wrap,
        num_proc,
        is_tokenized,
    )


def process_dataset(
    hf_dataset: Dataset,
    dataset_text_field: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    bos_text: str,
    eos_text: str,
    no_wrap: bool,
    num_proc: int | None = None,
    is_tokenized: bool = True,
):
    concat_tokens_dataset = ConcatTokensDataset(
        hf_dataset=hf_dataset,
        dataset_text_field=dataset_text_field,
        tokenizer=tokenizer,
        max_length=max_length,
        bos_text=bos_text,
        eos_text=eos_text,
        no_wrap=no_wrap,
        num_proc=num_proc,
        is_tokenized=is_tokenized,
        pre_tokenize=True,
    )

    packed_dataset_dict = {"tokens": []}
    for sample in concat_tokens_dataset:
        packed_dataset_dict["tokens"].append(sample["tokens"])
    packed_dataset = Dataset.from_dict(packed_dataset_dict)
    return packed_dataset


def prepare_dataset(
    hf_dataset: Dataset,
    dataset_text_field: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    bos_text: str,
    eos_text: str,
    no_wrap: bool,
    num_proc: int | None = None,
    num_partitions: int | None = 4,
) -> torch.Tensor:
    num_proc = mp.cpu_count() - 1 if num_proc is None else num_proc
    num_partitions = num_proc if num_partitions is None else num_partitions

    hf_dataset = hf_dataset.map(
        lambda x: tokenizer(
            x[dataset_text_field],
            padding=False,
            truncation=False,
            return_tensors="pt",
        ),
        num_proc=num_proc,
        batched=True,
        batch_size=1,
    )

    sz = len(hf_dataset) // num_partitions
    print(
        f"Done tokenizing the dataset. Splitting into {num_partitions} partitions of size {sz}. \
            Total size: {len(hf_dataset)}::{sz*num_partitions}"
    )
    # Split the tokenized dataset into num_partitions
    partitions = [
        hf_dataset.shard(num_partitions, i, contiguous=True)
        for i in range(num_partitions)
    ]

    # Prepare arguments for parallel execution
    args_list = [
        (
            partitions[i],
            "text",
            tokenizer,
            max_length,
            bos_text,
            eos_text,
            no_wrap,
            num_proc,
            True,
        )
        for i in range(num_partitions)
    ]
    """
    hf_dataset_partition,
    tokenizer,
    max_length,
    bos_text,
    eos_text,
    no_wrap,
    num_proc,
    is_tokenized,
    """

    print("Creating a pool of worker processes")
    # Create a pool of worker processes
    pool = mp.Pool(processes=num_partitions)

    print("Processing dataset partitions in parallel")
    # Process dataset partitions in parallel
    results = pool.map(process_partition, args_list)

    # Terminate the pool of workers
    pool.close()
    pool.join()

    print("Merging the results")
    # Merge the results into a single dataset
    merged_dataset_dict = {"tokens": torch.tensor([])}
    for result in tqdm(results):
        if len(merged_dataset_dict["tokens"]) == 0:
            merged_dataset_dict["tokens"] = torch.Tensor(result["tokens"])
            continue

        merged_dataset_dict["tokens"] = torch.cat(
            (merged_dataset_dict["tokens"], torch.Tensor(result["tokens"])), dim=0
        )

    # Convert the merged dictionary to a Dataset
    # merged_dataset = Dataset.from_dict(merged_dataset_dict)
    return merged_dataset_dict["tokens"]
