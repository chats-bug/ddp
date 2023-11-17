from typing import Optional, Union, List

import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, DistributedSampler
from trl.trainer.utils import ConstantLengthDataset
from transformers import PreTrainedTokenizerBase
import multiprocessing as mp
if __name__ == "__main__":
    from concat_dataset import ConcatTokensDataset
else:
    from .concat_dataset import ConcatTokensDataset
from tqdm import tqdm
from multiprocessing.pool import ThreadPool as Pool


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

    packed_dataset_dict = {"tokens": torch.tensor([], dtype=torch.long)}
    for sample in concat_tokens_dataset:
        sample["tokens"] = sample["tokens"].unsqueeze(0)
        if len(packed_dataset_dict["tokens"]) == 0:
            packed_dataset_dict["tokens"] = sample["tokens"]
            continue
        packed_dataset_dict["tokens"] = torch.cat(
            (packed_dataset_dict["tokens"], sample["tokens"]), dim=0
        )
    return packed_dataset_dict


def prepare_dataset(
    hf_dataset: Dataset,
    dataset_text_field: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    bos_text: str,
    eos_text: str,
    no_wrap: bool = False,
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
    pool = Pool(processes=num_partitions)

    print("Processing dataset partitions in parallel")
    # Process dataset partitions in parallel
    results = pool.map(process_partition, args_list)

    # Terminate the pool of workers
    pool.close()
    pool.join()

    print("Merging the results")
    # Merge the results into a single dataset
    merged_dataset_dict = {"tokens": torch.tensor([], dtype=torch.long)}
    for result in tqdm(results):
        if len(merged_dataset_dict["tokens"]) == 0:
            merged_dataset_dict["tokens"] = result["tokens"]
            continue

        merged_dataset_dict["tokens"] = torch.cat(
            (merged_dataset_dict["tokens"], result["tokens"]), dim=0
        )
    return merged_dataset_dict["tokens"]



if __name__ == "__main__":
    import argparse
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--subset", type=float, default=0.0)
    parser.add_argument("--dataset_text_field", type=str, default="text")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--num_partitions", type=int, default=4)

    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset_name}, subset: {args.subset if args.subset > 0.0 else 'all'}")
    dataset = load_dataset(args.dataset_name, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    if args.subset > 0.0:
        if args.subset > 1.0:
            dataset = dataset.select(range(int(args.subset)))
        else:
            dataset = dataset.select(range(int(len(dataset) * args.subset)))

    prepared_dataset = prepare_dataset(
        hf_dataset=dataset,
        dataset_text_field=args.dataset_text_field,
        tokenizer=tokenizer,
        max_length=args.max_length,
        bos_text=tokenizer.bos_token,
        eos_text=tokenizer.eos_token,
        num_proc=args.num_proc,
        num_partitions=args.num_partitions,
    )

    
    dataloader = DataLoader(
        prepared_dataset,
        batch_size=args.bsz,
        shuffle=False,
        pin_memory=True,
    )

    for batch in dataloader:
        print(batch.shape)
        break
