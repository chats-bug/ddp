from typing import Optional, Union, List
import torch
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerBase
from tqdm import tqdm
from multiprocessing import Pool, RLock
if __name__ == "__main__":
    from concat_dataset import ConcatTokensDataset
else:
    from .concat_dataset import ConcatTokensDataset
import os
import warnings

# ignore future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


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


def iterable_to_list(iterable):
    samples = []
    for sample in tqdm(iterable):
        samples.append(sample)
    return samples


def process_dataset(
    dataset: Dataset,
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
        hf_dataset=dataset,
        dataset_text_field=dataset_text_field,
        tokenizer=tokenizer,
        max_length=max_length,
        bos_text=bos_text,
        eos_text=eos_text,
        no_wrap=no_wrap,
        num_proc=num_proc,
        pre_tokenize=True,
        is_tokenized=is_tokenized,
    )
    samples = iterable_to_list(concat_tokens_dataset)
    samples_tensor = torch.stack([sample["tokens"] for sample in samples], dim=0)
    samples_tensor_dict = {"tokens": samples_tensor}
    return samples_tensor_dict


def process_dataset_wrapper(args):
    return process_dataset(*args)


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
    truncate: int = None,
    disable_tqdm: bool = False,
    split_rows: bool = False,
) -> torch.Tensor:
    num_proc = num_proc or 1
    num_partitions = num_proc if num_partitions is None else num_partitions
    chars_per_token = 5
    words_per_token = 1

    # remove all columns except the dataset_text_field
    hf_dataset = hf_dataset.select_columns(dataset_text_field)

    if truncate is not None:
        truncate = int(truncate)
        hf_dataset = hf_dataset.map(
            lambda x: {
                dataset_text_field: x[dataset_text_field][: truncate * chars_per_token]
            },
            num_proc=num_proc,
            # batched=True,
            # batch_size=1,
        )
    
    # split each row in the dataset into multiple rows
    # each containing chars_per_token * max_length characters
    # this is to avoid the tokenizer from crashing due to
    # memory issues
    if split_rows:

        def process_rows(x):
            chunks = []
            for sentence in x[dataset_text_field]:
                sentence = sentence.split(" ")
                chunks += [
                    " ".join(sentence[i : i + words_per_token * max_length])
                    for i in range(0, len(sentence), words_per_token * max_length)
                ]
            if len(chunks) > 1:
                if len(chunks[-1].split(" ")) < words_per_token * max_length // 2:
                    chunks[-2] += " " + chunks[-1]
                    chunks = chunks[:-1]
            return {dataset_text_field: chunks}

        hf_dataset = hf_dataset.map(
            process_rows,
            num_proc=num_proc,
            batched=True,
            batch_size=1,
        )
    
    hf_dataset = hf_dataset.map(
        lambda x: tokenizer(
            x[dataset_text_field],
            max_length=truncate,
            padding=False,
            truncation=bool(truncate),
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

    split_kwds = []
    for index in tqdm(range(num_partitions), desc="Sharding dataset"):
        shard = hf_dataset.select(range(index*sz, (index+1)*sz))
        split_kwds.append(
            (
                shard,
                dataset_text_field,
                tokenizer,
                max_length,
                bos_text,
                eos_text,
                no_wrap,
                num_proc,
                True,
            )
        )

    print(
        f"Spawning {num_proc} processes for {len(hf_dataset)} objects in slices of {[len(i[0]) for i in split_kwds]}"
    )
    initargs, initializer = None, None
    if not disable_tqdm:
        initargs, initializer = (RLock(),), tqdm.set_lock
    with Pool(num_partitions, initargs=initargs, initializer=initializer) as pool:
        mapped = pool.map(process_dataset_wrapper, split_kwds)
    print(f"Finished {num_partitions} processes.")

    mapped_tokens = torch.cat([i["tokens"] for i in mapped], dim=0)
    return mapped_tokens


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
    parser.add_argument(
        "--tokenizer_name", type=str, default="meta-llama/Llama-2-7b-hf"
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--split_rows", action=argparse.BooleanOptionalAction)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--num_partitions", type=int, default=8)
    parser.add_argument("--save", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default=None)

    args = parser.parse_args()

    print(
        f"Loading dataset: {args.dataset_name}, subset: {args.subset if args.subset > 0.0 else 'all'}"
    )
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
        bos_text="",
        eos_text="",
        num_proc=args.num_proc,
        num_partitions=args.num_partitions,
        disable_tqdm=False,
        split_rows=args.split_rows,
    )

    if args.save:
        path = args.save_dir or "../dumps"
        if not os.path.exists(path):
            os.makedirs(path)
        
        d_name = f"{args.dataset_name.split('/')[-1]}_{args.subset if args.subset > 0.0 else 'all'}_{args.max_length}_prepared.pt"
        path = os.path.join(path, d_name)
        torch.save(
            prepared_dataset,
            path,
        )
        print(f"Saved prepared dataset to {path}")