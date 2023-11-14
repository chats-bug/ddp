from typing import Optional, Union
import multiprocess as mp
import torch
from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from trl.trainer.utils import ConstantLengthDataset


def get_dataset(
    dataset_name: str,
    subset: Optional[str] = None,
    split: Optional[str] = None,
    **kwargs,
) -> HFDataset:
    data_path = dataset_name
    if subset:
        data_path += f"/{subset}"
    return load_dataset(data_path, split=split, **kwargs)


def prepare_dataset(
    hf_dataset: HFDataset,
    tokenizer,
    dataset_text_field: Optional[str] = None,
    seq_length: int = 1024,
    packing: bool = True,
    truncation: bool = False,
    padding: Optional[str] = None,
    formatting_func=None,
    batch_size: int = 8,
    *args,
    **kwargs,
) -> Union[HFDataset, ConstantLengthDataset]:
    if packing:
        kwargs.pop("dataset_num_proc")
        return ConstantLengthDataset(
            dataset=hf_dataset,
            tokenizer=tokenizer,
            seq_length=seq_length,
            dataset_text_field=dataset_text_field,
            *args,
            **kwargs,
        )
    else:
        use_formatting_func = formatting_func is not None and dataset_text_field is not None
        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field] if not use_formatting_func else formatting_func(element),
                truncation=truncation,
                padding=padding,
                max_length=seq_length,
                return_overflowing_tokens=False,
                return_length=False,
                return_tensors="pt",
            )

            if use_formatting_func:
                if not isinstance(formatting_func(element), list):
                    raise ValueError(
                        "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                    )

            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}
        
        max_workers = None
        if "dataset_num_proc" in kwargs:
            max_workers = kwargs.pop("dataset_num_proc")
        print(f"Using {max_workers} workers to tokenize the dataset")
        return hf_dataset.map(
            tokenize,
            batched=True,
            remove_columns=hf_dataset.column_names,
            num_proc=max_workers,
            batch_size=batch_size,
            *args,
            **kwargs,
        )


if __name__ == "__main__":
    dataset_name = "roneneldan/TinyStories"
    dataset = get_dataset(dataset_name, split="train")
    dataset = dataset.select(range(100_000))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = prepare_dataset(
        dataset, 
        tokenizer, 
        dataset_text_field="text", 
        seq_length=1024, 
        packing=False, 
        truncation=True, 
        padding="max_length", 
        batch_size=8, 
        dataset_num_proc=16
    )

    for batch in dataset:
        print(batch)
        print(batch['input_ids'].shape)
        break

