from datasets import load_dataset, Dataset as HFDataset
from typing import Optional, Union, Dict, List, Tuple
from torch.utils.data import Dataset
from trl.trainer.utils import ConstantLengthDataset
import torch


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


class CustomDataset(Dataset):
    def __init__(
        self,
        hf_dataset: HFDataset,
        tokenizer,
        dataset_text_field: Optional[str] = None,
        seq_length: int = 1024,
        packing: bool = True,
        trunctation: bool = False,
        padding: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.dataset_text_field = dataset_text_field
        self.seq_length = seq_length
        self.packing = packing
        self.trunctation = trunctation
        self.padding = padding
        self.kwargs = kwargs

        if packing:
            self.dataset = ConstantLengthDataset(
                dataset=hf_dataset,
                tokenizer=tokenizer,
                seq_length=seq_length,
                dataset_text_field=dataset_text_field,
                **kwargs,
            )

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __iter__(self):
        # The iterator of the custom dataset is the iterator of the constant length dataset
        # CustomLengthDataset iterator returns {'input_ids': ..., 'labels': ...}
        # We want to return {'source': ..., 'target': ...}
        # where 'source' is the input_ids and 'target' is input_ids shifted by one

        if self.packing:
            for data in self.dataset:
                d = dict()
                # The input_ids are already shifted by one in huggingface's model forward
                # >>> source = input_ids[:-1] and target = input_ids[1:]
                # So we don't need to shift them here
                # This behavior must be consistent with the model forward
                # Change the model forward if you want to shift the input_ids here
                # source and targets are shifted by one like below
                d["source"] = data["input_ids"]
                d["target"] = data["input_ids"]
                yield d
        else:
            for data in self.hf_dataset:
                d = dict()
                input_ids = self.tokenizer(
                    data[self.dataset_text_field],
                    max_length=self.seq_length,
                    padding=self.padding,
                    truncation=self.trunctation,
                    return_tensors="pt",
                )["input_ids"][0]
                # source and targets are shifted by one like below
                # >>> source = input_ids[:-1] and target = input_ids[1:]
                # The input_ids are already shifted by one in huggingface's model forward
                # So we don't need to shift them here
                # This behavior must be consistent with the model forward
                # Change the model forward if you want to shift the input_ids here
                d["source"] = input_ids
                d["target"] = input_ids
                yield d


class PoorMansDataLoader:
    def __init__(self, dataset: CustomDataset, batch_size: int = 8):
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size

    def get_batch_size(self) -> int:
        return self.batch_size

    def __iter__(self):
        counter = 1
        batch = [[], []]
        for data in self.dataset:
            batch[0].append(data["source"])
            batch[1].append(data["target"])
            if counter % self.batch_size == 0:
                # convert batch to torch.tensor
                batch[0] = torch.stack(batch[0])
                batch[1] = torch.stack(batch[1])
                yield batch
                batch = [[], []]
            counter += 1
