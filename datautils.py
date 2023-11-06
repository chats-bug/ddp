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
            self, hf_dataset: HFDataset, tokenizer, seq_len: int = 1024, **kwargs
    ) -> None:
        self.hf_dataset = hf_dataset
        self.dataset = ConstantLengthDataset(
            dataset=hf_dataset, tokenizer=tokenizer, seq_length=seq_len, **kwargs
        )

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __iter__(self):
        # The iterator of the custom dataset is the iterator of the constant length dataset
        # CustomLengthDataset iterator returns {'input_ids': ..., 'labels': ...}
        # We want to return {'source': ..., 'target': ...}
        # where 'source' is the input_ids and 'target' is input_ids shifted by one
        for data in self.dataset:
            d = dict()
            d["source"] = data["input_ids"][:-1]
            d["target"] = data["input_ids"][1:]
            yield d


class PoorMansDataLoader:
    def __init__(self, dataset: CustomDataset, batch_size: int = 8):
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size

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
