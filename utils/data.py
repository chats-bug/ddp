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
                d["source"] = data["input_ids"][:-1]
                d["target"] = data["input_ids"][1:]
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
                d["source"] = input_ids[:-1]
                d["target"] = input_ids[1:]
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


if __name__ == "__main__":
    from transformers import AutoTokenizer

    dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    unpacked_dataset = CustomDataset(
        hf_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        seq_length=1024,
        padding="max_length",
        packing=False,
    )
    unpacked_dataloader = PoorMansDataLoader(unpacked_dataset, batch_size=2)

    packed_dataset = CustomDataset(
        hf_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        seq_length=1024,
        packing=True,
    )
    packed_dataloader = PoorMansDataLoader(packed_dataset, batch_size=2)

    for source, target in packed_dataloader:
        print(source.shape, target.shape)
        break

    for source, target in unpacked_dataloader:
        print(source.shape, target.shape)
        break
