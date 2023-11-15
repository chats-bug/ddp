import os
from typing import Dict, Iterable, Union

import torch
from datasets import Dataset
from rich.console import Console
from transformers import PreTrainedTokenizerBase

console = Console()


class ConcatTokensDataset:
    """An IterableDataset that returns token samples.

    Returns dicts of {'tokens': torch.tensor} where tokens is a tensor of concatenated tokens
    """

    def __init__(
        self,
        hf_dataset: Union[Dataset, Dict[str, list]],
        dataset_text_field: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        bos_text: str,
        eos_text: str,
        no_wrap: bool,
        num_proc: int | None = None,
        pre_tokenize: bool = True,
        is_tokenized: bool = False,
        formatting_func=None,
    ):
        self.hf_dataset = hf_dataset
        self.dataset_text_field = dataset_text_field
        self.tokenizer = tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.max_length = max_length
        self.bos_text = bos_text
        self.eos_text = eos_text
        self.should_wrap = not no_wrap
        self.formatting_func = formatting_func
        self.pre_tokenize = pre_tokenize
        self.is_tokenized = is_tokenized

        if isinstance(self.hf_dataset, dict):
            self.hf_dataset = Dataset.from_dict(hf_dataset)

        self.bos_tokens = self.tokenizer(
            self.bos_text,
            truncation=False,
            padding=False,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]
        # flatten the bos_tokens
        self.bos_tokens = self.bos_tokens.view(-1)
        if len(self.bos_tokens) > 1:
            console.log(
                f"You specified --concat_tokens with --bos_text, but your BOS text is not tokenizing to one token\
                , instead we got {self.bos_tokens}. Quit if this was in error.",
                style="yellow",
            )

        self.eos_tokens = self.tokenizer(
            self.eos_text,
            truncation=False,
            padding=False,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]
        # flatten the eos_tokens
        self.eos_tokens = self.eos_tokens.view(-1)
        if len(self.eos_tokens) > 1:
            console.log(
                f"You specified --concat_tokens with --eos_text, but your EOS text is not tokenizing to one token\
                , instead we got {self.eos_tokens}. Quit if this was in error.",
                style="yellow",
            )

        eos_text_provided = self.eos_text != ""
        bos_text_provided = self.bos_text != ""
        test_text = self.tokenizer("")
        if len(test_text["input_ids"]) > 0 and (eos_text_provided or bos_text_provided):
            message = (
                "both eos and bos"
                if eos_text_provided and bos_text_provided
                else ("eos_text" if eos_text_provided else "bos_text")
            )
            console.log(
                f"The provided tokenizer adds special tokens, but you also specified {message}. This may result "
                + "in duplicated special tokens. Please be sure this is what you intend.",
                style="yellow",
            )

        def tokenize(batch, dataset_text_field, formatting_func=None, *args, **kwargs):
            return tokenizer(
                batch[dataset_text_field]
                if not formatting_func
                else formatting_func(batch),
                padding=False,
                truncation=False,
                *args,
                **kwargs,
            )

        if pre_tokenize:
            if is_tokenized or hasattr(self.hf_dataset, "input_ids"):
                self.encoded_dataset = self.hf_dataset
            else:
                self.encoded_dataset = self.hf_dataset.map(
                    tokenize,
                    dataset_text_field=self.dataset_text_field,
                    formatting_func=formatting_func,
                    batched=True,
                    batch_size=1,
                    num_proc=num_proc,
                    return_tensors="pt",
                )

    def __iter__(self) -> Iterable[Dict[str, torch.Tensor]]:
        buffer = torch.tensor([])
        dataset_to_iter = (
            self.encoded_dataset
            if hasattr(self, "encoded_dataset")
            else self.hf_dataset
        )
        for sample in dataset_to_iter:
            if "input_ids" not in sample:
                if self.is_tokenized:
                    raise ValueError(
                        "You specified --concat_tokens, but your dataset does not have input_ids. "
                        + "Please set --is_tokenized to False"
                    )
                encoded = self.tokenizer(
                    sample[self.dataset_text_field],
                    padding=False,
                    truncation=False,
                    return_tensors="pt",
                )
            else:
                encoded = sample
            iids = encoded["input_ids"]
            if not isinstance(iids, torch.Tensor):
                iids = torch.tensor(iids)
            buffer = torch.cat((buffer, self.bos_tokens, iids, self.eos_tokens), dim=0)
            while len(buffer) >= self.max_length:
                concat_sample = buffer[: self.max_length]
                buffer = buffer[self.max_length :] if self.should_wrap else []
                yield {"tokens": concat_sample}
