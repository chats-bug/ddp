import os

import torch
import torch.multiprocessing as mp
import torch.optim
from rich.console import Console
from rich.table import Table
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, DistributedSampler

from model import llama_1_7_model, smaller_llama
from single_gpu import Trainer
from utils import (
    get_dataset,
    num_trainable_params,
    WarmupCosineWithDecay,
    prepare_dataset,
)

console = Console()

DATASET_NAME = "togethercomputer/RedPajama-Data-1T-Sample"
VAL_SIZE = 0.01
SEQ_LENGTH = 512
BATCH_SIZE = 8
DATASET_TEXT_FIELD = "text"
LR = 3e-4
WEIGHT_DECAY = 0.1
NUM_EPOCHS = 1
EVAL_EVERY = 500
SAVE_EVERY = 500
MAX_CHECKPOINT_LIMIT = 5
LOG_EVERY = 1
GRAD_ACCUMULATION_STEPS = 4
TORCH_DTYPE = "fp16"
GPU_ID = 0
SMALLER = False
MAX_GRAD_NORM = None
REPORT_TO = None
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", None)
WANDB_RUN = None
ANNEAL_STRATEGY = "cos"
WARMUP_STEPS = 0.1
MIN_LR_FACTOR = 10
DATASET_NUM_PROC = None


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "1234"

    # initialize the process group
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--small_model", action=argparse.BooleanOptionalAction)
    parser.add_argument("--seq_len", type=int, default=SEQ_LENGTH)
    parser.add_argument("--dataset_text_field", type=str, default=DATASET_TEXT_FIELD)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dataset_num_proc", type=int, default=DATASET_NUM_PROC)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--warmup", type=float, default=WARMUP_STEPS)
    parser.add_argument("--min_lr_factor", type=float, default=MIN_LR_FACTOR)
    parser.add_argument("--anneal", type=str, default=ANNEAL_STRATEGY)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--max_grad_norm", type=float, default=MAX_GRAD_NORM)
    parser.add_argument("--dataset_name", type=str, default=DATASET_NAME)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--eval_every", type=int, default=EVAL_EVERY)
    parser.add_argument("--save_every", type=int, default=SAVE_EVERY)
    parser.add_argument("--log_every", type=int, default=LOG_EVERY)
    parser.add_argument("--report_to", type=str, default=REPORT_TO)
    parser.add_argument("--wandb_project", type=str, default=WANDB_PROJECT)
    parser.add_argument("--wandb_run", type=str, default=WANDB_RUN)
    parser.add_argument(
        "--grad_accumulation_steps", type=int, default=GRAD_ACCUMULATION_STEPS
    )
    parser.add_argument("--torch_dtype", type=str, default=TORCH_DTYPE)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=GPU_ID)
    parser.add_argument("--ddp", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    return args


def load_train_objs(
    dataset: str,
    dataset_text_field: str,
    seq_length: int,
    lr: float,
    weight_decay: float,
    smaller_model: bool = False,
    dataset_num_proc: int = None,
):
    hf_dataset = get_dataset(dataset, split="train")
    hf_dataset = hf_dataset.train_test_split(test_size=VAL_SIZE)
    train_dataset = hf_dataset["train"]
    val_dataset = hf_dataset["test"]

    # Add special tokens here
    # - PAD token is a special token that is used for padding
    special_tokens = {"pad_token": "[PAD]"}
    if smaller_model:
        packed_obj = smaller_llama(seq_length, special_tokens)
    else:
        packed_obj = llama_1_7_model(seq_length, special_tokens)
    model = packed_obj["model"]
    tokenizer = packed_obj["tokenizer"]

    train_dataset = prepare_dataset(
        hf_dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=seq_length,
        dataset_text_field=dataset_text_field,
        bos_text=tokenizer.bos_token,
        eos_text=tokenizer.eos_token,
        num_proc=dataset_num_proc,
        num_partitions=dataset_num_proc,
    )
    val_dataset = prepare_dataset(
        hf_dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=seq_length,
        dataset_text_field=dataset_text_field,
        bos_text=tokenizer.bos_token,
        eos_text=tokenizer.eos_token,
        num_proc=dataset_num_proc,
        num_partitions=dataset_num_proc,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return train_dataset, val_dataset, model, tokenizer, optimizer


def main(rank: int, world_size: int, args):
    ddp_setup(rank, world_size)

    # Setting the device
    # Using GPUs significantly speeds up the training process
    # as well as inference
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    device = args.device or device

    # Setting the torch dtype
    # The default is 32-bit floating point
    # Using mixed precision training can speed up the training process
    # Mixed precision automatically casts the model weights to 16-bit floating point
    # and rescales the gradients to 32-bit floating point
    # NOTE:
    # `mps` (Apple Silicon) currently doesn't support mixed precision training
    if args.torch_dtype == "fp32":
        torch_dtype = torch.float32
    # Using fp16 reduces training time by a substantial margin
    # All `cuda` GPUs support fp16
    elif args.torch_dtype == "fp16":
        torch_dtype = torch.float16
    # bf16: Brain Floating Point
    # Not supported by all GPUs
    # Only Ampere GPUs support bf16, namely A100 and A6000
    elif args.torch_dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        console.log(f"Invalid torch_dtype: {args.torch_dtype}")
        console.log("Setting torch_dtype to fp16")
        torch_dtype = torch.float16

    # Training Objects are loaded here
    # Datasets(train and validation)
    # model, tokenizer and optimizer
    train_dataset, val_dataset, model, tokenizer, optimizer = load_train_objs(
        dataset=args.dataset_name,
        seq_length=args.seq_len,
        dataset_text_field=args.dataset_text_field,
        lr=args.lr,
        weight_decay=args.weight_decay,
        smaller_model=args.small_model,
        dataset_num_proc=args.dataset_num_proc,
    )

    sampler = None
    if args.ddp:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    # DataLoaders process the datasets
    # and provide an iterator
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
    )

    # Number of steps is used for the learning rate scheduler
    num_steps = (
        len(train_dataloader)
        * args.num_epochs
        // (args.grad_accumulation_steps * world_size)
    )
    if args.anneal == "cos":
        lr_scheduler = WarmupCosineWithDecay(
            optimizer,
            warmup_steps=int(args.warmup * num_steps // args.num_epochs),
            t_total=num_steps,
            steps_per_epoch=num_steps // args.num_epochs,
            eta_max=args.lr,
            eta_min=args.lr / args.min_lr_factor,
        )
    else:
        console.log(f"Invalid anneal strategy: {args.anneal}", style="bold red")
        console.log("Not using a learning rate scheduler")
        lr_scheduler = None

    if rank == 0:
        log_arguments(
            args, model, optimizer, lr_scheduler, device, torch_dtype, num_steps
        )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        lr_schedular=lr_scheduler,
        gpu_id=rank,
        device=device,
        eval_every=args.eval_every,
        save_every=args.save_every,
        log_every=args.log_every,
        max_checkpoint_limit=MAX_CHECKPOINT_LIMIT,
        grad_accumulation_steps=args.grad_accumulation_steps,
        torch_dtype=torch_dtype,
        max_grad_norm=args.max_grad_norm,
        report_to=args.report_to,
        world_size=world_size,
    )

    trainer.train(
        max_epochs=args.num_epochs,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
    )

    destroy_process_group()


def log_arguments(args, model, optimizer, lr_scheduler, device, torch_dtype, num_steps):
    # Print all the important model, data and training parameters
    table = Table(title="Training Parameters")
    table.add_column("Parameter")
    table.add_column("Value")

    # Model parameters
    table.add_row("Model Type", str(model.config.model_type))
    table.add_row("Model Size", f"{(num_trainable_params(model) / 1e9):.2f}B")
    table.add_row("Dataset", str(args.dataset_name))
    table.add_row("Dataset Text Field", str(args.dataset_text_field))
    table.add_row("Sequence Length", str(args.seq_len))

    table.add_row("", "")
    table.add_row("-------------------", "-------------------")
    # Optimizer parameters
    table.add_row("Optim", "AdamW")
    table.add_row("Beta 1", str(optimizer.defaults["betas"][0]))
    table.add_row("Beta 2", str(optimizer.defaults["betas"][1]))
    table.add_row("Epsilon", str(optimizer.defaults["eps"]))
    table.add_row("Weight Decay", str(optimizer.defaults["weight_decay"]))

    table.add_row("", "")
    table.add_row("-------------------", "-------------------")
    # Learning rate scheduler parameters
    if isinstance(lr_scheduler, WarmupCosineWithDecay):
        table.add_row("Anneal Strategy", "Cosine")
        table.add_row("Warmup Steps", str(lr_scheduler.warmup_steps))
        table.add_row("Eta Max", str(lr_scheduler.eta_max))
        table.add_row("Eta Min", str(lr_scheduler.eta_min))

    table.add_row("", "")
    table.add_row("-------------------", "-------------------")
    # Training parameters
    table.add_row("Batch Size", str(args.batch_size))
    table.add_row("Max Grad Norm", str(args.max_grad_norm))
    table.add_row("Grad Accumulation Steps", str(args.grad_accumulation_steps))
    table.add_row("Number of Steps", f"{num_steps:,}")
    table.add_row("Number of Epochs", str(args.num_epochs))

    table.add_row("", "")
    table.add_row("-------------------", "-------------------")
    # Logging and saving parameters
    table.add_row("Eval Every", str(args.eval_every))
    table.add_row("Save Every", str(args.save_every))
    table.add_row("Log Every", str(args.log_every))

    table.add_row("", "")
    # Device and torch dtype
    table.add_row("Torch Dtype", str(args.torch_dtype))
    table.add_row("Device", str(device))
    table.add_row("GPU ID", str(args.gpu_id))

    console.print("Training info:")
    console.print(table)

    console.log(f"Starting training with {device} using {torch_dtype}...")


if __name__ == "__main__":
    args = parse_args()
    world_size = torch.cuda.device_count()
    if args.ddp:
        mp.spawn(
            main,
            args=(
                world_size,
                args,
            ),
            nprocs=world_size,
        )
    else:
        main(args, rank=0, world_size=1)
