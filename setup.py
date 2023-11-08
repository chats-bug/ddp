from typing import Optional
import torch
import torch.optim
from rich.console import Console
from rich.table import Table

from datautils import get_dataset, CustomDataset, PoorMansDataLoader
from model import llama_1_7_model, smaller_llama
from modelutils import num_trainable_params
from single_gpu import Trainer
from optimutils import WarmupCosineWithDecay

console = Console()

DATASET_NAME = "togethercomputer/RedPajama-Data-1T-Sample"
VAL_SIZE = 0.001
SEQ_LEN = 512
BATCH_SIZE = 8
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
WANDB_PROJECT = "pytorch-training"
WANDB_RUN = None
ANNEAL_STRATEGY = "cos"
WARMUP_STEPS = 0.1
MIN_LR_FACTOR = 10


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--small_model", action=argparse.BooleanOptionalAction)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
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

    args = parser.parse_args()
    return args


def load_train_objs(
    dataset: str,
    seq_len: int,
    lr: float,
    weight_decay: float,
    smaller_model: bool = False,
):
    console.log("Loading dataset...")
    hf_dataset = get_dataset(dataset, split="train")
    # Shuffle the dataset
    hf_dataset = hf_dataset.shuffle()
    console.log("Splitting dataset...")
    hf_dataset = hf_dataset.train_test_split(test_size=VAL_SIZE)
    train_dataset = hf_dataset["train"]
    val_dataset = hf_dataset["test"]

    console.log("Loading model and tokenizer...")
    if smaller_model:
        packed_obj = smaller_llama(seq_len)
    else:
        packed_obj = llama_1_7_model(seq_len)
    model = packed_obj["model"]
    tokenizer = packed_obj["tokenizer"]

    console.log("Making custom dataset...")
    train_dataset = CustomDataset(
        train_dataset, tokenizer, seq_len=seq_len, dataset_text_field="text"
    )
    val_dataset = CustomDataset(
        val_dataset, tokenizer, seq_len=seq_len, dataset_text_field="text"
    )

    console.log("Setting up optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return train_dataset, val_dataset, model, tokenizer, optimizer


def prepare_dataloader(dataset: CustomDataset, batch_size: int):
    return PoorMansDataLoader(dataset, batch_size=batch_size)


if __name__ == "__main__":
    args = parse_args()

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
        seq_len=args.seq_len,
        lr=args.lr,
        weight_decay=args.weight_decay,
        smaller_model=args.small_model,
    )

    # DataLoaders process the datasets
    # and provide an iterator
    train_dataloader = prepare_dataloader(train_dataset, batch_size=args.batch_size)
    val_dataloader = prepare_dataloader(val_dataset, batch_size=args.batch_size)

    # Number of steps is used for the learning rate scheduler
    num_steps = len(train_dataloader) * args.num_epochs // args.grad_accumulation_steps
    if args.anneal == "cos":
        lr_scheduler = WarmupCosineWithDecay(
            optimizer,
            warmup_steps=int(args.warmup * num_steps // args.num_epochs),
            t_total=num_steps,
            steps_per_epoch=len(train_dataloader),
            eta_max=args.lr,
            eta_min=args.lr / args.min_lr_factor,
        )
    else:
        console.log(f"Invalid anneal strategy: {args.anneal}", style="bold red")
        console.log("Not using a learning rate scheduler")
        lr_scheduler = None

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        lr_schedular=lr_scheduler,
        gpu_id=args.gpu_id,
        device=device,
        eval_every=args.eval_every,
        save_every=args.save_every,
        log_every=args.log_every,
        max_checkpoint_limit=MAX_CHECKPOINT_LIMIT,
        grad_accumulation_steps=args.grad_accumulation_steps,
        torch_dtype=torch_dtype,
        max_grad_norm=args.max_grad_norm,
        report_to=args.report_to,
    )

    # Print all the important model, data and training parameters
    table = Table(title="Training Parameters")
    table.add_column("Parameter")
    table.add_column("Value")

    # Model parameters
    table.add_row("Model Type", str(model.config.model_type))
    table.add_row("Model Size", f"{(num_trainable_params(model) / 1e9):.2f}B")
    table.add_row("Dataset", str(args.dataset_name))
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
    trainer.train(
        max_epochs=args.num_epochs,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
    )
