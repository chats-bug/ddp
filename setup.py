import os
import torch
import torch.multiprocessing as mp
import torch.optim
from rich.console import Console
from rich.table import Table
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler, random_split
import warnings
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from trainer_fsdp import Trainer
from model import llama_1_7_model, smaller_llama, original_llama
from utils import (
    get_dataset,
    num_trainable_params,
    WarmupCosineWithDecay,
    prepare_dataset,
    prepare_model_for_fsdp,
)


# Suppressing future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
console = Console()

DATASET_NAME = "togethercomputer/RedPajama-Data-1T-Sample"
VAL_SIZE = 0.0001
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


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--small_model", action=argparse.BooleanOptionalAction)
    parser.add_argument("--seq_len", type=int, default=SEQ_LENGTH)
    parser.add_argument("--dataset_text_field", type=str, default=DATASET_TEXT_FIELD)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--subset", type=float, default=0.0)
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
    parser.add_argument("--local_path", type=str, default=None)
    parser.add_argument("--fsdp", action=argparse.BooleanOptionalAction)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction)
    parser.add_argument("--original_llama", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    return args


def load_train_objs(
    dataset: str,
    dataset_text_field: str,
    seq_length: int,
    bsz: int,
    smaller_model: bool = False,
    orig_llama: bool = False,
    dataset_num_proc: int = None,
    subset: float = 0.0,
    local_path: str = None,
):
    if local_path:
        path = local_path
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")
        console.log(f"Loading dataset from {path}")
        dataset = torch.load(path)
        # cut the seq len to the desired length
        dataset = dataset[:, :seq_length]

        train_dataset = dataset[:int((1-VAL_SIZE)*len(dataset)), :]
        val_dataset = dataset[int((1-VAL_SIZE)*len(dataset)):, :]

        # # Cut the train and val datasets to the closest multiple of the batch size
        # # This is done to avoid the `IndexError: Caught IndexError in DataLoader worker process 0.` error in the DataLoader
        train_dataset = train_dataset[:(len(train_dataset) - (len(train_dataset) % bsz)), :]
        val_dataset = val_dataset[:(len(val_dataset) - (len(val_dataset) % bsz)), :]
        train_dataset = TensorDataset(train_dataset)
        val_dataset = TensorDataset(val_dataset)

    else:
        hf_dataset = get_dataset(dataset, split="train")
        if subset > 0.0:
            if subset > 1.0:
                hf_dataset = hf_dataset.select(range(int(subset)))
            else:
                hf_dataset = hf_dataset.select(range(int(subset * len(hf_dataset))))
        hf_dataset = hf_dataset.train_test_split(test_size=VAL_SIZE)
        train_dataset = hf_dataset["train"]
        val_dataset = hf_dataset["test"]

    # Add special tokens here
    # - PAD token is a special token that is used for padding
    special_tokens = {"pad_token": "[PAD]"}
    if orig_llama:
        packed_obj = original_llama(seq_length, special_tokens)
    elif smaller_model:
        packed_obj = smaller_llama(seq_length, special_tokens)
    else:
        packed_obj = llama_1_7_model(seq_length, special_tokens)
    model = packed_obj["model"]
    tokenizer = packed_obj["tokenizer"]

    if not local_path:
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
            hf_dataset=val_dataset,
            tokenizer=tokenizer,
            max_length=seq_length,
            dataset_text_field=dataset_text_field,
            bos_text=tokenizer.bos_token,
            eos_text=tokenizer.eos_token,
            num_proc=dataset_num_proc,
            num_partitions=dataset_num_proc,
        )
    
    return train_dataset, val_dataset, model, tokenizer


def main(args):
    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    # ddp = False
    if ddp:
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        init_process_group(backend="nccl", rank=ddp_rank, world_size=ddp_world_size)
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert args.grad_accumulation_steps % ddp_world_size == 0
        args.grad_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        ddp_local_rank = args.gpu_id
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
    torch.manual_seed(1337 + seed_offset)

    # Training Objects are loaded here
    # Datasets(train and validation)
    # model, tokenizer and optimizer
    train_dataset, val_dataset, model, tokenizer = load_train_objs(
        dataset=args.dataset_name,
        seq_length=args.seq_len,
        dataset_text_field=args.dataset_text_field,
        bsz=args.batch_size,
        smaller_model=args.small_model,
        orig_llama=args.original_llama,
        dataset_num_proc=args.dataset_num_proc,
        subset=args.subset,
        local_path=args.local_path,
    )
    model.to(device)

    # Setting the torch dtype
    # The default is 32-bit floating point
    # Using mixed precision training can speed up the training process
    # Mixed precision automatically casts the model weights to 16-bit floating point
    # and rescales the gradients to 32-bit floating point
    # NOTE:
    # `mps` (Apple Silicon) currently doesn't support mixed precision training
    if args.torch_dtype == "fp32":
        args.torch_dtype = torch.float32
    # Using fp16 reduces training time by a substantial margin
    # All `cuda` GPUs support fp16
    elif args.torch_dtype == "fp16":
        args.torch_dtype = torch.float16
    # bf16: Brain Floating Point
    # Not supported by all GPUs
    # Only Ampere GPUs support bf16, namely A100 and A6000
    elif args.torch_dtype == "bf16":
        args.torch_dtype = torch.bfloat16
    else:
        console.log(f"Invalid torch_dtype: {args.torch_dtype}")
        console.log("Setting torch_dtype to fp16")
        args.torch_dtype = torch.float16
    
    if args.compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)  # requires PyTorch 2.0
    if ddp:
        prefix = "_orig_mod." if compile else ""
        model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
        if args.fsdp:
            model = prepare_model_for_fsdp(
                model,
                transformer_layer_cls=LlamaDecoderLayer,
                torch_dtype=args.torch_dtype,
                backward_prefetch="pre",
                sharding_strategy="grad_op",
            )
        else:
            model = DDP(model, device_ids=[ddp_local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train(
        rank=ddp_local_rank,
        world_size=ddp_world_size,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        args=args,
        ddp=ddp,
        master_process=master_process
    )

    if ddp:
        destroy_process_group()


def train(
    rank: int,
    world_size: int,
    train_dataset,
    val_dataset,
    model,
    tokenizer,
    optimizer,
    args,
    master_process,
    ddp
):
    # Setting the device
    # Using GPUs significantly speeds up the training process
    # as well as inference
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    device = args.device or device

    train_sampler = None
    val_sampler = None
    if ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_kwargs = {'batch_size': args.batch_size, 'sampler': train_sampler}
    val_kwargs = {'batch_size': args.batch_size, 'sampler': val_sampler}
    cuda_kwargs = {'num_workers': 2, 'pin_memory': True, 'shuffle': False}
    # DataLoaders process the datasets
    # and provide an iterator
    train_dataloader = DataLoader(
        train_dataset,
        **train_kwargs,
        **cuda_kwargs,
    )
    val_dataloader = DataLoader(
        val_dataset,
        **val_kwargs,
        **cuda_kwargs,
    )

    # Number of steps is used for the learning rate scheduler
    num_steps = len(train_dataloader) * args.num_epochs // args.grad_accumulation_steps
    if master_process:
        console.print(
            f"Train dataloader length: {len(train_dataset)}(train_dataset) // {args.batch_size}(batch_size) * {world_size}(world_size) = {len(train_dataloader)}"
        )
        console.print(
            f"Val dataloader length: {len(val_dataset)}(val_dataset) // {args.batch_size}(batch_size) * {world_size}(world_size) = {len(val_dataloader)}"
        )
        console.print(
            f"Number of training steps: {len(train_dataloader)}(train_dataloader) * {args.num_epochs}(num_epochs) // {args.grad_accumulation_steps}(grad_accumulation_steps) = {num_steps}"
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
        if master_process:
            console.log(f"Invalid anneal strategy: {args.anneal}", style="bold red")
            console.log("Not using a learning rate scheduler")
        lr_scheduler = None

    if master_process:
        log_arguments(
            args, model, optimizer, lr_scheduler, device, args.torch_dtype, num_steps
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
        torch_dtype=args.torch_dtype,
        max_grad_norm=args.max_grad_norm,
        report_to=args.report_to,
        world_size=world_size,
        dist=ddp,
        master_process=master_process,
    )

    trainer.train(
        max_epochs=args.num_epochs,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
    )


def log_arguments(args, model, optimizer, lr_scheduler, device, torch_dtype, num_steps):
    # Print all the important model, data and training parameters
    table = Table(title="Training Parameters")
    table.add_column("Parameter")
    table.add_column("Value")

    # Model parameters
    # table.add_row("Model Type", str(model.config.model_type))
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
    main(args)