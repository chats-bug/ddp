import os
from typing import Union, Optional, Any

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
import wandb
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import format_float_to_str

console = Console()


class Trainer:
    checkpoint_val_losses: list[dict[str, Any]] = []

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        device: Union[str, torch.device],
        eval_every: int,
        save_every: int,
        max_checkpoint_limit: int = 5,
        lr_schedular: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        log_every: int = 1,
        grad_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = None,
        torch_dtype: torch.dtype = torch.float32,
        output_dir: Optional[str] = None,
        report_to: Optional[str] = None,
        world_size: int = 1,
        ddp: bool = False,
    ):
        assert (
            save_every % eval_every == 0
        ), "Can only save checkpoints at eval intervals"
        assert (
            max_grad_norm is None or max_grad_norm > 0.0
        ), "Max grad norm must be greater than 0.0"

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.grad_accumulation_steps = grad_accumulation_steps
        self.gpu_id = gpu_id
        self.eval_every = eval_every
        self.save_every = save_every
        self.log_every = log_every
        self.max_checkpoint_limit = max_checkpoint_limit
        self.torch_dtype = torch_dtype
        self.device = device
        self.lr_schedular = lr_schedular
        self.max_grad_norm = max_grad_norm
        self.output_dir = output_dir
        self.report_to = report_to
        self.world_size = world_size
        if self.device != "cuda":
            # If the device is not cuda, then the gpu_id is the device
            # This because the device is either "cpu" or probably a "mps" device
            self.gpu_id = self.device  # type: ignore
        self.ddp = ddp

        # Set the
        # torch.set_default_device(self.device)

        self.scaler = None
        if self.torch_dtype != torch.float32:
            # assert self.device in ["cpu", "cuda", torch.device("cpu"), torch.device("cuda")], \
            #     "Mixed precision training is only supported on CPU and CUDA devices"
            if self.device not in [
                "cpu",
                "cuda",
            ]:
                # print a warning statement in red
                console.log(
                    "[bold red]Mixed precision training is only supported on CPU and CUDA devices[/bold red]"
                )
                console.log(
                    f"Setting the default torch type to the {self.torch_dtype}",
                    style="bold red",
                )
                # set the default torch type to the specified torch type
                torch.set_default_dtype(self.torch_dtype)
            else:
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)

        # Wrap the model with DDP if the device is cuda
        self.model.to(self.gpu_id)
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.gpu_id])

        if not self.output_dir:
            self.output_dir = "output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _run_batch(
        self, source, target, step=1, attention_mask: Optional[torch.Tensor] = None
    ):
        if self.scaler:
            with torch.autocast(device_type=self.device, dtype=self.torch_dtype):
                output = self.model(
                    source, labels=target, attention_mask=attention_mask
                )
                loss = output.loss
                # Normalize the loss for the grad_accumulation_steps
                loss = loss / self.grad_accumulation_steps
            self.scaler.scale(loss).backward()
            if step % self.grad_accumulation_steps == 0:
                if self.max_grad_norm:
                    # Un-scale the gradients of optimizer's assigned parameters in-place
                    self.scaler.unscale_(self.optimizer)
                    # Since the gradients of optimizer's assigned parameters are now unscaled, clips as usual.
                    # You may use the same value for max_norm here as you would without gradient scaling.
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.max_grad_norm
                    )

                self.scaler.step(self.optimizer)
                if self.lr_schedular:
                    self.lr_schedular.step()
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            output = self.model(source, labels=target)
            loss = output.loss
            # Normalize the loss for the grad_accumulation_steps
            loss = loss / self.grad_accumulation_steps
            loss.backward()
            if step % self.grad_accumulation_steps == 0:
                # No need to unscale gradients here
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.max_grad_norm
                    )
                self.optimizer.step()
                if self.lr_schedular:
                    self.lr_schedular.step()
                self.optimizer.zero_grad()
        return loss.item() * self.grad_accumulation_steps

    def _run_eval(self, source, target):
        self.model.eval()
        with torch.no_grad():
            output = self.model(source, labels=target)
            loss = output.loss
        return loss

    def _run_epoch(self, epoch: int):
        train_bsz = self.train_dataloader.batch_size
        val_bsz = self.val_dataloader.batch_size
        total_steps = len(self.train_dataloader) // self.grad_accumulation_steps
        step = 1

        # Change the progress bar to add the following things:
        # 1. Add the current step
        # 2. Add the total number of steps
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),
            "•",
            TextColumn("[progress.percentage]{task.completed}"),
            "•",
            TextColumn(f"[progress.percentage]{total_steps:,}"),
        ) as progress:
            training_task = progress.add_task(
                "Training...",
                total=total_steps,
            )

            loss = 0
            for data in self.train_dataloader:
                data = data.to(self.gpu_id)

                opt_step = step // self.grad_accumulation_steps
                self.model.train()
                loss += (
                    self._run_batch(source=data, target=data, step=step)
                    / self.grad_accumulation_steps
                )

                if step % (self.log_every * self.grad_accumulation_steps) == 0:
                    # Write to wandb
                    if self.report_to == "wandb":
                        metrics = {
                            "train/loss": loss,
                            "train/epoch": epoch,
                            "train/global_step": opt_step,
                            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                        }
                        wandb.log(metrics)
                    formatted_lr = format_float_to_str(
                        self.optimizer.param_groups[0]["lr"]
                    )
                    progress.console.log(
                        f"Epoch: {epoch}, Step: {opt_step}, Learning Rate: {formatted_lr}, Loss: {loss:.3f}"
                    )
                    progress.update(training_task, advance=1, step=step)

                if step % self.grad_accumulation_steps == 0:
                    loss = 0

                if step % (self.eval_every * self.grad_accumulation_steps) == 0:
                    self.model.eval()
                    val_loss = 0
                    progress.console.log(
                        f"Running validation on {len(self.val_dataloader)*val_bsz} samples"
                    )
                    for ev_source, ev_target in self.val_dataloader:
                        ev_source = ev_source.to(self.gpu_id)
                        ev_target = ev_target.to(self.gpu_id)
                        val_loss += self._run_eval(ev_source, ev_target)
                    val_loss /= len(self.val_dataloader)  # type: ignore
                    # Write to wandb
                    if self.report_to == "wandb":
                        metrics = {
                            "eval/loss": val_loss,
                        }
                        wandb.log(metrics)

                    progress.console.log("=" * 80)
                    progress.console.log(
                        f"Epoch: {epoch}, Step: {opt_step}, Val loss: {val_loss:.4f}"
                    )
                    if (
                        step % (self.save_every * self.grad_accumulation_steps) == 0
                        and self.gpu_id == 0
                    ):
                        path = self._save_checkpoint(
                            epoch, opt_step, progress.console.log
                        )
                        if path:
                            self.checkpoint_val_losses.append(
                                {"ckp_path": path, "val_loss": val_loss}
                            )
                            # Sort the checkpoints by val_loss
                            self._sort_checkpoints_by_val_loss()
                            # progress.console.log(self.checkpoint_val_losses)
                        if len(self.checkpoint_val_losses) > self.max_checkpoint_limit:
                            progress.console.log(
                                "Comparing all checkpoints since the max checkpoint limit has been reached"
                            )
                            # Delete the worst checkpoint
                            # Since the checkpoints are sorted by val_loss in ascending order,
                            # the last checkpoint is the worst checkpoint
                            worst_checkpoint_path: str = self.checkpoint_val_losses[-1][
                                "ckp_path"
                            ]
                            path = self._delete_checkpoint(worst_checkpoint_path)
                            if path:
                                progress.console.log(
                                    f"Deleted checkpoint at {self.checkpoint_val_losses.pop()['ckp_path']}"
                                )
                    progress.console.log("=" * 80)

                step += 1

    def _save_checkpoint(self, epoch: int, step: int, log_fn):
        ckp = self.model.module.state_dict() if self.ddp else self.model.state_dict()
        ckp_path = f"{self.output_dir}/ckp_epoch_{epoch}_step_{step}.pt"
        try:
            torch.save(ckp, ckp_path)
            log_fn(f"Saved checkpoint at ckp_epoch_{epoch}_step_{step}.pt")
            return ckp_path
        except Exception as e:
            log_fn(f"Error saving checkpoint: {e}")
            return None

    def _delete_checkpoint(self, ckp_path: str):
        if os.path.exists(ckp_path):
            os.remove(ckp_path)
            return ckp_path
        return None

    def _sort_checkpoints_by_val_loss(self):
        self.checkpoint_val_losses = sorted(
            self.checkpoint_val_losses, key=lambda x: x["val_loss"]
        )

    def train(
        self,
        max_epochs: int,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
    ):
        # Initialize wandb
        if self.report_to == "wandb":
            wandb_config = {
                "train_batch_size": self.train_dataloader.get_batch_size(),
                "val_batch_size": self.val_dataloader.get_batch_size(),
                "eval_every": self.eval_every,
                "save_every": self.save_every,
                "log_every": self.log_every,
                "grad_accumulation_steps": self.grad_accumulation_steps,
                "max_grad_norm": self.max_grad_norm,
                "torch_dtype": self.torch_dtype,
                "device": self.device,
                "lr": self.optimizer.param_groups[0]["lr"],
                "weight_decay": self.optimizer.param_groups[0]["weight_decay"],
                "model_config": self.model.config,
                "epochs": max_epochs,
            }
            if wandb_project:
                if wandb_run_name:
                    wandb.init(
                        project=wandb_project, name=wandb_run_name, config=wandb_config
                    )
                else:
                    wandb.init(project=wandb_project, config=wandb_config)
            else:
                wandb.init(config=wandb_config)

        for epoch in range(max_epochs):
            if self.ddp and self.gpu_id != 0:
                self._run_epoch_no_log(epoch)
            else:
                self._run_epoch(epoch)

        # Finish wandb
        if self.report_to == "wandb":
            wandb.finish()

    def _run_epoch_no_log(self, epoch: int):
        train_bsz = self.train_dataloader.batch_size
        val_bsz = self.val_dataloader.batch_size
        total_steps = len(self.train_dataloader) // self.grad_accumulation_steps
        step = 1

        # Change the progress bar to add the following things:
        # 1. Add the current step
        # 2. Add the total number of steps
        loss = 0
        for data in self.train_dataloader:
            data.to(self.gpu_id)

            opt_step = step // self.grad_accumulation_steps
            self.model.train()
            loss += (
                self._run_batch(source=data, target=data, step=step)
                / self.grad_accumulation_steps
            )

            if step % self.grad_accumulation_steps == 0:
                loss = 0

            if step % (self.eval_every * self.grad_accumulation_steps) == 0:
                self.model.eval()
                val_loss = 0
                for ev_source, ev_target in self.val_dataloader:
                    ev_source = ev_source.to(self.gpu_id)
                    ev_target = ev_target.to(self.gpu_id)
                    val_loss += self._run_eval(ev_source, ev_target)
                val_loss /= len(self.val_dataloader)  # type: ignore

            step += 1
