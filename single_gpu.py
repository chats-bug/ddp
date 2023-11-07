from typing import Union, Optional
import os
import torch
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    track,
)
from tqdm import tqdm

from datautils import PoorMansDataLoader

console = Console()


class Trainer:
    def __init__(
			self,
			model: torch.nn.Module,
			train_dataloader: PoorMansDataLoader,
			val_dataloader: PoorMansDataLoader,
			optimizer: torch.optim.Optimizer,
			gpu_id: int,
			device: Union[str, torch.device],
			eval_every: int,
			save_every: int,
			lr_schedular: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
			log_every: int = 1,
			grad_accumulation_steps: int = 1,
			max_grad_norm: Optional[float] = None,
			torch_dtype: torch.dtype = torch.float32,
			output_dir: Optional[str] = None,
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
        self.torch_dtype = torch_dtype
        self.device = device
        self.lr_schedular = lr_schedular
        self.max_grad_norm = max_grad_norm
        self.output_dir = output_dir
        if self.device != "cuda":
            self.gpu_id = self.device

        # Set the
        torch.set_default_device(self.device)

        self.scaler = None
        if self.torch_dtype != torch.float32:
            # assert self.device in ["cpu", "cuda", torch.device("cpu"), torch.device("cuda")], \
            #     "Mixed precision training is only supported on CPU and CUDA devices"
            if self.device not in ["cpu", "cuda",]:
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

        if not self.output_dir:
            self.output_dir = "output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _run_batch(self, source, target, step=1):
        if self.scaler:
            with torch.autocast(device_type=self.device, dtype=self.torch_dtype):
                output = self.model(source, labels=target)
                loss = output.loss
            self.scaler.scale(loss).backward()
            if step % self.grad_accumulation_steps == 0:
                if self.max_grad_norm:
                    # Unscales the gradients of optimizer's assigned parameters in-place
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
        return loss.item()

    def _run_eval(self, source, target):
        self.model.eval()
        with torch.no_grad():
            output = self.model(source, labels=target)
            loss = output.loss
        return loss

    def _run_epoch(self, epoch: int):
        train_bsz = self.train_dataloader.get_batch_size()
        val_bsz = self.val_dataloader.get_batch_size()
        total_steps = len(self.train_dataloader) // self.grad_accumulation_steps
        step = 1
        # Change the progress bar to add the following things:
        # 1. Add the current step
        # 2. Add the total number of steps
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            "🏀",
            TimeElapsedColumn(),
            "🏀",
            TimeRemainingColumn(),
            "🏀",
            TextColumn("[progress.percentage]{task.completed}"),
            "🏀",
            TextColumn(f"[progress.percentage]{total_steps:,}"),
        ) as progress:
            training_task = progress.add_task(
                "Training...",
                total=len(self.train_dataloader) // self.grad_accumulation_steps,
            )
            for source, target in self.train_dataloader:
                opt_step = step // self.grad_accumulation_steps
                self.model.train()
                source = source.to(self.gpu_id)
                target = target.to(self.gpu_id)
                loss = self._run_batch(source, target, step)
                if step % (self.log_every * self.grad_accumulation_steps) == 0:
                    progress.console.log(
                        f"Epoch: {epoch}, Step: {opt_step}, Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}, Loss: {loss:.3f}"
                    )
                    progress.update(training_task, advance=1, step=step)

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
                    progress.console.log("=" * 80)
                    progress.console.log(
                        f"Epoch: {epoch}, Step: {opt_step}, Val loss: {val_loss:.4f}"
                    )
                    progress.console.log("=" * 80)

                if step % (self.save_every * self.grad_accumulation_steps) == 0:
                    self._save_checkpoint(epoch, opt_step, progress.console.log)

                step += 1

    def _save_checkpoint(self, epoch: int, step: int, log_fn: callable):
        ckp = self.model.state_dict()
        torch.save(ckp, f"{self.output_dir}/ckp_epoch_{epoch}_step_{step}.pt")
        log_fn(
            f"Epoch: {epoch}, Step: {step}, Saved checkpoint at ckp_epoch_{epoch}_step_{step}.pt"
        )

    def train(self, max_epochs: int):
        print(f"Transferring model to {self.gpu_id}...")
        self.model.to(self.gpu_id)
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
