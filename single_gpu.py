import torch
from rich.console import Console
from tqdm import tqdm
from typing import Union, Optional
from rich.progress import Progress

from datautils import PoorMansDataLoader

console = Console()


def convert_target_to_logits(vocab_size, target):
    # target is a tensor of shape (seq_len,)
    # convert it to a tensor of shape (seq_len, vocab_size)
    # where each row is a one-hot vector
    batch_size = target.size(0)
    seq_len = target.size(1)
    target = target.unsqueeze(-1)
    one_hot = torch.zeros(batch_size, seq_len, vocab_size, device=target.device)
    return one_hot.scatter_(2, target, 1)


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
            torch_dtype: torch.dtype = torch.float32,
    ):
        assert (
                save_every % eval_every == 0
        ), "Can only save checkpoints at eval intervals"
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
        if self.device not in ["cuda", torch.device("cuda")]:
            self.gpu_id = self.device

        self.scaler = None
        if self.torch_dtype is not torch.float32:
            # assert self.device in ["cpu", "cuda", torch.device("cpu"), torch.device("cuda")], \
            #     "Mixed precision training is only supported on CPU and CUDA devices"
            if self.device not in ["cpu", "cuda", torch.device("cpu"), torch.device("cuda")]:
                # print a warning statement in red
                console.log(
                    "[bold red]Mixed precision training is only supported on CPU and CUDA devices[/bold red]"
                )
                console.log(f"Currently setting the default torch type to the {self.torch_dtype}", style="bold red")
                # set the default torch type to the specified torch type
                torch.set_default_dtype(self.torch_dtype)
            else:
                self.scaler = torch.cuda.amp.GradScaler(enabled=False)

    def _run_batch(self, source, target, step=1):
        self.optimizer.zero_grad()
        if self.scaler:
            with torch.amp.autocast(device_type=self.device, dtype=self.torch_dtype):
                output = self.model(source, labels=target)
                loss = output.loss
            self.scaler.scale(loss).backward()
            if step % self.grad_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.lr_schedular:
                    self.lr_schedular.step()
        else:
            output = self.model(source, labels=target)
            loss = output.loss
            loss.backward()
            if step % self.grad_accumulation_steps == 0:
                self.optimizer.step()
                if self.lr_schedular:
                    self.lr_schedular.step()
        return loss.item()

    def _run_eval(self, source, target):
        self.model.eval()
        with torch.no_grad():
            output = self.model(source, labels=target)
            loss = output.loss
        return loss

    def _run_epoch(self, epoch: int):
        batch_size = len(next(iter(self.train_dataloader))[0])
        console.print(
            f"Gpu id: {self.gpu_id}, Epoch: {epoch}, Batch size: {batch_size}, Steps: {len(self.train_dataloader)}"
        )
        step = 1
        with Progress() as progress:
            for source, target in progress.track(self.train_dataloader):
                self.model.train()
                source = source.to(self.gpu_id)
                target = target.to(self.gpu_id)
                loss = self._run_batch(source, target, step)
                if step % (self.log_every * self.grad_accumulation_steps) == 0:
                    progress.console.log(
                        f"Epoch: {epoch}, Step: {step // self.grad_accumulation_steps}, Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}, Loss: {loss:.3f}"
                    )

                if step % self.eval_every == 0:
                    progress.console.print(f"Running eval at epoch: {epoch}, step: {step}")
                    self.model.eval()
                    val_loss = 0
                    for ev_source, ev_target in self.val_dataloader:
                        ev_source = ev_source.to(self.gpu_id)
                        ev_target = ev_target.to(self.gpu_id)
                        val_loss += self._run_eval(ev_source, ev_target)
                    val_loss /= len(self.val_dataloader)  # type: ignore
                    progress.console.print("=" * 80)
                    progress.console.print(f"Epoch: {epoch}, Step: {step}, Val loss: {val_loss:.4f}")
                    progress.console.print("=" * 80)

                if step % self.save_every == 0:
                    self._save_checkpoint(epoch, step)

                step += 1

    def _save_checkpoint(self, epoch: int, step: int):
        ckp = self.model.state_dict()
        torch.save(ckp, f"ckp_epoch_{epoch}_step_{step}.pt")
        console.log(
            f"Epoch: {epoch}, Step: {step}, Saved checkpoint at ckp_epoch_{epoch}_step_{step}.pt"
        )

    def train(self, max_epochs: int):
        print(f"Transferring model to {self.gpu_id}...")
        self.model.to(self.gpu_id)
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
