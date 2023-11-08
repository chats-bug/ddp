import torch
import math


class WarmupCosineWithDecay(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        t_total,
        steps_per_epoch,
        eta_max,
        eta_min,
        last_epoch=-1,
    ):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.steps_per_epoch = steps_per_epoch
        self.eta_min = eta_min
        self.eta_max = eta_max
        super(WarmupCosineWithDecay, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        step = step % self.steps_per_epoch

        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        else:
            return (
                self.eta_min
                + 0.5
                * (self.eta_max - self.eta_min)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (step - self.warmup_steps)
                        / (self.steps_per_epoch - self.warmup_steps)
                    )
                )
            ) / self.eta_max
