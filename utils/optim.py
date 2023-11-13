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


if __name__ == "__main__":
    model = torch.nn.Linear(10, 10)
    lr_scheular = WarmupCosineWithDecay(
        optimizer=torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-2,
            betas=(0.9, 0.999),
            eps=1e-6,
        ),
        warmup_steps=2,
        t_total=20,
        steps_per_epoch=10,
        eta_max=1,
        eta_min=0.1,
    )

    for epoch in range(2):
        for step in range(10):
            lr_scheular.step()
            print(lr_scheular.get_last_lr()[0])
