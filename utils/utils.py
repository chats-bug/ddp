import math

import torch


def format_float_to_str(num: float) -> str:
    """
    Format a float to a string
    >>> format_float_to_str(0.0003324) -> "3.324e-04"
    >>> format_float_to_str(0.00000283) -> "2.830e-06"
    """
    return f"{num:.3e}".replace("e-0", "e-").replace("e+0", "e+").replace("e0", "e")


def num_trainable_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
