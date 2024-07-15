from typing import TYPE_CHECKING, List, Optional, Tuple

import time
import torch


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by 2
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param

def recursive_round(item, precision=3):
    if isinstance(item, dict):
        return {key: recursive_round(value, precision) for key, value in item.items()}
    elif isinstance(item, list):
        return [recursive_round(value, precision) for value in item]
    elif isinstance(item, float):
        return round(item, precision)
    else:
        return item


def estimate_remaining_time(start_time, completed, total):
    elapsed_time = time.time() - start_time
    if completed == 0:
        return None
    time_per_unit = elapsed_time / completed
    remaining_units = total - completed
    remaining_time = int(time_per_unit * remaining_units)
    if remaining_time > 3600:
        remaining_time = round(remaining_time / 3600, 2)
        unit = "h"
    elif remaining_time > 60:
        remaining_time = round(remaining_time / 60, 2)
        unit = "min"
    else:
        unit = "sec"
    return f"{remaining_time} {unit}"
