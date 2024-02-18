from typing import Callable
import numpy as np
import torch

def numpy_to_torch(func: Callable) -> Callable:
    """
    Decorator that converts numpy arrays returned by the function to torch tensors.
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, tuple):
            return tuple(torch.from_numpy(r[..., None]).float() if isinstance(r, np.ndarray) else r for r in result)
        elif isinstance(result, np.ndarray):
            return torch.from_numpy(result[..., None]).float()
        return result
    return wrapper
