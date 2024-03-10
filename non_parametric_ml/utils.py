import torch
import numpy as np
import random
import os

def set_all_random_seeds(seed=1):
    """
    Set the seed for reproducibility across multiple runs.
    """
    # Set the seed for generating random numbers in PyTorch
    torch.manual_seed(seed)

    # Set the seed for Python's built-in random module
    random.seed(seed)

    # Set the seed for NumPy's random number generator
    np.random.seed(seed)

    torch.use_deterministic_algorithms(True)

    # Ensure that CUDA operations are deterministic
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


    # # Environment variable for CUDA version >= 10.2 for additional determinism
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

# Example usage
set_all_random_seeds(42)