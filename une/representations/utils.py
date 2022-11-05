import numpy as np
import torch


def to_tensor(array: np.ndarray, add_batch_dim: bool = True) -> torch.Tensor:
    tensor = torch.from_numpy(array).float()
    if add_batch_dim and tensor.ndim == 3:
        tensor.unsqueeze(0)
    return tensor
