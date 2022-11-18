from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class AbstractRepresentation(nn.Module):
    def __init__(self, input_shape: Union[int, Tuple[int]], features_dim: int) -> None:
        super().__init__()
        self._input_shape = input_shape
        self._features_dim = features_dim

    @abstractmethod
    def forward(self) -> torch.Tensor:
        raise NotImplementedError()

    @property
    def features_dim(self) -> int:
        return self._features_dim
