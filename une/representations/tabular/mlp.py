from typing import Union, Tuple

import numpy as np
import torch
import torch.nn as nn

from une.representations.abstract import AbstractRepresentation
from une.representations.utils import to_tensor


class GymMlp(AbstractRepresentation):
    def __init__(self, input_shape: Union[int, Tuple[int]], features_dim: int) -> None:
        super().__init__(input_shape=input_shape, features_dim=features_dim)

        if isinstance(self._input_shape, tuple):
            self._input_shape = self._input_shape[0]

        self.mlp = nn.Sequential(
            nn.Linear(self._input_shape, self._features_dim),
            nn.ReLU(),
            nn.Linear(self._features_dim, self._features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        #observations = to_tensor(observations)
        return self.mlp(observations)
