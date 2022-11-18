from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from une.representations.abstract import AbstractRepresentation


class AtariCnn(AbstractRepresentation):
    def __init__(self, input_shape: Tuple[int], features_dim: int) -> None:
        super().__init__(input_shape=input_shape, features_dim=features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(self._input_shape[0], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        # self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.fc4 = nn.Linear(3136, self._features_dim)

        with torch.no_grad():
            rand_observation = np.random.rand(*self._input_shape)
            if rand_observation.ndim == 3:
                rand_observation = np.expand_dims(rand_observation, 0)
            rand_observation = torch.as_tensor(rand_observation).float()
            n_flatten = self.cnn(rand_observation).shape[1]
            print("n_flatten : ", n_flatten)

        #self.linear = nn.Sequential(nn.Linear(3136, self._features_dim), nn.ReLU())
        self.linear = nn.Sequential(nn.Linear(n_flatten, self._features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        #x = self.cnn(observations)
        #return self.linear(x)
        observations = observations / 255.0
        return self.linear(self.cnn(observations))
        # x = self.conv1(observations)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # return self.fc4(x.view(x.size(0), -1))
