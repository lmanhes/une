from pathlib import Path
import random
from typing import Union

from .abstract import AbstractAgent


class RandomAgent(AbstractAgent):

    def act(self):
        return random.choice(range(self.config['n_actions']))

    def reset(self):
        raise NotImplementedError()

    def save(self, filename: Union[str, Path]):
        raise NotImplementedError()

    def load(self, filename: Union[str, Path]):
        raise NotImplementedError()