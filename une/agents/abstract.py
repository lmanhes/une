from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union


class AbstractAgent(ABC):

    @abstractmethod
    def act(self, observation: dict) -> int:
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def save(self, filename: Union[str, Path]):
        raise NotImplementedError()

    @abstractmethod
    def load(self, filename: Union[str, Path]):
        raise NotImplementedError()
