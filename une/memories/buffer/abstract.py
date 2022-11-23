from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np


class AbstractBuffer(ABC):

    def __init__(self, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def add(self, transition: Tuple[Any, ...]) -> Tuple[Any, ...]:
        raise NotImplementedError()

    @abstractmethod
    def sample(self) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()
    
    # @abstractmethod
    # def reset(self):
    #     raise NotImplementedError()

    # @abstractmethod
    # def save(self, filename: Union[str, Path]):
    #     raise NotImplementedError()

    # @abstractmethod
    # def load(self, filename: Union[str, Path]):
    #     raise NotImplementedError()
