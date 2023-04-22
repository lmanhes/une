from abc import ABC, abstractmethod
from pathlib import Path
import pickle
from typing import Any, Tuple, Union, Type

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

    def save(self, filename: Union[str, Path]):
        # Use protocol>=4 to support saving replay buffers >= 4Gb
        pickle.dump(self, open(filename, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    def load(cls, filename: Union[str, Path]) -> Type['AbstractBuffer']:
        return pickle.load(open(filename, "rb"))
