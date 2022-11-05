from collections import deque
from typing import Tuple, Union

import cv2
from loguru import logger
import numpy as np
import torch


class FrameStack:
    def __init__(self, n_stack: int) -> None:
        self.n_stack = n_stack
        self.frames = deque(maxlen=self.n_stack)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not len(self.frames):
            [self.frames.append(image) for _ in range(self.n_stack)]
        else:
            self.frames.append(image)

        return np.stack(self.frames, 0)


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    assert len(image.shape) == 3 and image.shape[-1] == 3
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def resize_square(image: np.ndarray, shape: int) -> np.ndarray:
    assert shape > 0
    return cv2.resize(image, (shape, shape), interpolation=cv2.INTER_AREA)


def normalize(image: np.ndarray) -> np.ndarray:
    low, high = np.min(image), np.max(image)
    if low >= 0 and high > 1 and image.dtype == np.uint8:
        return image / 255.0
    else:
        return image


def transpose_to_channel_first(image: np.ndarray) -> np.ndarray:
    if image.ndim == 4:
        transposed_idxs = [0, 3, 1, 2]
    else:
        transposed_idxs = [2, 0, 1]
        #transposed_idxs = [image.ndim - 1] + list(range(image.ndim - 1))
    return np.transpose(image, transposed_idxs)


def add_batch_dim(
    observation: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(observation, np.ndarray):
        return np.expand_dims(observation, 0)
    else:
        return observation.unsqueeze(0)


class VisionPreprocessing:
    def __init__(
        self,
        to_grayscale: bool = False,
        resize: bool = False,
        new_size: int = 84,
        normalize: bool = False,
        channel_first: bool = False,
        stack: bool = False,
        n_stack: int = 4,
    ) -> None:
        self.to_grayscale = to_grayscale
        self.resize = resize
        self.new_size = new_size
        self.normalize = normalize
        self.channel_first = channel_first
        self.stack = stack
        self.n_stack = n_stack

        if self.stack:
            self.image_stack = FrameStack(n_stack=self.n_stack)

    def reset(self):
        self.image_stack = FrameStack(n_stack=self.n_stack)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.to_grayscale:
            image = convert_to_grayscale(image)
        elif self.channel_first:
            image = transpose_to_channel_first(image)
        
        if self.resize:
            image = resize_square(image, self.new_size)
        if self.normalize:
            image = normalize(image)
        if self.stack:
            image = self.image_stack(image)

        if image.ndim == 2:
            image = np.expand_dims(image, 0)

        return image


def preprocess_obs(
    observation: Union[np.ndarray, torch.Tensor],
    is_batch: bool = False,
    to_batch: bool = True,
    normalize: bool = False,
    to_tensor: bool = False,
) -> Union[np.ndarray, torch.Tensor]:

    if normalize:
        observation = normalize_image(observation)

    is_space_channel_first, smallest_dimension = is_image_space_channels_first(
        observation=observation, is_batch=is_batch
    )
    if not is_space_channel_first:
        observation = transpose_smallest_dimension(
            observation=observation, smallest_dimension=smallest_dimension
        )
    if to_tensor and isinstance(observation, np.ndarray):
        observation = torch.from_numpy(observation).float()
    if not is_batch and to_batch:
        observation = add_batch_dim(observation)

    return observation
