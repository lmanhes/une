from dataclasses import dataclass
from typing import Union

import numpy as np
import torch


@dataclass(kw_only=True)
class Transition:
    observation: Union[np.ndarray, torch.Tensor]
    action: Union[np.ndarray, torch.Tensor]
    next_observation: Union[np.ndarray, torch.Tensor]
    reward: Union[np.ndarray, torch.Tensor]
    done: Union[np.ndarray, torch.Tensor]

@dataclass(kw_only=True)
class NStepTransition(Transition):
    next_nstep_observation: Union[np.ndarray, torch.Tensor]

@dataclass(kw_only=True)
class PERTransition(Transition):
    indices: Union[np.ndarray, torch.Tensor]
    weights: Union[np.ndarray, torch.Tensor]

@dataclass(kw_only=True)
class NStepPERTransition(NStepTransition, PERTransition):
    pass

@dataclass(kw_only=True)
class RecurrentTransition(Transition):
    h_recurrent: Union[np.ndarray, torch.Tensor]
    c_recurrent: Union[np.ndarray, torch.Tensor]
    next_h_recurrent: Union[np.ndarray, torch.Tensor]
    next_c_recurrent: Union[np.ndarray, torch.Tensor]
    mask: Union[np.ndarray, torch.Tensor] = None
    length: Union[np.ndarray, torch.Tensor] = None

@dataclass(kw_only=True)
class PERRecurrentTransition(PERTransition, RecurrentTransition):
    pass

@dataclass(kw_only=True)
class NStepRecurrentTransition(NStepTransition, RecurrentTransition):
    pass

@dataclass(kw_only=True)
class NStepPERRecurrentTransition(NStepRecurrentTransition, PERRecurrentTransition):
    pass
