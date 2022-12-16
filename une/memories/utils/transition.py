from collections import namedtuple


Transition = namedtuple(
    "Transition",
    field_names=["observation", "action", "reward", "done", "next_observation"],
)


TransitionPER = namedtuple(
    "Transition",
    field_names=[
        "observation",
        "action",
        "reward",
        "done",
        "next_observation",
        "indices",
        "weights",
    ],
)
