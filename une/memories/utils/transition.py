from collections import namedtuple


Transition = namedtuple(
    "Transition",
    field_names=["observation", "action", "reward", "done", "next_observation"],
)


TransitionNStep = namedtuple(
    "Transition",
    field_names=[
        "observation",
        "action",
        "reward",
        "done",
        "next_observation",
        "next_nstep_observation"
    ],
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


TransitionNStepPER = namedtuple(
    "Transition",
    field_names=[
        "observation",
        "action",
        "reward",
        "done",
        "next_observation",
        "next_nstep_observation",
        "indices",
        "weights",
    ],
)
