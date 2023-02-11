from collections import namedtuple


Transition = namedtuple(
    "Transition",
    field_names=["observation", "action", "reward", "done", "next_observation"],
)


TransitionEpisodic = namedtuple(
    "Transition",
    field_names=[
        "observation",
        "action",
        "reward",
        "episodic_reward",
        "done",
        "next_observation",
    ],
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


TransitionNStepEpisodic = namedtuple(
    "Transition",
    field_names=[
        "observation",
        "action",
        "reward",
        "episodic_reward",
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


TransitionPEREpisodic = namedtuple(
    "Transition",
    field_names=[
        "observation",
        "action",
        "reward",
        "episodic_reward",
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


TransitionNStepPEREpisodic = namedtuple(
    "Transition",
    field_names=[
        "observation",
        "action",
        "reward",
        "episodic_reward",
        "done",
        "next_observation",
        "next_nstep_observation",
        "indices",
        "weights",
    ],
)