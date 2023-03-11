from collections import namedtuple


Transition = namedtuple(
    "Transition",
    field_names=["observation", "action", "reward", "done", "next_observation"],
)


TransitionRecurrentIn = namedtuple(
    "Transition",
    field_names=[
        "observation",
        "h_recurrent",
        "c_recurrent",
        "action",
        "reward",
        "done",
        "next_observation",
        "next_h_recurrent",
        "next_c_recurrent",
        #"next_last_action"
    ],
)


TransitionRecurrentOut = namedtuple(
    "Transition",
    field_names=[
        "observation",
        "h_recurrent",
        "c_recurrent",
        #"last_action",
        #"last_reward",
        "action",
        "reward",
        "done",
        "next_observation",
        "next_h_recurrent",
        "next_c_recurrent",
        #"next_last_action",
        "mask",
        "lengths"
    ],
)


TransitionRecurrentPEROut = namedtuple(
    "Transition",
    field_names=[
        "observation",
        "h_recurrent",
        "c_recurrent",
        #"last_action",
        #"last_reward",
        "action",
        "reward",
        "done",
        "next_observation",
        "next_h_recurrent",
        "next_c_recurrent",
        #"next_last_action",
        "mask",
        "lengths",
        "indices",
        "weights",
    ],
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