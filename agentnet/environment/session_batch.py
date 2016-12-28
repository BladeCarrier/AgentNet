from warnings import warn

from theano import tensor as T

from ..environment import BaseEnvironment
from ..objective import BaseObjective
from ..utils.format import check_list


class SessionBatchEnvironment(BaseEnvironment, BaseObjective):
    """
    A generic pseudo-environment that replays sessions defined on creation by theano expressions
    ignoring agent actions completely.

    The environment takes symbolic expression for sessions represented as (.observations, .actions, .rewards)
    Unlike SessionPoolEnvironment, this one does not store it's own pool of sessions.

    To create experience-replay sessions, call Agent.get_sessions with this as an environment.

    :param observations: a tensor or a list of tensors matching agent observation sequence [batch, tick, whatever]
    :type observations: theano tensor or a list of such
    :param single observation shapes: shapes of one-tick one-batch-item observations.
                            E.g. if lasagne shape is [None, 25(ticks), 3,210,160],
                            than single_observation_shapes must contain [3,210,160]
    :type single_observation_shapes: a list of tuples of integers
    :param actions: a tensor or a list of tensors matching agent actions sequence [batch, tick, whatever]
    :type actions: theano tensor or a list of such
    :param single action shapes: shapes of one-tick one-batch-item actions. Similar to observations.
                                    All scalar means that each action has shape (,),
                                    lasagne sequence layer being of shape (None, seq_length)
    :type single_observation_shapes: a list of tuples of integers
    :param rewards: a tensor matching agent rewards sequence [batch, tick]
    :type rewards: theano tensor

    :param is_alive: whether or not session has still not finished by a particular tick. Always alive by default.
    :type is_alive: theano tensor or None
    :param preceding_agent_memory: a tensor or a list of such storing what was in agent's memory prior
                                    to the first tick of the replay session.
    :type actions: theano tensor or a list of such


    How does it tick:

    During experience replay sessions,
     - observations, actions and rewards match original ones
     - agent memory states, Q-values and all in-agent expressions (but for actions) will correspond to what
       agent thinks NOW about the replay (not what it thought when he commited actions)
     - preceding_agent_memory [optional] - what was agent's memory state prior to the first tick of the replay session.


    Although it is possible to get rewards via the regular functions, it is usually faster to take self.rewards as rewards
    with no additional computation.

    """

    def __init__(self, observations, single_observation_shapes,
                 actions=None, single_action_shapes="all_scalar",
                 rewards=None, is_alive=None, preceding_agent_memory=None):

        # setting environmental variables. Their shape is [batch,time,something]
        self.observations = check_list(observations)
        self.single_observation_shapes = check_list(single_observation_shapes)

        for obs, obs_shape in zip(self.observations, self.single_observation_shapes):
            assert obs.ndim == len(obs_shape) + 2

        if actions is not None:
            self.actions = check_list(actions)
            self.single_action_shapes = single_action_shapes
            if self.single_action_shapes == "all_scalar":
                self.single_action_shapes = [tuple()] * len(self.actions)

        self.rewards = rewards
        self.is_alive = is_alive

        if preceding_agent_memory is not None:
            self.preceding_agent_memory = check_list(preceding_agent_memory)

        self.padded_observations = [
            T.concatenate([obs, T.zeros_like(obs[:, :1])], axis=1)
            for obs in self.observations
            ]

        self.batch_size = self.observations[0].shape[0]
        self.sequence_length = self.observations[0].shape[1]

        BaseEnvironment.__init__(self,
                                 state_shapes=tuple(()),
                                 observation_shapes=self.single_observation_shapes,
                                 action_shapes=self.single_action_shapes,
                                 state_dtypes=("int32",),
                                 observation_dtypes=tuple(obs.dtype for obs in self.observations),
                                 action_dtypes=tuple(act.dtype for act in self.actions)
                                 )

    def get_action_results(self, last_states, actions, **kwargs):
        """Computes environment state after processing agent's action.

        :param last_states: Environment state on previous tick.
        :type last_states: float[batch_id, memory_id0[, memory_id1,...]]

        :param actions: Agent action after observing last state.
        :type actions: int[batch_id]

        :returns:
            new_state float[batch_id, memory_id0[, memory_id1,...]]: Environment state after processing agent's action.
            observation float[batch_id, n_agent_inputs]: What agent observes after commiting the last action.
        """
        warn("Warning - a session pool has all the observations already stored as .observations property."
             "Recomputing them this way is probably just a slower way of calling your_session_pool.observations")

        time_i = check_list(last_states)[0]

        batch_range = T.arange(time_i.shape[0])

        new_observations = [obs[batch_range, time_i + 1] for obs in self.padded_observations]
        return [time_i + 1], new_observations

    def get_reward(self, session_states, session_actions, batch_id):
        """
        WARNING! this runs on a single session, not on a batch.
        Reward given for taking the action in current environment state.

        :param session_states: Environment state before taking action.
        :type session_states: float[batch_id, memory_id]

        :param session_actions: Agent action at this tick.
        :type session_actions: int[batch_id]

        :returns:
            reward float[batch_id]: Reward for taking action from the given state.
        """
        warn("Warning - a session pool has all the rewards already stored as .rewards property."
             "Recomputing them this way is probably just a slower way of calling your_session_pool.rewards")
        return self.rewards[batch_id, :]
