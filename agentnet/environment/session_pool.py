from theano import tensor as T

import numpy as np
import theano

from collections import OrderedDict

from .base import BaseEnvironment
from .session_batch import SessionBatchEnvironment

from ..objective import BaseObjective

from ..utils import create_shared, set_shared
from ..utils.format import check_list
from ..utils.layers import get_layer_dtype

from warnings import warn


class SessionPoolEnvironment(BaseEnvironment, BaseObjective):
    """A generic pseudo-environment that replays sessions loaded via .load_sessions(...),
    ignoring agent actions completely.

    This environment can be used either as a tool to run experiments with non-theano environments or to actually
    train via experience replay [http://rll.berkeley.edu/deeprlworkshop/papers/database_composition.pdf]

    It has a single scalar integer env_state, corresponding to time tick.

    The environment maintains it's own pool of sessions represented as (.observations, .actions, .rewards)

    To load sessions into pool, use
        - .load_sessions - replace existing sessions with new ones
        - .append_sessions - add new sessions to existing ones up to a limited size
        - .get_session_updates - a symbolic update of experience replay pool via theano.updates.

    To use SessionPoolEnvironment for experience replay, one can
        - feed it into agent.get_sessions (with optimize_experience_replay=True recommended) to use all sessions
        - subsample sessions via .select_session_batch or .sample_sessions_batch to use random session subsample
                    [ this option creates SessionBatchEnvironment that can be used with agent.get_sessions ]

    During experience replay sessions
        - states are replaced with a fake one-unit state
        - observations, actions and rewards match original ones
        - agent memory states, Q-values and all in-agent expressions (but for actions) will correspond to
            what agent thinks NOW about the replay.

    Although it is possible to get rewards via the regular functions, it is usually faster to take self.rewards as rewards
    with no additional computation.


    :param observations: number of floatX flat observations or a list of observation inputs to mimic
    :type observations: int or lasagne.layers.Layer or list of lasagne.layers.Layer

    :param actions: number of int32 scalar actions or a list of resolvers to mimic
    :type actions: int or lasagne.layers.Layer or list of lasagne.layers.Layer

    :param agent_memories: number of agent states [batch,tick,unit] each or a list of memory layers to minic
    :type agent_memories: int or lasagne.layers.Layer or a list of lasagne.layers.Layer

    :param default_action_dtype: if actions are given as lasagne layers with valid dtypes, this is a default dtype of action
                            Otherwise agentnet.utils.layers.get_layer_dtype is used on a per-layer basis
    :type default_action_dtype: string or dtype

    To setup custom dtype, set the .output_dtype property of layers you send as actions, observations of memories.

    WARNING! this session pool is stored entirely as a set of theano shared variables.
    GPU-users willing to store a __large__ pool of sessions to sample from are recommended to store them
    somewhere outside (e.g. as numpy arrays) to avoid overloading GPU memory.
    """

    def __init__(self, observations=1,
                 actions=1,
                 agent_memories=1,
                 default_action_dtype="int32",
                 rng_seed=1337):

        def _create_shareds(vars, first_shape, second_shape, default_dtype, name_prefix):
            # Helper function for initializing storages for observations, actions and agent memories.
            if type(vars) is int:
                return [create_shared(name_prefix+str(i),
                                      np.zeros(first_shape),
                                      dtype=default_dtype)
                        for i in range(vars)]
            else:
                if isinstance(vars, dict):  # This is only possible with `agent_memories`.
                    vars = list(vars.keys())

                vars = check_list(vars)
                return [create_shared(name_prefix+str(i),
                                      np.zeros(second_shape+tuple(var.output_shape[1:])),
                                      dtype=get_layer_dtype(var))
                        for i, var in enumerate(vars)]

        # Observations.
        self.observations = _create_shareds(observations, (10, 5, 2), (10, 5),
                                            theano.config.floatX, "sessions.observations_history.")

        # Padded observations (to avoid index error when interacting with agent).
        self.padded_observations = [
            T.concatenate([obs, T.zeros_like(obs[:, :1])], axis=1)
            for obs in self.observations
            ]

        # Actions log.
        self.actions = _create_shareds(actions, (10, 5), (10, 5),
                                       default_action_dtype, "session.actions_history.")

        # Agent memory at state 0: floatX[batch_i,unit].
        self.preceding_agent_memories = _create_shareds(agent_memories, (10, 5), (10,),
                                                        theano.config.floatX, "session.prev_memory.")

        # rewards
        self.rewards = create_shared("session.rewards_history", np.zeros([10, 5]), dtype=theano.config.floatX)

        # is_alive
        self.is_alive = create_shared("session.is_alive", np.ones([10, 5]), dtype='uint8')

        # shapes
        self.batch_size = self.pool_size = self.rewards.shape[0]
        self.sequence_length = self.rewards.shape[1]

        # rng used to .sample_session_batch
        self.rng = T.shared_randomstreams.RandomStreams(rng_seed)

        BaseEnvironment.__init__(self,
                                 state_shapes=(tuple()),
                                 observation_shapes=tuple(obs.get_value().shape[2:] for obs in self.observations),
                                 action_shapes=tuple(act.get_value().shape[2:] for act in self.actions),
                                 state_dtypes=("int32",),
                                 observation_dtypes=tuple(obs.dtype for obs in self.observations),
                                 action_dtypes=tuple(act.dtype for act in self.actions)
                                 )

    def get_action_results(self, last_states, actions, **kwargs):
        """Compute environment state after processing agent's action.

        :param last_states: Environment state on previous tick.
        :type last_states: float[batch_id, memory_id0,[memory_id1],...] or list of such.

        :param actions: Agent action after observing last state.
        :type actions: int[batch_id] or list of such.

        :returns:
            new_state float[batch_id, memory_id0,[memory_id1],...]: Environment state after processing agent's action.
            observation float[batch_id,n_agent_inputs]: What agent observes after committing the last action.
        """
        warn("Warning - a session pool has all the observations already stored as .observations property."
             "Recomputing them this way is probably just a slower way of calling your_session_pool.observations")

        time_i = check_list(last_states)[0]

        batch_range = T.arange(time_i.shape[0])

        new_observations = [obs[batch_range, time_i + 1]
                            for obs in self.padded_observations]
        return [time_i + 1], new_observations

    def get_reward(self, session_states, session_actions, batch_id):
        """
        WARNING! this runs on a single session, not on a batch
        reward given for taking the action in current environment state

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

    def load_sessions(self, observation_sequences, action_sequences, reward_seq, is_alive=None, prev_memories=None):
        """Load a batch of sessions into env. The loaded sessions are that used during agent interactions.
        """
        observation_sequences = check_list(observation_sequences)
        action_sequences = check_list(action_sequences)

        assert len(observation_sequences) == len(self.observations)
        assert len(action_sequences) == len(self.actions)
        if prev_memories is not None:
            prev_memories = check_list(prev_memories)
            assert len(prev_memories) == len(self.preceding_agent_memories)

        for observation_var, observation_seq in zip(self.observations, observation_sequences):
            set_shared(observation_var, observation_seq)
        for action_var, action_seq in zip(self.actions, action_sequences):
            set_shared(action_var, action_seq)

        set_shared(self.rewards, reward_seq)

        if is_alive is not None:
            set_shared(self.is_alive, is_alive)

        if prev_memories is not None:
            for prev_memory_var, prev_memory_value in zip(self.preceding_agent_memories, prev_memories):
                set_shared(prev_memory_var, prev_memory_value)

    def append_sessions(self, observation_sequences, action_sequences, reward_seq, is_alive=None, prev_memories=None,
                        max_pool_size=None):
        """Add a batch of sessions to the existing sessions. The loaded sessions are that used during agent interactions.
        
        If max_pool_size != None, only last max_pool_size sessions are kept.
        """

        observation_sequences = check_list(observation_sequences)
        action_sequences = check_list(action_sequences)

        assert len(observation_sequences) == len(self.observations)
        assert len(action_sequences) == len(self.actions)
        if prev_memories is not None:
            prev_memories = check_list(prev_memories)
            assert len(prev_memories) == len(self.preceding_agent_memories)

        try:
            # in case numpy concatenate throws ValueError
            # meaning "can't concatenate dummy shared values with acutal new memory"
            # call load_sessions instead

            # observations
            observation_tensors = [np.concatenate((obs.get_value(), new_obs), axis=0)
                                   for obs, new_obs in zip(self.observations, observation_sequences)]

            # actions
            action_tensors = [np.concatenate((act.get_value(), new_act), axis=0)
                              for act, new_act in zip(self.actions, action_sequences)]

            # rewards
            rwd = self.rewards.get_value()
            reward_tensor = np.concatenate((rwd, reward_seq), axis=0)

            is_alive_tensor = None
            preceding_memory_states = None
            # is_alives
            if is_alive is not None:
                is_alive_tensor = np.concatenate((self.is_alive.get_value(), is_alive), axis=0)
            else:
                is_alive_tensor = None

            # prev memories
            if prev_memories is not None:
                prev_memory_tensors = [np.concatenate((prev_mem.get_value(), new_prev_mem), axis=0)
                                           for prev_mem, new_prev_mem in
                                           zip(self.preceding_agent_memories, prev_memories)]
            else:
                prev_memory_tensors = None

            # crop to pool size
            if max_pool_size is not None:
                new_size = len(observation_tensors[0])
                if new_size > max_pool_size:
                    observation_tensors = [obs[-max_pool_size:] for obs in observation_tensors]
                    action_tensors = [act[-max_pool_size:] for act in action_tensors]
                    reward_tensor = reward_tensor[-max_pool_size:]
                    if is_alive_tensor is not None:
                        is_alive_tensor = is_alive_tensor[-max_pool_size:]
                    if prev_memory_tensors is not None:
                        prev_memory_tensors = [mem[-max_pool_size:] for mem in prev_memory_tensors]
        except ValueError:
            warn("Warning! Appending sessions to empty or broken pool. Old pool sessions, if any, are disposed.")
            observation_tensors = observation_sequences
            action_tensors = action_sequences
            reward_tensor = reward_seq
            is_alive_tensor = is_alive
            prev_memory_tensors = prev_memories

        #load everything into the environmnet
        self.load_sessions(observation_tensors,action_tensors,reward_tensor,is_alive_tensor,prev_memory_tensors)



    def get_session_updates(self, observation_sequences, action_sequences, reward_seq, is_alive=None, prev_memory=None,
                            cast_dtypes=True):
        """Return a dictionary of updates that will set shared variables to argument state.
        If cast_dtypes is True, casts all updates to the dtypes of their respective variables.
        """
        observation_sequences = check_list(observation_sequences)
        action_sequences = check_list(action_sequences)

        assert len(observation_sequences) == len(self.observations)
        assert len(action_sequences) == len(self.actions)
        if prev_memory is not None:
            assert len(prev_memory) == len(self.preceding_agent_memories)

        updates = OrderedDict()

        for observation_var, observation_sequences in zip(self.observations, observation_sequences):
            updates[observation_var] = observation_sequences
        for action_var, action_sequences in zip(self.actions, action_sequences):
            updates[action_var] = action_sequences

        updates[self.rewards] = reward_seq

        if is_alive is not None:
            updates[self.is_alive] = is_alive

        if prev_memory is not None:
            for prev_memory_var, prev_memory_value in zip(self.preceding_agent_memories, check_list(prev_memory)):
                updates[prev_memory_var] = prev_memory_value

        if cast_dtypes:
            casted_updates = OrderedDict({})
            for var, upd in list(updates.items()):
                casted_updates[var] = upd.astype(var.dtype)
            updates = casted_updates

        return updates

    def select_session_batch(self, selector):
        """Returns SessionBatchEnvironment with sessions (observations, actions, rewards) from pool at given indices.

        :param selector: An array of integers that contains all indices of sessions to take.

        Note that if this environment did not load is_alive or preceding_memory, 
        you won't be able to use them at the SessionBatchEnvironment
        """
        selected_observations = [observation_seq[selector] for observation_seq in self.observations]
        selected_actions = [action_seq[selector] for action_seq in self.actions]
        selected_prev_memories = [prev_memory[selector] for prev_memory in self.preceding_agent_memories]

        return SessionBatchEnvironment(selected_observations, self.observation_shapes,
                                       selected_actions, self.action_shapes,
                                       self.rewards[selector],
                                       self.is_alive[selector],
                                       selected_prev_memories)

    def sample_session_batch(self, max_n_samples, replace=False, selector_dtype='int32'):
        """Return SessionBatchEnvironment with sessions (observations, actions, rewards) that will be sampled uniformly
        from this session pool.

        If replace=False, the amount of samples is min(max_n_sample, current pool). Otherwise it equals max_n_samples.
        
        The chosen session ids will be sampled at random using self.rng on each iteration.
        P.S. There is no need to propagate rng updates! It does so by itself.
        Unless you are calling it inside theano.scan, ofc, but i'd recommend that you didn't.
        unroll_scan works ~probably~ perfectly fine btw
        """
        if replace:
            n_samples = max_n_samples
        else:
            n_samples = T.minimum(max_n_samples, self.pool_size)

        sample_ids = self.rng.choice(size=(n_samples,), a=self.pool_size, dtype=selector_dtype, replace=replace)
        return self.select_session_batch(sample_ids)
