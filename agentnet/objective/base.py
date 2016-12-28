from __future__ import division, print_function, absolute_import
import theano.tensor as T
import theano

from ..utils.format import check_list, unpack_list


class BaseObjective(object):
    """
    Instance, that:
        - determines rewards for all actions agent takes given environment state and agent action.
    """
    def reset(self, batch_size):
        """Performs this action each time a new session [batch] is loaded
        batch size: size of the new batch
        """
        pass

    def get_reward(self, last_environment_states, agent_actions, batch_id):
        """WARNING! This function is computed on a single session, not on a batch!
        Reward given for taking the action in current environment state.

        :param last_environment_states: Environment state before taking action.
        :type last_environment_states: float[time_i, memory_id]

        :param agent_actions: Agent action at this tick.
        :type agent_actions: int[time_i]

        :param batch_id: Session id.
        :type batch_id: int

        :return: Reward for taking action.
        :rtype: float[time_i]
        """

        raise NotImplementedError

    def get_reward_sequences(self, env_state_sessions, agent_action_sessions):
        """Computes the rewards given to agent at each time step for each batch.

        :param env_state_sessions: Environment state [batch_i,seq_i,state_units] history for all sessions.
        :type env_state_sessions: theano tensor [batch_i,seq_i,state_units]

        :param agent_action_sessions: Actions chosen by agent at each tick for all sessions.
        :type agent_action_sessions: int[batch_i,seq_i]

        :return rewards: What reward was given to an agent for corresponding action from state in that batch.
        :rtype: float[batch_i,seq_i]
        """
        env_state_sessions = check_list(env_state_sessions)
        n_states = len(env_state_sessions)
        agent_action_sessions = check_list(agent_action_sessions)
        n_actions = len(agent_action_sessions)

        def compute_reward(batch_i, *args):
            session_states, session_actions = unpack_list(args, [n_states, n_actions])
            return self.get_reward(session_states, session_actions, batch_i)

        sequences = [T.arange(agent_action_sessions[0].shape[0], ), ] + env_state_sessions + agent_action_sessions

        rewards, updates = theano.map(compute_reward, sequences=sequences)

        assert len(updates) == 0
        return rewards.reshape(agent_action_sessions[0].shape)  # reshape bach to original
