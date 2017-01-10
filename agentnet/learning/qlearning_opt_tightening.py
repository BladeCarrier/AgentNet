"""
Q-learning algorithm with optimality tightening
(as described in arXiv:1611.01606)
"""

import theano
import theano.tensor as T

from lasagne.objectives import squared_error

from .helpers import get_action_Qvalues



def get_elementwise_objective(
    Qvalues, actions, rewards, n_steps,
    is_alive='always',
    gamma=0.95, lambda_constr=1e3):
    """
    Return squared error from standard Q-learning plus the penalty for violating the lower bound.
    
    :param Qvalues: [batch, tick, action_id] - predicted qvalues
    :param actions: [batch, tick] - committed actions
    :param rewards: [batch, tick] - immediate rewards for taking actions at given time ticks
    :param n_steps: the length of the horizon used for constrain computation.
    :param is_alive: [batch, tick] - whether given session is still active at given tick.
        Default value implies simplified construction of error function.
    :param gamma: delayed reward discount (defaults to 0.95).
    :param lambda_constr: penalty coefficient for vonstrain violation (default to 1000).

    :return: tensor [batch, tick] of error function values for every tick in every batch.
    """

    # Computing standard Q-learning error.
    opt_Qvalues = T.max(Qvalues, axis=-1)
    act_Qvalues = get_action_Qvalues(Qvalues, actions)
    ref_Qvalues = rewards + gamma * T.concatenate((opt_Qvalues[:, 1:], T.zeros_like(opt_Qvalues[:, 0:1])), axis=1)
    classic_error = squared_error(ref_Qvalues, act_Qvalues)

    gamma_pows, gamma_pows_upd = theano.scan(
        fn=(lambda prior_result, gamma: prior_result * gamma),
        outputs_info=gamma**-1,
        non_sequences=gamma,
        n_steps=n_steps
    )

    reward_shifts, reward_shifts_upd = theano.scan(
        fn=lambda prior_result: T.concatenate(
            (prior_result[:, 1:],
            T.zeros_like(prior_result[:, 0:1])), axis=1
        ),
        outputs_info=T.concatenate(
            (T.zeros_like(rewards[:, 0:1]),
            rewards), axis=1
        ),
        n_steps=n_steps
    )
    reward_shifts = reward_shifts[:, :, :-1].dimshuffle(1,0,2)

    if is_alive != 'always':
        is_alive_shifts, is_alive_shifts_upd = theano.scan(
            fn=lambda prior_result: T.concatenate(
                (prior_result[:, 1:],
                T.zeros_like(prior_result[:, 0:1])), axis=1
            ),
            outputs_info=is_alive,
            n_steps=n_steps
        )
        is_alive_shifts = is_alive_shifts.dimshuffle(1,0,2)

    lower_bound_rewards_raw, lower_bound_rewards_raw_updates = theano.map(
        lambda x,y: x*y,
        sequences=(
            T.tile(gamma_pows, reward_shifts.shape[0]),
            reward_shifts.reshape((n_steps*rewards.shape[0],rewards.shape[1]))
        )
    )
    lower_bound_rewards = lower_bound_rewards_raw.reshape(reward_shifts.shape)
    lower_bound_rewards = lower_bound_rewards.cumsum(axis=1)

    lower_bound_qvals, lower_bound_qvals_updates = theano.scan(
        fn=lambda prior_result: T.concatenate(
            (gamma*prior_result[:, 1:],
             T.zeros_like(prior_result[:, 0:1])
            ),
            axis=1
        ),
        outputs_info=opt_Qvalues,
        n_steps=n_steps
    )
    lower_bound_qvals = lower_bound_qvals.dimshuffle(1,0,2)
    if is_alive != 'always':
        lower_bound_qvals *= is_alive_shifts

    lower_bound_total = T.max(lower_bound_rewards + lower_bound_qvals, axis=1)
    lower_bound_error = T.maximum(0, lower_bound_total - act_Qvalues)**2
    error_fn = classic_error + lambda_constr * lower_bound_error
    return error_fn
