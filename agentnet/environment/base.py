"""
Base environment class that all experiments must inherit from (or at least implement this interface).
"""

import theano
from lasagne.layers import InputLayer

from ..utils.format import check_list, check_tuple
from ..utils.layers import DictLayer
from collections import OrderedDict


class BaseEnvironment(object):
    """Base for environment layers.
    This is the class you want to inherit from when designing your own custom environments.
    
    To define an environment, one has to describe
        - it's internal state(s),
        - observations it sends to agent
        - actions it accepts from agent
        - environment inner logic

    States, observations and actions are theano tensors (matrices, vectors, etc),
    their shapes should be defined via state_shape, state_dtype, action_shape, etc.

    The default dtypes are floatX for state and observation, int32 for action.
    This suits most of the cases, one can usually use inherited dtypes.

    Finally, one has to implement get_action_results, which converts
    a tuple of (old environment state, agent action) -> (new state, new observation)

    Developer tips:
    [when playing with non-float observations and states]
    if you implemented a new environment, but keep getting a _.grad illegally returned an
    integer-valued variable exception. (Input index _, dtype _), please make sure that any non-float environment
    states are excluded from gradient computation or are cast to floatX.

    To find out which variable causes the problem, find all expressions of the dtype mentioned in
    the expression and then iteratively replace their type with a similar one (like int8 -> uint8 or int32) 
    until the error message dtype changes. Once id does, you have found the cause of the exception.
    """

    def __init__(self,
                 state_shapes,
                 observation_shapes,
                 action_shapes,
                 state_dtypes=None,
                 observation_dtypes=None,
                 action_dtypes=None):
        """
        :param state_shapes: Environment state size: a single shape tuple or several such tuples in a list/tuple.
        :type state_shapes: tuple or list/tuple of tuples.

        :param observation_shapes: Single observation size: a single shape tuple or several such tuples in a list/tuple.
        :type observation_shapes: tuple or list/tuple of tuples.

        :param action_shapes: Single agent action size: a single shape tuple or several such tuples in a list/tuple.
        :type action_shapes: tuple or list/tuple of tuples.

        :param state_dtypes: Types of respective observations, None means all theano.config.floatX.
        :type state_dtypes: tuple or None

        :param observation_dtypes: tTpes of observations. None means all theano.config.floatX.
        :type observation_dtypes: tuple or None

        :param action_dtypes: Types of actions. None means all "int32".
        :type action_dtypes: tuple or None
        
        """

        # shapes
        self.state_shapes = check_tuple(state_shapes)
        self.observation_shapes = check_tuple(observation_shapes)
        self.action_shapes = check_tuple(action_shapes)

        # types
        floatX = theano.config.floatX
        self.state_dtypes = check_tuple(state_dtypes or [floatX for _ in self.state_shapes])
        self.observation_dtypes = check_tuple(observation_dtypes or [floatX for _ in self.observation_shapes])
        self.action_dtypes = check_tuple(action_dtypes or ["int32" for _ in self.action_shapes])

    # Interaction with agent (one step).
    def get_action_results(self, last_states, actions, **kwargs):
        """Computes environment state after processing agent's action.

        An example of implementation:
        ``# a dummy update rule where new state is equal to last state
        new_states = prev_states
        # mdp with full observability
        observations = new_states
        return prev_states, observations``
        
        :param last_states: Environment state on previous tick.
        :type last_states: list(float[batch_id, memory_id0,[memory_id1],...])

        :param actions: Agent action after observing last state.
        :type actions: list(int[batch_id])
        
        :returns: a tuple of new_states, actions
            new_states: Environment state after processing agent's action.
            observations: What agent observes after committing the last action.
        :rtype: tuple of
            new_states: list(float[batch_id, memory_id0,[memory_id1],...]),
            observations: list(float[batch_id,n_agent_inputs])
        """

        raise NotImplementedError

    def as_layers(self, prev_state_layers=None, action_layers=None,
                  environment_layer_name='EnvironmentLayer'):
        """Lasagne Layer compatibility method.
        Understanding this is not required when implementing your own environments.

        Creates a lasagne layer that makes one step environment updates given agent actions.

        :param prev_state_layers: a layer or a list of layers that provide previous environment state.
            None means create InputLayers automatically

        :param action_layers: a layer or a list of layers that provide agent's chosen action.
            None means create InputLayers automatically.

        :param environment_layer_name: layer's name
        :type environment_layer_name: str

        :return:
            [new_states], [observations]: 2 lists of Lasagne layers
            new states - all states in the same order as in self.state_shapes
            observations - all observations in the order of self.observation_shapes
                
        """
        outputs = EnvironmentStepLayer(self, prev_state_layers, action_layers, name=environment_layer_name).values()

        pivot = len(self.state_shapes)
        return outputs[:pivot], outputs[pivot:]


class EnvironmentStepLayer(DictLayer):
    def __init__(self,
                 environment,
                 prev_state_layers=None,
                 action_layers=None,
                 name=None):
        """Creates a lasagne layer that makes one step environment updates given agent actions.

        :param environment: Environment to interact with.
        :type environment: BaseEnvironment

        :param prev_state_layers: a layer or a list of layers that provide previous environment state.
            None means create automatically

        :param action_layers: a layer or a list of layers that provide agent's chosen action
            None means create automatically

        :param name: layer's name
        """
        self.env = environment

        # default name
        if name is None:
            name = self.env.__class__.__name__ + ".OneStep"

        # create default prev_state and action inputs, if none provided
        if prev_state_layers is None:
            prev_state_layers = [InputLayer((None,) + check_tuple(shape), name=name + ".prev_state[%i]" % i)
                                 for i, shape in enumerate(check_list(self.env.state_shapes))]

        if action_layers is None:
            action_layers = [InputLayer((None,) + check_tuple(shape), name=name + ".agent_action_input[%i]" % i)
                             for i, shape in enumerate(check_list(self.env.action_shapes))]

        self.prev_state_layers = check_list(prev_state_layers)
        self.action_layers = check_list(action_layers)

        # incomings
        incomings = prev_state_layers + action_layers

        output_names = ["environment_new_state.%i" % i for i in range(len(self.prev_state_layers))] + \
                       ["new_observations.%i" % i for i in range(len(self.env.observation_shapes))]

        # output shapes
        state_shapes = [(None,) + check_tuple(shape) for shape in check_list(self.env.state_shapes)]
        observation_shapes = [(None,) + check_tuple(shape) for shape in check_list(self.env.observation_shapes)]
        output_shapes = state_shapes + observation_shapes
        output_shapes = OrderedDict(zip(output_names, output_shapes))

        # output_dtypes
        output_dtypes = check_list(self.env.state_dtypes) + check_list(self.env.observation_dtypes)
        output_dtypes = OrderedDict(zip(output_names, output_dtypes))

        # dict layer init
        super(EnvironmentStepLayer, self).__init__(incomings,
                                                   output_shapes=output_shapes,
                                                   output_dtypes=output_dtypes,
                                                   name=name)

    def get_output_for(self, inputs, **kwargs):
        """Computes get_action_results as a lasagne layer.

        :param inputs: list of previous states and agent actions
            first, all previous states, then all actions in the original order
        :type inputs: list or tuple

        :return: an OrderedDict of [all new states, then all observations]
        :rtype: OrderedDict
        """

        # slice inputs into prev states and actions
        pivot = len(self.prev_state_layers)
        prev_states, actions = inputs[:pivot], inputs[pivot:]

        new_states, observations = self.env.get_action_results(prev_states, actions, **kwargs)

        return OrderedDict(zip(self.keys(), check_list(new_states) + check_list(observations)))
