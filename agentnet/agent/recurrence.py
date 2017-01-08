"""
AgentNet core abstraction is Recurrence - a lasagne container-layer that can hold
  arbitrary graph and roll it for specified number of steps.

Apart from from MDP Agent, recurrence is also useful for arbitrary recurrent constructs
e.g. convolutional RNN, attentive and/or augmented architectures etc. etc.

As Recurrence is a lasagne layer, one recurrence can be used as a part of computational graph of another recurrence.

"""

from __future__ import division, print_function, absolute_import

from collections import OrderedDict
from itertools import chain
from warnings import warn

import lasagne
from lasagne.layers import InputLayer
from lasagne.utils import unroll_scan
import theano
from theano import tensor as T


from ..utils import insert_dim
from ..utils.format import check_list, check_ordered_dict, unpack_list, supported_sequences,is_theano_object
from ..utils.layers import DictLayer, get_layer_dtype



class Recurrence(DictLayer):
    """
    A generic recurrent layer that works with a custom graph.
    Recurrence is a lasagne layer that takes an inner graph and rolls it for several steps using scan.
    Conversely, it can be used as any other lasagne layer, even as a part of another recurrence.

    Parameters
    --------

    :param input_nonsequences: inputs that are same at each time tick.
        Technically it's a dictionary that maps InputLayer from one-step graph
        to layers from the outer graph.

    :param input_sequences: layers that represent time sequences, fed into graph tick by tick.
        This has to be a dict (one-step input -> sequence layer).
        All such sequences are iterated over FIRST AXIS (axis=1),
        since we consider their shape to be [batch, time, whatever_else...]

    :param tracked_outputs: any layer from the one-state graph which outputs should be
        recorded at every time tick.
        Note that all state_variables are tracked separately, so their inclusion is not needed.


    :param state_variables: a dictionary that maps next state variables to their respective previous state
        keys (new states) must be lasagne layers and values (previous states) must be InputLayers

        Note that state dtype is defined thus:
         - if state key layer has output_dtype, than that type is used for the entire state
         - otherwise, theano.config.floatX is used

    :param state_init: what are the default values for state_variables. In other words, what is
        prev_state for the first iteration. By default it's T.zeros of the appropriate shape.
        Can be a dict mapping state OUTPUTS to their initialisations.
        if so, any states not mentioned in it will be considered zeros
        Can be a list of initializer layers for states in the order of dict.items()
        if so, it's length must match len(state_variables)

    :param unroll_scan: whether or not to use lasagne.utils.unroll_scan instead of theano.scan.
        Note that if unroll_scan == False, one should use .get_rng_updates after .get_output to collect
        automatic updates

    :param n_steps: how many time steps will the recurrence roll for. If n_steps=None, tries to infer it.
       n_steps == None will only work when unroll_scan==False and there are at least some input sequences

    :param batch_size: if the process has no inputs, this expression (int or theano scalar),
        this variable defines the batch size

    :param delayed_states: any states mentioned in this list will be shifted 1 turn backwards
        - from init to n_steps -1. They will be padded with their initial values
        This is intended to allow flipping the recurrence graph to synchronize corresponding values.
        E.g. for MDP, if environment reaction follows agent action,  synchronize observations with actions
        [at i-th turn agent sees i-th observation, than chooses i-th action and gets i-th reward]

    :param verify_graph: whether to assert that all inner graph input layers are registered for the recurrence
        as inputs or prev states and all inputs/prev states are actually needed to compute next states/outputs.
        NOT the same as theano.scan(strict=True).

    :returns: a tuple of sequences with shape [batch,tick, ...]
            - state variable sequences in order of dict.items()
            - tracked_outputs in given order

        WARNING! can not be used further as an atomic lasagne layer.
        Instead, consider calling .get_sequences() or unpacking it

        state_sequence_layers, output_sequence_layers = Recurrence(...).get_sequences()
        (see .get_sequences help for more info)

        OR

        state_seq_layer, ... , output1_seq_layer, output2_seq_layer, ... = Recurrence(...)

    Examples
    --------

    >>> import numpy as np
    >>> import theano
    >>> import agentnet
    >>> from agentnet.memory import RNNCell
    >>> from lasagne.layers import *
    >>> sequence = InputLayer((None,None,3),name='input sequence')
    >>> #one step
    >>> inp = InputLayer((None,3))
    >>> prev_rnn = InputLayer((None,10))
    >>> rnn = RNNCell(prev_rnn,inp,name='rnn')
    >>> #recurrence roll of the one-step graph above.
    >>> rec = agentnet.Recurrence(input_sequences={inp:sequence},
    ...                          state_variables={rnn:prev_rnn},
    ...                          unroll_scan=False)
    >>> weights = get_all_params(rec) #get weights
    >>> print(weights)
    >>> rnn_states = rec[rnn] #get rnn state sequence
    >>> run = theano.function([sequence.input_var], get_output(rnn_states)) #compile applier function as any lasagne network
    >>> run(np.random.randn(5,25,3)) #demo run
    """

    def __init__(self,
                 input_nonsequences=OrderedDict(),
                 input_sequences=OrderedDict(),
                 tracked_outputs=tuple(),
                 state_variables=OrderedDict(),
                 state_init='zeros',
                 unroll_scan=True,
                 n_steps=None,
                 batch_size=None,
                 mask_input=None,
                 delayed_states=tuple(),
                 verify_graph=True,
                 name="YetAnotherRecurrence",
                 ):
        self.n_steps = n_steps
        self.unroll_scan = unroll_scan
        self.updates=theano.OrderedUpdates()
        self.mask_input = mask_input
        assert mask_input is None or isinstance(self.mask_input,lasagne.layers.Layer)

        self.input_nonsequences = check_ordered_dict(input_nonsequences)
        self.input_sequences = check_ordered_dict(input_sequences)

        if is_theano_object(self.n_steps) and self.unroll_scan:
            raise ValueError("Symbolic n_steps is only available when unroll_scan = False.")

        #check consistency of sequences (if any), possibly infer n_steps
        for seq_layer in self.input_sequences.values():
            inp_seq_shape = seq_layer.output_shape
            if len(inp_seq_shape) <2:
                raise ValueError("All input sequences must have at least 2 dimensions: (batch, tick, *whatever)")
            inp_seq_length = inp_seq_shape[1]

            if inp_seq_length is None:
                continue
            elif self.n_steps is None:  # infer n_ticks
                self.n_steps = inp_seq_length
            else:   # verify n_steps
                if inp_seq_length != self.n_steps:
                    warn("number of steps is inconsistent. Either n_steps != input_sequence.output_shape[1] or "
                         "several input sequences have different lengths. "
                         "The Recurrence will roll for {} steps".format(self.n_steps))


        if self.n_steps is None and (self.unroll_scan or len(self.input_sequences) ==0):
            raise ValueError("Please provide n_steps param or meet conditions of inferring it. "
                             "Inferring n_steps is only possible when there is at least one sequence in input_sequences and"
                             "when scan is not unrolled (unroll_scan=False) OR some input_sequences have constant time axis "
                             "(output_shape[1] is not None). n_steps can be symbolic if unroll_scan=False.")


        self.tracked_outputs = check_list(tracked_outputs)

        self.state_variables = check_ordered_dict(state_variables)
        if type(state_variables) is not OrderedDict:
            if len(self.state_variables) > 1:
                warn("""State_variables recommended type is OrderedDict.
                Otherwise, order of agent state outputs from get_sessions and get_agent_reaction methods
                may depend on python configuration.

                Current order is: {state_variables}
                You may find OrderedDict in standard collections module: from collections import OrderedDict
                """.format(state_variables=list(self.state_variables.keys())))


        self.delayed_states = delayed_states

        # initial states

        # convert zeros to empty dict
        if state_init == "zeros":
            state_init = OrderedDict()
        # convert list to dict
        elif state_init in supported_sequences:
            assert len(state_init) == len(self.state_variables)
            state_init = OrderedDict(list(zip(list(self.state_variables.keys()), state_init)))

        # cast to dict and save
        self.state_init = check_ordered_dict(state_init)

        # init base class
        incomings = list(chain(self.state_init.values(),
                               self.input_nonsequences.values(),
                               self.input_sequences.values()))

        if self.mask_input is not None:
            incomings = [self.mask_input]+incomings

        #try to infer batch_size
        rec_inputs = list(chain(self.state_init.keys(),
                               self.input_nonsequences.keys(),
                               self.input_sequences.keys(),
                               self.state_variables.values(),))
        for layer in incomings+rec_inputs:
            layer_bsize = layer.output_shape[0]
            if layer_bsize is not None:
                batch_size = batch_size or layer_bsize
                if batch_size != layer_bsize:
                    warn("Batch size is inconsistent within recurrence. Either batch_size doesn't match output_shape[0]"
                         "of some of the layers OR several incoming or internal layers have different values for "
                         "output_shape[0] (excluding None")

        # if we have no inputs or initialisations, make sure batch_size is specified or infered from layers
        if len(incomings) == 0:
            if batch_size is None:
                raise ValueError("Please specify batch size (int/symbolic). if no inputs (seq/nonseq) are used,"
                                 "agentnet can't infer the batch size. ")
        self.batch_size = batch_size
        

        #all inputs and outputs
        #outputs = all new_state+output dependencies (input layers) lie inside one-step recurrence
        self.all_outputs = list(self.state_variables.keys()) + self.tracked_outputs

        #inputs = all prev states, initializers, nonsequences and sequences
        self.all_inputs = list(chain(self.state_variables.values(),
                                self.state_init.values(),
                                self.input_nonsequences.keys(),
                                self.input_sequences.keys()))


        #output shapes and dtypes
        output_shapes = [tuple(layer.output_shape) for layer in self.all_outputs]
        output_shapes =  [shape[:1] + (self.n_steps,) + shape[1:] for shape in output_shapes]
        output_shapes = OrderedDict(zip(self.all_outputs,output_shapes))
        
        output_dtypes = [get_layer_dtype(layer) for layer in self.all_outputs]
        output_dtypes = OrderedDict(zip(self.all_outputs,output_dtypes))

        super(Recurrence, self).__init__(incomings, 
                                         output_shapes = output_shapes,
                                         output_dtypes = output_dtypes,
                                         name = name)


        # assertions and only assertions. From now this function only asserts stuff.

        if verify_graph:
            # verifying graph topology (assertions)

            # all recurrent graph inputs and prev_states are unique (no input/prev_state is used more than once)
            assert len(self.all_inputs) == len(set(self.all_inputs))

            # all state_init correspond to defined state variables
            for state_out in list(self.state_init.keys()):
                assert state_out in list(self.state_variables.keys())


            # all input layers must be in inputs list (but not all inputs must be input layers)
            for layer in lasagne.layers.get_all_layers(self.all_outputs,treat_as_input=self.all_inputs):
                if type(layer) is InputLayer:
                    if layer not in self.all_inputs:
                        #note: all_inptus is a list, so technically this is a O(n) lookup,
                        #but it's negligible if network has reasonable input count :)
                        message = "One of your network dependencies ({layer_name}) isn't mentioned in Recurrence inputs"
                        raise ValueError(message.format(layer_name=layer.name))

            # verifying shapes (assertions)
            nonseq_pairs = list(chain(self.state_variables.items(),
                                      self.state_init.items(),
                                      self.input_nonsequences.items()))

            for layer_out, layer_in in nonseq_pairs:
                assert tuple(layer_in.output_shape) == tuple(layer_out.output_shape)

            for seq_onestep, seq_input in list(self.input_sequences.items()):
                seq_shape = tuple(seq_input.output_shape)
                step_shape = seq_shape[:1] + seq_shape[2:]
                assert tuple(seq_onestep.output_shape) == step_shape

                seq_len = seq_shape[1]
                assert seq_len is None or n_steps is None or seq_len >= n_steps
                if seq_len is None:
                    warn("You are giving Recurrence an input sequence of undefined length (None).\n" \
                         "Make sure it is always above {}(n_steps) you specified for recurrence".format(n_steps or "<unspecified>"))

    def get_sequence_layers(self):
        """
        returns history of agent interaction with environment for given number of turns.
            [state_sequences] , [output sequences] - a list of all state sequences and  a list of all output sequences
            Shape of each such sequence is [batch, tick, shape_of_one_state_or_output...]
        """
        state_keys = list(self.state_variables.keys())
        state_dict = OrderedDict(zip(state_keys, self[state_keys]))

        output_dict = self[self.tracked_outputs]
        return state_dict, output_dict

    def get_one_step(self, prev_states={}, current_inputs={}, **get_output_kwargs):
        """
        Applies one-step recurrence.
        :param prev_states: a dict {memory output: prev state} or a list of theano expressions for each prev state

        :param current_inputs: a dictionary of inputs that maps {input layers -> theano expressions for them},
            Alternatively, it can be a list where i-th input corresponds to
            i-th input slot from concatenated sequences and nonsequences
            self.input_nonsequences.keys() + self.input_sequences.keys()

        :param get_output_kwargs: any flag that should be passed to the lasagne network for lasagne.layers.get_output method

        :returns:
            new_states: a list of all new_state values, where i-th element corresponds
            to i-th self.state_variables key
            new_outputs: a list of all outputs  where i-th element corresponds
            to i-th self.tracked_outputs key

        """

        # standardize prev_states to a dictionary
        if not isinstance(prev_states, dict):
            # if only one layer given, make a single-element list of it
            prev_states = check_list(prev_states)
            prev_states = OrderedDict(list(zip(list(self.state_variables.keys()), prev_states)))
        else:
            prev_states = check_ordered_dict(prev_states)

        assert len(prev_states) == len(self.state_variables)

        # input map
        ## prev state input layer: prev state expression
        prev_states_kv = [(self.state_variables[s], prev_states[s])
                          for s in list(self.state_variables.keys())]  # prev states

        # prepare both sequence and nonsequence inputs in one dict
        input_layers = list(self.input_nonsequences.keys()) + list(self.input_sequences.keys())

        # standardize current_inputs to a dictionary

        if not isinstance(current_inputs, dict):
            # if only one layer given, make a single-element list of it
            current_inputs = check_list(current_inputs)
            current_inputs = OrderedDict(list(zip(input_layers, current_inputs)))
        else:
            current_inputs = check_ordered_dict(current_inputs)

        assert len(current_inputs) == len(input_layers)

        # second half of input map
        ## external input layer: input expression
        inputs_kv = list(current_inputs.items())

        # compose input map
        input_map = OrderedDict(prev_states_kv + inputs_kv)

        # compose output_list
        output_list = list(self.state_variables.keys()) + self.tracked_outputs

        # call get output
        results = lasagne.layers.get_output(
            layer_or_layers=output_list,
            inputs=input_map,
            **get_output_kwargs
        )

        # parse output array
        n_states = len(self.state_variables)
        n_outputs = len(self.tracked_outputs)

        new_states, new_outputs = unpack_list(results, [n_states, n_outputs])

        return new_states, new_outputs

    def get_automatic_updates(self, recurrent=True):
        """
        Gets all random state updates that happened inside scan.
        :param recurrent: if True, appends automatic updates from previous layers
        :return: theano.OrderedUpdates with all automatic updates
        """
        updates = theano.OrderedUpdates(self.updates)
        if recurrent:
            # add previous layers if any
            for layer in lasagne.layers.get_all_layers(self):
                if layer is self: continue
                if hasattr(layer, 'get_automatic_updates'):
                    layer_updates = layer.get_automatic_updates(recurrent=False)
                    updates += theano.OrderedUpdates(layer_updates)

        # assert there is no inner updates that we can't handle
        inner_graph_outputs = list(self.state_variables.keys()) + self.tracked_outputs
        for inner_layer in lasagne.layers.get_all_layers(inner_graph_outputs):
            if hasattr(inner_layer, 'get_automatic_updates'):
                inner_updates = inner_layer.get_automatic_updates(recurrent=False)
                if len(inner_updates) != 0:
                    raise ValueError(
                        "Currently AgentNet only supports non-unrolled scan if there is no lower-level"
                        "scan with automatic updates. In other words, if you are playing with hierarchical MDP, "
                        " all recurrences must fall into three categories:\n"
                        " - recurrence with unroll_scan - works anywhere\n"
                        " - recurrence with theano.scan - on the bottom level\n"
                        " - recurrence with theano.scan - if all lower levels are unrolled OR have no random state updates\n"
                    )
        return updates

    def get_params(self,**tags):
        """returns all params, including recurrent params from one-step network"""

        params = super(Recurrence, self).get_params(**tags)


        #get inner graph
        inner_layers = lasagne.layers.get_all_layers(self.all_outputs,treat_as_input=self.all_inputs)
        #remove input layers themselves since their params are not used
        inner_layers = [l for l in inner_layers if l not in self.all_inputs]

        inner_params = chain.from_iterable(l.get_params(**tags) for l in inner_layers)
        inner_params = lasagne.utils.unique(inner_params)

        params += inner_params

        return params

    def get_output_for(self, inputs, recurrence_flags={}, **kwargs):
        """
        returns history of agent interaction with environment for given number of turns.
        
        parameters:
            inputs - [state init]  + [input_nonsequences] + [input_sequences]
                Each part is a list of theano expressions for layers in the order they were
                provided when creating this layer.
            recurrence_flags - a set of flags to be passed to the one step agent (anything that lasagne supports)
                e.g. {deterministic=True}
        returns:
            [state_sequences] + [output sequences] - a list of all states and all outputs sequences
            Shape of each such sequence is [batch, tick, shape_of_one_state_or_output...]
        """
        #aliases
        n_states = len(self.state_variables)
        n_state_inits = len(self.state_init)
        n_input_nonseq = len(self.input_nonsequences)
        n_input_seq = len(self.input_sequences)
        n_outputs = len(self.tracked_outputs)

        #slice inputs

        if self.mask_input is not None:
            mask,inputs = inputs[0],inputs[1:]

        initial_states_provided, nonsequences, sequences = unpack_list(inputs, [n_state_inits, n_input_nonseq, n_input_seq])

        # infer batch size
        if self.batch_size is not None:
            batch_size = self.batch_size
        elif len(inputs) != 0:
            batch_size = inputs[0].shape[0]
        else:
            raise ValueError("Need to set batch_size explicitly for recurrence")


        # reshape sequences from [batch, time, ...] to [time,batch,...] to fit scan
        sequences = [seq.swapaxes(1, 0) for seq in sequences]

        #here we create outputs_info for scan
        ## initial states that are given as input
        initial_states_provided = OrderedDict(list(zip(self.state_init, initial_states_provided)))

        def get_initial_state(state_out_layer,batch_size=batch_size):
            """Pick dedicated initial state or create zeros of appropriate shape and dtype"""
            # if we have a dedicated init, use it
            if state_out_layer in initial_states_provided:
                initial_state = initial_states_provided[state_out_layer]
            # otherwise initialize with zeros
            else:
                dtype = get_layer_dtype(state_out_layer)
                initial_state = T.zeros((batch_size,) + tuple(state_out_layer.output_shape[1:]),dtype=dtype)

                #cast to non-broadcastable tensortype
                t_state = T.TensorType(dtype, (False,) * initial_state.ndim)
                initial_state = t_state.convert_variable(initial_state)
                assert initial_state is not None #if None, conversion failed. report ASAP

            return initial_state

        initial_states = list(map(get_initial_state, self.state_variables))


        #dummy values for initial outputs. They have no role in computation, but if nonsequences are present,
        # AND scan is not unrolled, the step function will not receive prev outputs as parameters, while
        # if unroll_scan, these parameters are present. we forcibly initialize outputs to prevent
        # complications during parameter parsing in step function below.
        initial_output_fillers = list(map(get_initial_state, self.tracked_outputs))
        
        
        outputs_info = initial_states + initial_output_fillers
        
        # recurrent step function
        def step(*args):

            sequence_slices, prev_states, prev_outputs, nonsequences = \
                unpack_list(args, [n_input_seq, n_states, n_outputs, n_input_nonseq])
            # make dicts of prev_states and inputs
            prev_states_dict = OrderedDict(zip(list(self.state_variables.keys()), prev_states))

            input_layers = list(chain(self.input_nonsequences.keys(), self.input_sequences.keys()))
            assert len(input_layers) == len(nonsequences + sequence_slices)

            inputs_dict = OrderedDict(zip(input_layers, nonsequences + sequence_slices))

            # call one step recurrence
            new_states, new_outputs = self.get_one_step(prev_states_dict, inputs_dict, **recurrence_flags)


            #make sure output variable is of exactly the same type as corresponding input
            new_states = [prev_state.type.convert_variable(state)
                            for (prev_state,state) in zip(prev_states,new_states)]
            assert None not in new_states, "Some state variables has different dtype/shape from init ."

            new_outputs = [prev_out.type.convert_variable(out)
                            for (prev_out,out) in zip(prev_outputs,new_outputs)]
            assert None not in new_outputs, "Some of the tracked outputs has shape/dtype changing over time. Please report this."

            return new_states + new_outputs

        ###handling mask_input###

        #a step function that utilizes a mask
        def step_masked(mask_t,*args):
            #unpack arrays
            sequence_slices, prev_states, prev_outputs, nonsequences = \
                unpack_list(args, [n_input_seq, n_states, n_outputs, n_input_nonseq])

            #get regular step
            new_states_and_outputs = step(*args)
            old_states_and_outputs = prev_states+prev_outputs

            #if mask_t, return new ones, else return old ones
            def apply_mask(mask_t,new_state,old_state):
                assert new_state.ndim == old_state.ndim
                ndim = new_state.ndim
                #append dims to mask
                pattern = list(range(mask_t.ndim)) + ['x'] * (ndim - mask_t.ndim)

                return T.switch(mask_t.dimshuffle(pattern),
                                new_state, old_state)


            next_states_and_outputs = [apply_mask(mask_t,new_state,old_state)
                                       for new_state,old_state in zip(new_states_and_outputs,
                                                                      old_states_and_outputs)]

            return next_states_and_outputs


        if self.mask_input is not None:
            sequences = [mask.swapaxes(1, 0)]+sequences
            step_function = step_masked
        else:
            step_function = step


        #scan itself
        if self.unroll_scan:
            # call scan itself
            history = unroll_scan(step_function,
                                  sequences=sequences,
                                  outputs_info=outputs_info,
                                  non_sequences=nonsequences,
                                  n_steps=self.n_steps
                                  )
            self.updates=OrderedDict()
        else:
            history,updates = theano.scan(step_function,
                                  sequences=sequences,
                                  outputs_info=outputs_info,
                                  non_sequences=nonsequences,
                                  n_steps=self.n_steps
                                  )
            self.updates = updates
            if len(updates) !=0:
                warn("Warning: recurrent loop without unroll_scan got nonempty random state updates list. That happened"
                     " because there is some source of randomness (e.g. dropout) inside recurrent step graph."
                     " To compile such graph, one must either call .get_automatic_updates() right after .get_output"
                     " and pass these updates to a function, or use no_defalt_updates=True when compiling theano.function.")


        # reordering from [time,batch,...] to [batch,time,...]
        history = [(var.swapaxes(1, 0) if var.ndim > 1 else var) for var in check_list(history)]

        assert len(history) == n_states+n_outputs

        state_seqs, output_seqs = unpack_list(history, [n_states, n_outputs])

        # handle delayed_states
        # selectively shift state sequences by 1 tick into the past, padding with their initialisations
        for i in range(len(state_seqs)):
            if list(self.state_variables.keys())[i] in self.delayed_states:
                state_seq = state_seqs[i]
                state_init = initial_states[i]
                state_seq = T.concatenate([insert_dim(state_init, 1), state_seq[:, :-1]], axis=1)
                state_seqs[i] = state_seq


        #keys corresponding to output sequences. Note that we do not use self.keys() to correctly
        # handle cases where some variable is present in both state_variables and tracked_outputs
        output_keys = list(self.state_variables.keys()) + list(self.tracked_outputs)
        output_values = state_seqs + output_seqs
        assert len(output_keys) == len(output_values)
        return OrderedDict(zip(output_keys,output_values))

