"""
'Layers' introduces a number of auxiliary lasagne layers that are used throughout AgentNet and examples
"""

import theano
import theano.tensor as T

from lasagne.layers import MergeLayer, Layer
from lasagne.layers import NonlinearityLayer, ElemwiseMergeLayer
from collections import OrderedDict

from .format import supported_sequences, check_list, check_ordered_dict
from ..utils.logging import warn


# shortcut functions
def mul(*args, **kwargs):
    """Element-wise multiply layers"""
    inp_names = [layer.name or "layer" + str(i) for i, layer in enumerate(args)]
    kwargs["name"] = kwargs.get("name",
                                "sum(%s)" % (', '.join(inp_names)))

    return ElemwiseMergeLayer(args, T.mul, **kwargs)


def add(*args, **kwargs):
    """Element-wise sum of layers"""
    inp_names = [layer.name or "layer" + str(i) for i, layer in enumerate(args)]
    kwargs["name"] = kwargs.get("name",
                                "sum(%s)" % (', '.join(inp_names)))

    return ElemwiseMergeLayer(args, T.add, **kwargs)


def clip_grads(layer, clipping_bound):
    """Clips grads passing through a lasagne.layers.layer"""
    clipping_bound = abs(clipping_bound)
    grad_clip_op = lambda v: theano.gradient.grad_clip(v, -clipping_bound, clipping_bound)
    name = layer.name or "clipping_layer"
    return NonlinearityLayer(layer, grad_clip_op, name=name + ".grad_clip")


# dtype checker
def get_layer_dtype(layer, default=None):
    """ takes layer's output_dtype property if it is defined,
    otherwise defaults to default or (if it's not given) theano.config.floatX"""
    return layer.output_dtype if hasattr(layer, "output_dtype") else default or theano.config.floatX


class DictLayer(MergeLayer):
    """
    A base class for Lasagne layer that returns several outputs.

    For a custom dictlayer you should implement get_output_for so that it returns a dict of {key:tensor_for_that_key}

    By default it just outputs all the inputs IF their number matches, otherwise it raises an exception.

    In other words, if you return 'foo' and 'bar' of shapes (None, 25) and (None,15,5,7),
    self.get_output_shape must be {'foo': (None,25), 'bar': (None,15,5,7)}

    warning: this layer is needed for the purpose of graph optimization,
        it slightly breaks Lasagne conventions, so it is hacky.

    :param incomings: Incoming layers.
    :type incomings: lasagne.layers.Layer or a list of such

    :param output_shapes: Shapes of key-value outputs from the DictLayer.
    :type output_shapes: dict of { output_key: tuple of shape dimensions (like input layer shape) }
        or a list of shapes, in which case keys are integers from 0 to len(output_shapes)

    :param output_dtypes: If provided, defines the dtypes of all key-value outputs. None means all float32.
    :type output_dtypes: None, dict of {key:dtype of output} or a list of dtypes.
        Key names must match those in output_shapes.

    """

    def __init__(self, incomings, output_shapes, output_dtypes=None, **kwargs):

        # infer keys
        if isinstance(output_shapes, dict):
            keys = list(output_shapes.keys())
            if not isinstance(output_shapes, OrderedDict):
                warn("DictLayer output_shapes should be collections.OrderedDict, instead given a regular dict. "
                     "Assuming keys order to be {}. If you want a different order, consider sending an OrderedDict"
                     " instead.".format(keys),verbosity_level=2)

        elif isinstance(output_dtypes, dict):
            keys = list(output_dtypes.keys())
            warn("Warning: DictLayer running with keys inferred from dtypes:" + str(keys))
        else:
            keys = list(range(len(check_list(output_shapes))))
            warn("Warning: DictLayer running with default keys:" + str(keys))

        # convert * to OrderedDict with same order of keys
        if not isinstance(output_shapes, dict):
            output_shapes = OrderedDict(
                zip(keys, check_list(output_shapes)))
        else:
            # output_shapes is dict
            output_shapes = check_ordered_dict(output_shapes)

        if not isinstance(output_dtypes, dict):
            if output_dtypes is None:
                output_dtypes = {key: theano.config.floatX for key in keys}
            else:
                output_dtypes = OrderedDict(
                    zip(keys, check_list(output_dtypes)))
        else:
            # output_dtypes is dict
            output_dtypes = OrderedDict([(key, output_dtypes[key]) for key in keys])

        # save them all
        self.output_shapes = output_shapes
        self.output_dtypes = output_dtypes
        self.output_keys = keys

        super(DictLayer, self).__init__(check_list(incomings), **kwargs)

    def get_output_for(self, inputs, **flags):
        """By default returns a dict of {key:theano_tensor}
        please override this method to implement layer functionality (if any)."""
        if len(inputs) == len(self.output_keys):
            return OrderedDict(zip(self.output_keys, inputs))
        else:
            raise NotImplementedError("One must implement get_output_for logic for DictLayer")

    def get_output_shape_for(self, input_shape, **flags):
        """
        DictLayer's shape is a dictionary of shapes for each output (each value is a tuple)
        """
        return self.output_shapes

    @property
    def output_dtype(self):
        """ dtypes of dict outputs"""
        return self.output_dtypes

    def keys(self):
        """a dict-like memthod that returns all keys"""
        return self.output_keys

    def values(self):
        """ list of output layers"""
        return self[list(self.keys())]

    def __len__(self):
        """an amount of output layers in a tuple"""
        return len(self.output_shapes)

    def __iter__(self):
        """Dict layer should not be iterated"""
        raise TypeError("DictLayer.__iter__ not supported. One must use .keys() or .values().")

    def __getitem__(self, key):
        """ returns a lasagne layer that yields value corresponding to i-th key.
        parameters:
            key - an key or an iterable of keys or a slice.
        returns:
            - a particular layer if a single index is given
            - a list of layers if list/tuple of keys is given
            - a slice of values if slice is given
        """

        if type(key) in supported_sequences:
            return tuple(self[key_i] for key_i in key)
        else:
            assert key in self.keys()
            return DictElementLayer(self, key)


class DictElementLayer(Layer):
    """A special-purpose layer that returns a particular element of DictLayer"""

    def __init__(self, incoming, key, name=None):
        assert isinstance(incoming, DictLayer)
        assert key in incoming.keys()

        self.input_layer = incoming
        self.input_shape = incoming.output_shape
        self.key = key

        self.name = name
        self.params = OrderedDict()
        self.get_output_kwargs = []

        #deliberately NOT calling Layer's consructor

    def get_output_for(self, inputs_dict, **flags):
        return inputs_dict[self.key]

    def get_output_shape_for(self, input_shapes_dict, **flags):
        return input_shapes_dict[self.key]

    @property
    def output_dtype(self):
        input_dtypes = get_layer_dtype(self.input_layer)
        assert isinstance(input_dtypes, dict)
        return input_dtypes[self.key]
