"""
A few auxiliary methods that work with supported collection formats.
Used widely within AgentNet.
"""

from collections import OrderedDict
from warnings import warn

import lasagne
import numpy as np


def is_layer(var):
    """checks if var is lasagne layer"""
    return isinstance(var, lasagne.layers.Layer)


def is_theano_object(var):
    """checks if var is a theano input, transformation, constant or shared variable"""
    return type(var).__module__.startswith("theano")


def is_numpy_object(var):
    """checks if var is a theano input, transformation, constant or shared variable"""
    return type(var).__module__.startswith("numpy")


supported_sequences = (tuple, list)


def check_sequence(variables):
    """
    Ensure that variables is one of supported_sequences or converts to one.
    If naive conversion fails, throws an error.
    """
    if any(isinstance(variables, seq) for seq in supported_sequences):
        return variables
    else:
        # If it is a numpy or theano array, excluding numpy array of objects, return a list with single element
        # Yes, i know it's messy. Better options are welcome for pull requests :)
        if (is_theano_object(variables) or is_numpy_object(variables)) and variables.dtype != np.object:
            return [variables]
        elif hasattr(variables, '__iter__'):
            # Elif it is a different kind of sequence try casting to tuple. If cannot, treat that it will be treated
            # as an atomic object.
            try:
                tupled_variables = tuple(variables)
                message = """{variables} of type {var_type} will be treated as a sequence of {len_casted} elements,
                not a single element.
                If you want otherwise, please pass it as a single-element list/tuple.
                """
                warn(message.format(variables=variables, var_type=type(variables), len_casted=len(tupled_variables)))
                return tupled_variables
            except:
                message = """
                {variables} of type {var_type} will be treated as a single input/output tensor,
                and not a collection of such.
                If you want otherwise, please cast it to list/tuple.
                """
                warn(message.format(variables=variables, var_type=type(variables)))
                return [variables]
        else:
            # otherwise it's a one-element list
            return [variables]


def check_list(variables):
    """Ensure that variables is a list or converts to one.
    If naive conversion fails, throws an error
    :param variables: sequence expected
    """
    return list(check_sequence(variables))


def check_tuple(variables):
    """Ensure that variables is a list or converts to one.
    If naive conversion fails, throws an error
    :param variables: sequence expected
    """
    return tuple(check_sequence(variables))


def check_ordered_dict(variables):
    """Ensure that variables is an OrderedDict
    :param variables: dictionary expected
    """
    assert isinstance(variables, dict)
    try:
        return OrderedDict(list(variables.items()))
    except:
        raise ValueError("Could not convert {variables} to an ordered dictionary".format(variables=variables))


def unpack_list(array, parts_lengths):
    """
    Returns slices of the input list a.
    unpack_list(a, [2,3,5]) -> a[:2], a[2:2+3], a[2+3:2+3+5]

    :param array: array-like or tensor variable
    :param parts_lengths: lengths of subparts

    """
    borders = np.concatenate([[0], np.cumsum(parts_lengths)])

    groups = []
    for low, high in zip(borders[:-1], borders[1:]):
        groups.append(array[low:high])

    return groups
