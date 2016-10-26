"""
tests for recurrence class
"""
import numpy as np
import theano
import agentnet
from agentnet.memory import RNNCell,LSTMCell
import lasagne
from lasagne.layers import *


def test_recurrence():
    sequence = InputLayer((None, None, 3), name='input sequence')
    initial = InputLayer((None, 10), name='rnn zero tick')

    # step
    inp = InputLayer((None, 3))
    prev_rnn = InputLayer((None, 10))
    rnn = RNNCell(prev_rnn, inp, name='rnn')


    rec = agentnet.Recurrence(input_sequences={inp: sequence},
                              state_variables={rnn: prev_rnn},
                              state_init={rnn: initial},  # defaults to zeros
                              unroll_scan=False)

    weights = get_all_params(rec)
    print(weights)

    rnn_states = rec[rnn]

    run = theano.function([sequence.input_var, initial.input_var], get_output(rnn_states), )

    assert tuple(run(np.random.randn(5, 25, 3), np.random.randn(5, 10)).shape) == (5, 25, 10)


def test_recurrence_weird():
    sequence = InputLayer((None, None, 3), name='input sequence')
    initial_cell = InputLayer((None, 20), name='lstm cell zero tick')

    # step
    inp = InputLayer((None, 3))
    prev_rnn = InputLayer((None, 10))
    rnn = RNNCell(prev_rnn, inp, name='rnn')

    prev_lstm_cell = InputLayer((None,20)) #lstm cell
    prev_lstm_hid = InputLayer((None, 20)) #lstm output
    lstm_cell,lstm_hid = LSTMCell(prev_lstm_cell,prev_lstm_hid,input_or_inputs=rnn)

    lstm_hid = DropoutLayer(lstm_hid,p=0.5) #dropout hid, but not cell. Just to check it works

    rec = agentnet.Recurrence(input_sequences={inp: sequence},
                              state_variables={rnn: prev_rnn,
                                               lstm_hid:prev_lstm_hid,
                                               lstm_cell:prev_lstm_cell
                                               },
                              state_init={lstm_cell: initial_cell},  # defaults to zeros
                              unroll_scan=False)

    weights = get_all_params(rec)
    print(weights)

    rnn_states = rec[rnn]
    lstm_cell_states = rec[lstm_cell]
    lstm_hid_states = rec[lstm_hid]

    run = theano.function([sequence.input_var, initial_cell.input_var],
                          get_output([rnn_states,lstm_cell_states,lstm_hid_states]),
                          updates = rec.get_automatic_updates() #if any randomness is used AND unroll_scan,
                                                                # one has to pass automatic updates
                          )

    out = run(np.random.randn(5, 25, 3), np.random.randn(5, 10))

    assert tuple(out[0].shape) == (5, 25, 10) #rnn
    assert tuple(out[1].shape) == (5, 25, 20) #lstm cell
    assert tuple(out[2].shape) == (5, 25, 20) #lstm hid (aka output)


def test_recurrence_mask():
    np.random.seed(1337)

    sequence = InputLayer((None, None, 2), name='input sequence')
    mask = InputLayer((None, None), name="rnn mask [batch,tick]")

    # step
    inp = InputLayer((None, 2))
    prev_rnn = InputLayer((None, 3))
    rnn = RNNCell(prev_rnn, inp, name='rnn',
                  nonlinearity=lasagne.nonlinearities.linear,
                  b=lasagne.init.Constant(100.0))  # init with positive constant to make sure hiddens change

    rec = agentnet.Recurrence(input_sequences={inp: sequence},
                              state_variables={rnn: prev_rnn},
                              unroll_scan=False,
                              mask_input=mask)

    rnn_states = rec[rnn]
    run = theano.function([sequence.input_var, mask.input_var], get_output(rnn_states))

    seq = np.random.randn(4, 5, 2)
    mask = np.zeros([4, 5])
    mask[:2, :3] = 1
    mask[2:, 2:] = 1
    out = run(seq, mask)

    assert tuple(out.shape) == (4, 5, 3)

    diff_out = np.diff(out, axis=1)
    assert (np.diff(out, axis=1)[:2, 2:] == 0).all()
    assert (np.diff(out, axis=1)[:2, :2] != 0).all()
    assert (np.diff(out, axis=1)[2:, 1:] != 0).all()
    assert (np.diff(out, axis=1)[2:, :1] == 0).all()




