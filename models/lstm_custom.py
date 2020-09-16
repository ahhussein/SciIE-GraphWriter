import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.jit as jit
import warnings
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
import numbers
import torch.nn.functional as nnf

'''
Some helper classes for writing custom TorchScript LSTMs.

Goals:
- Classes are easy to read, use, and extend
- Performance of custom LSTMs approach fused-kernel-levels of speed.

'''


LSTMState = namedtuple('LSTMState', ['hx', 'cx'])


def reverse(lst):
    # type: (List[Tensor]) -> List[Tensor]
    return lst[::-1]


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.zeros(4 * hidden_size))
        self.bias_hh = Parameter(torch.zeros(4 * hidden_size))

    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)

class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(input_size + hidden_size, 3 * hidden_size))
        #self.weight_hh = Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(3 * hidden_size))
        #self.bias_hh = Parameter(torch.randn(3 * hidden_size))

        self.weight_ih.weight = self.weights_init_cat(self.weight_ih.shape)
        #self.weight_hh.weight = self.weights_init(self.weight_hh.shape)

    def forward(self, input, state, dropout_mask):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state

        hx *= dropout_mask

        gates = torch.mm(torch.cat((input, hx), 1), self.weight_ih) + self.bias_ih
        i, j, o = gates.chunk(3, 1)

        i = torch.sigmoid(i)
        new_cx = (1 - i) * cx + i * torch.tanh(j)
        new_h = torch.tanh(new_cx) * torch.sigmoid(o)

        return new_h, (new_h, new_cx)

    def weights_init_cat(self, shape):
        return torch.cat([self.weights_init(shape[0], shape[1]//3) for i in range(3)], 1)

    def weights_init(self, shapex, shapey):
        M1 = torch.randn(shapex, shapey, dtype=torch.float32)
        M2 = torch.randn(shapex, shapey, dtype=torch.float32)
        Q1, R1 = torch.qr(M1)
        Q2, R2 = torch.qr(M2)
        Q1 = Q1 * torch.sign(torch.diag(R1))
        Q2 = Q2 * torch.sign(torch.diag(R2))
        n_min = min(shapex, shapey)
        params = torch.mm(Q1[:, :n_min], Q2[:n_min, :])
        return params

class LSTMLayer(nn.Module):
    def __init__(self, cell, dropout, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)
        self.dropout = dropout

    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])

        dropout_mask = nnf.dropout(torch.ones_like(state[0]), self.dropout)

        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state, dropout_mask)
            outputs += [out]
        return torch.stack(outputs), state


class ReverseLSTMLayer(nn.Module):
    def __init__(self, cell, dropout, *cell_args):
        super(ReverseLSTMLayer, self).__init__()
        self.cell = cell(*cell_args)
        self.dropout = dropout

    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = reverse(input.unbind(0))
        outputs = jit.annotate(List[Tensor], [])

        dropout_mask = nnf.dropout(torch.ones_like(state[0]), self.dropout)

        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state, dropout_mask)
            outputs += [out]
        return torch.stack(reverse(outputs)), state


class BidirLSTMLayer(nn.Module):
    __constants__ = ['directions']

    def __init__(self, cell, dropout, *cell_args):
        super(BidirLSTMLayer, self).__init__()
        self.directions = nn.ModuleList([
            LSTMLayer(cell, dropout, *cell_args),
            ReverseLSTMLayer(cell, dropout, *cell_args),
        ])

    def forward(self, input, states):
        # type: (Tensor, List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        # List[LSTMState]: [forward LSTMState, backward LSTMState]
        outputs = jit.annotate(List[Tensor], [])
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for direction in self.directions:
            state = states[i]
            out, out_state = direction(input, state)
            outputs += [out]
            output_states += [out_state]
            i += 1
        return torch.cat(outputs, -1), output_states


def flatten_states(states):
    states = list(zip(*states))
    assert len(states) == 2
    return [torch.stack(state) for state in states]


def double_flatten_states(states):
    # XXX: Can probably write this in a nicer way
    states = flatten_states([flatten_states(inner) for inner in states])
    return [hidden.view([-1] + list(hidden.shape[2:])) for hidden in states]
