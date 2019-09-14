import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import sigmoid, tanh

class LSTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, Wy, bias, by, old_h, old_cell):
        X = torch.concat(1, [old_h, input])
        gate_weights = torch.addmm(bias, X, weights.T)
        gates = gate_weights.chunk(4,1)

        hf, hi, ho, hc = sigmoid(gates[0]), sigmoid(gates[1]), sigmoid(gates[2]), tanh(gates[3])

        new_cell = hf * old_cell + hi * hc
        new_h = ho * tanh(new_cell)

        y = torch.addmm(by, h, Wy.T)

        prob = torch.softmax(y)
        state = (new_h, new_cell)

        cache = [new_h, new_cell, y, new_h, new_cell, hf, hi, ho, hc, X, gate_weights]

        ctx.save_for_backward(*cache)

        return prob, state, cache

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        outputs = lstm_cpp.backward(grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_variables)
        d_input, d_weights, dWy, d_bias, dby, dh, dc = outputs
        return d_input, d_weights, dWy, d_bias, dby, dh, dc

class LSTM(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super(LSTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = torch.nn.Parameter(torch.empty(4 * state_size, input_features + state_size))
        self.Wy = torch.nn.Parameter(torch.empty(input_features, state_size))
        self.bias = torch.nn.Parameter(torch.empty(4 * state_size))
        self.by = torch.nn.Parameter(torch.empty(input_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return LSTMFunction.apply(input, self.weights, self.Wy, self.bias, self.by, *state)


batch_size = 16
input_features = 32  # D
state_size = 128  # H

X = torch.randn(batch_size, input_features)
h = torch.randn(batch_size, state_size)
C = torch.randn(batch_size, state_size)
