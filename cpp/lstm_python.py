import time
import torch

import math

# Our module!
import lstm_cpp


class LLTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, Wy, bias, by, old_h, old_cell):
        outputs = lstm_cpp.forward(
            input, weights, Wy, bias, by, old_h, old_cell)
        print(outputs[-2].T.size())
        new_h, new_cell = outputs[1:3]
        variables = outputs[2:] + [weights]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        print('--------ok----   ', grad_h.size())
        outputs = lstm_cpp.backward(grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_variables)
        dWf, dbf, dWi, dbi, dWo, dbo, dWc, dbc, dh, dc = outputs
        return dWf, dbf, dWi, dbi, dWo, dbo, dWc, dbc, dh, dc


class LLTM(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = torch.nn.Parameter(torch.empty(
            4 * state_size, input_features + state_size))
        self.Wy = torch.nn.Parameter(torch.empty(input_features, state_size))
        self.bias = torch.nn.Parameter(torch.empty(4 * state_size))
        self.by = torch.nn.Parameter(torch.empty(input_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return LLTMFunction.apply(input, self.weights, self.Wy, self.bias, self.by, *state)


batch_size = 16
input_features = 32  # D
state_size = 128  # H

X = torch.randn(batch_size, input_features)
h = torch.randn(batch_size, state_size)
C = torch.randn(batch_size, state_size)

rnn = LLTM(input_features, state_size)

forward = 0
backward = 0
for _ in range(100000):
    start = time.time()
    new_h, new_C = rnn(X, (h, C))
    forward += time.time() - start

    start = time.time()
    (new_h.sum() + new_C.sum()).backward()
    backward += time.time() - start

print('Forward: {:.3f} us | Backward {:.3f} us'.format(
    forward * 1e6/1e5, backward * 1e6/1e5))
