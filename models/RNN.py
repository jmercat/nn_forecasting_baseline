import torch.nn as nn
import torch
from torch.nn import Parameter
import torch.jit as jit
import math
from models.reshape import Reshape, Permute

class LSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        scale = math.sqrt(hidden_size)
        self.weight_hh = Parameter(torch.rand(4 * hidden_size, hidden_size)*2*scale - scale)
        self.bias_hh = Parameter(torch.rand(4 * hidden_size)*2*scale - scale)
        # self.bias_hi = Parameter(torch.rand(4 * hidden_size)*2*scale - scale)

    @jit.script_method
    def forward(self, hx, cx):
        # type: (Tensor, Tensor) ->  Tuple[Tensor, Tensor]
        gates = (torch.mm(hx, self.weight_hh.t()) + self.bias_hh.t())
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy

class LSTM_model(nn.Module):
    def __init__(self, args):
        super(LSTM_model, self).__init__()
        self.args = args
        self.n_pred_step = int(args.time_pred/args.dt)
        self.encoder = nn.LSTM(2, args.feature_size)
        self.decoder = LSTMCell(args.feature_size, args.feature_size)
        self.out = nn.Linear(args.feature_size, 2)
        self.h = nn.Parameter(torch.zeros([1, 1, args.feature_size]))
        self.c = nn.Parameter(torch.zeros([1, 1, args.feature_size]))

    def forward(self, inpt):

        h = self.h.repeat(1, inpt.shape[1], 1)
        c = self.c.repeat(1, inpt.shape[1], 1)

        _, (h, c) = self.encoder(inpt, (h, c))

        outputs = []
        (h, c) = (h.squeeze(0), c.squeeze(0))
        for i in range(self.n_pred_step):
            h, c = self.decoder(h, c)
            outputs += [self.out(h)]
        return torch.stack(outputs)


class LSTM_model2(nn.Module):
    def __init__(self, args):
        super(LSTM_model2, self).__init__()
        self.args = args
        self.n_pred_step = int(args.time_pred/args.dt)
        self.encoder = nn.LSTM(2, args.feature_size)
        self.decoder = nn.LSTM(args.feature_size, args.feature_size)
        self.out = nn.Linear(args.feature_size, 2)
        self.h = nn.Parameter(torch.zeros([1, 1, args.feature_size]))
        self.c = nn.Parameter(torch.zeros([1, 1, args.feature_size]))

    def forward(self, inpt):

        h = self.h.repeat(1, inpt.shape[1], 1)
        c = self.c.repeat(1, inpt.shape[1], 1)

        _, (h, c) = self.encoder(inpt, (h, c))

        # out = h.repeat(self.n_pred_step, 1, 1)
        out = torch.zeros(self.n_pred_step, inpt.shape[1], h.shape[2])
        out, _ = self.decoder(out, (h, c))
        return self.out(out)
