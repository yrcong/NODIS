import torch
import torch.nn as nn
from torchdiffeq import odeint

class odeFunc1(nn.Module):
    def __init__(self, bidirectional):
        super(odeFunc1, self).__init__()

        self.bidirectional = bidirectional
        if self.bidirectional:
            self.lstm = nn.LSTM(input_size=1024, hidden_size=512, bidirectional=True)
        else:
            self.lstm = nn.LSTM(input_size=1024, hidden_size=1024, bidirectional=False)
        self.nfe = 0
        self.hidden_state = None

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if self.bidirectional:
            return (torch.zeros(2, batch_size, 512).cuda(),
                    torch.zeros(2, batch_size, 512).cuda())
        else:
            return (torch.zeros(1, batch_size, 1024).cuda(),
                    torch.zeros(1, batch_size, 1024).cuda())

    def forward(self, t, x):

        x, self.hidden_state = self.lstm(x, self.hidden_state)
        self.nfe += 1
        return x


class odeFunc2(nn.Module):
    def __init__(self, use_cuda):
        super(odeFunc2, self).__init__()
        self.use_cuda = use_cuda
        self.lstm = nn.LSTM(input_size=1024, hidden_size=512, bidirectional=True)
        self.nfe = 0
        self.hidden_state = None

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if self.use_cuda:
            return (torch.zeros(2, batch_size, 512).cuda(),
                    torch.zeros(2, batch_size, 512).cuda())
        else:
            return (torch.zeros(2, batch_size, 512),
                    torch.zeros(2, batch_size, 512))

    def forward(self, t, x):

        x, self.hidden_state = self.lstm(x, self.hidden_state)
        self.nfe += 1
        return x


class odeBlock(nn.Module):
    def __init__(self, odeFunc):
        super(odeBlock, self).__init__()
        self.odeFunc = odeFunc
        self.time = torch.tensor([0, 1.5]) # time interval

    def forward(self, x, batch_size):
        self.odeFunc.hidden_state = self.odeFunc.init_hidden(batch_size)
        self.odeFunc.nfe = 0
        out = odeint(self.odeFunc, x, self.time, rtol=0.01, atol=0.01)
        return out