import torch.nn as nn

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.reshape(self.shape)

class Permute(nn.Module):
    def __init__(self, *args):
        super(Permute, self).__init__()
        self.permute = args

    def forward(self, x):
        return x.permute(self.permute)
