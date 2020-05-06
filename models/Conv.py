import torch
from models.reshape import Reshape, Permute

class PrintShape(torch.nn.Module):
    def __init__(self, *args):
        super(PrintShape, self).__init__()

    def forward(self, x, msg=''):
        if msg != '':
            print(msg)
        print(x.shape)
        return x

class KeepLast(torch.nn.Module):
    def __init__(self, *args):
        super(KeepLast, self).__init__()

    def forward(self, x, msg=''):
        return x[:, :, -1]

def conv_model(args):
    n_inputs = int(args.time_hist/args.dt)
    n_outputs = int(args.time_pred/args.dt * 2)
    shape_output = (-1, int(args.time_pred/args.dt), 2)
    kernel_size = 3
    size_1 = n_inputs - (kernel_size-1)
    size_2 = int(size_1/2)
    size_3 = size_2 - (kernel_size-1)
    # size_3 = 1


    model = torch.nn.Sequential(
        Permute(1, 2, 0),
        torch.nn.Conv1d(2, args.feature_size, kernel_size, 1),
        torch.nn.ReLU(),
        torch.nn.MaxPool1d(2, 2),
        torch.nn.Conv1d(args.feature_size, args.feature_size, kernel_size, 1),
        torch.nn.ReLU(),
        # KeepLast(),
        Reshape(-1, size_3*args.feature_size),
        torch.nn.Linear(size_3*args.feature_size, n_outputs),
        Reshape(*shape_output),
        Permute(1, 0, 2)
    )
    return model