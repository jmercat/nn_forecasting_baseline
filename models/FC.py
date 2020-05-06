import torch
from models.reshape import Reshape, Permute

def FC_model(args):

    n_inputs = int(args.time_hist/args.dt * 2)
    n_outputs = int(args.time_pred/args.dt * 2)
    shape_output = (-1, int(args.time_pred/args.dt), 2)

    model = torch.nn.Sequential(
        Permute(1, 0, 2),
        Reshape(-1, n_inputs),
        torch.nn.Linear(n_inputs, args.feature_size),
        torch.nn.ReLU(),
        torch.nn.Linear(args.feature_size, args.feature_size),
        torch.nn.ReLU(),
        torch.nn.Linear(args.feature_size, n_outputs),
        Reshape(*shape_output),
        Permute(1, 0, 2)
    )

    return model