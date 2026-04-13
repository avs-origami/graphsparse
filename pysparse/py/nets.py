import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class LinHead(nn.Module):
    def __init__(self, mode: str, d_model: int, hidden_size: int, num_hidden: int, num_actions: int):
        super().__init__()
        layers = []

        layers.append(layer_init(nn.Linear(
            d_model,
            int(d_model / 4)
        )))

        layers.append(nn.ReLU())

        layers.append(layer_init(nn.Linear(
            int(d_model / 4),
            hidden_size
        )))

        layers.append(nn.ReLU())

        for i in range(num_hidden - 2):
            layers.append(layer_init(nn.Linear(hidden_size, hidden_size)))
            layers.append(nn.ReLU())

        if mode == "actor":
            layers.append(layer_init(nn.Linear(hidden_size, num_actions), std=0.01))
        elif mode == "critic":
            layers.append(layer_init(nn.Linear(hidden_size, 1), std=1.0))
        else:
            raise RuntimeError(f"LinHead: unknown mode {mode}, valid modes are 'actor', 'critic', and 'shared'")
        
        self.layers = nn.Sequential(*layers)


    def forward(self, x: Tensor) -> Tensor:
        return self.layers.forward(x)