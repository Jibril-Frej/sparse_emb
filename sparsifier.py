import torch
from torch import nn


class MLP(nn.Module):
    """
    Multilayer Perceptron.
    """

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 20, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(20, 20, bias=False),
            torch.nn.GELU(),
        )

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)

    def avg_zeros(self):
        zero_count = 0
        param_count = 0
        for param in self.parameters():
            zero_count += torch.sum(param == 0).item()
            param_count += param.numel()
        return zero_count / param_count
