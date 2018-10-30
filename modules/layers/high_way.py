# coding = utf-8
# author = xy

from torch import nn
import torch


class Highway(nn.Module):

    def __init__(self, input_size, n):
        super(Highway, self).__init__()

        self.input_size = input_size
        self.n = n
        self.highway_linear = nn.ModuleList([nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU())
                                             for _ in range(n)])
        self.highway_gate = nn.ModuleList([nn.Sequential(nn.Linear(input_size, input_size), nn.Sigmoid())
                                           for _ in range(n)])

    def forward(self, x1, x2):
        """
        x1_dim + x2_dim = input_size
        :param x1: (.., x1_dim)
        :param x2: (.., x2_dim)
        :return: (.., input_size)
        """
        x = torch.cat([x1, x2], dim=-1)
        for i in range(self.n):
            h = self.highway_linear[i](x)
            g = self.highway_gate[i](x)
            x = g * h + (1 - g) * x

        return x
