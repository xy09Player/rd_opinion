# encoding = utf-8
# author = xy

import torch
from torch import nn


class Choose(nn.Module):
    def __init__(self, input_size, output_size):
        super(Choose, self).__init__()

        self.W = nn.Linear(input_size, output_size)

    def forward(self, p_vec, a_vec):
        """
         no softmax
        :param p_vec: (batch_size, input_size)
        :param a_vec: (batch_size, .., output_size)
        :return: (batch_size, ..)
        """

        p_vec = self.W(p_vec)
        p_vec = p_vec.unsqueeze(2)  # (batch_size, output_size, 1)
        s = torch.bmm(a_vec, p_vec).squeeze(2)  # (batch_size, ..)

        return s
