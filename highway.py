import torch
import torch.nn as nn
import torch.nn.functional


class Highway(nn.Module):
    # paper: https://arxiv.org/pdf/1505.00387.pdf
    def __init__(self, input_dim, activation_function):
        super(Highway, self).__init__()
        self.activation_affine = nn.Linear(input_dim, input_dim)
        self.gate_affine = nn.Linear(input_dim, input_dim)
        self.activation = activation_function

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, input_dim]
            :return: tensor with shape of [batch_size, input_dim]
            applies g(x) ⨀ x + (1 - g(x)) ⨀ f(x) transformation,
            f is non-linear transformation, g(x) is transformation gate obtained by sigmoid
            and ⨀ is element-wise multiplication
            """
        gate = torch.nn.functional.sigmoid(self.gate_affine(x))
        act = self.activation(self.activation_affine(x))
        output = act + (1 - gate) * x
        return output