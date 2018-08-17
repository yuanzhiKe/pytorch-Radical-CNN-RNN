import torch
import torch.nn as nn
import torch.nn.functional as F


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
        gate = torch.sigmoid(self.gate_affine(x))
        act = self.activation(self.activation_affine(x))
        output = act + (1 - gate) * x
        return output


class SelfAttention(nn.Module):
    """
    self attention. Detailed functions follows https://www.jstage.jst.go.jp/article/tjsai/33/4/33_D-I23/_article/-char/ja/
    """
    def __init__(self, max_input_length, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_sent_length = max_input_length
        self.W = nn.Linear(self.hidden_dim, max_input_length)
        self.u = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x):
        """
        sum x with self attention over time steps
        :param x: shape: [max_input_length, hidden_dim]
        :return: the output of each time step. Each one is the weighted sum of all the time steps
                 y_i = A_i*X
                 shape: [max_input_length, hidden_dim]
        """
        out_u = torch.tanh(self.u(x))  # shape: [max_input_length, hidden_dim]
        attention_weights = F.softmax(self.W(out_u), dim=1)  # shape:[max_input_length, max_input_length]
        assert attention_weights.size(0) == self.max_sent_length
        assert attention_weights.size(1) == self.max_sent_length
        attention_applied = torch.mm(attention_weights, x)  # shape: [max_input_length, hidden_dim]
        return attention_applied