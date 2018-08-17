import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from nnBlocks import Highway, SelfAttention


class RNNSAClassifier(nn.Module):
    """
    The top layer of the whole text. with self-attention
    (define bottom <- subword/char/radical/word, middle <- phrase/clause, etc.)
    Note: do not contain embedding layer
    """
    def __init__(self, word_embedding_dim, hidden_dim, target_size, max_input_length):
        super(RNNSAClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_sent_length = max_input_length
        self.lstm = nn.LSTM(word_embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, target_size)
        self.self_attention = SelfAttention(max_input_length, hidden_dim)

    def init_hidden(self):
        # LSTM has 2 hidden state
        return (autograd.Variable(torch.cuda.FloatTensor(1, 1, self.hidden_dim).fill_(0)),
                autograd.Variable(torch.cuda.FloatTensor(1, 1, self.hidden_dim).fill_(0)))

    def forward(self, sentence, hidden):
        """
        :param sentence: shape: [max_input_length, word_embedding_dim]
        """
        assert sentence.size(0) == self.max_sent_length
        out, hidden = self.lstm(sentence.unsqueeze(1), hidden) # out's shape: [max_input_length, 1, hidden_dim]
        assert out.size(0) == self.max_sent_length
        assert out.size(2) == self.hidden_dim
        out = out.squeeze(1)  # shape: [max_input_length, hidden_dim]
        attention_applied = self.self_attention(out)
        summed = torch.sum(attention_applied, 0)  # shape: [hidden_dim]
        normalized = summed / self.max_sent_length
        tag_prediction = self.hidden2tag(normalized)  # shape: [target_size]
        return tag_prediction, hidden  # do not forget softmax/log_softmax outside! - pytorch style
