import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional
from collections import OrderedDict
from highway import Highway


class RadicalCNN(nn.module):
    # encode word feature from radicals

    def __init__(self, radical_vocab_size, radical_emb_dim, basic_filter_dim, max_word_length, max_character_length):
        super(RadicalCNN, self).__init__()
        self.radical_emb_lookup = nn.Embedding(radical_vocab_size, radical_emb_dim, scale_grad_by_freq=False)
        # filter_out_channels = [50, 50*2, 50*3] for 2 kinds of filters. [100, 100*2, 100*3] for 1 kind
        self.radical_level_filter1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv1d(in_channels=radical_emb_dim, out_channels=basic_filter_dim, kernel_size=1, stride=1)),
            ('activation', torch.nn.ReLU()),  # maybe can try Leaky, PReLU or RReLU
            ('maxpooling', nn.MaxPool1d(kernel_size=max_word_length * max_character_length))
        ]))
        self.radical_level_filter2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv1d(in_channels=radical_emb_dim, out_channels=2 * basic_filter_dim, kernel_size=2, stride=1)),
            ('activation', torch.nn.ReLU()),  # maybe can try Leaky, PReLU or RReLU
            ('maxpooling', nn.MaxPool1d(kernel_size=max_word_length * max_character_length - 1))
        ]))
        self.radical_level_filter3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv1d(in_channels=radical_emb_dim, out_channels=3 * basic_filter_dim, kernel_size=3, stride=1)),
            ('activation', torch.nn.ReLU()),  # maybe can try Leaky, PReLU or RReLU
            ('maxpooling', nn.MaxPool1d(kernel_size=max_word_length * max_character_length - 2))
        ]))
        self.char_level_filter1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv1d(in_channels=radical_emb_dim, out_channels=basic_filter_dim, kernel_size=max_character_length,
                      stride=max_character_length)),
            ('activation', torch.nn.ReLU()),  # maybe can try Leaky, PReLU or RReLU
            ('maxpooling', nn.MaxPool1d(kernel_size=max_word_length))
        ]))
        self.char_level_filter2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv1d(in_channels=radical_emb_dim, out_channels=2 * basic_filter_dim,
                      kernel_size=2 * max_character_length,
                      stride=max_character_length)),
            ('activation', torch.nn.ReLU()),  # maybe can try Leaky, PReLU or RReLU
            ('maxpooling', nn.MaxPool1d(kernel_size=max_word_length - 1))
        ]))
        self.char_level_filter3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv1d(in_channels=radical_emb_dim, out_channels=3 * basic_filter_dim,
                      kernel_size=3 * max_character_length,
                      stride=max_character_length)),
            ('activation', torch.nn.ReLU()),  # maybe can try Leaky, PReLU or RReLU
            ('maxpooling', nn.MaxPool1d(kernel_size=max_word_length - 2))
        ]))

    def forward(self, x):
        #x shape: [max_sent_length, max_word_length * max_character_length]
        radical_embedding_representations = self.radical_emb_lookup(x)
        radical_feature1 = self.radical_level_filter1(radical_embedding_representations)
        pass


class TextRNN(nn.module):
    def __init__(self):
        pass
    def forward(self, x):
        #x shape: [batch_size, max_sent_length, max_word_length * max_character_length]
        pass