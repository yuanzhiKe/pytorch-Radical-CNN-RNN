import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from nnBlocks import Highway, SelfAttention


class Radical2WordCNN(nn.Module):
    """
    encode word feature from radicals
    """

    def __init__(self, radical_vocab_size, radical_emb_dim, basic_filter_dim, max_word_length, max_character_length):
        super(Radical2WordCNN, self).__init__()
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
        self.highway = Highway(input_dim=12*basic_filter_dim, activation_function=F.relu)

    def forward(self, sentence):
        """
        :param sentence: sentence of shape: [max_sent_length, max_word_length * max_character_length]
        :return: shape: [max_sent_length, basic_filter_dim * 12]
        """
        radical_embedding_representations = self.radical_emb_lookup(sentence)
        radical_embedding_representations \
            = torch.transpose(radical_embedding_representations, 1, 2)  # to (N_batch,C_in,L_sequence)
        radical_feature1 = self.radical_level_filter1(radical_embedding_representations)
        radical_feature2 = self.radical_level_filter2(radical_embedding_representations)
        radical_feature3 = self.radical_level_filter3(radical_embedding_representations)
        character_feature1 = self.char_level_filter1(radical_embedding_representations)
        character_feature2 = self.char_level_filter2(radical_embedding_representations)
        character_feature3 = self.char_level_filter3(radical_embedding_representations)
        # concatenate the outputs of filters
        word_features = torch.cat((radical_feature1, radical_feature2, radical_feature3, character_feature1, character_feature2, character_feature3), dim=1)
        # flatten the outputs of each word
        word_features = word_features.view(sentence.size(0), -1)
        output = self.highway.forward(word_features)
        return output
