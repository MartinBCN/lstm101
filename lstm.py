import os
from typing import Tuple

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from data import one_hot_encode, get_batches, get_data


class CharRNN(nn.Module):
    """
    Character Level LSTM
    """

    def __init__(self, tokens, n_hidden=256, n_layers=2,
                 drop_prob=0.5, lr=0.001):
        """

        Parameters
        ----------
        tokens
        n_hidden
        n_layers
        drop_prob
        lr
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        input_size = len(tokens)

        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        # define the LSTM, self.lstm
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.n_hidden,
            num_layers=self.n_layers,
            batch_first=True)

        # define a dropout layer, self.dropout
        self.drop_out = nn.Dropout(drop_prob)

        # Define the final, fully-connected output layer, self.fc
        self.fc = nn.Linear(self.n_hidden, input_size)

        # initialize the weights
        self.init_weights()

    def forward(self, x: Tensor, hc: Tuple[Tensor, Tensor]) -> (Tensor, Tuple[Tensor, Tensor]):
        ''' Forward pass through the network.
            These inputs are x, and the hidden/cell state `hc`. '''

        # Get x, and the new hidden state (h, c) from the lstm

        x, (h, c) = self.lstm(x, hc)

        # pass x through a dropout layer
        x = self.drop_out(x)

        # Stack up LSTM outputs using view
        x = x.reshape(-1, self.n_hidden)

        # put x through the fully-connected layer
        x = self.fc(x)

        # return x and the hidden state (h, c)
        return x, (h, c)

    def predict(self, char, h=None, cuda=False, top_k=None):
        ''' Given a character, predict the next character.

            Returns the predicted character and the hidden state.
        '''
        if cuda:
            self.cuda()
        else:
            self.cpu()

        if h is None:
            h = self.init_hidden(1)

        x = np.array([[self.char2int[char]]])
        x = one_hot_encode(x, len(self.chars))
        inputs = torch.from_numpy(x)
        if cuda:
            inputs = inputs.cuda()

        h = tuple([each.data for each in h])
        out, h = self.forward(inputs, h)

        p = F.softmax(out, dim=1).data
        if cuda:
            p = p.cpu()

        if top_k is None:
            top_ch = np.arange(len(self.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()

        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p / p.sum())

        return self.int2char[char], h

    def init_weights(self):
        ''' Initialize weights for fully connected layer '''
        initrange = 0.1

        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1)

    def init_hidden(self, n_seqs: int) -> Tuple[Tensor, Tensor]:
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x n_seqs x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, n_seqs, self.n_hidden).zero_(),
                weight.new(self.n_layers, n_seqs, self.n_hidden).zero_())


if __name__ == '__main__':
    data_dir = os.environ.get('DATA_DIR', 'data')
    fn = f'{data_dir}/anna.txt'
    anna = get_data(fn)
    chars = tuple(set(anna))
    n_chars = len(chars)
    print('Number of characters', n_chars)

    n_seqs = 10
    n_steps = 50


    lstm = CharRNN(chars)

    # Initiate hidden layers
    val_h = lstm.init_hidden(n_seqs)
    batches = get_batches(anna, n_seqs=n_seqs, n_steps=n_steps)
    x, y = next(batches)

    x = one_hot_encode(x, n_chars)
    x, y = torch.from_numpy(x), torch.from_numpy(y)

    print('Input: ', x.shape)
    print('Target: ', y.shape)
    h, c = val_h
    print('Initial h: ', h.shape)
    print('Initial c:', c.shape)

    val_h = tuple([each.data for each in val_h])
    x, hc = lstm.forward(x, val_h)

    # print(x.shape)
    # print(h.shape)
