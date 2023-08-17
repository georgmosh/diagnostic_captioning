import os
import torch
import torch.nn as nn

class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, device='cuda:0'):
        super(RecurrentNeuralNetwork, self).__init__()

        # Defining some parameters
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.device = device

        # Defining the layers
        # RNN Layer with Gated Recurrent Units
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=False)

    def forward(self, x):
        batch_size = x.size(1)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.gru(x, hidden)

        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        rep = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.device)

        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return rep