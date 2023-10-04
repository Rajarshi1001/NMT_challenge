import torch
import torchtext
from torchtext import vocab
import torch.nn as nn
import math
import random
import torch.nn.functional as F
from torchtext.data import Field, BucketIterator, TabularDataset
from torch import Tensor
import torch.optim as optim
from data_process import device, l, eng, lang


DROPOUT_RATE = 0.2
TAR_VOCAB_SIZE = len(lang.vocab)

class Encoder(nn.Module):
    """
    The Encoder module is part of the Seq2Seq network and is responsible for 
    converting the input sequences into a context vector. This context vector 
    is then used by the Decoder to generate the output sequences.

    Attributes:
    - embedding (nn.Embedding): Transforms input tokens into dense vectors.
    - LSTM (nn.LSTM): Gated Recurrent Unit to process the embedded sequences.
    - dropout (nn.Dropout): Regularization layer to prevent overfitting.
    """

    def __init__(self, input_dim, hidden_dim, embed_dim, num_layers):
        """
        Initializes the Encoder module.

        Parameters:
        - input_dim (int): Size of the vocabulary of the input sequences.
        - hidden_dim (int): Number of features in the hidden state.
        - embed_dim (int): Number of features in the embedded space.
        - num_layers (int): Number of recurrent layers.
        """
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, dropout = DROPOUT_RATE)
        self.dropout = nn.Dropout(p = DROPOUT_RATE)

    def forward(self, x):
        """
        Forward pass for the Encoder module.

        Parameters:
        - x (Tensor): Input sequence.

        Returns:
        - hidden_state (Tensor): Context vector for the input sequence.
        """
        embedded = self.dropout(self.embedding(x))
        _, hidden_state., cell_state = self.lstm(embedded)
        return hidden_state, cell_state

class Decoder(nn.Module):
    """
    The Decoder module is part of the Seq2Seq network and is responsible for 
    generating the output sequences from the context vector produced by the Encoder.

    Attributes:
    - embedding (nn.Embedding): Transforms input tokens into dense vectors.
    - lstm (nn.LSTM): Gated Recurrent Unit to process the embedded sequences.
    - ll (nn.Linear): Linear layer to transform the LSTM outputs to predicted tokens.
    - dropout (nn.Dropout): Regularization layer to prevent overfitting.
    """

    def __init__(self, input_dim, hidden_dim, embed_dim, output_dim, num_layers):
        """
        Initializes the Decoder module.

        Parameters:
        - input_dim (int): Size of the vocabulary of the input sequences.
        - hidden_dim (int): Number of features in the hidden state.
        - embed_dim (int): Number of features in the embedded space.
        - output_dim (int): Size of the vocabulary of the output sequences.
        - num_layers (int): Number of recurrent layers.
        """
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, dropout = DROPOUT_RATE)
        self.ll = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p = DROPOUT_RATE)

    def forward(self, x, hidden_state, cell_state):
        """
        Forward pass for the Decoder module.

        Parameters:
        - x (Tensor): Input token.
        - hidden_state (Tensor): Hidden state from the previous time step or the Encoder's context vector.

        Returns:
        - preds (Tensor): Predicted token probabilities for the next time step.
        - hidden_state (Tensor): Updated hidden state.
        """
        x = x.unsqueeze(0)
        embedded = self.dropout(self.embedding(x))
        output, hidden_state, cell_state= self.lstm(embedded, hidden_state, cell_state)
        preds = self.ll(output).squeeze(0)
        return preds, hidden_state, cell_state

class LSTMSeq2Seq(nn.Module):
    """
    A Sequence-to-Sequence network using LSTMLSTM as the recurrent unit. This module 
    combines the Encoder and Decoder into a unified model.

    Attributes:
    - encoder (Encoder): The Encoder module.
    - decoder (Decoder): The Decoder module.
    """

    def __init__(self, encoder, decoder):
        """
        Initializes the Seq2Seq module.

        Parameters:
        - encoder (Encoder): An instance of the Encoder module.
        - decoder (Decoder): An instance of the Decoder module.
        """
        super(LSTMSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.1):
        """
        Forward pass for the Seq2Seq module.

        Parameters:
        - source (Tensor): Input sequence for the Encoder.
        - target (Tensor): Actual target sequence.
        - tfratio (float): Threshold for teacher forcing.

        Returns:
        - outputs (Tensor): Predicted sequence.
        """
        outputs = torch.zeros(target.shape[0], source.shape[1], TAR_VOCAB_SIZE).to(device)
        hidden_state = self.encoder(source)
        x = target[0]
        
        for t_id in range(1, target.shape[0]):
            pred, hidden_state, cell_state = self.decoder(x, hidden_state, cell_state)
            outputs[t_id] = pred
            pred_best = pred.argmax(dim=1)
            x = target[t_id] if random.random() > teacher_force_ratio else pred_best
        return outputs
s