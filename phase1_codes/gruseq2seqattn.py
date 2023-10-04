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
    Encoder module for the Seq2Seq architecture with attention.

    Attributes:
    - embedding: An embedding layer that transforms input tokens into embeddings.
    - gru: A bi-directional GRU (Gated Recurrent Unit) layer.
    - fc_hidden: A linear layer that reduces the combined forward and backward hidden states to the desired hidden size.
    - dropout: A dropout layer for regularization.
    """
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_dim

        # Define the layers
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, bidirectional=True)
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(p = DROPOUT_RATE)

    def forward(self, x):
        """
        Forward pass of the encoder.

        Arguments:
        - x: Source sequence.

        Returns:
        - encoder_states: Outputs of the GRU for each step.
        - hidden_state: Combined hidden state for forward and backward GRU.
        """
        embedding = self.dropout(self.embedding(x))
        encoder_states, hidden = self.gru(embedding)

        # Combine forward and backward hidden states
        forward_hidden = hidden[0:1]
        backward_hidden = hidden[1:2]
        hidden_concat = torch.cat((forward_hidden, backward_hidden), dim = 2)
        hidden_state = self.fc_hidden(hidden_concat)
        return encoder_states, hidden_state


class Decoder(nn.Module):
    """
    Decoder module for the Seq2Seq architecture with attention.

    Attributes:
    - embedding: An embedding layer that transforms target tokens into embeddings.
    - gru: A GRU (Gated Recurrent Unit) layer.
    - attention_layer: A linear layer to compute attention scores.
    - fc_layer: A linear layer to produce the output tokens.
    - dropout: A dropout layer for regularization.
    - softmax_layer: Softmax activation for attention scores.
    - gelu: GELU activation function used in attention mechanism.
    """
    def __init__(self, input_dim, embed_dim, hidden_dim, output_dim, num_layers):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_dim

        # Define the layers
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.num_layers = num_layers
        self.gru = nn.GRU(hidden_dim * 2 + embed_dim, hidden_dim, num_layers)
        self.attention_layer = nn.Linear(hidden_dim * 3, 1)
        self.fc_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p = DROPOUT_RATE)
        self.softmax_layer = nn.Softmax(dim = 0)
        self.gelu = nn.GELU()

    def forward(self, x, encoder_states, hidden_state):
        """
        Forward pass of the decoder.

        Arguments:
        - x: Target sequence.
        - encoder_states: Output from the encoder.
        - hidden_state: Last hidden state from the encoder.

        Returns:
        - predictions: Predicted output tokens.
        - hidden_state: Hidden state after passing through the GRU.
        """
        x = x.unsqueeze(0)
        sequence_length = encoder_states.shape[0]
        embedding = self.dropout(self.embedding(x))

        # Attention mechanism
        hidden_state_reshaped = hidden_state.repeat(sequence_length, 1, 1)
        inp_state = torch.cat((hidden_state_reshaped, encoder_states), dim = 2)
        attention_score = self.gelu(self.attention_layer(inp_state))
        attention_score = self.softmax_layer(attention_score)
        context_vector = torch.einsum("snk,snl->knl", attention_score, encoder_states)

        gru_input = torch.cat((context_vector, embedding), dim=2)
        outputs, hidden_state = self.gru(gru_input, hidden_state)
        predictions = self.fc_layer(outputs).squeeze(0)
        return predictions, hidden_state


class GRUSeq2SeqAttn(nn.Module):
    """
    GRU-based Seq2Seq model with attention mechanism.

    Attributes:
    - encoder: Encoder module.
    - decoder: Decoder module.
    """
    def __init__(self, encoder, decoder):
        super(GRUSeq2SeqAttn, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        """
        Forward pass of the Seq2Seq model.

        Arguments:
        - source: Source sequence.
        - target: Target sequence.
        - teacher_force_ratio: Probability to use true target tokens as next input instead of predictions.

        Returns:
        - outputs: Predicted target sequence.
        """
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = TAR_VOCAB_SIZE
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        # Pass source through encoder
        encoder_states, hidden_state = self.encoder(source)
        x = target[0]

        # Decode the encoder's output
        for t in range(1, target_len):
            output, hidden = self.decoder(x, encoder_states, hidden_state)
            outputs[t] = output
            best_guess = output.argmax(dim = 1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs



