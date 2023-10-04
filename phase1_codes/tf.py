import torch
import torchtext
from torchtext import vocab
import torch.nn as nn
import math
import torch.nn.functional as F
from torchtext.data import Field, BucketIterator, TabularDataset
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerDecoder,TransformerEncoderLayer, TransformerDecoderLayer
import torch.optim as optim
from data_process import device, l, eng, lang


DROPOUT_RATE = 0.1
PAD_IDX = eng.vocab.stoi["<pad>"]
# warnings.filterwarnings("ignore")


class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal Positional Encoding for Transformer models.
    
    The positional encoding module uses sine and cosine functions of different frequencies to 
    encode the position of tokens in the sequence.
    
    Parameters:
    - embed_size (int): Dimension of the embedding vector.
    - dropout (float): Dropout rate for the dropout layer.
    - max_len (int, optional): Maximum length of the sequence. Default is 5000.
    """
    
    def __init__(self, embed_size, dropout, max_len = 5000):
        super(SinusoidalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        
        # Compute the sinusoidal positional encodings
        denom = max_len*2
        pdist = torch.exp(- torch.arange(0, embed_size, 2) * math.log(denom) / embed_size)
        position = torch.arange(0, max_len).reshape(max_len, 1)
        position_embedding = torch.zeros((max_len, embed_size))
        position_embedding[:, 0::2] = torch.sin(position * pdist)
        position_embedding[:, 1::2] = torch.cos(position * pdist)
        position_embedding = position_embedding.unsqueeze(-2)
        
        # Register the position embeddings so they get saved with the model's state_dict
        self.register_buffer('position_embedding', position_embedding)

    def forward(self, token_embed):
        """
        Forward pass of the SinusoidalEmbedding.
        
        Parameters:
        - token_embed (torch.Tensor): Token embeddings.
        
        Returns:
        - torch.Tensor: Token embeddings added with positional encodings.
        """
        outputs = token_embed + self.position_embedding[:token_embed.size(0),:]
        outputs = self.dropout(outputs)
        return outputs


class TokenEmbedding(nn.Module):
    """
    Token Embedding module for Transformer models.
    
    This module converts token indices into dense vectors of fixed size, embed_size. 
    The embeddings are scaled by the square root of their dimensionality.
    
    Parameters:
    - vocab_size (int): Size of the vocabulary.
    - embed_size (int): Dimension of the embedding vector.
    """
    
    def __init__(self, vocab_size, embed_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embed_size = embed_size
        
    def forward(self, tokens):
        """
        Forward pass of the TokenEmbedding.
        
        Parameters:
        - tokens (torch.Tensor): Tensor of token indices.
        
        Returns:
        - torch.Tensor: Scaled token embeddings.
        """
        outputs = self.embedding(tokens.long())
        outputs_scaled = outputs * math.sqrt(self.embed_size)
        return outputs_scaled


def generate_square_subsequent_mask(sz):
    """
    Generate a square mask for the sequence, where the mask indicates subsequent positions.
    This mask is used to ensure that a position cannot attend to subsequent positions in the sequence.
    
    Parameters:
    - sz (int): Size of the sequence.
    
    Returns:
    - torch.Tensor: Mask tensor of shape (sz, sz) with 0s in positions that can be attended to and negative infinity elsewhere.
    """
    
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(source, target):
    """
    Create masks for the source and target sequences.
    
    Parameters:
    - source (torch.Tensor): Source sequence tensor.
    - target (torch.Tensor): Target sequence tensor.
    
    Returns:
    - tuple: A tuple containing source mask, target mask, source padding mask, and target padding mask.
    """
    
    source_seq_len = source.shape[0]
    target_seq_len = target.shape[0]
    batch_size = source.shape[1]
    source_mask = torch.zeros((source_seq_len, source_seq_len), device=device).type(torch.bool)
    target_mask = generate_square_subsequent_mask(target_seq_len)
    source_padding_mask = (source == PAD_IDX).transpose(0, 1)
    target_padding_mask = (target == PAD_IDX).transpose(0, 1)
    return source_mask, target_mask, source_padding_mask, target_padding_mask


class TransformerMT(nn.Module):
    """
    Custom Transformer model for Machine Translation (MT).
    
    This model consists of an encoder and a decoder, both built using the Transformer architecture.
    
    Parameters:
    - nhead (int): Number of heads in the multihead attention mechanism.
    - embed_size (int): Dimension of the embedding vector.
    - source_vocab_size (int): Size of the source vocabulary.
    - target_vocab_size (int): Size of the target vocabulary.
    - num_encoder_layers (int): Number of layers in the transformer encoder.
    - num_decoder_layers (int): Number of layers in the transformer decoder.
    - ffnn_size (int, optional): Size of the feedforward neural network inside transformer layers. Default is 512.
    """
    
    def __init__(self, 
                 nhead: int,
                 embed_size: int, 
                 source_vocab_size: int, 
                 target_vocab_size: int,
                 num_encoder_layers: int, 
                 num_decoder_layers: int,
                 ffnn_size:int = 512):
        
        super(TransformerMT, self).__init__()
        
        # Define encoder and decoder layers
        encoder_layer = TransformerEncoderLayer(d_model = embed_size, nhead = nhead, dim_feedforward = ffnn_size)
        decoder_layer = TransformerDecoderLayer(d_model = embed_size, nhead = nhead, dim_feedforward = ffnn_size)
        
        # Initialize transformer encoder, decoder, and final fully connected layer
        self.tf_encoder = TransformerEncoder(encoder_layer, num_layers = num_encoder_layers)
        self.tf_decoder = TransformerDecoder(decoder_layer, num_layers = num_decoder_layers)
        self.fc_layer = nn.Linear(embed_size, target_vocab_size)
        
        # Embedding layers for tokens and positional information
        self.src_token_embedding = TokenEmbedding(source_vocab_size, embed_size)
        self.tar_token_embedding = TokenEmbedding(target_vocab_size, embed_size)
        self.positional_embedding = SinusoidalEmbedding(embed_size, dropout = DROPOUT_RATE)

    def forward(self, 
                source: Tensor,
                target: Tensor,
                source_mask: Tensor,
                target_mask: Tensor,
                source_padding_mask: Tensor,
                target_padding_mask: Tensor, 
                memory_key_padding_mask: Tensor):
        """
        Forward pass of the TransformerMT model.
        
        Parameters:
        - source (torch.Tensor): Source sequence tensor.
        - target (torch.Tensor): Target sequence tensor.
        - source_mask (torch.Tensor): Source sequence mask.
        - target_mask (torch.Tensor): Target sequence mask.
        - source_padding_mask (torch.Tensor): Source padding mask.
        - target_padding_mask (torch.Tensor): Target padding mask.
        - memory_key_padding_mask (torch.Tensor): Memory key padding mask.
        
        Returns:
        - torch.Tensor: Output tensor after passing through the transformer and the fully connected layer.
        """
        
        source_embedding = self.positional_embedding(self.src_token_embedding(source))
        target_embedding = self.positional_embedding(self.tar_token_embedding(target))
        memory = self.tf_encoder(source_embedding, source_mask, source_padding_mask)
        outputs = self.tf_decoder(target_embedding, 
                                  memory, 
                                  target_mask, 
                                  None,
                                  target_padding_mask,
                                  memory_key_padding_mask)
        outputs = self.fc_layer(outputs)
        return outputs

    def encode(self, source: Tensor, source_mask: Tensor):
        """
        Encode the source sequence using the transformer encoder.
        
        Parameters:
        - source (torch.Tensor): Source sequence tensor.
        - source_mask (torch.Tensor): Source sequence mask.
        
        Returns:
        - torch.Tensor: Encoded memory tensor.
        """
        token_rep = self.src_token_embedding(source)
        positional_rep = self.positional_embedding(token_rep)
        encoder_output = self.tf_encoder(positional_rep, source_mask)
        return encoder_output

    def decode(self, target: Tensor, memory: Tensor, target_mask: Tensor):
        """
        Decode the memory tensor using the transformer decoder.
        
        Parameters:
        - target (torch.Tensor): Target sequence tensor.
        - memory (torch.Tensor): Encoded memory tensor.
        - target_mask (torch.Tensor): Target sequence mask.
        
        Returns:
        - torch.Tensor: Decoded tensor.
        """
        token_rep = self.src_token_embedding(target)
        positional_rep = self.positional_embedding(token_rep)
        decoder_output = self.tf_decoder(positional_rep, memory, target_mask)
        return decoder_output
        
