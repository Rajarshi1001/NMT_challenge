import torch
import numpy as np
import torch.nn as nn
from data_process import eng, lang, train_iterator, val_iterator, l, device
from tf import create_mask, generate_square_subsequent_mask

SOS_IDX = eng.vocab.stoi["<sos>"]
EOS_IDX = eng.vocab.stoi["<eos>"]


def get_tokens(model, source, source_mask, max_len, start_symbol):
    """
    Get token indices from a given source sequence using a trained model.
    
    Parameters:
    - model (nn.Module): The trained transformer model.
    - source (torch.Tensor): The source sequence tensor.
    - source_mask (torch.Tensor): The mask for the source sequence.
    - max_len (int): Maximum length of the target sequence.
    - start_symbol (int): The starting symbol for the target sequence.
    
    Returns:
    - result (torch.Tensor): The tensor containing token indices of the target sequence.
    """
    
    # Move source and its mask to device
    source_mask = source_mask.to(device)
    source = source.to(device)
    
    # Encode the source sequence
    memory = model.encode(source, source_mask)
    
    # Initialize result tensor with the start symbol
    result = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
                
    # Decode the memory tensor to get the target sequence
    for index in range(max_len - 1):
        memory = memory.to(device)
        memory_mask = torch.zeros(result.shape[0], memory.shape[0]).to(device).type(torch.bool)
        target_mask = (generate_square_subsequent_mask(result.size(0)).type(torch.bool)).to(device)
        output = model.decode(result, memory, target_mask)
        output = output.transpose(0, 1)
        
        # Get the next word's probability distribution and find the word with the maximum probability
        probs = model.fc_layer(output[:, -1])
        _, next_word = torch.max(probs, dim = 1)
        next_word = next_word.item()
        
        # Add the next word to the result
        result = torch.cat([result,torch.ones(1, 1).type_as(source.data).fill_(next_word)], dim=0)
        
        # Break if the next word is the start of sequence symbol
        if next_word == SOS_IDX:
            break
            
    return result


def translate(model, source, source_vocab, target_vocab):
    """
    Translate a given source sequence to a target sequence using a trained model.
    
    Parameters:
    - model (nn.Module): The trained transformer model.
    - source (str or torch.Tensor): The source sequence (string or tensor).
    - source_vocab (Vocab): Vocabulary object for the source language.
    - target_vocab (Vocab): Vocabulary object for the target language.
    
    Returns:
    - str: The translated sequence.
    """
    
    # Set the model to evaluation mode
    model.eval()
    
    # Convert source string to tensor, if it's a string
    if type(source) == str:
        tokens = [SOS_IDX] + [source_vocab.vocab.stoi[tok] for tok in source.split()] + [EOS_IDX]
        num_tokens = len(tokens)
        source = (torch.LongTensor(tokens).reshape(num_tokens, 1))
        source_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        
        # Get target token indices from the source tensor
        target_tokens = get_tokens(model, source, source_mask, max_len = num_tokens + 5, start_symbol = SOS_IDX).flatten()
        
        # Convert target token indices to string
        return " ".join([target_vocab.vocab.itos[token] for token in target_tokens]).replace("<sos>", "").replace("<eos>", "").replace("<unk>", "")
