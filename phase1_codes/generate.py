import torch
import numpy as np
import torch.nn as nn
from data_process import eng, lang, device, train_iterator, val_iterator, l

def translate(text, model, eng, lang, max_len = 20):
    """
    Translates a given text from the source language to the target language using the provided trained model.
    
    Args:
    - text (str or list): The input text to be translated. Can be a string or a list of tokens.
    - model (nn.Module): The trained sequence-to-sequence model used for translation.
    - eng (torchtext.data.Field): The Field object for the source language (English in this case).
    - lang (torchtext.data.Field): The Field object for the target language.
    - max_len (int, optional): Maximum length of the translated output. Defaults to 20.

    Returns:
    - str: The translated text in the target language.
    """
    
    # If the input text is a string, tokenize it.
    if type(text) == str:
        tokens = [tok for tok in text.split()]
    else:
        tokens = text
    # Add the start and end tokens to the tokenized text.
    tokens.insert(0, eng.init_token)
    tokens.append(eng.eos_token)

    # Convert tokens to their respective indices from the vocabulary.
    txt2idx = [eng.vocab.stoi[tok] for tok in tokens]

    # Convert token indices to a tensor and move it to the specified device (GPU or CPU).
    st = torch.LongTensor(txt2idx).unsqueeze(1).to(device)

    # Initialize the result list with the index of the start token.
    res = [eng.vocab.stoi[0]]

    # Generate the translation iteratively.
    for i in range(1, max_len):
        tt = torch.LongTensor(res).unsqueeze(1).to(device)
        with torch.no_grad():
            output = model(st, tt)
            best_guess = output.argmax(2)[-1, :].item()

            # If the end token is predicted, stop the translation.
            if best_guess == lang.vocab.stoi["<eos>"]:
                break
            res.append(best_guess)

    # Convert the indices in the result list back to tokens.
    tsent = [lang.vocab.itos[index] for index in res]

    # Return the translated sentence as a string, replacing any unknown tokens with a space.
    return " ".join(tsent[1:]).replace("<unk>", " ")
