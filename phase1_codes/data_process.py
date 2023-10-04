# Import necessary libraries
import torch
import torchtext
from torchtext import vocab
from torchtext.data import Field, BucketIterator, TabularDataset
from torch import Tensor
import os
import warnings
import re, string
import math
import random
from collections import Counter
from string import digits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# warnings.filterwarnings("ignore")

## Choose the language
l = "bengali"
BATCH_SIZE = 32


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess(text):
    """
    Convert all the text into lower letters
    Remove the words betweent brakets ()
    Remove these characters: {'$', ')', '?', '"', '’', '.',  '°', '!', ';', '/', "'", '€', '%', ':', ',', '('}
    Replace these special characters with space:
    Replace extra white spaces with single white spaces
    """
    text = re.sub(r"([?.!,])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    text = re.sub('[$)\"’°;\'€%:,(/]', '', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\u200d', ' ', text)
    text = re.sub('\u200c', ' ', text)
    text = re.sub('-', ' ', text)
    text = re.sub('  ', ' ', text)
    text = re.sub('   ', ' ', text)
    text =" ".join(text.split())
    return text

def tokenizer(text): 
    """
    Tokenize the input text.
    
    Parameters:
    - text (str): Input text to be tokenized.
    
    Returns:
    - list: List of tokens.
    """
    return [tok for tok in preprocess(text).split()]

# Define Fields for tokenization and preprocessing
lang = Field(tokenize = tokenizer, lower = True, init_token = "<sos>", eos_token = "<eos>")
eng = Field(tokenize = tokenizer, lower = True, init_token = "<sos>", eos_token = "<eos>")

# Define data fields for loading the dataset
datafields = [("english", eng), ("{}".format(l), lang)]
# Load the dataset from a CSV file
dataset = TabularDataset(path="./../{}.csv".format(l), format='csv', skip_header=True, fields=datafields)
# Split the dataset into training and validation sets
train_data, val_data = dataset.split(split_ratio = 0.80)

# Build vocabulary for each language from the training data
lang.build_vocab(train_data, min_freq = 1, max_size = 50000)
eng.build_vocab(train_data, min_freq = 1, max_size = 50000)

# creating the train and validation data iterator for training
train_iterator, val_iterator = BucketIterator.splits(
    (train_data, val_data), 
    batch_size = BATCH_SIZE, 
    device = device, 
    sort_key = lambda x: getattr(x, l),  # change the language after x.
    sort_within_batch = True)

