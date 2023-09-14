
import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import Field, BucketIterator, TabularDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
import random
from collections import Counter
from torchtext import vocab
import warnings
import re, string
from string import digits
warnings.filterwarnings("ignore")


def tokenizer(text): 
    return [tok for tok in text.split()]

def load(l):
    data = pd.read_csv('../Data/{}.csv'.format(l))
    data.drop(["Unnamed: 0", "entry_id"], inplace = True, axis = 1)
    data = data.rename(columns = {"entry_id" : "id"})
    data.head(10)
    data.to_csv("{}.csv".format(l), index = False)

    data = pd.read_csv("kannada.csv")
    # Fields
    lang = Field(tokenize = tokenizer, lower = True, init_token = "<sos>", eos_token = "<eos>")
    eng = Field(tokenize = tokenizer, lower = True, init_token = "<sos>", eos_token = "<eos>")
    # Dataset and split
    datafields = [("english", eng), ("{}".format(l), lang)]
    dataset = TabularDataset(path="{}.csv".format(l), format='csv', skip_header=True, fields=datafields)
    train_data, val_data = dataset.split(split_ratio = 0.85)
    # Vocabulary
    lang.build_vocab(train_data, min_freq = 1, max_size = 50000)
    eng.build_vocab(train_data, min_freq = 1, max_size = 50000)

    return train_data, val_data, eng, lang
