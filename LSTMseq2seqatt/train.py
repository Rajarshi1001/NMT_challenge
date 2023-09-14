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
import os
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import Counter
import warnings
import re, string
from string import digits
warnings.filterwarnings("ignore")
from model import Encoder, Decoder, GRUSeq2Seq
from train import load

# Hyperparameters
DATA_DIR = "./../Data"
DROPOUT_RATE = 0.5
EPOCHS = 30
BATCH_SIZE = 64
TEACHER_FORCE_RATIO = 0.2
NUM_LAYERS = 3
HIDDEN_SIZE = 1024
EMBEDDING_SIZE = 300
TRAIN_SIZE = 0.85
VAL_SIZE = 0.15
SRC_VOCAB_SIZE = len(eng.vocab)
TAR_VOCAB_SIZE = len(lang.vocab)
INPUT_SIZE_EN = SRC_VOCAB_SIZE
INPUT_SIZE_DR = TAR_VOCAB_SIZE
OUTPUT_SIZE_DR = TAR_VOCAB_SIZE
LEARNING_RATE = 0.001
WEIGHT_DEACAY = 0.0008
eng_tokens = []
bn_tokens = []
device = ("cuda" if torch.cuda.is_available() else "cpu")


# loading the dataset and vocabulary
language = "kannada"
train_data, val_data, eng, lang = load(language)


# creating the dataloaders for initiating training
trainiterator, valiterator = BucketIterator.splits(
        (train_data, val_data), 
        batch_size = BATCH_SIZE, 
        device=device, 
        sort_key=lambda x: len(x.kannada), 
        sort_within_batch=True)


# defining the model and its parameters
encoder = Encoder(INPUT_SIZE_EN, EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
decoder = Decoder(INPUT_SIZE_DR, EMBEDDING_SIZE, HIDDEN_SIZE, OUTPUT_SIZE_DR, NUM_LAYERS).to(device)
model = GRUSeq2Seq(encoder, decoder).to(device)
print(model)
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
pad_idx = eng.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=0)


def translate(text, model, eng, lang, max_len = 20):

    if type(text) == str:
        tokens = [tok.lower() for tok in text.split()]
    tokens.insert(0, eng.init_token)
    tokens.append(eng.eos_token)
    txt2idx = [eng.vocab.stoi[tok] for tok in tokens]
    st = torch.LongTensor(txt2idx).unsqueeze(1).to(device)
    res = [eng.vocab.stoi[0]]
    for i in range(1, max_len):
        tt = torch.LongTensor(res).unsqueeze(1).to(device)
        with torch.no_grad():
            output = model(st, tt)
            best_guess = output.argmax(2)[-1, :].item()
            if best_guess == lang.vocab.stoi["<eos>"]:
                break
            res.append(best_guess)
    tsent = [lang.vocab.itos[index] for index in res]
    return " ".join(tsent[1:])

def trainvalscript(plot_loss = False):
    train_losses, val_losses = [], []
    device = "cuda"
    for epoch in range(EPOCHS):
        train_loss=0
        valid_loss =0
        print(f"[Epoch no: {epoch} / {EPOCHS}]")
        model.train()
        for batch_idx, batch in enumerate(trainiterator):
            # print(src)
            inp_data = batch.english.to(device)
            target = batch.kannada.to(device)
            output = model(inp_data, target)
            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)
            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            train_loss+= ((1 / (batch_idx + 1)) * (loss.data.item() - train_loss))
            if batch_idx%100==0:
                print('Train loss -> {} steps: {:.3f}'.format(batch_idx, train_loss))
                print(translate("Football is a tough game", model, eng, lang, max_len=20))         
        
        model.eval()
        for batch_idx, batch in enumerate(valiterator):
            inp_data = batch.english.to(device)
            target = batch.kannada.to(device)
            output = model(inp_data, target)
            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)
            loss = criterion(output, target)
            valid_loss+= ((1 / (batch_idx + 1)) * (loss.data.item() - valid_loss))
            
        train_losses.append(train_loss)
        val_losses.append(val_losses)
        print('Epoch no: {} \tTraining Loss: {:.5f} \tValidation Loss: {:.5f}'.format(epoch, train_loss,valid_loss))

        if plot_loss is True:
            train_loss = np.array(train_loss)
            val_loss = np.array(val_loss)
            plt.figure(figsize = (20,14))
            plt.plot(range(EPOCHS), train_loss, color = "blue", marker = "o")
            plt.plot(range(EPOCHS), val_loss, color = "black", marker = "^")
            plt.legend(["Train", "Validation"])
            plt.grid()
            plt.save_fig("lossplot.png")


if not os.path.exists("Translations"):
    os.makedirs("Translations")

def evaluate(language):
    predictions = []
    data = pd.read_csv("./../valData/valEnglish-{}.csv".format(language))
    for idx, row in data.iterrows():
        en = row["english"]
        pred = translate(en, model, eng, lang, max_len=20)
        predictions.append(pred)
    data["translated"] = predictions
    data.drop(["Unnamed: 0"], inplace = True, axis = 1)
    data.to_csv(os.path.join("Translations", "answer1_{}.csv".format(language)))


if __name__ == "__main__":

    trainvalscript(plot_loss = False)
    evaluate("Kannada")






