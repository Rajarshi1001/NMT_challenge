import torch
import torchtext
from torchtext import vocab
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import Field, BucketIterator, TabularDataset
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerDecoder,TransformerEncoderLayer, TransformerDecoderLayer
import torch.optim as optim
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
from data_process import train_iterator, val_iterator, lang, eng, device, l
from lstmseq2seq import Encoder, Decoder, LSTMSeq2Seq
from generate import translate

# Model Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters and model configuration values
DROPOUT_RATE = 0.2 # Dropout rate used for regularization in the model
EPOCHS = 1 # Total number of training epochs (full passes over the training dataset)
BATCH_SIZE = 16 # Number of training examples processed in a single batch during training
TEACHER_FORCE_RATIO = 0.1 # Probability with which true target tokens are used as the next input instead of the predicted tokens during training (used in sequence-to-sequence models)
NUM_LAYERS =  1 # Number of recurrent layers in the model
HIDDEN_SIZE = 600 # Number of features in the hidden state of the recurrent unit (e.g., GRU or LSTM)
EMBEDDING_SIZE = 300 # Size of the embedding vectors used to represent tokens
SRC_VOCAB_SIZE = len(eng.vocab) # Vocabulary size for the source language (English in this case)
TAR_VOCAB_SIZE = len(lang.vocab) # Vocabulary size for the target language
PAD_IDX = eng.vocab.stoi["<pad>"]  # Index for the padding token
SOS_IDX = eng.vocab.stoi["<sos>"] # index for start token
EOD_IDX = eng.vocab.stoi["<eos>"] # index for end token
INPUT_SIZE_EN = SRC_VOCAB_SIZE # Input size for the encoder (equal to the source vocabulary size)
INPUT_SIZE_DR = TAR_VOCAB_SIZE
OUTPUT_SIZE_DR = TAR_VOCAB_SIZE # Input and output sizes for the decoder (equal to the target vocabulary size)
LEARNING_RATE = 0.001 # Learning rate for the optimizer
WEIGHT_DECAY = 0.0008 # Weight decay parameter for regularization in the optimizer
eng_tokens = [] # List to store tokenized sentences for the source language
bn_tokens = [] # List to store tokenized sentences for the target language (Bengali in this example)
device = ("cuda" if torch.cuda.is_available() else "cpu") # Device configuration (uses GPU if available, otherwise falls back to CPU)
pad_idx = eng.vocab.stoi["<pad>"]


# Define the directory to save the plots.
PLOT_DIR = "lstm_plots"

# Check if the directory exists. If not, create it.
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

def plotresults():
    """
    Function to plot training and validation loss over epochs.

    The function uses the matplotlib library to plot the loss curves for 
    training and validation data. It saves the generated plot in the specified
    directory (`plot_dir`) with a filename based on the language (`l`).
    """
    # Plotting the training loss (in black color with circle markers).
    plt.plot(range(len(train_losses)), train_losses, marker = "o", color = "black")

    # Plotting the validation loss (in blue color with circle markers).
    plt.plot(range(len(val_losses)), val_losses, marker = "o", color = "blue")

    # Adding legend to distinguish between train and validation curves.
    plt.legend(["Train loss", "Val loss"])

    # Adding title and axis labels to the plot.
    plt.title("Loss curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss values")

    # Displaying grid for better visualization.
    plt.grid()

    # Save the plot as a PNG image in the specified directory with a filename
    # based on the language (`l`).
    plt.savefig(os.path.join(PLOT_DIR,"loss_{}.png".format(l)))

# Call the function to plot the results.

# Instantiate the Encoder module with the specified input size, embedding size, hidden size, and number of layers.
# The model is moved to the specified device (GPU or CPU).
encoder = Encoder(INPUT_SIZE_EN, EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)

# Instantiate the Decoder module with the specified input size, embedding size, hidden size, output size, 
# and number of layers. The model is also moved to the specified device.
decoder = Decoder(INPUT_SIZE_DR, EMBEDDING_SIZE, HIDDEN_SIZE, OUTPUT_SIZE_DR, NUM_LAYERS).to(device)

# Instantiate the main GRU-based Sequence-to-Sequence model by combining the Encoder and Decoder modules. 
model = LSTMSeq2Seq(encoder, decoder).to(device)

# Initialize the Adam optimizer with the specified learning rate to optimize the model parameters.
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

# Define the loss criterion
# CrossEntropyLoss is used since this is a classification task, and we ignore the loss computed on padding tokens
criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)


# List to store training losses after each epoch.
train_losses = []

# List to store validation losses after each epoch.
val_losses = []


# Start the training process over specified number of epochs.
for epoch in range(EPOCHS):
    # Initialize the epoch-level training and validation loss.
    train_loss = 0
    valid_loss = 0

    # Print out the current epoch number.
    print(f"[Epoch no: {epoch} / {EPOCHS}]")

    # Set the model to training mode.
    model.train()

    # Iterate over each batch in the training data.
    for batch_idx, batch in enumerate(train_iterator):
        # Move the input and target data to the specified device.
        inp_data = batch.english.to(device)
        target = getattr(batch, l).to(device)

        # Forward pass: Get model predictions for the current batch.
        output = model(inp_data, target)

        # Reshape the output and target for loss calculation.
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        # Zero out any previously calculated gradients.
        optimizer.zero_grad()

        # Compute the loss between model predictions and actual target.
        loss = criterion(output, target)

        # Backward pass: Compute gradient of loss w.r.t. model parameters.
        loss.backward()

        # Clip the gradients to prevent them from exploding (a common issue in RNNs).
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Update the model parameters using the computed gradients.
        optimizer.step()

        # Update the training loss.
        train_loss += ((1 / (batch_idx + 1)) * (loss.data.item() - train_loss))

        # Print training loss every 100 steps and a sample translation.
        if batch_idx % 100 == 0:
            print('Train loss -> {} steps: {:.3f}'.format(batch_idx, train_loss))
            print(translate("Football is a tough game", model, eng, lang, max_len=20))

    # Set the model to evaluation mode for validation.
    model.eval()

    # Iterate over each batch in the validation data.
    for batch_idx, batch in enumerate(val_iterator):
        inp_data = batch.english.to(device)
        target = getattr(batch, l).to(device)
        output = model(inp_data, target)
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        # Compute the loss between model predictions and actual target.
        loss = criterion(output, target)

        # Update the validation loss.
        valid_loss += ((1 / (batch_idx + 1)) * (loss.data.item() - valid_loss))

    # Append epoch-level train and validation loss to respective lists.
    train_losses.append(train_loss)
    val_losses.append(valid_loss)

    # Print epoch-level summary.
    print('Epoch no: {} \tTraining Loss: {:.5f} \tValidation Loss: {:.5f}'.format(epoch, train_loss, valid_loss))


plotresults()

# Inference

# Check if the "Translations" directory exists. If not, create it.
if not os.path.exists("lstm_translations"):
    os.makedirs("lstm_translations")

def evaluate(language):
    """
    Function to evaluate and generate translations for given test data.
    
    This function reads a CSV file containing English sentences, 
    translates each sentence to the target language using the 
    trained model, and then saves the translations to a new CSV file.

    Parameters:
    - language: The target language for translation.

    Outputs:
    - A CSV file named "answer1_{language}_test.csv" saved in the "Translations" directory.
      This file contains the original English sentences and their corresponding translations.
    """
    # List to store the predicted translations.
    predictions = []

    # Read the test data from the specified CSV file.
    data = pd.read_csv("./../../testData/testEnglish-{}.csv".format(language))
    data = data.iloc[:100,:] # first 100 samples
    # Loop through each row (sentence) in the test data.
    for idx, row in data.iterrows():
        # Extract the English sentence.
        en = row["english"]

        # Translate the English sentence to the target language.
        pred = translate(en, model, eng, lang, max_len=20)

        # Print the translated sentence (optional, can be commented out).
        print(pred)

        # Append the translated sentence to the predictions list.
        predictions.append(pred)

    # Add the predicted translations as a new column to the original dataframe.
    data["translated"] = predictions

    # Drop the unwanted column "Unnamed: 0" (assuming it exists in the CSV).
    data.drop(["Unnamed: 0"], inplace=True, axis=1)

    # Save the dataframe with translations to a new CSV file.
    data.to_csv(os.path.join("lstm_translations", "answer_{}_test.csv".format(language)))

# Evaluate the model on the Bengali test set.
evaluate(l.title())