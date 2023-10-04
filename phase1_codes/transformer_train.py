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
import time
import random
from collections import Counter
from string import digits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# warnings.filterwarnings("ignore")
from data_process import train_iterator, val_iterator, lang, eng, device, l
from tf import SinusoidalEmbedding, TokenEmbedding, generate_square_subsequent_mask, create_mask, TransformerMT
from transformer_translate import translate, get_tokens
# Model Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters and model configuration values

SRC_VOCAB_SIZE = len(eng.vocab)  # Source vocabulary size (English)
TAR_VOCAB_SIZE = len(lang.vocab)  # Target vocabulary size (Other language, inferred from 'lang' variable)
EMBEDDING_SIZE = 512  # Size of the embedding vector
NHEAD = 8  # Number of heads in the multihead attention mechanism
FFNN_DIM = 256  # Dimension of the feed-forward neural network inside transformer layers
BATCH_SIZE = 32  # Size of each training batch
NUM_ENCODER_LAYERS = 3  # Number of layers in the transformer encoder
NUM_DECODER_LAYERS = 3  # Number of layers in the transformer decoder
LEARNING_RATE = 0.0001  # Learning rate for the optimizer
DROPOUT_RATE = 0.1  # Dropout rate for the dropout layer
NUM_EPOCHS = 50  # Number of epochs for training
PAD_IDX = eng.vocab.stoi["<pad>"]  # Index for the padding token
SOS_IDX = eng.vocab.stoi["<sos>"] # index for start token
EOD_IDX = eng.vocab.stoi["<eos>"] # index for end token



# Define the directory to save the plots.
PLOT_DIR = "transformer_plots"

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



# Instantiate the TransformerMT model with the specified configuration
model = TransformerMT(NHEAD,
                      EMBEDDING_SIZE,
                      SRC_VOCAB_SIZE, 
                      TAR_VOCAB_SIZE,
                      NUM_ENCODER_LAYERS, 
                      NUM_DECODER_LAYERS, 
                      FFNN_DIM)

# Initialize the model's parameters using the Xavier uniform initializer
# This helps in achieving a better distribution of activations
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

# Move the model to the specified device (either GPU or CPU)
model = model.to(device)

# Define the loss criterion
# CrossEntropyLoss is used since this is a classification task, and we ignore the loss computed on padding tokens
criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# Define the optimizer to be used for training
# Adam optimizer is used with specific betas and epsilon values
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

def train():
    """
    Train the model using the training data.
    
    This function carries out a single epoch of training. For each batch of data in the training dataset:
    - The model's gradients are zeroed.
    - The data is passed through the model to get predictions.
    - The loss between the predictions and actuals is computed.
    - The gradients are computed via backpropagation.
    - The model's parameters are updated.
    
    Returns:
    - float: Average training loss for the epoch.
    """
    
    # Set the model to training mode and move it to the device
    model.train().to(device)
    losses = 0  # Accumulator for the total loss

    # Iterate over each batch in the training data
    for batch_idx, batch in enumerate(train_iterator):
        train_loss = 0  # Accumulator for batch loss

        # Extract source and target sequences from the batch and move them to the device
        src = batch.english.to(device)
        target = getattr(batch, l).to(device)
        
        # Exclude the last token for target input
        target_input = target[:-1,:]

        # Generate masks for the source and target sequences
        source_mask, target_mask, src_padding_mask, tar_padding_mask = create_mask(src, target_input)
        
        # Pass data through the model
        output = model(src, 
                       target_input, 
                       None, 
                       target_mask, 
                       src_padding_mask, 
                       tar_padding_mask, 
                       src_padding_mask)
        
        # Reset model gradients
        optimizer.zero_grad()
        
        # Reshape output for loss computation
        output = output.reshape(-1, output.shape[-1])
        output_target = target[1:,:].reshape(-1)
        
        # Compute the loss and backpropagate
        loss = criterion(output, output_target)
        loss.backward()
        
        # Clip gradients to avoid exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 3)
        
        # Update model parameters
        optimizer.step()
        
        # Update loss accumulators
        losses += loss.item()
        train_loss += ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
        
        # Print the loss for every 100 batches and also translate a sample sentence
        if batch_idx % 100 == 0:
            print(f'Train loss at step {batch_idx}: {train_loss:.3f}')
            print(translate(model, "Football is a tough game", eng, lang))
    
    # Return average training loss for the epoch
    return losses/len(train_iterator)

def validate():
    """
    Validate the model using the validation data.
    
    This function computes the model's performance on the validation dataset. 
    No parameter updates are performed during this stage.
    
    Returns:
    - float: Average validation loss.
    """
    
    # Set the model to evaluation mode and move it to the device
    model.eval().to(device)
    losses = 0  # Accumulator for the total loss

    # Ensure no computation graph is built during validation
    with torch.no_grad():
        
        # Iterate over each batch in the validation data
        for batch_idx, batch in enumerate(val_iterator):
            
            # Extract source and target sequences from the batch and move them to the device
            src = batch.english.to(device)
            target = getattr(batch, l).to(device)
            
            # Exclude the last token for target input
            target_input = target[:-1,:]

            # Generate masks for the source and target sequences
            source_mask, target_mask, src_padding_mask, tar_padding_mask = create_mask(src, target_input)
            
            # Pass data through the model
            output = model(src, 
                           target_input, 
                           source_mask, 
                           target_mask, 
                           src_padding_mask, 
                           tar_padding_mask, 
                           src_padding_mask)
            
            # Reshape output for loss computation
            output = output.reshape(-1, output.shape[-1])
            output_target = target[1:,:].reshape(-1)
            
            # Compute the loss
            loss = criterion(output, output_target)
            
            # Update loss accumulator
            losses += loss.item()

    # Return average validation loss
    return losses/len(val_iterator)


def train_and_validate(num_epochs, patience):
    """
    Trains and validates the model over a specified number of epochs.
    Implements early stopping based on the validation loss.
    
    Parameters:
    - num_epochs (int): Maximum number of epochs to train.
    - patience (int): Number of epochs to wait without improvement before stopping training.
    
    Returns:
    - list: A list of training losses over the epochs.
    - list: A list of validation losses over the epochs.
    """
    
    # Initialize the best validation loss to a high value
    best_val_loss = float('inf')
    epochs_without_improvement = 0  # Counter for epochs without validation loss improvement
    train_losses, val_losses = [], []  # Lists to store training and validation losses
    
    # Loop over epochs
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # Training phase
        train_loss = train()
        train_losses.append(train_loss)
        
        # Validation phase
        val_loss = validate()
        val_losses.append(val_loss)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Print training and validation statistics for the epoch
        print(f'Epoch: {epoch}/{num_epochs}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tVal Loss: {val_loss:.3f}')
        print(f'\tEpoch time: {elapsed_time:.3f}s')
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the current state of the model if validation loss is the best seen so far
            torch.save(model.state_dict(), "best_model_{}.pt".format(l))
        else:
            epochs_without_improvement += 1
            # Stop training if validation loss hasn't improved for a number of epochs specified by 'patience'
            if epochs_without_improvement >= patience:
                print("Stopping training due to early stopping criteria!")
                break

    # Load the best model weights after all epochs are completed or early stopping is triggered
    model.load_state_dict(torch.load("best_model_{}.pt".format(l)))
    
    return train_losses, val_losses

# Train and validate the model for a specified number of epochs with early stopping criteria
train_losses, val_losses = train_and_validate(NUM_EPOCHS, 2)


plotresults()

# Inference
# Check if the "Translations" directory exists. If not, create it.
if not os.path.exists("transformer_translations"):
    os.makedirs("transformer_translations")

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
    data = pd.read_csv("./../testData/testEnglish-{}.csv".format(language))

    # Loop through each row (sentence) in the test data.
    for idx, row in data.iterrows():
        # Extract the English sentence.
        en = row["english"]

        # Translate the English sentence to the target language.
        pred = translate(model, en eng, lang)

        # Print the translated sentence (optional, can be commented out).
        print(pred)

        # Append the translated sentence to the predictions list.
        predictions.append(pred)

    # Add the predicted translations as a new column to the original dataframe.
    data["translated"] = predictions

    # Drop the unwanted column "Unnamed: 0" (assuming it exists in the CSV).
    data.drop(["Unnamed: 0"], inplace=True, axis=1)

    # Save the dataframe with translations to a new CSV file.
    data.to_csv(os.path.join("transformer_translations", "answer_{}_test.csv".format(language)))

# Evaluate the model on the Bengali test set.
evaluate(l.title())