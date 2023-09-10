import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import Field, BucketIterator, TabularDataset
from torch.optim import Adam
from sklearn.metrics import mean_squared_error
from torch.utils.tensorboard import SummaryWriter


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


# implementation of the Encoder part -> (seq_length, batch_size) as input and hidden_states of the GRU as output
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, num_layers):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p = DROPOUT_RATE)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, embed_dim)
        # Use GRU instead of LSTM
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, dropout = DROPOUT_RATE)
        
    def forward(self, x):
        embedding_x = self.embedding(x)
        embedding_drop = self.dropout(embedding_x)
        _, hidden_state = self.gru(embedding_drop)  # Note: GRU doesn't return cell state
        return hidden_state

# implementation of the decoder part of the Seq2Seq model -> (batch_size) as input
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, output_dim, num_layers):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p = DROPOUT_RATE)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, embed_dim)
        # Use GRU instead of LSTM
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, dropout = DROPOUT_RATE)
        self.ll = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, hidden_state):
        x = x.unsqueeze(0)
        embedding_x = self.embedding(x)
        embedding_drop = self.dropout(embedding_x)
        output, hidden_state = self.gru(embedding_drop, hidden_state)  # GRU doesn't use cell state
        preds = self.ll(output)
        preds = preds.squeeze(0)
        return preds, hidden_state
        
# implementation of the GRUSeq2Seq model
class GRUSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(GRUSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source , target, tfratio = TEACHER_FORCE_RATIO):
        tar_vocab_size = TAR_VOCAB_SIZE
        tar_len = target.shape[0]
        batch_size = source.shape[1]
        hidden_state = self.encoder(source)
        first_token = target[0]
        x = first_token
        outputs = torch.zeros(tar_len, batch_size, TAR_VOCAB_SIZE).to(device)
        for t_id in range(1, tar_len):
            pred, hidden_state = self.decoder(x, hidden_state)
            outputs[t_id] = pred
            pred_best = pred.argmax(dim = 1)
            x = target[t_id] if random.random() > tfratio else pred_best
        return outputs

