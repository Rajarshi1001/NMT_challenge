import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import Field, BucketIterator, TabularDataset
from torch.optim import Adam
from sklearn.metrics import mean_squared_error
from torch.utils.tensorboard import SummaryWriter


DATA_DIR = "./../Data"
DROPOUT_RATE = 0.5
EPOCHS = 10
BATCH_SIZE = 64
TEACHER_FORCE_RATIO = 0.2
NUM_LAYERS = 1
HIDDEN_SIZE = 512
EMBEDDING_SIZE = 100
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



class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, num_layers):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p = DROPOUT_RATE)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, bidirectional = True)
        self.hidden_fc = nn.Linear(hidden_dim*2, hidden_dim)
        self.cell_fc = nn.Linear(hidden_dim*2, hidden_dim)
        
    def forward(self, x):
        embedding_x = self.embedding(x)
        embedding_drop = self.dropout(embedding_x)
        enc_state, (hidden_state, cell_state) = self.lstm(embedding_drop)  
        hidden_state = self.hidden_fc(torch.cat((hidden_state[0:1], hidden_state[1:2]), dim = 2))
        cell_state = self.cell_fc(torch.cat((cell_state[0:1], cell_state[1:2]), dim = 2))
        return enc_state, hidden_state, cell_state

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, output_dim, num_layers):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p = DROPOUT_RATE)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(hidden_dim*2 + embed_dim, hidden_dim, num_layers)
        self.func = nn.Linear(hidden_dim*3, 1)
        self.softmax = nn.Softmax(dim = 0)
        self.gelu = nn.GELU()
        self.ll = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, enc_state, x, hidden_state, cell_state): # implementation of attention in decoder
        x = x.unsqueeze(0)
        embedding_x = self.embedding(x)
        embedding_drop = self.dropout(embedding_x)
        seqlen = enc_state.shape[0]
        hidden_state_updated = hidden_state.repeat(seqlen, 1, 1) #(seq_len, batch_size, hidden_dim*2)
        attention = self.softmax(self.gelu(self.func(torch.cat((hidden_state_updated, enc_state), dim = 2))))
        # attention dim = (seqlen, batch_size, 1)
        attention = attention.permute(1, 2, 0) #(batch_size, 1, seq_len)
        enc_state = enc_state.permute(1, 0, 2) #(batch_size, seq_len, hidden_size*2)
        context = torch.einsum("snk,snl->knl", attention, enc_state)
        lstm_inp = torch.cat((context, embedding_drop), dim = 2)
        output, (hidden_state, cell_state) = self.lstm(lstm_inp, (hidden_state, cell_state))  # Note: GRU doesn't use cell state
        preds = self.ll(output)
        preds = preds.squeeze(0) #(batch_size, hidde_size)
        return preds, hidden_state, cell_state
        
class LSTMSeq2SeqAtt(nn.Module):
    def __init__(self, encoder, decoder):
        super(LSTMSeq2SeqAtt, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source , target, tfratio = TEACHER_FORCE_RATIO):
        tar_vocab_size = TAR_VOCAB_SIZE
        tar_len = target.shape[0]
        batch_size = source.shape[1]
        enc_state, hidden_state, cell_state = self.encoder(source)
        first_token = target[0]
        x = first_token
        outputs = torch.zeros(tar_len, batch_size, TAR_VOCAB_SIZE).to(device)
        for t_id in range(1, tar_len):
            pred, hidden_state = self.decoder(x, enc_state, hidden_state, cell_state)
            outputs[t_id] = pred
            pred_best = pred.argmax(dim = 1)
            x = target[t_id] if random.random() < tfratio else pred_best
        return outputs