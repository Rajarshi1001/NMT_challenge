# GRU Sequence 2 Sequence Model

The folder contains the implementation of the __GRUSeq2Seq__ model in Pytorch. The folder structure is as follows:

- `model.py`: contains the classes pertaining to the implementation of the __Encoder__ and the __Decoder__ part and the collective model in separate classes inheriting the pytorch's __nn.Module__ class. The input to the Encoder is a tensor of `shape: (seq_length, batch_size)` where _seq_length_ denotes the length of the tokenized numerical representations of each sentence after the creationg of the vocabulary. The input to the Decoder is rather a tensor of `shape: (batch_size)` and spits out a tensor of `shape: (seq_length, batch_size, target_vocab)`

- `preprocess.py`: contains basic functions for creation of the vocabulary and cleaning the dataset

- `train.py`: contains the main *train* function *translate* function for performing the machine translations on the validation dataset. 

Initiate your training by changing the hyperparemeters defined in `train.py` as of now (might be updated later):
```bash
python3 train.py
```

