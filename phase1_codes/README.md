# Machine Translation Chnallenge

This folder has all the python scripts and the notebooks containing implementation of the models like __LSTM Sequence to Sequence__, __GRU Sequence to Sequence__, __GRU Sequence to Sequence with Attention__, __Transformer__ model. The entire file structure for the following folder is shown:

```bash
├── data_process.py
├── execute.ipynb
├── generate.py
├── gru_attention_plots
│   └── loss_bengali.png
├── gru_attention_translations
│   └── answer_Bengali_test.csv
├── gru_plots
│   └── loss_bengali.png
├── gruseq2seqattn.py
├── gruseq2seqattn_train.py
├── gruseq2seq.py
├── gruseq2seq_train.py
├── gru_translations
│   └── answer_Bengali_test.csv
├── lsmseq2seqattn.py
├── lstmseq2seq.py
├── lstmseq2seq_train.py
├── Notebooks
│   ├── GRUSeq2Seq_ATTN_notebook.ipynb
│   ├── GRUSeq2Seq_notebook.ipynb
│   ├── LSTM_Seq2Seq_notebook.ipynb
│   └── transformer_notebook.ipynb
├── __pycache__
│   ├── architecture.cpython-310.pyc
│   ├── data_process.cpython-310.pyc
│   ├── generate.cpython-310.pyc
│   ├── gruseq2seqattn.cpython-310.pyc
│   ├── gruseq2seq.cpython-310.pyc
│   ├── model.cpython-310.pyc
│   ├── tf.cpython-310.pyc
│   ├── transformer_train.cpython-310.pyc
│   └── transformer_translate.cpython-310.pyc
├── README.md
├── tf.py
├── transformer_train.py
└── transformer_translate.py
├── CS779A_Competition_report.pdf
└── NMT_final_script.ipynb
```

- The files `gruseq2seqattn.py`, `tf.py`, `lstmseq2seq.py`, `gruseq2seq.py` contains the code for the model architectures written in pytorch
- The `_train.py` files essentially contains the entire training and validation loops as well as starts inferencing on the test set once the training ends. The __train__ and __validation__ loss curves generated are essentially dumped in a separate folder with 
`modelname_plots` like (gru_attention_plots for the case of __GRU__ with __Attention__ model)
- The `data_process.py` contains the code for reading the csv files of individual languages, basic preprocessing functions and hence it creates the vocabulary for source and target set using __Pytorch Field__ property. It also has the lines for creation of the train and validation dataloaders. 

## Training steps:
     
1. Firstly the change the variable `l` to the desired language (like bengali) all in lower letters in the file `data_process.py`
    ```py
    l = "bengali"
    ```
2. Choose which model you want to train and perform inference on your test set.

    - For training with `LSTM without Attention model`:
    ```bash
    $ python3 lstmseq2seq_train.py
    ```
    - For training with `GRU without Attention model`:
    ```bash
    $ python3 gruseq2seq_train.model
    ```
    - For training with `GRU with Attention model`:
    ```bash
    $ python3 gruseq2seq2attn_train.py
    ```
    - For training with `Transformer model`:
<<<<<<< HEAD
    ```bash
    $ python3 transformer_train.py
    ```
### Inference steps:

1. There are transformer models params saved on a google drive but these models are not that efficient since they are only trained for 1 epoch for reference and stored as `.pth files`. For inferencing on a sample test set with the same format as the `train_ata.json`.
        - Firstly download the `test_data.json` from the required source.
        - Save it in a google drive link and download the drive link in a cell present in he script `NMT_final_script.ipynb`
        - Run the script

The script esssentially takes first 10 sentences from each of the language csv files and produces translations and combines them into a single `answer.csv` file under the `transformer_translations` folder creates while running the script. 

=======
    ```py
    > python3 transformer_train.py
    ```
>>>>>>> 50320f48ba44df3cf55b9d052bbe1e84dea70523
