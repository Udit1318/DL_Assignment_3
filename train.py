import requests,zipfile,io
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import optim
import numpy as np
import random
import torch.nn.functional as F
import warnings
import wandb
import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
warnings.filterwarnings("ignore")
device = 'cuda:0'
# %env WANDB_MODE=disabled

def get_corpus(data):
    eng_corpus = set()  # Set to store English characters
    hin_corpus = set()  # Set to store Hindi characters

    for eng_word, hin_word in zip(data[1], data[0]):
        # Check if both are strings before proceeding
        if isinstance(eng_word, str):
            eng_corpus.update(eng_word)
        else:
            print(f"Skipping non-string eng_word: {eng_word} (type: {type(eng_word)})")

        if isinstance(hin_word, str):
            hin_corpus.update(hin_word)
        else:
            print(f"Skipping non-string hin_word: {hin_word} (type: {type(hin_word)})")

    # Add end delimiter characters to both corpora
    eng_corpus.add('#')
    hin_corpus.add('#')
    hin_corpus.add('$')
    eng_corpus.add('$')

    # Add start delimiter character to the Hindi corpus
    hin_corpus.add('^')

    return hin_corpus, eng_corpus


def word2index(data):
    hin_corpus, eng_corpus = get_corpus(data)  # Get Hindi and English corpora from data

    engchar_idx = {}  # Dictionary to map English characters to indices
    hinchar_idx = {}  # Dictionary to map Hindi characters to indices
    idx_engchar = {}  # Dictionary to map indices to English characters
    idx_hinchar = {}  # Dictionary to map indices to Hindi characters

    # Assign indices to English characters and vice versa
    for i, char in enumerate(eng_corpus):
        engchar_idx[char] = i
        idx_engchar[i] = char

    # Assign indices to Hindi characters and vice versa
    for i, char in enumerate(hin_corpus):
        hinchar_idx[char] = i
        idx_hinchar[i] = char

    eng_vocab_size = len(eng_corpus)  # Vocabulary size of English corpus
    hin_vocab_size = len(hin_corpus)  # Vocabulary size of Hindi corpus

    return engchar_idx, hinchar_idx, idx_engchar, idx_hinchar, eng_vocab_size, hin_vocab_size


def get_data():
    # Read the train, test, and validation datasets from TSV files
    tpath = "/home/narayan/Desktop/DL_Assignment_3_S038/dataset/DakshinaDataSet_Hindi/hindi_Train_dataset.csv"
    vpath = "/home/narayan/Desktop/DL_Assignment_3_S038/dataset/DakshinaDataSet_Hindi/hindi_Validation_dataset.csv"
    tspath = "/home/narayan/Desktop/DL_Assignment_3_S038/dataset/DakshinaDataSet_Hindi/hindi_Test_dataset.csv"
    train_df = pd.read_csv(tpath, sep='\t', header=None, dtype=str, 
        keep_default_na=False)
    val_df = pd.read_csv(vpath, sep='\t', header=None, dtype=str,
        keep_default_na=False)
    test_df = pd.read_csv(tspath, sep='\t', header=None, dtype=str,
        keep_default_na=False)

    train_df = train_df.drop(columns=[2])
    val_df = val_df.drop(columns=[2])
    test_df = test_df.drop(columns=[2])

    # train_df = train_df.dropna()
    # val_df = val_df.dropna()
    # test_df = test_df.dropna()

    # Convert words to indices and retrieve vocabulary information
    eng_to_idx, hin_to_idx, idx_to_eng, idx_to_hin, input_len, target_len = word2index(train_df)

    # Return the datasets and vocabulary information
    return train_df, test_df, val_df, eng_to_idx, hin_to_idx, idx_to_eng, idx_to_hin, input_len, target_len


def maxlen(data):
    maxlen_eng = max(len(word) for word in data[1])  # Maximum length of English words
    maxlen_hin = max(len(word) for word in data[0])  # Maximum length of Hindi words
    return maxlen_eng, maxlen_hin


def pre_process(data, eng_to_idx, hin_to_idx):
    eng = []  # List to store pre-processed English sentences
    hin = []  # List to store pre-processed Hindi sentences

    maxlen_eng, maxlen_hin = maxlen(data)  # Get the maximum lengths of English and Hindi words

    unknown = eng_to_idx['$']  # Index for unknown character in English corpus
    print(len(data))
    for i in range(0, len(data)):
        sz = 0  # Variable to track the size of the sentence
        eng_word = data[1][i]  # English word at index i
        hin_word = '^' + data[0][i]  # Add start delimiter (^) to Hindi word

        # Pad the English and Hindi words to their respective maximum lengths
        eng_word = eng_word.ljust(maxlen_eng + 1, '#')
        hin_word = hin_word.ljust(maxlen_hin + 1, '#')

        idx = []
        for char in eng_word:
            if eng_to_idx.get(char) is not None:
                idx.append(eng_to_idx[char])  # Append the index of the character if it exists in the corpus
            else:
                idx.append(unknown)  # Append the index of unknown character otherwise
        eng.append(idx)

        idx = []
        for char in hin_word:
            if hin_to_idx.get(char) is not None:
                idx.append(hin_to_idx[char])  # Append the index of the character if it exists in the corpus
            else:
                idx.append(unknown)  # Append the index of unknown character otherwise
        hin.append(idx)

    return eng, hin


def accuracy(predictions, y):
    count = 0
    for p, target in zip(predictions, y):
        if np.array_equal(p, target):
            count += 1
    return (count / len(predictions)) * 100

class MyDataset(Dataset):
    def __init__(self, train_x, train_y, transform=None):
        self.train_x = train_x  # Input data (train_x)
        self.train_y = train_y  # Target data (train_y)
        self.transform = transform  # Optional data transformation

    def __len__(self):
        return len(self.train_x)  # Return the length of the dataset

    def __getitem__(self, idx):
        if self.transform:
            sample = self.transform(sample)  # Apply the transformation (if any) to the sample

        # Return the input and target data tensors for the given index
        return torch.tensor(self.train_x[idx]).to(device), torch.tensor(self.train_y[idx]).to(device)


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, num_layers, batch_size, bidirectional, dropout_p=0.1):
        super(EncoderLSTM, self).__init__()
        # Initialize the hidden size, batch size, embedding size, number of layers, and bidirectional flag
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional == "Yes"

        # Create an embedding layer to convert input tokens to dense vectors
        self.embedding = nn.Embedding(input_size, embedding_size)
        
        # Create an LSTM layer with the specified parameters
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_p, bidirectional=self.bidirectional)
        
        # Create a dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, hidden, state):
        # Pass the input through the embedding layer and apply dropout
        embedded = self.dropout(self.embedding(input).view(-1, self.batch_size, self.embedding_size))
        
        # Pass the embedded input and the hidden state to the LSTM layer
        output, (hidden, state) = self.lstm(embedded, (hidden, state))

        # If the LSTM is bidirectional, adjust the hidden and state tensors
        if self.bidirectional:
            # Reshape the hidden and state tensors to separate the directions
            hidden = hidden.view(2, self.num_layers, self.batch_size, self.hidden_size)
            state = state.view(2, self.num_layers, self.batch_size, self.hidden_size)
            # Average the hidden and state tensors from both directions
            hidden = (hidden[0] + hidden[1]) / 2
            state = (state[0] + state[1]) / 2

        return output, hidden, state

    def initHidden(self):
        # Determine the number of layers based on whether the LSTM is bidirectional
        layers = 2 * self.num_layers if self.bidirectional else self.num_layers
        # Initialize the hidden state with zeros
        return torch.zeros(layers, self.batch_size, self.hidden_size, device=device)

    def initState(self):
        # Determine the number of layers based on whether the LSTM is bidirectional
        layers = 2 * self.num_layers if self.bidirectional else self.num_layers
        # Initialize the cell state with zeros
        return torch.zeros(layers, self.batch_size, self.hidden_size, device=device)


class DecoderLSTM(nn.Module):
    def __init__(self, output_size, hidden_size, embedding_size, num_layers, batch_size, dropout_p=0.1):
        super(DecoderLSTM, self).__init__()
        # Initialize hidden size, embedding size, number of layers, and batch size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        # Create an embedding layer for output tokens
        self.embedding = nn.Embedding(output_size, embedding_size)
        
        # Create an LSTM layer with the specified parameters
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_p)
        
        # Create a linear layer to map the LSTM output to the target vocabulary size
        self.out = nn.Linear(hidden_size, output_size)
        
        # Apply log softmax activation function along the specified dimension
        self.softmax = nn.LogSoftmax(dim=2)
        
        # Initialize the dropout layer
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, hidden, state):
        # Pass the input through the embedding layer and reshape
        embedded = self.embedding(input).view(-1, self.batch_size, self.embedding_size)
        
        # (Optional) Apply dropout to the embedded input (commented out in this version)
        # embedded = self.dropout(embedded)
        
        # Pass the embedded input, hidden state, and cell state to the LSTM layer
        output, (hidden, state) = self.lstm(embedded, (hidden, state))
        
        # Pass the LSTM output through the linear layer and apply softmax
        output = self.softmax(self.out(output))
        
        return output, hidden, state


class AttnDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, embedding_size, decoder_layers, batch_size, rnn_type, dropout_p=0.1):
        super(AttnDecoder, self).__init__()
        # Initialize attributes including hidden size, output size, dropout rate, batch size, RNN type, embedding size, and number of decoder layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.batch_size = batch_size
        self.rnn_type = rnn_type
        self.embedding_size = embedding_size
        self.decoder_layers = decoder_layers

        # Create an embedding layer for output tokens
        self.embedding = nn.Embedding(output_size, embedding_size)
        
        # Create a dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout_p)

        # Attention mechanism linear layers
        self.attn_U = nn.Linear(hidden_size, hidden_size, bias=False).to(device)
        self.attn_W = nn.Linear(hidden_size, hidden_size, bias=False).to(device)
        self.attn_V = nn.Linear(hidden_size, 1, bias=False).to(device)

        # Output linear layer
        self.output_linear = nn.Linear(hidden_size, output_size, bias=True)
        
        # Softmax activation functions
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=2)

        # Determine the input size for the RNN based on the embedding size and hidden size
        rnn_input_size = embedding_size + hidden_size

        # Choose the appropriate RNN type based on the provided argument
        if rnn_type == "GRU":
            self.rnn = nn.GRU(rnn_input_size, hidden_size, decoder_layers, dropout=dropout_p)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(rnn_input_size, hidden_size, decoder_layers, dropout=dropout_p)
        elif rnn_type == "RNN":
            self.rnn = nn.RNN(rnn_input_size, hidden_size, decoder_layers, dropout=dropout_p)


def forward(self, input, hidden, encoder_outputs, seq_len, state=None):
    # Embed the input and reshape it to match the expected input shape
    embedded = self.embedding(input).view(-1, self.batch_size, self.embedding_size)

    # Compute attention weights using the Bahdanau attention mechanism
    temp1 = self.attn_W(hidden[-1])
    temp2 = self.attn_U(encoder_outputs)
    c = torch.zeros(self.batch_size, 1, self.hidden_size).to(device)
    temp1 = temp1.unsqueeze(0)
    e_j = self.attn_V(F.tanh(temp1 + temp2))
    alpha_j = self.softmax(e_j)

    # Compute the context vector using the attention weights and encoder outputs
    c = torch.bmm(alpha_j.permute(1, 2, 0), encoder_outputs.permute(1, 0, 2))

    # Concatenate the embedded input and context vector
    combined_input = torch.cat((embedded[0], c.squeeze(1)), 1).unsqueeze(0)
    combined_input = F.relu(combined_input)

    # Pass the combined input through the RNN layer
    if self.rnn_type == "GRU" or self.rnn_type == "RNN":
        output, hidden = self.rnn(combined_input, hidden)
    elif self.rnn_type == "LSTM":
        output, (hidden, state) = self.rnn(combined_input, (hidden, state))

    # Pass the RNN output through the linear layer and apply log softmax
    output = self.log_softmax(self.output_linear(output))

    # Return the output, updated hidden state (and cell state for LSTM), and attention weights
    if self.rnn_type in ["GRU", "RNN"]:
        return output, hidden, alpha_j
    elif self.rnn_type == "LSTM":
        return output, hidden, state, alpha_j



def train(training_data, encoder, decoder, loss_function, encoder_optimizer, decoder_optimizer, num_encoder_layers, num_decoder_layers, batch_sz, hidden_dim, bidirectional, cell_type, use_attention):
    total_loss = 0
    teacher_forcing_ratio = 0.5

    for i, (input_seq, target_seq) in enumerate(training_data):
        loss = 0
        # Zero the gradients of both optimizers
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        input_seq = input_seq.T
        target_seq = target_seq.T
        time_steps = len(input_seq)

        if cell_type in ['GRU', 'RNN']:
            encoder_hidden = encoder.initHidden()
            # Initialize encoder hidden state
            encoder_output, encoder_hidden = encoder(input_seq, encoder_hidden)
            # Prepare decoder hidden state based on the number of layers
            if num_decoder_layers > num_encoder_layers:
                decoder_hidden = encoder_hidden
                while num_decoder_layers > num_encoder_layers:
                    decoder_hidden = torch.cat([decoder_hidden, encoder_hidden[-1].unsqueeze(0)], dim=0)
                    num_decoder_layers -= 1
            elif num_decoder_layers < num_encoder_layers:
                decoder_hidden = encoder_hidden[-num_decoder_layers:]
            else:
                decoder_hidden = encoder_hidden

            # Set initial decoder input as the first token of the target sequence
            decoder_input = target_seq[0]

            if bidirectional.lower() == "yes":
                split_tensor = torch.split(encoder_output, hidden_dim, dim=-1)
                encoder_output = (split_tensor[0] + split_tensor[1]) / 2
            # Determine if teacher forcing should be used
            use_teacher_forcing = random.random() < teacher_forcing_ratio

            if use_teacher_forcing:
                for i in range(len(target_seq)):
                    # Perform decoding with or without attention based on the parameter
                    if use_attention.lower() == "yes":
                        decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output, time_steps)
                        loss += loss_function(decoder_output.squeeze(), target_seq[i])
                        decoder_input = target_seq[i]
                    else:
                        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                        loss += loss_function(decoder_output.squeeze(), target_seq[i])
                        decoder_input = target_seq[i]
            else:
                for i in range(len(target_seq)):
                    if use_attention.lower() == "yes":
                        decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output, time_steps)
                        top_value, top_index = decoder_output.topk(1)
                        loss += loss_function(decoder_output.squeeze(), target_seq[i])
                        decoder_input = top_index.squeeze().detach()
                    else:
                        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                        top_value, top_index = decoder_output.topk(1)
                        loss += loss_function(decoder_output.squeeze(), target_seq[i])
                        decoder_input = top_index.squeeze().detach()

            # Backpropagation and optimization
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            total_loss += loss.item()

        elif cell_type == 'LSTM':
            encoder_hidden = encoder.initHidden()
            encoder_state = encoder.initState()
            encoder_output, encoder_hidden, encoder_state = encoder(input_seq, encoder_hidden, encoder_state)

            if num_decoder_layers > num_encoder_layers:
                decoder_hidden = encoder_hidden
                decoder_state = encoder_state
                while num_decoder_layers > num_encoder_layers:
                    decoder_hidden = torch.cat([decoder_hidden, encoder_hidden[-1].unsqueeze(0)], dim=0)
                    decoder_state = torch.cat([decoder_state, encoder_state[-1].unsqueeze(0)], dim=0)
                    num_decoder_layers -= 1
            elif num_decoder_layers < num_encoder_layers:
                decoder_hidden = encoder_hidden[-num_decoder_layers:]
                decoder_state = encoder_state[-num_decoder_layers:]
            else:
                decoder_hidden = encoder_hidden
                decoder_state = encoder_state

            if bidirectional.lower() == "yes":
                split_tensor = torch.split(encoder_output, hidden_dim, dim=-1)
                encoder_output = (split_tensor[0] + split_tensor[1]) / 2

            decoder_input = target_seq[0]
            use_teacher_forcing = random.random() < teacher_forcing_ratio

            if use_teacher_forcing:
                for i in range(len(target_seq)):
                    if use_attention.lower() == "yes":
                        decoder_output, decoder_hidden, decoder_state, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output, time_steps, decoder_state)
                        loss += loss_function(decoder_output.squeeze(), target_seq[i])
                        decoder_input = target_seq[i]
                    else:
                        decoder_output, decoder_hidden, decoder_state = decoder(decoder_input, decoder_hidden, decoder_state)
                        loss += loss_function(decoder_output.squeeze(), target_seq[i])
                        decoder_input = target_seq[i]
            else:
                for i in range(len(target_seq)):
                    if use_attention.lower() == "yes":
                        decoder_output, decoder_hidden, decoder_state, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output, time_steps, decoder_state)
                        top_value, top_index = decoder_output.topk(1)
                        loss += loss_function(decoder_output.squeeze(), target_seq[i])
                        decoder_input = top_index.squeeze().detach()
                    else:
                        decoder_output, decoder_hidden, decoder_state = decoder(decoder_input, decoder_hidden, decoder_state)
                        top_value, top_index = decoder_output.topk(1)
                        loss += loss_function(decoder_output.squeeze(), target_seq[i])
                        decoder_input = top_index.squeeze().detach()

            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            total_loss += loss.item()

    return total_loss / len(training_data), encoder, decoder


def train_iter(train_data, val_data, val_targets, input_dim, target_dim, epochs, batch_size, embed_dim, enc_layers, dec_layers, hidden_dim, rnn_type, bidirectional, dropout, use_attention, beam_size=0):
    learning_rate = 0.001

    # Initialize the encoder and decoder based on the RNN type and attention mechanism
    if rnn_type == 'GRU':
        encoder = EncoderGRU(input_dim, hidden_dim, embed_dim, enc_layers, batch_size, bidirectional, dropout).to(device)
        if use_attention == "Yes":
            decoder = AttnDecoder(target_dim, hidden_dim, embed_dim, dec_layers, batch_size, rnn_type, dropout).to(device)
        else:
            decoder = DecoderGRU(target_dim, hidden_dim, embed_dim, dec_layers, batch_size, dropout).to(device)
    elif rnn_type == 'RNN':
        encoder = EncoderRNN(input_dim, hidden_dim, embed_dim, enc_layers, batch_size, bidirectional, dropout).to(device)
        if use_attention == "Yes":
            decoder = AttnDecoder(target_dim, hidden_dim, embed_dim, dec_layers, batch_size, rnn_type, dropout).to(device)
        else:
            decoder = DecoderRNN(target_dim, hidden_dim, embed_dim, dec_layers, batch_size, dropout).to(device)
    elif rnn_type == 'LSTM':
        print('Entered LSTM')
        encoder = EncoderLSTM(input_dim, hidden_dim, embed_dim, enc_layers, batch_size, bidirectional, dropout).to(device)
        if use_attention == "Yes":
            decoder = AttnDecoder(target_dim, hidden_dim, embed_dim, dec_layers, batch_size, rnn_type, dropout).to(device)
        else:
            decoder = DecoderLSTM(target_dim, hidden_dim, embed_dim, dec_layers, batch_size, dropout).to(device)

    encoder_optimizer = optim.Adam(encoder.parameters(), learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        train_loss, encoder, decoder = train(train_data, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, enc_layers, dec_layers, batch_size, hidden_dim, bidirectional, rnn_type, use_attention)
        val_outputs, val_loss, attn_weights = eval(val_data, encoder, decoder, enc_layers, dec_layers, batch_size, hidden_dim, bidirectional, rnn_type, use_attention)

        train_losses.append(train_loss / len(train_data))
        val_losses.append(val_loss)

        val_accuracy = accuracy(val_outputs, val_targets)
        val_accuracies.append(val_accuracy)

    

    return train_losses, val_losses, val_accuracies, encoder, decoder, enc_layers, dec_layers

def eval(input_data, encoder, decoder, encoder_layers, decoder_layers, batch_size, hidden_size, bi_directional, cell_type, attention, build_matrix=False):
    with torch.no_grad():
        loss_fun = nn.CrossEntropyLoss(reduction="sum")
        total_loss = 0
        pred_words = []
        attention_matrix = []

        for X, y in input_data:
            attn = []
            loss = 0
            decoder_words = []
            x = x.T
            y = y.T

            # Initialize the encoder hidden state
            if cell_type == 'LSTM':
                encoder_hidden = encoder.initHidden()
                encoder_state = encoder.initState()
                encoder_output, encoder_hidden, encoder_state = encoder(x, encoder_hidden, encoder_state)
            else:
                encoder_hidden = encoder.initHidden()
                encoder_output, encoder_hidden = encoder(x, encoder_hidden)

            timesteps = len(x)

            if decoder_layers > encoder_layers:
                i = decoder_layers
                decoder_hidden = encoder_hidden
                decoder_state = encoder_state if cell_type == 'LSTM' else None

                while True:
                    if i == encoder_layers:
                        break
                    decoder_hidden = torch.cat([decoder_hidden, encoder_hidden[-1].unsqueeze(0)], dim=0)
                    if cell_type == 'LSTM':
                        decoder_state = torch.cat([decoder_state, encoder_state[-1].unsqueeze(0)], dim=0)
                    i -= 1
            elif decoder_layers < encoder_layers:
                decoder_hidden = encoder_hidden[-decoder_layers:]
                decoder_state = encoder_state[-decoder_layers:] if cell_type == 'LSTM' else None
            else:
                decoder_hidden = encoder_hidden
                decoder_state = encoder_state if cell_type == 'LSTM' else None

            decoder_input = y[0]

            if bi_directional == "Yes":
                split_tensor = torch.split(encoder_output, hidden_size, dim=-1)
                encoder_output = torch.add(split_tensor[0], split_tensor[1]) / 2

            for i in range(len(y)):
                if attention == "Yes":
                    if cell_type == 'LSTM':
                        decoder_output, decoder_hidden, decoder_state, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output, len(x), decoder_state)
                    else:
                        decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output, len(x))
                    max_prob, index = decoder_output.topk(1)
                    loss += loss_fun(torch.squeeze(decoder_output), y[i])
                    index = index.squeeze()
                    decoder_input = index
                    decoder_words.append(index.tolist())
                    if build_matrix:
                        attn.append(attn_weights)
                else:
                    if cell_type == 'LSTM':
                        decoder_output, decoder_hidden, decoder_state = decoder(decoder_input, decoder_hidden, decoder_state)
                    else:
                        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    max_prob, index = decoder_output.topk(1)
                    loss += loss_fun(torch.squeeze(decoder_output), y[i])
                    index = index.squeeze()
                    decoder_input = index
                    # print(f'Index shape {index.shape}')
                    decoder_words.append(index.tolist())

            if build_matrix:
                attention_matrix = torch.cat(attn, dim=2).to(device)

            decoder_words = np.array(decoder_words)
            pred_words.append(decoder_words.T)
            total_loss += loss.item()

        predictions = [word for batch in pred_words for word in batch]

    return predictions, total_loss / (len(predictions) * len(predictions[0])), attention_matrix





def wandb_run_sweeps(train_dataset, val_dataset, test_dataset, train_y, val_y, test_y, input_len, target_len):
    config = {
        "project":"joke",
        "method": 'grid',
        "metric": {
        'name': 'val_acc',
        'goal': 'maximize'
        },
        'parameters' :{
        "epochs": {"values":[5]},
        "batchsize": {"values": [512]},
        "embedding_size": {"values":[128]},
        "hidden_size": {"values":[256]},
        "encoder_layers": {"values":[2]},
        "decoder_layers": {"values":[2]},
        "cell_type": {"values":["LSTM"]},
        "bi_directional":{"values":["No"]},
        "dropout":{"values":[0.2, 0.3, 0.5]},
        "attention":{"values":["No"]},
        }
    }

    def train_rnn():
        wandb.init()
        name = '_CT_' + str(wandb.config.cell_type) + "_BS_" + str(wandb.config.batchsize) + "_EPOCH_" + str(
            wandb.config.epochs) + "_ES_" + str(wandb.config.embedding_size) + "_HS_" + str(
            wandb.config.hidden_size)
        wandb.run.name = name
        train_dataloader = DataLoader(train_dataset, batch_size=wandb.config.batchsize,drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=wandb.config.batchsize,drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=wandb.config.batchsize,drop_last=True)

        print(f'Len train dataset = {len(train_dataset)}')
        print(f'Len of train dataloader = {len(train_dataloader)}')

        x = next(iter(train_dataloader))
        print(x[0])
        epoch_train_loss, epoch_val_loss, epoch_val_acc, encoder, decoder, encoder_layers, decoder_layers = train_iter(
            train_dataloader, val_dataloader, val_y, input_len, target_len, wandb.config.epochs, wandb.config.batchsize,
            wandb.config.embedding_size, wandb.config.encoder_layers, wandb.config.decoder_layers,
            wandb.config.hidden_size, wandb.config.cell_type, wandb.config.bi_directional, wandb.config.dropout,
            wandb.config.attention)

        train_predictions, _, _ = eval(train_dataloader, encoder, decoder, wandb.config.encoder_layers,
                                       wandb.config.decoder_layers, wandb.config.batchsize, wandb.config.hidden_size,
                                       wandb.config.bi_directional, wandb.config.cell_type, wandb.config.attention)

        test_predictions, _, _ = eval(test_dataloader, encoder, decoder, wandb.config.encoder_layers,
                                      wandb.config.decoder_layers, wandb.config.batchsize, wandb.config.hidden_size,
                                      wandb.config.bi_directional, wandb.config.cell_type, wandb.config.attention)

        train_accuracy = accuracy(train_predictions, train_y)
        test_accuracy = accuracy(test_predictions, test_y)
        print("train_accuracy:", train_accuracy)
        print("test_accuracy:", test_accuracy)

        wandb.run.save()
        wandb.run.finish()

    wandb.login(key="")
    sweep_id = wandb.sweep(config, project="A3-DL")
    wandb.agent(sweep_id, function=train_rnn, count=1)




if '__name__' == 'main':
    train_df,test_df,val_df,eng_to_idx,hin_to_idx,idx_to_eng,idx_to_hin,input_len,target_len=get_data()

    train_x,train_y = pre_process(train_df,eng_to_idx,hin_to_idx)
    test_x,test_y = pre_process(test_df,eng_to_idx,hin_to_idx)
    val_x,val_y = pre_process(val_df,eng_to_idx,hin_to_idx)

    train_dataset=MyDataset(train_x,train_y)
    test_dataset=MyDataset(test_x,test_y)
    val_dataset=MyDataset(val_x,val_y)
    wandb_run_sweeps(train_dataset,val_dataset,test_dataset,train_y,val_y,test_y,input_len,target_len)