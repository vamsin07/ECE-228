import math
import time

import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        dropout=0.1,
        max_len=5000,
    ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_windows: int,
        embedding_dim: int,
        output_dim: int,
    ):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        self.transformer = nn.Transformer(d_model=embedding_dim)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(embedding_dim * num_windows, 512)
        self.fc2 = nn.Linear(512, output_dim)
        
    def forward(self, input, target):
        input = F.relu(self.embedding(input))
        input = self.pos_encoder(input)
        target = F.relu(self.embedding(target))
        target = self.pos_encoder(target)
        x = self.transformer(input, target)
        x = self.flatten(x).unsqueeze(1)
        x = x.repeat(1, 2, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def evaluate(
    model,
    dataloader,
    criterion,
    nbits,
    ntrials,
):
    demodulate = dataloader.dataset.rx.demodulate
    model.eval()
    num_model_errors = 0
    num_errors = 0
    total_loss = 0.0
    with torch.no_grad():
        for tx_bits, rx_bits, tx_iq, rx_iq, tx_iq_orig in dataloader:
            output = model(rx_iq, tx_iq)
            total_loss += criterion(output, tx_iq_orig).item()
            rx_bits_model = torch.zeros(tx_bits.shape)
            for i in range(output.shape[0]):
                iq = output[i].unsqueeze(0).unsqueeze(0)
                rx_bits_model[i] = torch.tensor(demodulate(iq=iq)).t()
            num_model_errors += torch.sum(torch.abs(rx_bits_model - tx_bits)).item()
            num_errors += torch.sum(torch.abs(rx_bits - tx_bits)).item()
    model_ber = float(num_model_errors) / float(nbits * ntrials)
    ber = float(num_errors) / float(nbits * ntrials)
    return total_loss / len(dataloader), model_ber, ber


def train(
    model,
    dataloader,
    criterion,
    optimizer,
    nbits,
    ntrials,
):
    demodulate = dataloader.dataset.rx.demodulate
    model.train()
    num_model_errors = 0
    num_errors = 0
    total_loss = 0.0
    for tx_bits, rx_bits, tx_iq, rx_iq, tx_iq_orig in dataloader:
        optimizer.zero_grad()
        output = model(rx_iq, tx_iq)
        loss = criterion(output, tx_iq_orig)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        rx_bits_model = torch.zeros(tx_bits.shape)
        for i in range(output.shape[0]):
            iq = output[i].unsqueeze(0).unsqueeze(0)
            rx_bits_model[i] = torch.tensor(demodulate(iq=iq)).t()
        num_model_errors += torch.sum(torch.abs(rx_bits_model - tx_bits)).item()
        num_errors += torch.sum(torch.abs(rx_bits - tx_bits)).item()
    model_ber = float(num_model_errors) / float(nbits * ntrials)
    ber = float(num_errors) / float(nbits * ntrials)
    avg_loss = total_loss / len(dataloader)           
    return model, avg_loss, model_ber, ber


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    train_nbits,
    val_nbits,
    train_ntrials,
    val_ntrials,
):
    train_losses, val_losses = [], []
    train_bers, val_bers = [], []
    best_val_loss = float("inf")
    best_model = None
    for epoch in range(num_epochs):
        start_time = time.time()
        model, train_loss, train_ber_model, train_ber  = train(model, train_loader, criterion, optimizer, train_nbits, train_ntrials)
        val_loss, val_ber_model, val_ber = evaluate(model, val_loader, criterion, val_nbits, val_ntrials)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} / {num_epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Elapsed Time: {elapsed:.2f} seconds")
        print(f"Train BER: {train_ber:.4f} - Train BER (Model): {train_ber_model:.4f} - Val BER: {val_ber:.4f} - Val BER (Model): {val_ber_model:.4f}") 
        train_bers.append(train_ber)
        val_bers.append(val_ber)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
    return best_model, best_val_loss, train_losses, val_losses, train_bers, val_bers


def plot_transformer_losses(
    train_losses,
    val_losses,
    train_bers,
    val_bers,
):
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(121)
    ax1.plot(train_losses, label='Train Reconstruction Loss', linestyle='--')
    ax1.plot(val_losses, label='Validation Reconstrunction Loss')
    ax1.set_ylabel('Reconstruction Loss (MSELoss)')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax2 = fig.add_subplot(122)
    ax2.plot(train_bers, label='Train BER', linestyle='--')
    ax2.plot(val_bers, label='Validation BER')
    ax2.set_ylabel('Bit Error Rate (BER)')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    fig.tight_layout()
    plt.show()