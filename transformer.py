import math
import time

import torch
from torch import nn
import torch.nn.functional as F


class StepFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output, min_val=0.0, max_val=1.0)


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
        self.input_embedding = nn.Linear(input_dim, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8),
            num_layers=6,
        )
        self.flatten = nn.Flatten()
        self.decoder = nn.Linear(embedding_dim * num_windows, output_dim)
        
    def forward(self, input):
        input = F.relu(self.input_embedding(input))
        input = self.pos_encoder(input)
        x = self.transformer_encoder(input)
        x = self.flatten(x)
        x = self.decoder(x)
        x = F.sigmoid(x)
        return StepFunction.apply(x)


def evaluate(
    model,
    dataloader,
    criterion,
    nbits,
    ntrials,
):
    model.eval()
    num_errors = 0
    total_loss = 0.0
    with torch.no_grad():
        for tx_bits, _, rx_iq in dataloader:
            tx_bits = tx_bits.float()
            output = model(rx_iq)
            total_loss += criterion(output, tx_bits).item()
            num_errors += torch.sum(torch.abs(output - tx_bits)).item()
    ber = float(num_errors) / float(nbits * ntrials)
    return total_loss / len(dataloader), ber


def train(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    nbits,
    ntrials,
):
    model.train()
    num_errors = 0
    total_loss = 0.0
    for tx_bits, _, rx_iq in dataloader:
        tx_bits = tx_bits.float()
        optimizer.zero_grad()
        output = model(rx_iq)
        loss = criterion(output, tx_bits)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        num_errors += torch.sum(torch.abs(output - tx_bits)).item()
    ber = float(num_errors) / float(nbits * ntrials)
    avg_loss = total_loss / len(dataloader)           
    return model, avg_loss, ber


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    train_nbits,
    val_nbits,
    train_ntrials,
    val_ntrials,
):
    best_val_loss = float("inf")
    best_model = None
    for epoch in range(num_epochs):
        start_time = time.time()
        model, train_loss, train_ber = train(model, train_loader, criterion, optimizer, scheduler, train_nbits, train_ntrials)
        val_loss, val_ber = evaluate(model, val_loader, criterion, val_nbits, val_ntrials)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} / {num_epochs} - Train Loss: {train_loss:.2f} - Val Loss: {val_loss:.2f} - Elapsed Time: {elapsed:.2f} seconds")
        print(f"Train BER: {train_ber:.4f} - Val BER: {val_ber:.4f}") 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
    return best_model, best_val_loss
