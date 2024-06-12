import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from rfml.ptradio import AWGN, Transmitter, Receiver
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# RFStaticDataset class
class RFStaticDataset(Dataset):
    modulation_schemes = {1: "BPSK", 2: "QPSK", 3: "8PSK", 4: "16QAM", 5: "32QAM", 6: "64QAM"}

    def __init__(self, n_symbols=10, n_trials=10, snr_min=0, snr_max=8, modulation_order=1):
        self.n_symbols = n_symbols
        self.n_trials = n_trials
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.modulation = self.modulation_schemes[modulation_order]
        self.tx = Transmitter(modulation=self.modulation)
        self.rx = Receiver(modulation=self.modulation)

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        snr = np.random.uniform(self.snr_min, self.snr_max)
        tx_bits = torch.randint(0, 2, (self.n_symbols,))
        channel = AWGN(snr)
        tx_iq = self.tx.modulate(bits=tx_bits)
        rx_iq = channel(tx_iq)
        rx_bits = self.rx.demodulate(iq=rx_iq)
        return tx_bits, rx_bits, tx_iq, rx_iq

# Transformer Encoder class
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, nhead, num_layers):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(input_dim, encoding_dim)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.linear(x)
        return x

# Transformer Decoder class
class TransformerDecoder(nn.Module):
    def __init__(self, encoding_dim, input_dim, nhead, num_layers):
        super(TransformerDecoder, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=encoding_dim, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(encoding_dim, input_dim)

    def forward(self, x, memory):
        x = self.transformer_decoder(x, memory)
        x = self.linear(x)
        return x

# Vector Quantization Layer
class VectorQuantizationLayer1D(nn.Module):
    def __init__(self, num_codewords, encoding_dim):
        super(VectorQuantizationLayer1D, self).__init__()
        self.num_codewords = num_codewords
        self.codewords = nn.Parameter(torch.randn(num_codewords, encoding_dim))

    def forward(self, input_data):
        input_data = input_data.view(-1, self.codewords.size(1))  # Ensure input data has the correct shape
        distances = torch.cdist(input_data.unsqueeze(0), self.codewords.unsqueeze(0)).squeeze(0)
        quantized_indices = torch.argmin(distances, dim=1)
        quantized_distances = distances.gather(1, quantized_indices.unsqueeze(1)).squeeze(1)
        quantized_data = self.codewords[quantized_indices]
        return quantized_indices, quantized_distances, quantized_data

# VQVAE model
class VQVAE(nn.Module):
    def __init__(self, input_dim, encoding_dim, num_codewords, nhead, num_layers):
        super(VQVAE, self).__init__()
        self.encoder = TransformerEncoder(input_dim, encoding_dim, nhead, num_layers)
        self.vq_layer = VectorQuantizationLayer1D(num_codewords, encoding_dim)
        self.decoder = TransformerDecoder(encoding_dim, input_dim, nhead, num_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        quantized_indices, quantized_distances, quantized = self.vq_layer(encoded)
        reconstructed = self.decoder(quantized, encoded)
        return quantized_indices, quantized_distances, reconstructed, quantized

# Training function
def train_vqvae(dataset, input_dim, encoding_dim, num_codewords, nhead, num_layers, num_epochs=100, lr=0.001):
    vqvae = VQVAE(input_dim, encoding_dim, num_codewords, nhead, num_layers)
    vqvae.to(device)
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=lr)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    quantization_losses = []
    reconstruction_losses = []

    for epoch in range(num_epochs):
        epoch_q_loss = 0.0
        epoch_r_loss = 0.0
        for tx_bits, rx_bits, tx_iq, rx_iq in dataloader:
            tx_bits = tx_bits.float().to(device)
            optimizer.zero_grad()
            _, quantized_distances, reconstructed, quantized = vqvae(tx_bits)
            q_loss = torch.mean(quantized_distances)  # Quantization loss
            r_loss = nn.functional.mse_loss(reconstructed, tx_bits)  # Reconstruction loss
            loss = q_loss + r_loss
            loss.backward()
            optimizer.step()
            epoch_q_loss += q_loss.item()
            epoch_r_loss += r_loss.item()

        quantization_losses.append(epoch_q_loss / len(dataloader))
        reconstruction_losses.append(epoch_r_loss / len(dataloader))
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Quantization Loss: {epoch_q_loss / len(dataloader)}, Reconstruction Loss: {epoch_r_loss / len(dataloader)}")

    return vqvae, quantization_losses, reconstruction_losses

# Function to plot losses for different numbers of codewords
def plot_losses_for_codewords(dataset, input_dim, encoding_dim, num_codewords_list, nhead, num_layers, num_epochs=100, lr=0.001):
    all_quantization_losses = {}
    all_reconstruction_losses = {}
    models = {}
    for num_codewords in num_codewords_list:
        print(f"Training with {num_codewords} codewords...")
        model, quantization_losses, reconstruction_losses = train_vqvae(dataset, input_dim, encoding_dim, num_codewords, nhead, num_layers, num_epochs, lr)
        all_quantization_losses[num_codewords] = quantization_losses
        all_reconstruction_losses[num_codewords] = reconstruction_losses
        models[num_codewords] = model

    plt.figure(figsize=(10, 6))
    for num_codewords, losses in all_quantization_losses.items():
        plt.plot(losses, label=f'{num_codewords} Codewords - Quantization')
    for num_codewords, losses in all_reconstruction_losses.items():
        plt.plot(losses, linestyle='--', label=f'{num_codewords} Codewords - Reconstruction')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Quantization and Reconstruction Loss for Different Numbers of Codewords')
    plt.legend()
    plt.show()

    return models

# Main
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for k in range(1, 7):
        print(f"Training with modulation scheme 2^{k}...")
        dataset = RFStaticDataset(n_symbols=1000, n_trials=100, modulation_order=k)
        input_dim = 1000
        encoding_dim = 128
        nhead = 8
        num_layers = 2
        num_codewords_list = [2 ** k]  # Number of codewords based on modulation scheme
        plot_losses_for_codewords(dataset, input_dim, encoding_dim, num_codewords_list, nhead, num_layers)
