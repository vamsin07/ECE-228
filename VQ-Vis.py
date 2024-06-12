import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from rfml.ptradio import AWGN, Transmitter, Receiver
from VQ import VectorQuantizationLayer1D
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class RFStaticDataset(Dataset):
    # Your RFStaticDataset implementation
    def __init__(self, n_symbols=1000, n_trials=1, modulation="QPSK"):
        self.n_symbols = n_symbols
        self.n_trials = n_trials
        self.tx = Transmitter(modulation=modulation)
        self.rx = Receiver(modulation=modulation)

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        # Generate data for one sample
        tx_bits = torch.randint(0, 2, (self.n_symbols,), dtype=torch.float32)
        # Simulate transmission and reception
        tx_iq = self.tx.modulate(tx_bits)
        snr = np.random.uniform(0, 8)  # Example SNR generation
        channel = AWGN(snr)
        rx_iq = channel(tx_iq)
        rx_bits = self.rx.demodulate(rx_iq)
        # Return relevant data as a dictionary
        return {
            "tx_bits": tx_bits,
            "rx_bits": rx_bits,
            "snr": snr
        }


# Define a function to train the VQ layer using the dataset
def train_vq_layer_with_dataset(dataset, num_codewords_list, num_epochs=100, lr=0.001):
    quantization_losses = []
    distances_data = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define device
    for num_codewords in num_codewords_list:
        # Initialize VQ layer
        vq_layer = VectorQuantizationLayer1D(num_codewords)
        vq_layer.to(device)  # Move VQ layer to GPU if available
        
        # Define optimizer
        optimizer = torch.optim.Adam(vq_layer.parameters(), lr=lr)
        
        # Define data loader
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for sample in data_loader:
                tx_bits = sample["tx_bits"].squeeze()  # Get transmitted bits
                tx_bits = tx_bits.to(device)  # Move data to GPU if available

                # Forward pass
                quantized_indices, quantized_distances = vq_layer(tx_bits)
                
                # Compute loss
                loss = torch.mean(quantized_distances)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(data_loader)}")

        quantization_losses.append(epoch_loss / len(data_loader))
        #distances_data.append(quantized_distances.cpu().detach().numpy())
    
    return num_codewords_list, quantization_losses

# Function to plot quantization loss and distances against number of codewords
def plot_quantization_loss_and_distances(num_codewords_list, quantization_losses):
    fig, ax1 = plt.subplots()


    ax1.set_xlabel('Number of Codewords')
    ax1.set_ylabel('Log Loss', color='tab:blue')

    log_quantization_losses = np.log(quantization_losses)
    ax1.plot(num_codewords_list, log_quantization_losses, marker='o', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    fig.tight_layout()
    plt.title('Loss vs Number of Codewords')
    plt.show()

# Example usage
if __name__ == "__main__":
    dataset = RFStaticDataset(n_symbols=1000, n_trials=1, modulation="BPSK")
    num_codewords_list = [10, 20, 30, 40 , 50, 60, 70, 80, 90, 100]  # Different numbers of codewords to test
    num_codewords, quantization_losses = train_vq_layer_with_dataset(dataset, num_codewords_list)
    plot_quantization_loss_and_distances(num_codewords, quantization_losses)
