import numpy as np
import torch
from torch.utils.data import Dataset
from rfml.ptradio import AWGN, Transmitter, Receiver

class RFStaticDataset(Dataset):
    def __init__(self, n_symbols: int = 10, n_trials: int = 10, snr_min: float = 0, snr_max: float = 8, modulation: str = "QPSK") -> None:
        self.n_symbols: int = n_symbols
        self.n_trials: int = n_trials
        self.snr_min: float = snr_min
        self.snr_max: float = snr_max
        self.modulation: str = modulation
        
        self.tx: Transmitter = Transmitter(modulation=modulation)
        self.rx: Receiver = Receiver(modulation=modulation)

    def __len__(self) -> int:
        return self.n_trials

    def __getitem__(self, idx: int) -> tuple:
        
        snr: float = np.random.uniform(self.snr_min, self.snr_max)

        
        tx_bits: torch.Tensor = torch.randint(0, 2, (self.n_symbols,))
        
        channel: AWGN = AWGN(snr)
        tx_iq: torch.Tensor = self.tx.modulate(bits=tx_bits)
        rx_iq: torch.Tensor = channel(tx_iq)
        rx_bits: torch.Tensor = self.rx.demodulate(iq=rx_iq)

        
        n_errors: torch.Tensor = torch.sum(torch.abs(tx_bits.float() - torch.tensor(rx_bits)))
        #ber: torch.Tensor = n_errors / self.n_symbols

        return tx_bits, rx_bits, rx_iq

# Example usage
if __name__ == "__main__":
    dataset: RFStaticDataset = RFStaticDataset(n_symbols=1000, n_trials=10)
    for i in range(len(dataset)):
        tx_bits, rx_bits, rx_iq = dataset[i]
        print(f"Sample {i + 1}: Receiver Bits = {rx_iq}")
