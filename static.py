import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from rfml.ptradio import AWGN, Transmitter, Receiver

class RFStaticDataset(Dataset):
    def __init__(
        self,
        n_symbols: int = 10,
        n_trials: int = 10,
        snr_min: float = 0,
        snr_max: float = 8,
        modulation: str = "QPSK",
        window: bool = False,
        window_size: int = 10,
        window_stride: int = 1,
        window_padding: int = 0,
    ):
        self.n_symbols: int = n_symbols
        self.n_trials: int = n_trials
        self.snr_min: float = snr_min
        self.snr_max: float = snr_max
        self.modulation: str = modulation
        
        self.window: bool = window
        self.window_size: int = window_size
        self.window_stride: int = window_stride
        self.window_padding: int = window_padding

        self.tx: Transmitter = Transmitter(modulation=modulation)
        self.rx: Receiver = Receiver(modulation=modulation)
        self.snr = np.random.uniform(snr_min, snr_max)
        self.channel: AWGN = AWGN(self.snr)

    def __len__(self) -> int:
        return self.n_trials

    def __getitem__(self, idx: int) -> tuple:        
        tx_bits: torch.Tensor = torch.randint(0, 2, (self.n_symbols,))
        
        tx_iq: torch.Tensor = self.tx.modulate(bits=tx_bits)
        rx_iq: torch.Tensor = self.channel(tx_iq)
        rx_bits: torch.Tensor = self.rx.demodulate(iq=rx_iq)
        
        n_errors: torch.Tensor = torch.sum(torch.abs(tx_bits.float() - torch.tensor(rx_bits)))
        #ber: torch.Tensor = n_errors / self.n_symbols
        
        rx_bits = torch.tensor(rx_bits).t()
        tx_iq = tx_iq.squeeze(0).squeeze(0)
        rx_iq = rx_iq.squeeze(0).squeeze(0)

        if self.window:
            rx_iq = F.pad(rx_iq, (self.window_padding, self.window_padding))
            rx_iq = rx_iq.unfold(-1, self.window_size, self.window_stride)
            _, n_windows, _ = rx_iq.shape
            rx_iq = rx_iq.permute(1, 0, 2).reshape(n_windows, -1)
            tx_iq_orig = tx_iq
            tx_iq = F.pad(tx_iq, (self.window_padding, self.window_padding))
            tx_iq = tx_iq.unfold(-1, self.window_size, self.window_stride)
            _, n_windows, _ = tx_iq.shape
            tx_iq = tx_iq.permute(1, 0, 2).reshape(n_windows, -1)
            return tx_bits, rx_bits, tx_iq, rx_iq, tx_iq_orig

        return tx_bits, rx_bits, rx_iq



# Example usage
if __name__ == "__main__":
    dataset: RFStaticDataset = RFStaticDataset(n_symbols=1000, n_trials=10)
    for i in range(len(dataset)):
        tx_bits, rx_bits, rx_iq = dataset[i]
        print(f"Sample {i + 1}: Receiver Bits = {rx_iq}, Shape = {rx_iq.shape}")
