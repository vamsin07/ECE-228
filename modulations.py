import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib.pyplot as plt
from rfml.ptradio import AWGN, Transmitter, Receiver, theoreticalBER

modulations = ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64"]
snrs = range(0, 8)
n_trials = 10

for modulation in modulations:
    tx = Transmitter(modulation=modulation)
    channel = AWGN()
    rx = Receiver(modulation=modulation)

    n_symbols = int(10e3)
    n_bits = int(tx.symbol_encoder.get_bps() * n_symbols)

    ber_values = []

    for snr in snrs:
        channel.set_snr(snr)
        n_errors = 0

        for _ in range(n_trials):
            tx_bits = np.random.randint(low=0, high=2, size=n_bits)
            tx_iq = tx.modulate(bits=tx_bits)

            rx_iq = channel(tx_iq)

            rx_bits = rx.demodulate(iq=rx_iq)
            rx_bits = np.array(rx_bits)

            n_errors += np.sum(np.abs(tx_bits - rx_bits))

        ber = float(n_errors) / float(n_bits * n_trials)
        ber_values.append(ber)

    snr_values = list(snrs)
    theory_values = [theoreticalBER(modulation=modulation, snr=snr) for snr in snr_values]

    plt.semilogy(snr_values, ber_values, marker='o', linestyle='-', label='Simulated {}'.format(modulation))
    plt.semilogy(snr_values, theory_values, 'r--', label='Theoretical {}'.format(modulation))

plt.xlabel('SNR(dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER vs SNR for Different Modulations')
plt.legend()
plt.grid(True)
plt.yscale('log')  # Set y-axis to log scale
plt.show()
