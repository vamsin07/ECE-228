import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, f_oneway
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class VectorQuantizationLayer1D(nn.Module):
    def __init__(self, num_codewords):
        super(VectorQuantizationLayer1D, self).__init__()
        self.num_codewords = num_codewords
        self.codewords = nn.Parameter(torch.randn(num_codewords))

    def forward(self, input_data):
        distances = torch.abs(input_data.unsqueeze(1) - self.codewords)
        quantized_indices = torch.argmin(distances, dim=1)
        quantized_distances = torch.abs(input_data - self.codewords[quantized_indices])
        return quantized_indices, quantized_distances

def hypothesis_test(num_codewords_list, significance_level=0.05):
    distances_data = []
    for num_codewords in num_codewords_list:
        input_data = torch.randn(1000)  # Generate random 1D data
        vq_layer = VectorQuantizationLayer1D(num_codewords)
        quantized_indices, quantized_distances = vq_layer(input_data)
        distances_data.append(quantized_distances.cpu().detach().numpy())

    _, p_value = f_oneway(*distances_data)
    return p_value

def plot_normal_distribution_with_hypothesis_test(num_codewords_list):
    input_data = torch.randn(100)  # Generate random 1D data

    for num_codewords in num_codewords_list:
        # Initialize and fit the VQ layer
        vq_layer = VectorQuantizationLayer1D(num_codewords)
        
        # Move the model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vq_layer.to(device)
        input_data = input_data.to(device)
        
        quantized_indices, quantized_distances = vq_layer(input_data)
        
        # Plot histogram
        sns.histplot(quantized_distances.cpu().detach().numpy(), kde=True, label=f"{num_codewords} codewords", stat="density")

        # Fit normal distribution
        mu, std = norm.fit(quantized_distances.cpu().detach().numpy())
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)

    plt.legend()
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.title('Distribution of Distances for Different Numbers of Codewords')
    plt.show()

# Example usage
num_codewords_list = [10, 20, 30]  # Different numbers of codewords to test
plot_normal_distribution_with_hypothesis_test(num_codewords_list)
