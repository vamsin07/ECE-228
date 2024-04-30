import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class RBM(nn.Module):
    def __init__(self, num_visible, num_hidden):
        super(RBM, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.weights = nn.Parameter(torch.randn(num_visible, num_hidden) * 0.1)
        self.visible_bias = nn.Parameter(torch.zeros(num_visible))
        self.hidden_bias = nn.Parameter(torch.zeros(num_hidden))
    
    def forward(self, visible_probabilities):
        hidden_activations = torch.sigmoid(torch.matmul(visible_probabilities, self.weights) + self.hidden_bias)
        hidden_states = torch.bernoulli(hidden_activations)
        visible_reconstructions = torch.sigmoid(torch.matmul(hidden_states, self.weights.t()) + self.visible_bias)
        return visible_reconstructions, hidden_activations
    
    def sample_hidden_given_visible(self, visible_probabilities):
        hidden_activations = torch.sigmoid(torch.matmul(visible_probabilities, self.weights) + self.hidden_bias)
        hidden_states = torch.bernoulli(hidden_activations)
        return hidden_states
    
    def sample_visible_given_hidden(self, hidden_states):
        visible_activations = torch.sigmoid(torch.matmul(hidden_states, self.weights.t()) + self.visible_bias)
        visible_states = torch.bernoulli(visible_activations)
        return visible_states

def hypothesis_test(rbm, data, num_samples=1000, significance_level=0.05):
    generated_samples = torch.zeros((num_samples, rbm.num_visible))
    for i in range(num_samples):
        visible_state = torch.bernoulli(torch.full((rbm.num_visible,), 0.5))
        for _ in range(10):  # Gibb's sampling to converge to a stationary distribution
            hidden_state = rbm.sample_hidden_given_visible(visible_state)
            visible_state = rbm.sample_visible_given_hidden(hidden_state)
        generated_samples[i] = visible_state
    _, p_value = stats.ks_2samp(data.flatten(), generated_samples.detach().numpy().flatten())
    if p_value < significance_level:
        print("Reject null hypothesis: The data does not follow the same distribution as RBM-generated data.")
    else:
        print("Fail to reject null hypothesis: The data follows the same distribution as RBM-generated data.")

def plot_normal_distribution(data):
    plt.hist(data.flatten(), bins=20, density=True, alpha=0.8, color='g', label='Sample data')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    mu, std = np.mean(data), np.std(data)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label='Normal distribution')
    plt.title('Normal Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# Example usage
data = np.random.binomial(3, 0.8, (100, 10)).astype(np.float32)  # Sample data
rbm = RBM(num_visible=1, num_hidden=1)  # Create RBM
optimizer = optim.SGD(rbm.parameters(), lr=0.01)  # Define optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
num_epochs = 10

for epoch in range(num_epochs):
    visible_reconstructions, _ = rbm(torch.tensor(data))
    loss = criterion(visible_reconstructions, torch.tensor(data))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

hypothesis_test(rbm, data)  # Hypothesis test
plot_normal_distribution(data)  # Plot normal distribution
