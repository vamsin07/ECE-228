<h1>ECE 228: Final Project</h1>

# Signal Reconstruction Using VQ-VAE and Transformer

## Description

This project explores the use of VQ-VAE and Transformer models for signal reconstruction of I/Q data under various modulation schemes and unknown noise distributions. The project aims to enhance signal representation, improve reconstruction quality, and achieve better noise robustness.

## Table of Contents

- [Installation](#installation)
- [Features](#features)
- [Contributing](#contributing)
- [Acknowledgements](#acknowldegements)
- [License](#license)


## Installation

Follow these steps to set up the project locally:

# Clone the repository
git clone https://github.com/your_username/ECE-228.git

# Navigate to the project directory
cd ECE-228

# Install dependencies
pip install -r requirements.txt


## Features
<h3>Log Loss Plot for the number of codewords v/s the Quantisation Loss</h3>

![Log_Quantisation_loss_3](https://github.com/vamsin07/ECE-228/assets/51058745/b4a41125-7c11-4ea9-8c01-226f0629a7df)

<h5>Here we look at the quantisation loss generated over the number of codewords for a Vector Quantisation Layer. As the the number of codewords increases(the distance) the quantisation loss is expected to decrease.</h5>

<h3>BER v/s SNR for Different Modulations</h3>

![Mult_modulations](https://github.com/vamsin07/ECE-228/assets/51058745/2adc3dd6-07ed-44ef-8185-5bf92ac8b2a3)

<h5>We generated a static dataset that returns the BER, SNR, transmitted and received bits, and we plotted the BER v/s SNR for the different modulation schemes.</h5>

<h3>Normal Distribution Curve for different number of codewords</h3>

![Plot_7](https://github.com/vamsin07/ECE-228/assets/51058745/4c6251a2-d403-44db-9d96-a43fd6b271ca)

<h5>We look at distribution of codewords in terms of normal distribution. We see that as the distance increases the distribution significantly decreases as it's because of the quantisation loss.</h5>
<h3>Quantisation and reconstruction loss for 2 codewords</h3>

![2 codewords](https://github.com/vamsin07/ECE-228/assets/51058745/ca951641-3fbd-4e16-9bb8-68340c47bd53)

<h5>Reconstruction and Quantisation loss for 2 codewords</h5>
<h3>Quantisation and reconstruction loss for 4 codewords</h3>

![4 codewords](https://github.com/vamsin07/ECE-228/assets/51058745/938c2ebd-1f94-4187-b9c0-2d8cd3ab98fc)
<h5>Reconstruction and Quantisation loss for 2 codewords</h5>

<h3>Quantisation and reconstruction loss for Transformer with the Vector Quantizer layer</h3>
![Transformer+VQVAE](https://github.com/vamsin07/ECE-228/assets/51058745/a9bfe7f5-40c0-4d78-945c-519288098825)


# Contributing
We welcome contributions! Follow these steps to contribute:

1. Fork the repository
2. Create a new branch (git checkout -b feature/your-feature)
3. Commit your changes (git commit -m 'Add your feature')
4. Push to the branch (git push origin feature/your-feature)
5. Create a new Pull Request

# Acknowledgements
 
Inspired by the works of Aaron van den Oord et al., and Ashish Vaswani et al.

# License

This project is licensed under the MIT License.



