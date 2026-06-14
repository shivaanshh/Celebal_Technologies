# MNIST Denoising Autoencoder

A deep learning model that removes noise from handwritten-digit images using autoencoders, built in PyTorch on the MNIST dataset.

## Week 6 Assessment - Project 

## Overview
The model is trained on noisy-to-clean image pairs using Gaussian noise and learns to reconstruct clean MNIST digits through an information bottleneck.

### Models Implemented
- FFNN Autoencoder
- Transpose CNN Autoencoder
- Upsample CNN Autoencoder

### Dataset
- MNIST (48k train / 12k validation / 10k test)
- Gaussian Noise Factor = 0.5
- Loss: MSE
- Optimizer: Adam
- Epochs: 20

### Results
Convolutional autoencoders produce cleaner and more legible reconstructions than the FFNN baseline and improve PSNR over noisy inputs.
