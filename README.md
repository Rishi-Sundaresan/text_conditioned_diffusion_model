# Training a Text-Conditioned Diffusion Model
This repo contains training + eval code for a text-conditioned diffusion model, evaluated on the MNIST and Quickdraw datasets.
Some of the code in this repo was taken from [this blog](https://towardsdatascience.com/diffusion-model-from-scratch-in-pytorch-ddpm-9d9760528946/)

## Architecture
We utilize the UNET architecture from the original [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239), and add text conditioning by adding pre-trained CLIP encodings of text input to the sinusoidal positional embeddings in diffusion training. Both of those are added to the initial image before passing it into the UNET: The model outputs the predicted noise for a specified t. 

<img width="907" height="193" alt="image" src="https://github.com/user-attachments/assets/d237d31b-9f09-406a-8913-f6f49a8233a4" />

## Diffusion Training
Our model is trained with 1000 diffusion timesteps. Note that increasing this value does not directly slow down training, since we calculate directly the noise added at a given time t in training, rather than iteratively playing it forward.

Our loss function is MSE loss between original noise and predicted noise at given time t. Gradient is accumulated and provided per batch. We use the following hyperparameters
1) a batch size of 642)
2) adam optimizer with an initial learning rate of 2e-6. 
3) 250 epochs

## 
