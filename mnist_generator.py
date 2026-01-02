import torch
from torchvision import datasets, transforms
from diffusion import train, inference

TEXT_CONDITIONED_ = True

def main():
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    train(batch_size=64, lr=2e-6, num_epochs=100, text_conditioned =TEXT_CONDITIONED_, train_dataset=train_dataset)
    inference('checkpoints/mnist/ddpm_checkpoint', num_images=10, text_conditioned =TEXT_CONDITIONED_)

if __name__ == '__main__':
    main()