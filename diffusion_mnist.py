# Imports
from unet import UNET
from text_conditioned_unet import TextConditionedUNET, TextEmbeddings
from utils import display_reverse

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List
import random
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
from timm.utils import ModelEmaV3
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np

# CONFIG
DEVICE_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE_", DEVICE_)
TEXT_CONDITIONED_ = True


class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int=1000):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False).to(DEVICE_)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False).to(DEVICE_)
    
    def forward(self, t):
        return self.beta[t], self.alpha[t]

def set_seed(seed: int=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def train(batch_size: int=64,
          num_time_steps: int=1000,
          num_epochs: int=15,
          seed: int=-1,
          ema_decay: float=0.9999,  
          lr=2e-5,
          text_conditioned=False,
          checkpoint_path: str=None):

    set_seed(random.randint(0, 2**32-1) if seed == -1 else seed)
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False, drop_last=True, num_workers=4)

    scheduler = DDPM_Scheduler(num_time_steps)
    model = TextConditionedUNET().to(DEVICE_) if text_conditioned else UNET().to(DEVICE_)
    optimizer = optim.Adam(model.parameters(), lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = nn.MSELoss(reduction='mean')

    # Precompute all Text Embeddings.
    print("Precomputing all Text Embeddings.")
    text_embedder = TextEmbeddings().to(DEVICE_)
    all_labels = [f"Drawing of the number {i}" for i in train_loader.dataset.targets.tolist()]
    all_label_embeddings = torch.Tensor([]).to(DEVICE_)
    for i in range(0, len(all_labels), len(all_labels)//5):
        labels = all_labels[i:i+len(all_labels)//5]
        embeddings = text_embedder(labels)
        all_label_embeddings = torch.cat([all_label_embeddings, embeddings], dim = 0)
    print("Done Precomputing all Text Embeddings.")

    for i in range(num_epochs):
        total_loss = 0
        for bidx, (x, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}")):
            x = x.to(DEVICE_)
            x = F.pad(x, (2,2,2,2)) # 32 by 32
            t = torch.randint(0, num_time_steps, (batch_size,))
            e = torch.randn_like(x, requires_grad=False) # the noise you are adding.
            a = scheduler.alpha[t].view(batch_size, 1, 1, 1).to(DEVICE_)
            x = torch.sqrt(a)*x + torch.sqrt(1-a)*e # Var is 1-a, stdev is sqrt of that, accumulated steps.
            # model predicts the noise e that was added (and accumulated to t)
            if text_conditioned:
                output = model(x, t, all_label_embeddings[bidx*batch_size:bidx*batch_size + batch_size])
            else:
                output = model(x,t) 
            optimizer.zero_grad()
            loss = criterion(output, e) # Main Loss
            total_loss += loss.item()

            # Compute gradient and update weights
            loss.backward()
            optimizer.step()
            ema.update(model)
        print(f'Epoch {i+1} | Avg Batch Loss {total_loss / (60000/batch_size):.5f}')

        checkpoint = {
            'weights': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'ema': ema.state_dict()
        }   
        torch.save(checkpoint, 'checkpoints/ddpm_checkpoint')
        with open("checkpoints/checkpoint.txt", "w") as f:
            f.write(f"Epoch for checkpoint: {i}\n")

        if i % 10 == 0:
            inference('checkpoints/ddpm_checkpoint', num_images=3) # store examples.




def inference(checkpoint_path: str=None,
              num_time_steps: int=1000,
              ema_decay: float=0.9999,
              num_images: int = 10,
              text_conditioned = True):


    print("Loading Checkpoint Info...")
    checkpoint = torch.load(checkpoint_path)
    model = TextConditionedUNET().to(DEVICE_) if text_conditioned else UNET().to(DEVICE_)
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPM_Scheduler(num_time_steps).to(DEVICE_)
    text_embedder = TextEmbeddings().to(DEVICE_)
    times = [0,15,50,100,200,300,400,550,700,999]
    images = []

    with torch.no_grad():
        model = ema.module.eval()
        print("Running Inference...")
        for i in range(num_images): # 10 images.
            text = f"Drawing of the number {i % 10}"
            embeddings = text_embedder([text])
            images.append([]) 
            z = torch.randn(1, 1, 32, 32).to(DEVICE_) # Start from noise, we will create image.

            for t in reversed(range(1, num_time_steps)):
                t = [t]
                temp = (scheduler.beta[t]/( (torch.sqrt(1-scheduler.alpha[t]))*(torch.sqrt(1-scheduler.beta[t])) ))
                model_output = model(z,t,embeddings) if text_conditioned else model(z,t)
                z = (1/(torch.sqrt(1-scheduler.beta[t])))*z - (temp*model_output) # weighted sum of prev z and model output.
                if t[0] in times:
                    images[-1].append(z)
                # Add some noise for stability
                e = torch.randn(1,1,32,32).to(DEVICE_)
                z = z + (e*torch.sqrt(scheduler.beta[t]))
            
            # Last step, same as before but don't add noise at end.
            temp = scheduler.beta[0]/( (torch.sqrt(1-scheduler.alpha[0]))*(torch.sqrt(1-scheduler.beta[0]))) 
            model_output = model(z,[0], embeddings) if text_conditioned else model(z,[0])
            x = (1/(torch.sqrt(1-scheduler.beta[0])))*z - (temp*model_output)

            images[-1].append(x)
            x = rearrange(x.squeeze(0), 'c h w -> h w c').detach()
            display_reverse(images, save_path="images/example.png")
            

def main():
    #train(lr=2e-6, num_epochs=100, text_conditioned =TEXT_CONDITIONED_)
    inference('checkpoints/ddpm_checkpoint', num_images=10, text_conditioned =TEXT_CONDITIONED_)

if __name__ == '__main__':
    main()