# Imports
from unet import UNET
from text_conditioned_unet import TextConditionedUNET, TextEmbeddings
from utils import display_reverse, display_losses

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

DEVICE_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXT_EMBEDDING_NUM_CHUNKS_ = 10
EARLY_TIMESTEP_THRESHOLD_ = 100
print("Using Device: ", DEVICE_)


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


def precompute_text_embeddings(train_dataset, label_name_lookup, batch_size) -> torch.Tensor:
    print("Precomputing all Text Embeddings.")
    text_embedder = TextEmbeddings().to(DEVICE_)
    text_embedder.eval()


    all_label_raw = []
    if "label" in train_dataset.features:
        all_label_raw = train_dataset['label']
    else:
        all_label_raw = train_dataset.targets.tolist()

    if label_name_lookup:
        all_label_raw = [label_name_lookup[int(i)] for i in all_label_raw]

    all_labels = [f"Drawing of {i}" for i in all_label_raw]
    all_label_embeddings = torch.empty(0, device=DEVICE_)

    chunk_size = batch_size

    for i in tqdm(
        range(0, len(all_labels), chunk_size),
        desc="Text embeddings",
        unit="chunk"
    ):
        labels = all_labels[i:i + chunk_size]
        embeddings = text_embedder(labels)
        all_label_embeddings = torch.cat([all_label_embeddings, embeddings], dim=0)
    return all_label_embeddings


def train(batch_size: int=64,
          num_time_steps: int=1000,
          num_epochs: int=15,
          seed: int=-1,
          ema_decay: float=0.9999,  
          lr=2e-5,
          text_conditioned=False,
          checkpoint_path: str=None,
          train_dataset = None,
          label_name_lookup = None):
    set_seed(random.randint(0, 2**32-1) if seed == -1 else seed)
    
    # Precompute all Text Embeddings.
    all_label_embeddings = precompute_text_embeddings(train_dataset, label_name_lookup, batch_size)
    
    def attach_embedding(example, idx):
        example["text_embedding"] = all_label_embeddings[idx]
        return example

    train_dataset = train_dataset.map(attach_embedding, with_indices=True)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True, num_workers=4)

    scheduler = DDPM_Scheduler(num_time_steps)
    model = TextConditionedUNET().to(DEVICE_) if text_conditioned else UNET().to(DEVICE_)
    optimizer = optim.Adam(model.parameters(), lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = nn.MSELoss(reduction='none')


    avg_epoch_losses = []
    avg_early_timestep_epoch_losses = []

    for i in range(num_epochs):
        all_loss = []
        all_early_timestep_loss = []
        for bidx, data in enumerate(tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}")):
            x = data['image']
            x = x.to(DEVICE_)
            x = F.pad(x, (2,2,2,2)) # 32 by 32
            t = torch.randint(0, num_time_steps, (batch_size,))
            e = torch.randn_like(x, requires_grad=False) # the noise you are adding.
            a = scheduler.alpha[t].view(batch_size, 1, 1, 1).to(DEVICE_)
            x = torch.sqrt(a)*x + torch.sqrt(1-a)*e # Var is 1-a, stdev is sqrt of that, accumulated steps.
            # model predicts the noise e that was added (and accumulated to t)
            if text_conditioned:
                output = model(x, t, data['text_embedding'].to(DEVICE_))
            else:
                output = model(x,t) 
            optimizer.zero_grad()
            loss_tensor = criterion(output, e) # Main Loss
            loss_per_sample = loss_tensor.mean(dim=(1, 2, 3)) 
            loss = loss_per_sample.mean()
            all_loss.append(loss)

            # Compute gradient and update weights
            loss.backward()
            optimizer.step()
            ema.update(model)

            # Accumulate early timesteps loss for logging.
            mask = t <= EARLY_TIMESTEP_THRESHOLD_
            if mask.any():
                all_early_timestep_loss.append(loss_per_sample[mask].mean())

        avg_batch_loss = torch.stack(all_loss).mean().item()
        avg_early_timestep_loss = torch.stack(all_early_timestep_loss).mean().item()
        avg_epoch_losses.append(avg_batch_loss)
        avg_early_timestep_epoch_losses.append(avg_early_timestep_loss)
        print(f'Epoch {i+1} | Avg Batch Loss {avg_batch_loss:.5f}\
         | T < {EARLY_TIMESTEP_THRESHOLD_} Loss {avg_early_timestep_loss:.5f}')

        checkpoint = {
            'weights': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'ema': ema.state_dict()
        }   
        torch.save(checkpoint, 'checkpoints/ddpm_checkpoint')
        with open("checkpoints/checkpoint.txt", "w") as f:
            f.write(f"Epoch for checkpoint: {i}\n")

        if (i+1) % 10 == 0:
            sample_texts = ["Drawing of apple", "Drawing of banana", "Drawing of strawberry"]
            inference('checkpoints/ddpm_checkpoint', text_conditioned=text_conditioned, texts=sample_texts) # store examples.
        
        if (i+1) % 5 == 0:
            display_losses(losses_list=[avg_epoch_losses, avg_early_timestep_epoch_losses], losses_names=["Avg loss", f"Avg t < {EARLY_TIMESTEP_THRESHOLD_} loss"], save_path = "images/quickdraw/losses.png")

        




def inference(checkpoint_path: str=None,
              num_time_steps: int=1000,
              ema_decay: float=0.9999,
              text_conditioned = True,
              texts = []):

    num_images = len(texts)

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
            embeddings = text_embedder([texts[i]])
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
            display_reverse(images, texts, save_path="images/quickdraw/example.png")