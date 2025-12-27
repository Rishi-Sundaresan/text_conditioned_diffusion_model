# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List
import math


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps:int, embed_dim: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float() #(time_steps, 1)
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings
    
    def forward(self, x: torch.Tensor, t) -> torch.Tensor:
        embeds = self.embeddings[t].to(x.device)
        return embeds[:,:,None,None]


class ResBlock(nn.Module):
    def __init__(self, C: int, num_groups: int, dropout_prob: float):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.gnorm1 = nn.GroupNorm(num_groups, C)
        self.gnorm2 = nn.GroupNorm(num_groups, C)
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout_prob, inplace=True) # check if inplace=true is needed.
    
    def forward(self, x: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        x = x + embeddings[:, :x.shape[1], :, :]
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        return r + x

class Attention(nn.Module):
    def __init__(self, C: int, num_heads: int, dropout_prob: float):
        super().__init__()
        self.proj1 = nn.Linear(C, C*3) # weights for q, k, v
        self.proj2 = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, height, width = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj1(x)
        x = rearrange(x, 'b L (c H K) -> K b H L c', K=3, H=self.num_heads) # Create 3 tensors for q, k, v
        q, k, v = x[0], x[1], x[2]
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout_prob, is_causal=False)
        x = rearrange(x, 'b H (h w) c -> b h w (c H)', h=height, w=width)
        x = self.proj2(x)
        return rearrange(x, 'b h w c -> b c h w')

class UNETLayer(nn.Module):
    def __init__(self, upscale: bool, attention: bool, num_groups: int, dropout_prob: float, num_heads: int, C: int):
        super().__init__()
        self.ResBlock1 = ResBlock(C, num_groups, dropout_prob)
        self.ResBlock2 = ResBlock(C, num_groups, dropout_prob)
        if upscale:
            self.conv = nn.ConvTranspose2d(C, C // 2, kernel_size = 4, stride = 2, padding = 1)
        else:
            self.conv = nn.Conv2d(C, C*2, kernel_size=3, stride=2, padding=1)
        
        if attention:
            self.attention_layer = Attention(C, num_heads, dropout_prob)
    
    def forward(self, x: torch.Tensor, embeddings: torch.Tensor):
        x = self.ResBlock1(x, embeddings)
        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)
        x = self.ResBlock2(x, embeddings)
        return self.conv(x), x

class UNET(nn.Module):
    def __init__(self,
            Channels: List = [64, 128, 256, 512, 512, 384],
            Attentions: List = [False, True, False, False, False, True],
            Upscales: List = [False, False, False, True, True, True],
            num_groups: int = 32,
            dropout_prob: float = 0.1,
            num_heads: int = 8,
            input_channels: int = 1,
            output_channels: int = 1,
            time_steps: int = 1000):
        super().__init__()
        self.num_layers = len(Channels)
        self.shallow_conv = nn.Conv2d(input_channels, Channels[0], kernel_size=3, padding=1)
        out_channels = (Channels[-1]//2) + Channels[0]
        self.late_conv = nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(out_channels//2, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.positional_embeddings = SinusoidalEmbeddings(time_steps=time_steps, embed_dim=max(Channels))

        for i in range(self.num_layers):
            layer = UNETLayer(
                upscale=Upscales[i],
                attention=Attentions[i],
                num_groups=num_groups,
                dropout_prob=dropout_prob,
                C=Channels[i],
                num_heads=num_heads
            )
            setattr(self, f'Layer{i+1}', layer)
    
    def forward(self, x , t):
        x = self.shallow_conv(x)
        residuals = []
        # Downwards part of U
        for i in range(self.num_layers//2):
            layer = getattr(self, f'Layer{i+1}')
            positional_embeddings = self.positional_embeddings(x, t)
            x, r = layer(x, positional_embeddings)
            residuals.append(r)
        # Upwards Part of U
        for i in range(self.num_layers//2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            x = torch.concat((layer(x, positional_embeddings)[0], residuals[self.num_layers - i - 1]), dim=1)
        return self.output_conv(self.relu(self.late_conv(x)))
