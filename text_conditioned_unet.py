# Imports
import torch
import torch.nn as nn
from transformers import CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPTextModel
from unet import UNET
from typing import List


# Clip Embeddings.
class TextEmbeddings(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name)

        ## Freeze Text Embeddings
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        self.embedding_dim = self.text_model.config.hidden_size

    def forward(self, text_prompts: List[str]) -> torch.Tensor: # [B, D]
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]

        tokens = self.tokenizer(
            text_prompts,
            padding="max_length",  # pad to max_length for *all batches*
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(self.text_model.device)

        self.text_model.eval()
        with torch.no_grad():
            outputs = self.text_model(**tokens)
            return outputs.pooler_output


class TextConditionedUNET(UNET):
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
        self.text_embeddings = TextEmbeddings()
    
    def forward(self, x , t, text_embeddings):
        x = self.shallow_conv(x)
        residuals = []
        # Downwards part of U
        for i in range(self.num_layers//2):
            layer = getattr(self, f'Layer{i+1}')
            positional_embeddings = self.positional_embeddings(x, t)

            embeddings = positional_embeddings + text_embeddings[:, :, None, None]
            x, r = layer(x, embeddings)
            residuals.append(r)
        # Upwards Part of U
        for i in range(self.num_layers//2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            x = torch.concat((layer(x, embeddings)[0], residuals[self.num_layers - i - 1]), dim=1)
        return self.output_conv(self.relu(self.late_conv(x)))