
from einops import rearrange
from typing import List
import matplotlib.pyplot as plt

def display_reverse(images: List[List], save_path: str = "images/example.png"):
    fig, axes = plt.subplots(len(images), len(images[0]), figsize=(len(images[0]),len(images)))
    flattened_image_list = [img for imgs in images for img in imgs]
    for i, ax in enumerate(axes.flat):
        x = flattened_image_list[i].squeeze(0)
        x = rearrange(x, 'c h w -> h w c')
        x = x.cpu().numpy()
        ax.imshow(x.squeeze(-1), cmap='gray')
        ax.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()