
from einops import rearrange
from typing import List
import matplotlib.pyplot as plt
from textwrap import wrap
import numpy as np

def display_reverse(raw_images: List[List], texts: List[str], save_path: str = "images/example.png"):

    images = []
    for i in range(len(raw_images)):
        images.append([texts[i]] + raw_images[i])

    fig, axes = plt.subplots(len(images), len(images[0]), figsize=(len(images[0]),len(images)))
    flattened_image_list = [img for imgs in images for img in imgs]
    for i, ax in enumerate(axes.flat):
        if isinstance(flattened_image_list[i], str):
            # Display text instead of image
            ax.imshow(np.zeros((32, 32, 1)), cmap='gray')  # tiny white image as background
            wrapped_text = "\n".join(wrap(flattened_image_list[i], width=30))  # adjust width as needed
            ax.text(0.5, 0.5, wrapped_text, color='white', fontsize=4,
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue
        x = flattened_image_list[i].squeeze(0)
        x = rearrange(x, 'c h w -> h w c')
        x = x.cpu().numpy()
        ax.imshow(x.squeeze(-1), cmap='gray')
        ax.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def display_losses(losses_list: List[List], losses_names: List[str], save_path: str = "images/example.png"):
    epochs = list(range(1, len(losses_list[0]) + 1))  # use length of the list, not num_epochs

    plt.figure(figsize=(10,6))
    for i, loss in enumerate(losses_list):
        plt.plot(epochs, loss, label=losses_names[i])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, None)
    plt.tight_layout()
    plt.savefig(save_path)