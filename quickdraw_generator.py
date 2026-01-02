import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset, Value, Dataset
from diffusion import train, inference
import numpy as np

# =========================
# Configuration
# =========================

TEXT_CONDITIONED_ = True

# Food-related categories you want
SELECTED_CATEGORIES = [
    "apple",
    "banana",
    "blackberry",
    "blueberry",
    "grapes",
    "pear",
    "pineapple",
    "strawberry",
    "watermelon"
]


def main():
    # =========================
    # Load dataset + metadata
    # =========================

    # Load Hugging Face dataset
    dataset = load_dataset("Xenova/quickdraw-small")

    train_ds = dataset["train"]
    train_ds = train_ds.with_format("torch")
    label_names = train_ds.features["label"].names

    label_number_to_name = {name:i for i, name in enumerate(label_names)}
    selected_category_numbers = set(label_number_to_name[name] for name in SELECTED_CATEGORIES)
    


    def filter_fn(example):
        return int(example["label"]) in selected_category_numbers

    def preprocess(example):
        example["image"] = example['image'] / 255.0
        return example


    filtered_train_ds = (
        train_ds
        .filter(filter_fn, num_proc=8)
        .map(preprocess, num_proc=4)

    )

    train(batch_size=64, lr=2e-6, num_epochs=250, text_conditioned =TEXT_CONDITIONED_, train_dataset=filtered_train_ds, label_name_lookup=label_names)

    inference('checkpoints/ddpm_checkpoint', text_conditioned =TEXT_CONDITIONED_, texts=[f"Drawing of {fruit}" for fruit in SELECTED_CATEGORIES])

if __name__ == '__main__':
    main()