from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2


def _transform(image_size, training):
    spatial = (
        v2.RandomResizedCrop((image_size, image_size), scale=(0.7, 1.0))
        if training
        else v2.Compose([
            v2.Resize(image_size + 32),
            v2.CenterCrop((image_size, image_size)),
        ])
    )
    return v2.Compose([
        spatial,
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])


def build_dataset(root, image_size=256, training=False):
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {root}. "
            "Use an ImageFolder layout with one subdirectory per class."
        )
    return ImageFolder(root, transform=_transform(image_size, training))


def build_dataloaders(
    train_root,
    valid_root,
    image_size=256,
    train_batch_size=16,
    valid_batch_size=16,
    num_workers=4,
):
    train_dataset = build_dataset(train_root, image_size, training=True)
    valid_dataset = build_dataset(valid_root, image_size, training=False)
    worker_options = {}
    if num_workers > 0:
        worker_options.update(prefetch_factor=2, persistent_workers=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **worker_options,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **worker_options,
    )
    return train_loader, valid_loader
