"""CIFAR-10 transforms, deterministic split, and data loaders."""
import logging
import random
import time
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

logger = logging.getLogger("vgg6.data")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

class Cutout:
    """Mask one random square after normalization, matching the reference notebook."""
    def __init__(self, length: int = 16):
        self.length = length

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        cy, cx = random.randrange(height), random.randrange(width)
        half = self.length // 2
        y1, y2 = max(0, cy-half), min(height, cy+half)
        x1, x2 = max(0, cx-half), min(width, cx+half)
        image[:, y1:y2, x1:x2] = 0
        return image

def build_transforms():
    """Return separate training and evaluation pipelines."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        Cutout(length=16),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_transform, eval_transform

def build_loaders(data_dir: str, batch_size: int, seed: int, num_workers: int = 2):
    """Build 45k/5k train/validation split and official 10k test loader.

    Independent dataset instances prevent stochastic training augmentation from
    affecting validation images.
    """
    started = time.time()
    logger.info("DATASET STAGE START | data_dir=%s batch_size=%d seed=%d workers=%d", data_dir, batch_size, seed, num_workers)
    train_tf, eval_tf = build_transforms()
    train_full = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    eval_full = datasets.CIFAR10(data_dir, train=True, download=True, transform=eval_tf)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_tf)
    logger.info("DATASETS READY | train_pool=%d validation_pool=%d test=%d", len(train_full), len(eval_full), len(test_set))

    split_generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_full), generator=split_generator).tolist()
    train_set = Subset(train_full, indices[:45000])
    val_set = Subset(eval_full, indices[45000:])

    common = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), **common
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **common)
    logger.info("DATASET STAGE COMPLETE | train_examples=%d validation_examples=%d test_examples=%d train_batches=%d validation_batches=%d test_batches=%d elapsed_sec=%.2f", len(train_set), len(val_set), len(test_set), len(train_loader), len(val_loader), len(test_loader), time.time()-started)
    return train_loader, val_loader, test_loader
