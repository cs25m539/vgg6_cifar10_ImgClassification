"""Random seed and deterministic backend configuration."""
import logging
import os
import random
import numpy as np
import torch

logger = logging.getLogger("vgg6.reproducibility")

def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Deterministic mode improves repeatability but may reduce performance or raise
    errors for operations without deterministic implementations.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except TypeError:  # compatibility with older supported PyTorch builds
        torch.use_deterministic_algorithms(deterministic)
    logger.info("REPRODUCIBILITY CONFIGURED | seed=%d deterministic=%s cuda_available=%s", seed, deterministic, torch.cuda.is_available())
