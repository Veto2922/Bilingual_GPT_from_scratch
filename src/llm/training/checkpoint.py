from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.llm.config import Modelconfig


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    step: int = 0,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Save model checkpoint including weights, configuration, and training state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config_obj = model.config() if callable(getattr(model, "config", None)) else model.config
    payload = {
        "model": model.state_dict(),
        "model_config": asdict(config_obj),
        "step": step,
        "metrics": metrics or {},
    }

    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()

    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()

    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    map_location: str = "cpu",
) -> int:
    """Load model checkpoint and restore state."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    saved_config = Modelconfig(**checkpoint["model_config"])

    if saved_config != model.config:
        raise ValueError(
            "Config mismatch!\n"
            f"Loaded config: {saved_config}\n"
            f"Current config: {model.config}"
        )

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint["step"]


def model_config_from_checkpoint(path: str | Path, map_location: str = "cpu") -> Modelconfig:
    """Extract Modelconfig from a saved checkpoint."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    return Modelconfig(**checkpoint["model_config"])
