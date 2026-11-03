"""LLM package exposing model, data loading, tokenizer, and training components."""

from src.llm.config import GenerationConfig, Modelconfig, TrainingConfig
from src.llm.data_loading.collate import collate_fn
from src.llm.data_loading.dataset import PackedDataset
from src.llm.model.GPT_model import GPTModel
from src.llm.tokenizer import Tokenizer
from src.llm.training.checkpoint import load_checkpoint, save_checkpoint
from src.llm.training.tracking import Tracker
from src.llm.training.trainer import Trainer

# Backward compatibility / aliases
GPT = GPTModel
ModelConfig = Modelconfig

__version__ = "0.1.0"

__all__ = [
    "GPT",
    "GPTModel",
    "GenerationConfig",
    "ModelConfig",
    "Modelconfig",
    "PackedDataset",
    "Tokenizer",
    "Tracker",
    "Trainer",
    "TrainingConfig",
    "collate_fn",
    "load_checkpoint",
    "save_checkpoint",
]
