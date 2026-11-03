from dataclasses import dataclass
from datetime import datetime
import math
import random


@dataclass
class Modelconfig:
    """Configuration for GPT model architecture."""

    vocab_size: int = 32_000
    d_model: int = 768  # embedding dim
    d_ff: int | None = None  # feedforward layer dim
    max_seq_len: int = 1024

    dropout: float = 0.1

    n_heads: int = 12
    n_layer: int = 12

    qkv_bias: bool = False
    pad_id: int = 0
    tie_weights: bool = True

    def __init__(
        self,
        vocab_size: int = 32_000,
        d_model: int = 768,
        d_ff: int | None = None,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        n_heads: int = 12,
        n_layer: int = 12,
        qkv_bias: bool = False,
        pad_id: int = 0,
        tie_weights: bool = True,
        n_layers: int | None = None,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.n_heads = n_heads
        self.n_layer = n_layers if n_layers is not None else n_layer
        self.qkv_bias = qkv_bias
        self.pad_id = pad_id
        self.tie_weights = tie_weights

    @property
    def d_k(self) -> int:
        """Attention head dimension."""
        return self.d_model // self.n_heads

    @property
    def n_layers(self) -> int:
        """Alias for n_layer."""
        return self.n_layer

    @classmethod
    def main_model(cls) -> "Modelconfig":
        """Default pretraining architecture (512-dim, 8 heads, 8 layers)."""
        return cls(d_model=512, n_heads=8, n_layer=8, dropout=0.05)

    @classmethod
    def tiny(cls) -> "Modelconfig":
        """Small test architecture (128-dim, 4 heads, 2 layers)."""
        return cls(d_model=128, n_heads=4, n_layer=2, max_seq_len=256)

    @classmethod
    def GPT_Small(cls) -> "Modelconfig":
        """Backward-compatible alias for main_model."""
        return cls.main_model()


# Alias for naming consistency
ModelConfig = Modelconfig


@dataclass
class GenerationConfig:
    """How to turn logits into text.

    Sampling is where a model stops being deterministic. Every field here
    trades coherence against variety, and there is no universally right
    setting.
    """

    max_new_tokens: int = 100
    temperature: float = 0.8

    top_k: int | None = 50
    top_p: float | None = 0.95

    use_cache: bool = True
    seed: int | None = None

    @classmethod
    def greedy(cls) -> "GenerationConfig":
        """Always take the likeliest token. Deterministic, and repetitive."""
        return cls(temperature=0.0, top_k=None, top_p=None)


@dataclass
class TrainingConfig:
    """How to train, kept separate from what to train.

    ModelConfig describes the model and has to match a checkpoint exactly.
    This does not: the same weights can be trained again at a different
    learning rate on a different corpus.
    Keeping them apart is what makes continued pretraining a config change
    rather than a code change.
    """

    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    # Schedule
    max_steps: int = 10_000
    warmup_steps: int = 500

    min_lr_ratio: float = 0.1

    batch_size: int = 24
    accumulation_steps: int = 1

    log_every: int = 10
    print_every: int = 0
    eval_every: int = 500
    eval_batches: int = 50

    save_every: int = 1_000

    project: str = "Bilingual_GPT_V1"
    run_name: str = "train-phase"

    checkpoint_dir: str = "checkpoints"
    seed: int = 0

    sample_every: int = 500
    sample_chat: bool = False  # wrap sample prompts in the chat template (finetuning)
    sample_max_new_tokens: int = 60
    sample_prompts: tuple[str, ...] = (
        "وُلد في مدينة بغداد عام",  # biography
        "The early life of",  # biography
        "تقع هذه المدينة في شمال",  # geography
        "This city is located on the eastern coast of",  # geography
        "شهدت الحرب العالمية الثانية",  # history
        "During the nineteenth century,",  # history
        "يُعرف هذا الفنان بأعماله في مجال",  # arts
        "The film was directed by",  # arts
        "فاز الفريق بالبطولة بعد",  # sports
        "The scientific study found that",  # science
    )


def unique_run_name(base: str) -> str:
    """Generate a unique run identifier using random seed and timestamp."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"run-{random.randint(0, 9999):04d}-{base}-{stamp}"


def steps_for_epochs(
    num_examples: int, batch_size: int, accumulation_steps: int, epochs: float | int
) -> int:
    """Compute total optimizer steps required for a given number of epochs."""
    effective_batch = batch_size * accumulation_steps
    steps_per_epoch = math.ceil(num_examples / effective_batch)
    return max(1, math.ceil(epochs * steps_per_epoch))
