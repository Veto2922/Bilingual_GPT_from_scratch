from dataclasses import dataclass


@dataclass
class Modelconfig:
    vocab_size: int = 32_000
    d_model: int = 768  # embedding dim
    d_ff: int = 4 * d_model  # feedforward layer dim
    max_seq_len: int = 1024

    dropout: float = 0.1

    n_heads: int = 12
    n_layer: int = 12

    qkv_bias: bool = False

    pad_id: int = 0

    tie_weights: bool = True

    @property
    def d_k(self):
        return self.d_model // self.n_heads

    @classmethod
    def GPT_Small(cls) -> "Modelconfig":
        return cls(
            vocab_size=32_000,
            d_model=512,
            d_ff=4 * 512,
            max_seq_len=1024,
            dropout=0.1,
            n_heads=8,
            n_layer=8,
            qkv_bias=False,
            pad_id=0,
            tie_weights=True,
        )


@dataclass
class GenerationConfig:
    """How to turn logits into text.

    Sampling is where a model stops being deterministic. Every field here
    trades coherence against variety, and there is no universally right
    setting -- ch11 exists to give a feel for the trade.
    """

    max_new_tokens: int = 100
    temperature: float = 0.8

    top_k: int | None = 50
    top_p: float | None = 0.95

    use_cache: bool = True
    seed: int | None = None

    @classmethod
    def greedy(cls):
        """Always take the likeliest token. Deterministic, and repetitive."""
        return cls(temperature=0.0, top_k=None, top_p=None)


@dataclass
class TrainingConfig:
    """How to train, kept separate from what to train.

    ModelConfig describes the model and has to match a checkpoint exactly.
    This does not: the same weights can be trained again at a different
    learning rate on a different corpus, which is precisely what ch10 does.
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

    project: str = "jabarti-llm"
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
