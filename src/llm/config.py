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
