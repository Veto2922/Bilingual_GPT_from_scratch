from dataclasses import dataclass


@dataclass
class Modelconfig:
    vocab_size: int = 32_000
    d_model: int = 768  # embedding dim
    d_ff: int = 4 * d_model  # feedforward layer dim
    max_seq_len: int = 1024

    dropout: float = 0.1

    n_heads: int = 12

    qkv_bias: bool = False

    @property
    def d_k(self):
        return self.d_model // self.n_heads
