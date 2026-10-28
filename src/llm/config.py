from dataclasses import dataclass


@dataclass
class Modelconfig:
    vocab_size: int = 32_000
    d_model: int = 768  # embedding dim

    max_seq_len: int = 1024

    dropout: float = 0.1
