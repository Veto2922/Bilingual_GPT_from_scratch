import torch
import torch.nn as nn
from typing import Dict


class InputEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.max_seq_len = config.max_seq_len
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model

        self.dropout = nn.Dropout(config.dropout)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.d_model)

    def forward(self, input_ids, position_offsets=0):
        _, seq_len = input_ids.shape

        if position_offsets + seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {position_offsets + seq_len} exceeds maximum sequence length {self.max_seq_len}"
            )

        positions = torch.arange(
            position_offsets, position_offsets + seq_len, device=input_ids.device
        )

        embeddings = self.token_embedding(input_ids) + self.position_embedding(
            positions
        )
        return self.dropout(embeddings)  # (Batch, Seq_len, D_model)
