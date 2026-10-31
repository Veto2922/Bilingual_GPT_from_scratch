import torch 
import torch.nn as nn

from src.llm.model.attention import MultiHeadAttention
from src.llm.model.feedforward import FeedForward

class GPTBlock(nn.Module):
    def __init__(self , config):
        super().__init__()


        self.mha = MultiHeadAttention(config)
        self.ff = FeedForward(config)

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self , x , kv_cache = None , use_cache = False):
        
        att_out , updated_kv_cache = self.mha(self.norm1(x), kv_cache=kv_cache, use_cache=use_cache)
        x = x + self.dropout(att_out)

        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)

        return x , updated_kv_cache