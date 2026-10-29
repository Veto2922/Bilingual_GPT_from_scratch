

import torch 
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_k
        self.qkv_bias = config.qkv_bias

        self.W_q = nn.Linear(self.d_model, self.d_model, bias=self.qkv_bias)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=self.qkv_bias)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=self.qkv_bias)

        self.W_o = nn.Linear(self.d_model, self.d_model, bias=True)

        mask = torch.tril(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)
        )
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x , kv_cache = None , use_cache = False ):

        

        batch_size, seq_len, d_model = x.shape

        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        past_len = 0
        
        if kv_cache is not None and use_cache:
            past_k, past_v = kv_cache
            past_len = past_k.size(2)

            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)  

        updated_cache = (k,v) if use_cache else None
        total_len = past_len + seq_len

        scores = q @ k.transpose(-2, -1)
        scores = scores / self.d_k**0.5

        causal_mask = self.mask[past_len:total_len, :total_len]
        causal_mask = causal_mask.to(q.device)
        
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)

        out = weights @ v

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.W_o(out), updated_cache

        
        