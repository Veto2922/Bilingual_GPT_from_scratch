import torch
import torch.nn as nn
import torch.nn.functional as F

import math 


from src.llm.model.GPT_block import GPTBlock
from src.llm.model.embedding import InputEmbeddings

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.w_embeddings = InputEmbeddings(config)

        self.blocks = nn.ModuleList(
                GPTBlock(config) for _ in range(config.n_layer) 
        )

        self.ln_final = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.w_embeddings.token_embedding.weight

        self.apply(self._init_weights)
        self._scale_residual_projections()

    

    

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor | None = None,
        kv_cache: list | None = None,
        use_cache: bool = True,
    ):
        past_length = kv_cache[0][0].shape[2] if kv_cache is not None else 0

        x = self.w_embeddings(input_ids, position_offsets=past_length)

        updated_caches = [] if use_cache else None
        for ix, block in enumerate(self.blocks):
            current_cache = kv_cache[ix] if kv_cache is not None else None
            x, updated_cache = block(x, use_cache=use_cache, kv_cache=current_cache)
            if use_cache:
                updated_caches.append(updated_cache)
        
        x = self.ln_final(x)

        logits = self.lm_head(x)
        
        if target_ids is None:
            return logits, None, updated_caches
        
        loss = F.cross_entropy(
            logits.reshape(
                -1, logits.size(-1)
            ), 
            target_ids.reshape(-1),
            ignore_index=self.config.pad_id,
        )

        return logits,  loss , updated_caches

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2's initialisation: normal(0, 0.02), zeroed biases.

        PyTorch's Linear default depends on fan-in, which leaves a 768-wide
        model's activations larger than GPT-2 assumes and makes the early
        steps of training less stable.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        """Scale weights of residual projections by 1 / sqrt(2 * n_layer).

        Follows GPT-2's initialization scheme to prevent activation variance
        from growing with model depth.
        """
        scale = math.sqrt(2 * self.config.n_layer)
        for block in self.blocks:
            with torch.no_grad():
                block.mha.W_o.weight.div_(scale)
                block.ff.fc2.weight.div_(scale)

    def num_parameters(self, non_embedding: bool = False) -> int:

        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n -= self.w_embeddings.token_embedding.weight.numel()
            n -= self.w_embeddings.position_embedding.weight.numel()

        return n
    
    def config_str(self) -> str:
        return f"d_model={self.config.d_model} d_ff={self.config.d_ff} n_heads={self.config.n_heads} n_layers={self.config.n_layer} vocab_size={self.config.vocab_size} max_seq_len={self.config.max_seq_len}"

