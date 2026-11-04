import torch
import torch.nn as nn


ATTENTION_PROJECTIONS = ("W_q", "W_k", "W_v", "W_o")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int = 16):
        super().__init__()

        self.base = base

        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.r = r
        self.scale = alpha / r

        self.A = nn.parameter(torch.empty(r, base.in_features))
        self.B = nn.parameter(torch.zeros(base.out_features, r))

        nn.init.kaiming_uniform_(self.A, a=5**0.5)

    def forward(self, x):
        detour = (x @ self.A.T) @ self.B.T

        return self.base(x) + self.scale * detour

    def merged_weight(self):
        return self.base.weight + self.scale * (self.B @ self.A)


def apply_lora(
    model, r: int = 8, alpha: int = 16, target_modules=ATTENTION_PROJECTIONS
):
    """Freeze the whole model, then wrap the named attention projections in
    every block with a LoRALinear.
    """

    for parameter in model.parameters():
        parameter.requires_grad = False

    replaced = 0
    for block in model.blocks:
        for name in target_modules:
            linear = getattr(block.attn, name)
            lora_linear = LoRALinear(base=linear, r=r, alpha=alpha)

            setattr(block.attn, name, lora_linear)
            replaced += 1

    return replaced


def merge_lora(model, target_modules=ATTENTION_PROJECTIONS):
    """Fold every LoRALinear's detour back into an ordinary nn.Linear."""

    merged = 0
    for block in model.blocks:
        for name in target_modules:
            module = getattr(block.attn, name)
            if not isinstance(module, LoRALinear):
                continue

            plain = nn.Linear(
                module.base.in_features,
                module.base.out_features,
                bias=module.base.bias is not None,
            ).to(module.base.weight.device)

            with torch.no_grad():
                plain.weight.copy_(module.merged_weight())
                if module.base.bias is not None:
                    plain.bias.copy_(module.base.bias)

            setattr(block.attn, name, plain)
            merged += 1

    return merged
