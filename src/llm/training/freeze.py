import torch.nn as nn


def freeze_all(model: nn.Module) -> None:
    """Freeze all model parameters."""
    for parameter in model.parameters():
        parameter.requires_grad = False


def freeze_embeddings(model: nn.Module) -> None:
    """Freeze input embeddings."""
    if hasattr(model, "w_embeddings"):
        for parameter in model.w_embeddings.parameters():
            parameter.requires_grad = False


def freeze_all_except_lora(model: nn.Module) -> None:
    """Freeze all model parameters except LoRA adapter weights."""
    for name, parameter in model.named_parameters():
        if "lora" in name.lower():
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False


def unfreeze_tail(model: nn.Module, n_blocks: int = 2, unfreeze_lm_head: bool = True) -> None:
    """Unfreeze the last n transformer blocks, final layer norm, and optionally lm_head."""
    if hasattr(model, "blocks"):
        for block in model.blocks[-n_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    if hasattr(model, "ln_final"):
        for parameter in model.ln_final.parameters():
            parameter.requires_grad = True

    if unfreeze_lm_head and hasattr(model, "lm_head"):
        for parameter in model.lm_head.parameters():
            parameter.requires_grad = True
