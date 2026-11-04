"""
main.py -- interactive text generation from a trained checkpoint

Load a pretrained (or finetuned) GPT checkpoint and generate text from
user prompts in the terminal.

Usage:
    python main.py
    python main.py --checkpoint checkpoints/pretrain_final.pt
    python main.py --max-tokens 200 --temperature 0.6
    python main.py --chat
    python main.py --greedy
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# The project's .venv has the correct library versions (e.g. tokenizers
# 0.23+) but its Python symlink may be broken. Prepend the venv's
# site-packages so the system Python can use the right versions.
# ---------------------------------------------------------------------------
_VENV_SITE = Path(__file__).resolve().parent / ".venv" / "Lib" / "site-packages"
if _VENV_SITE.is_dir() and str(_VENV_SITE) not in sys.path:
    sys.path.insert(0, str(_VENV_SITE))

import torch

# ---------------------------------------------------------------------------
# The ``src.llm`` package __init__ eagerly imports every sub-package,
# including the training tracker which depends on the optional ``trackio``
# library.  For inference we only need config, model, tokenizer, checkpoint
# and generation -- none of which require trackio.
#
# To avoid the import chain we pre-register just the ``src`` and ``src.llm``
# packages with empty module objects *before* importing any leaf modules.
# This lets ``from src.llm.config import …`` resolve without running the
# real ``src/llm/__init__.py``.
# ---------------------------------------------------------------------------
import types

_PROJECT_ROOT = Path(__file__).resolve().parent

_STUB_PACKAGES = {
    "src":                  _PROJECT_ROOT / "src",
    "src.llm":              _PROJECT_ROOT / "src" / "llm",
    "src.llm.model":        _PROJECT_ROOT / "src" / "llm" / "model",
    "src.llm.generation":   _PROJECT_ROOT / "src" / "llm" / "generation",
    "src.llm.data_loading": _PROJECT_ROOT / "src" / "llm" / "data_loading",
    "src.llm.training":     _PROJECT_ROOT / "src" / "llm" / "training",
}

for _pkg_name, _pkg_path in _STUB_PACKAGES.items():
    if _pkg_name not in sys.modules:
        _mod = types.ModuleType(_pkg_name)
        _mod.__path__ = [str(_pkg_path)]
        _mod.__package__ = _pkg_name
        sys.modules[_pkg_name] = _mod

from src.llm.config import GenerationConfig, Modelconfig  # noqa: E402
from src.llm.generation.generate import generate  # noqa: E402
from src.llm.model.GPT_model import GPT  # noqa: E402
from src.llm.tokenizer import Tokenizer  # noqa: E402
from src.llm.training.checkpoint import (  # noqa: E402
    load_checkpoint,
    model_config_from_checkpoint,
)


DEFAULT_CHECKPOINT = Path("checkpoints") / "pretrain_final.pt"


def _resolve_device(requested: str) -> torch.device:
    """Pick the best available device, falling back gracefully."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple["GPT", Modelconfig]:
    """Reconstruct the model from the checkpoint's saved config and weights.

    Args:
        checkpoint_path: Path to the saved ``.pt`` checkpoint.
        device: Device to place the model on.

    Returns:
        A tuple of (model, model_config).

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_config = model_config_from_checkpoint(checkpoint_path, map_location="cpu")
    model = GPT(model_config)
    load_checkpoint(checkpoint_path, model, map_location=str(device))
    model.to(device)
    model.eval()

    return model, model_config


def _build_generation_config(args: argparse.Namespace) -> GenerationConfig:
    """Build a GenerationConfig from parsed CLI arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Configured GenerationConfig instance.
    """
    if args.greedy:
        config = GenerationConfig.greedy()
        config.max_new_tokens = args.max_tokens
        return config

    return GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )


def _run_interactive_loop(
    model: "GPT",
    tokenizer: Tokenizer,
    gen_config: GenerationConfig,
    device: torch.device,
    chat_mode: bool,
) -> None:
    """Run an interactive prompt loop, generating text from user input.

    Args:
        model: The loaded GPT model.
        tokenizer: Tokenizer instance.
        gen_config: Generation sampling configuration.
        device: Device the model lives on.
        chat_mode: Whether to wrap prompts in the chat template.
    """
    mode_label = "chat" if chat_mode else "completion"
    print(f"\nMode: {mode_label} | max_new_tokens={gen_config.max_new_tokens} "
          f"| temperature={gen_config.temperature}")
    print("Type a prompt and press Enter. Use 'quit' or Ctrl+C to exit.\n")

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            config=gen_config,
            device=device,
            chat=chat_mode,
        )

        print(f"\n{output}\n")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for interactive generation."""
    parser = argparse.ArgumentParser(
        description="Interactive text generation from a trained GPT checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"path to the .pt checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="device to run on: 'cpu', 'cuda', or 'auto' (default: auto)",
    )

    # Generation knobs
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="maximum new tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="top-k filtering (default: 50)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="nucleus (top-p) filtering (default: 0.95)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for reproducible generation",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="use greedy decoding (ignores temperature, top-k, top-p)",
    )

    # Mode
    parser.add_argument(
        "--chat",
        action="store_true",
        help="wrap prompts in the chat template ([SYS]...[USER]...[ASST])",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point: load the model and start the interactive generation loop."""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = _parse_args()
    device = _resolve_device(args.device)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, model_config = _load_model(args.checkpoint, device)
    print(
        f"Model loaded: {model.num_parameters():,} parameters | "
        f"d_model={model_config.d_model} | "
        f"layers={model_config.n_layer} | "
        f"heads={model_config.n_heads} | "
        f"device={device}"
    )

    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file()
    print(f"Tokenizer ready: vocab_size={tokenizer.vocab_size}")

    gen_config = _build_generation_config(args)

    _run_interactive_loop(
        model=model,
        tokenizer=tokenizer,
        gen_config=gen_config,
        device=device,
        chat_mode=args.chat,
    )


if __name__ == "__main__":
    main()
