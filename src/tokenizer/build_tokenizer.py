import sys
from pathlib import Path

import pandas as pd
from src.data_processing.text_normalization import build_normalizer
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
data_DIR = PROJECT_ROOT / "outputs" / "data"
tokenizer_DIR = PROJECT_ROOT / "outputs" / "tokenizer"

CLEAN_PATHS = [
    data_DIR / "cleaned" / "phase1_train_clean.parquet",
    # data_DIR / "cleaned" / "phase1_eval_clean.parquet",
    data_DIR / "cleaned" / "phase2_train_clean.parquet",
    # data_DIR / "cleaned" / "phase2_eval_clean.parquet",
]

OUTPUT_PATH = tokenizer_DIR / "tokenizer.json"

VOCAB_SIZE = 32_000
MIN_FREQUENCY = 2

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[BOS]",
    "[EOS]",
    "[SEP]",
    "[AR]",
    "[EN]",
    "[SYS]",
    "[USER]",
    "[ASST]",
]


def build_tokenizer():
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    tokenizer.normalizer = build_normalizer()

    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(prepend_scheme="always")

    tokenizer.decoder = decoders.Metaspace(prepend_scheme="always")

    return tokenizer


def corpus_iterator(df, batch_size=1000):
    texts = df["text"].tolist()

    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size]


def train_tokenizer(tokenizer, df):
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train_from_iterator(corpus_iterator(df), trainer=trainer, length=len(df))

    return tokenizer


def main():
    print("Loading data from {CLEAN_PATHS}")

    frames = []
    for p in CLEAN_PATHS:
        print(f"Loading {p}")
        df_split = pd.read_parquet(p)
        print(f"Loaded {len(df_split):,} rows")

        frames.append(df_split)

    df = pd.concat(frames, ignore_index=True)

    print(f"Loaded {len(df):,} rows from {len(CLEAN_PATHS)} files")

    print("Training tokenizer...")
    tokenizer = build_tokenizer()
    tokenizer = train_tokenizer(tokenizer, df)

    tokenizer.save(str(OUTPUT_PATH))
    print(f"Tokenizer saved to {OUTPUT_PATH}")

    # Example Usage

    example_text = [
        "مرحبا بك في عالم الذكاء الاصطناعي!",
        "Welcome to the world of artificial intelligence!",
        "جُمْهُورِيَّةُ مِصْرَ  الْعَرَبِيَّة",
        "أحمد إبراهيم آدم علىّ",
        "The   United  States",
    ]

    for text in example_text:
        enc = tokenizer.encode(text, add_special_tokens=False)
        print(f"\nOriginal text: {text}")
        print(f"Tokens: {enc.tokens}")
        print(f"IDs:    {enc.ids}")

        decoded = tokenizer.decode(enc.ids)
        print(f"Decoded: {decoded}")

        print("============================\n")


if __name__ == "__main__":
    main()
