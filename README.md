# Bilingual GPT From Scratch (Arabic & English)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Architecture: Decoder-Only GPT](https://img.shields.io/badge/Architecture-Decoder--Only%20GPT-purple.svg)]()
[![Streaming: Memmap Zero--RAM](https://img.shields.io/badge/Dataset-Memory--Mapped%20uint16-orange.svg)]()

A clean, production-grade, from-scratch implementation of an autoregressive **Bilingual (Arabic & English) GPT model** in PyTorch. 

Every component is written from first principles without relying on high-level modeling frameworks like Hugging Face `transformers`:
- Custom **BPE Tokenizer** with Arabic-specialized normalization baked into the serialization format.
- Two-tier **data cleaning pipeline** separating offline corpus sanitization from runtime normalization.
- **Pre-LayerNorm Decoder-only Transformer** with **Key-Value (KV) Caching**, **Weight Tying**, and **GPT-2 scaled residual initialization**.
- **Zero-RAM memory-mapped binary dataset packing** (`uint16`) supporting seamless streaming over tens of millions of tokens.
- **Document-balanced cross-entropy loss**, matrix-selective AdamW weight decay, cosine warmup learning rate scheduling, and prompt-loss masking for chat/instruction tuning.

---

## Table of Contents

- [1. Project Overview & Motivation](#1-project-overview--motivation)
- [2. Model Architecture Deep Dive](#2-model-architecture-deep-dive)
  - [High-Level Topology](#high-level-topology)
  - [Multi-Head Causal Attention & KV Caching](#multi-head-causal-attention--kv-caching)
  - [Feed-Forward Network (FFN)](#feed-forward-network-ffn)
  - [Weight Tying & Residual Scaling](#weight-tying--residual-scaling)
- [3. Model Specifications & Parameter Counts](#3-model-specifications--parameter-counts)
- [4. Data Engineering Philosophy (Type A vs. Type B)](#4-data-engineering-philosophy-type-a-vs-type-b)
  - [Type A: Heavy Corpus Cleansing (Offline)](#type-a-heavy-corpus-cleansing-offline)
  - [Type B: Lightweight Normalization (Train & Inference Invariant)](#type-b-lightweight-normalization-train--inference-invariant)
- [5. Custom Bilingual BPE Tokenizer](#5-custom-bilingual-bpe-tokenizer)
  - [Special Role Tokens](#special-role-tokens)
- [6. Zero-RAM Binary Packing & Windowing](#6-zero-ram-binary-packing--windowing)
  - [Token Concatenation with Document Seams](#token-concatenation-with-document-seams)
  - [Memory Mapping with `PackedDataset`](#memory-mapping-with-packeddataset)
  - [Next-Token Shift in Collation](#next-token-shift-in-collation)
- [7. Training & Optimization Innovations](#7-training--optimization-innovations)
  - [Document-Balanced Cross-Entropy](#document-balanced-cross-entropy)
  - [Selective Weight Decay (AdamW)](#selective-weight-decay-adamw)
  - [Cosine Warmup Schedule](#cosine-warmup-schedule)
  - [Instruction Tuning & Prompt-Loss Masking](#instruction-tuning--prompt-loss-masking)
- [8. Step-by-Step Pipeline Guide](#8-step-by-step-pipeline-guide)
  - [Environment Setup](#environment-setup)
  - [Step 1: Download & Filter Raw Data](#step-1-download--filter-raw-data)
  - [Step 2: Clean the Dataset Splits](#step-2-clean-the-dataset-splits)
  - [Step 3: Train the Bilingual Tokenizer](#step-3-train-the-bilingual-tokenizer)
  - [Step 4: Pack the Tokenized Corpus](#step-4-pack-the-tokenized-corpus)
  - [Step 5: Run Pretraining](#step-5-run-pretraining)
  - [Step 6: Text Generation & Sampling](#step-6-text-generation--sampling)
- [9. Exploratory Jupyter Notebooks](#9-exploratory-jupyter-notebooks)
- [10. Project Directory Layout](#10-project-directory-layout)

---

## 1. Project Overview & Motivation

Most open-source foundation models underperform on Arabic due to uneven tokenizer vocabularies (where Arabic characters decompose into multiple byte fragments, causing high compression ratios and ballooning sequence lengths) and poor orthographic normalization.

This project implements a complete, self-contained Arabic-English bilingual Large Language Model from raw text scraping up to autoregressive generation:

```
[Raw Shards / Hugging Face]
            │
            ▼
[Data Cleaning: Type A & B] ──► [32,000 Vocab BPE Tokenizer]
            │                               │
            ▼                               ▼
[Flat uint16 Binary Packing] ◄──────────────┘
            │
            ▼
[Memory-Mapped PackedDataset]
            │
            ▼
[Pre-LN Transformer + KV Cache]
            │
            ▼
[Autoregressive Bilingual Generation / Chat]
```

### Key Highlights
1. **True Bilingual Parity**: A 32,000-token BPE vocabulary trained on balanced Arabic and English corpora with `Metaspace` pre-tokenization.
2. **Mathematically Rigorous Arabic Normalization**: Diacritic and elongation removal while consciously keeping grammatically vital suffixes (such as *Ta-Marbuta* `ة`).
3. **Hardware Efficient**: Memory-mapped binary streaming enables training on large corpora with virtually zero RAM footprint.
4. **Instruction & Chat Ready**: Native architectural and data-collator support for multi-turn role tokens (`[SYS]`, `[USER]`, `[ASST]`) with prompt loss masking.

---

## 2. Model Architecture Deep Dive

The model is a modern, decoder-only Transformer following the **Pre-LayerNorm** formulation (similar to GPT-2/3 and LLaMA) rather than original Post-LN Transformer designs. Pre-LN ensures smooth gradient flow across deep networks without gradient explosion or vanishing during early training phases.

### High-Level Topology

```
Input Token IDs [B, T]
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
Token Embedding [B, T, D]     Position Embedding [1, T, D]
       │                                 │
       └───────────────┬─────────────────┘
                       ▼
                 Vector Addition
                       │
                    Dropout
                       │
          ┌────────────┴─────────────┐
          │  GPT Block 1 .. N_layer  │
          │  ┌────────────────────┐  │
          │  │     LayerNorm      │  │
          │  │        │           │  │
          │  │ Multi-Head Causal  │  │
          │  │     Attention      │  │
          │  │   (+ KV Cache)     │  │
          │  └────────┬───────────┘  │
          │           ▼              │
          │       Residual ───(+)    │
          │           │              │
          │  ┌────────┴───────────┐  │
          │  │     LayerNorm      │  │
          │  │        │           │  │
          │  │  Feed-Forward MLP  │  │
          │  │   (GELU tanh)      │  │
          │  └────────┬───────────┘  │
          │           ▼              │
          │       Residual ───(+)    │
          └───────────┬──────────────┘
                      ▼
               Final LayerNorm
                      │
                   LM Head (Weights Tied to Token Embedding)
                      │
               Logits [B, T, V]
```

---

### Multi-Head Causal Attention & KV Caching

Attention heads project the normalized representation $X \in \mathbb{R}^{B \times T \times d_{\text{model}}}$ into Query, Key, and Value spaces:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

where $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ and head dimension $d_k = \frac{d_{\text{model}}}{n_{\text{heads}}}$.

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

The causal mask $M$ is a lower-triangular matrix set to $-\infty$ above the diagonal, strictly forbidding tokens from attending to subsequent tokens in the sequence.

```
       Token 0   Token 1   Token 2   Token 3
Tok 0 [   0        -inf      -inf      -inf  ]
Tok 1 [   0          0       -inf      -inf  ]
Tok 2 [   0          0         0       -inf  ]
Tok 3 [   0          0         0         0   ]
```

#### Key-Value (KV) Caching for Generation
During autoregressive text generation, recomputing $K$ and $V$ matrices for all preceding tokens at every generation step incurs an $O(T^2)$ computational overhead. 

The attention layer implements dynamic **KV Caching**:
1. At prompt encoding step: $K$ and $V$ tensors across all prompt tokens are computed and stored.
2. At subsequent token generations: Only the single newly sampled token is passed through ($T=1$). Its projected $K_{\text{new}}$ and $V_{\text{new}}$ are concatenated along the sequence length dimension with the cached tensors:

$$K_{\text{total}} = [K_{\text{past}} \,;\, K_{\text{new}}], \quad V_{\text{total}} = [V_{\text{past}} \,;\, V_{\text{new}}]$$

This reduces the generation time complexity per step from $O(T)$ down to $O(1)$ matrix-vector operations.

---

### Feed-Forward Network (FFN)

Each transformer block includes a position-wise feed-forward sublayer that expands the hidden dimension by a factor of 4 before projecting back:

$$\text{FFN}(x) = \left(\text{GELU}\left(x W_1 + b_1\right)\right) W_2 + b_2$$

where $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ ($d_{\text{ff}} = 4 \cdot d_{\text{model}}$), $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$, and $\text{GELU}$ uses the tanh approximation:

$$\text{GELU}(z) \approx 0.5 \cdot z \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(z + 0.044715 \cdot z^3\right)\right)\right)$$

---

### Weight Tying & Residual Scaling

1. **Weight Tying**:
   The output projection layer (`lm_head`) shares identical weight tensors with the input token embedding layer (`w_embeddings.token_embedding`):
   ```python
   self.lm_head.weight = self.w_embeddings.token_embedding.weight
   ```
   This halves the parameter footprint of the vocabulary projection ($32,000 \times 512 = 16.38\text{M}$ parameters saved) while forcing token representations and next-token prediction targets to live in an identical geometric manifold.

2. **Residual Projections Initialization Scaling**:
   Standard normal initialization ($\mu=0, \sigma=0.02$) can lead to activation variance growing proportionally to model depth $L$. Following GPT-2, all output projection weights (`W_o` in attention and `fc2` in feed-forward) are scaled at initialization by:
   
   $$\text{scale} = \frac{1}{\sqrt{2 \cdot N_{\text{layer}}}}$$

---

## 3. Model Specifications & Parameter Counts

| Parameter | Main Model (`main_model()`) | Tiny / Smoke-Test (`tiny()`) |
| :--- | :--- | :--- |
| **Vocabulary Size ($V$)** | 32,000 | 32,000 |
| **Hidden Dimension ($d_{\text{model}}$)** | 512 | 128 |
| **FFN Dimension ($d_{\text{ff}}$)** | 2,048 ($4 \times d_{\text{model}}$) | 512 ($4 \times d_{\text{model}}$) |
| **Number of Layers ($N_{\text{layer}}$)** | 8 | 2 |
| **Attention Heads ($n_{\text{heads}}$)** | 8 | 4 |
| **Head Dimension ($d_k$)** | 64 | 32 |
| **Maximum Sequence Length** | 1,024 | 256 |
| **Dropout** | 0.05 | 0.1 |
| **Weight Tying** | Enabled | Enabled |
| **Total Parameter Count** | **42,116,096** (~42.1M) | **4,524,800** (~4.5M) |
| **Non-Embedding Parameters** | **25,207,808** (~25.2M) | **396,032** (~396K) |

---

## 4. Data Engineering Philosophy (Type A vs. Type B)

A core principle of this project is the **strict architectural separation of corpus preparation from text normalization**.

```
┌────────────────────────────────────────────────────────────────────────┐
│ TYPE A: Heavy, Destructive Corpus Preparation (Executed Once, Offline) │
│ - Removes MediaWiki junk, citation templates, navigational dumps       │
│ - Unifies Markdown headings, eliminates duplicate lines                │
│ - Discards low-density articles (< 40 chars, < 8 words/section)        │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ produces clean parquet
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TYPE B: Lightweight Invariant Normalization (Train & Inference Shared) │
│ - NFC Unicode Normalization                                            │
│ - Diacritic stripping (Tashkeel: Fatha, Damma, Kasra, Sukun, Tanwin)   │
│ - Tatweel (Kashida) deletion                                           │
│ - Alef variant collapse (أ, إ, آ -> ا) & Alef Maqsura (ى -> ي)         │
│ - Deliberate Ta-Marbuta preservation (ة kept)                          │
│ - Serialized directly into tokenizer.json (zero-drift guarantee)       │
└────────────────────────────────────────────────────────────────────────┘
```

### Type A: Heavy Corpus Cleansing (Offline)
Located in [`src/data_processing/data_cleaning.py`](file:///d:/GAN_AI/From%20Scratch/Bilingual_GPT_from_scratch/src/data_processing/data_cleaning.py):
- **Template Stripping**: MediaWiki citation remnants (`استشهاد بويب`, `cite web`, `citation needed`) left glued to words during raw scraping are pruned via regular expressions.
- **Navigational Section Pruning**: Reference, bibliography, and portal sections (`مراجع`, `وصلات خارجية`, `انظر أيضا`, `references`, `external links`) contain low linguistic density and are stripped in their entirety.
- **Sparse Section Filtering**: Any section with fewer than 8 words (`MIN_SECTION_WORDS = 8`) is dropped.
- **Heading Standardization**: MediaWiki `== heading ==` syntax is mapped to Markdown `## heading`.
- **Deduplication**: Immediate repeating lines caused by scrapers duplicating header text are eliminated.

### Type B: Lightweight Normalization (Train & Inference Invariant)
Located in [`src/data_processing/text_normalization.py`](file:///d:/GAN_AI/From%20Scratch/Bilingual_GPT_from_scratch/src/data_processing/text_normalization.py):
The cardinal rule of language modeling: **Whatever normalization the tokenizer is trained on must be applied identically to user prompts at inference time.**

To guarantee zero drift, Type B normalization is implemented as a Hugging Face `tokenizers.normalizers.Sequence` and stored directly inside `outputs/tokenizer/tokenizer.json`:

```python
Sequence([
    NFC(),
    Replace(Regex(r"[\u064B-\u0652\u0640]"), ""),  # Tashkeel & Tatweel
    Replace("أ", "ا"), Replace("إ", "ا"), Replace("آ", "ا"),  # Unify Alef
    Replace("ى", "ي"),                             # Alef Maqsura -> Yaa
    Replace(Regex(r"[ \t]+"), " "),               # Collapse horizontal whitespace
])
```

#### Why keep *Ta-Marbuta* (`ة`)?
Many naive Arabic preprocessing scripts replace `ة` with `ه`. While this collapses inflectional variants, it is destructive for generation: the model would permanently lose the ability to output proper feminine nouns and grammatically valid Arabic sentence endings. We deliberately preserve `ة`.

---

## 5. Custom Bilingual BPE Tokenizer

The tokenizer is a custom Byte-Pair Encoding (BPE) model built with Hugging Face's `tokenizers` engine and wrapped in [`src/llm/tokenizer.py`](file:///d:/GAN_AI/From%20Scratch/Bilingual_GPT_from_scratch/src/llm/tokenizer.py).

- **Vocabulary**: 32,000 subwords.
- **Pre-tokenizer**: `Metaspace(prepend_scheme="always")` replaces spaces with the ` ` (U+2581) meta-character. This prevents space-stripping ambiguities and models whitespace as first-class tokens across both Latin and Arabic scripts.
- **Decoder**: `decoders.Metaspace` cleanly reconstructs the exact original text with spaces.

### Special Role Tokens

The first 10 vocabulary indices are reserved for structural signals and chat instruction templates:

| Index | Token | Purpose |
| :---: | :---: | :--- |
| `0` | `[PAD]` | Batch padding token (ignored in loss computations) |
| `1` | `[UNK]` | Out-of-vocabulary fallback |
| `2` | `[BOS]` | Beginning of Document / Sequence |
| `3` | `[EOS]` | End of Document / Seam between documents in packed binary |
| `4` | `[SEP]` | General purpose separator |
| `5` | `[AR]` | Language identification flag: Arabic |
| `6` | `[EN]` | Language identification flag: English |
| `7` | `[SYS]` | System prompt boundary marker |
| `8` | `[USER]` | User instruction marker |
| `9` | `[ASST]` | Assistant response marker |

---

## 6. Zero-RAM Binary Packing & Windowing

Training Large Language Models by reading Parquet, JSON, or text files during the training loop incurs severe CPU bottlenecks, GC stalls, and heavy RAM bloat. This project uses a **flat memory-mapped binary format** (`uint16`).

```
Document 1: [BOS] The Nile flows north. [EOS]
Document 2: [BOS] مصر دولة عربية. [EOS]
Document 3: [BOS] Alexandria was founded in... [EOS]
                         │
                         ▼
Flat uint16 Stream (train.bin):
[2, 7776, 10245, 29003, 29146, 3, 2, 7954, 13322, 18518, 8297, 3, 2, 451, ...]
 |<─────── Doc 1 ────────────>| |<────────── Doc 2 ───────────>| |<── Doc 3 ...
 │                                                             │
 └──────────────────────── Window 0 ───────────────────────────┘
```

### Token Concatenation with Document Seams
Located in [`src/tokenizer/pack_courpus.py`](file:///d:/GAN_AI/From%20Scratch/Bilingual_GPT_from_scratch/src/tokenizer/pack_courpus.py):
1. Each document is tokenized independently.
2. Every document is wrapped with `[BOS]` at the start and `[EOS]` at the end.
3. Documents are appended sequentially into `outputs/train.bin` and `outputs/eval.bin`.
4. Because the vocabulary size is 32,000 ($< 65,536$), every token fits comfortably in a 2-byte unsigned integer (`np.uint16`), minimizing disk storage by 50% compared to `int32`.

### Memory Mapping with `PackedDataset`
[`src/llm/data_loading/dataset.py`](file:///d:/GAN_AI/From%20Scratch/Bilingual_GPT_from_scratch/src/llm/data_loading/dataset.py) opens binary files using `np.memmap(mode='r')`.
- Zero tokens are loaded into RAM at startup.
- The operating system page cache transparently fetches required pages from disk into memory on demand.
- A 100M+ token dataset can be initialized instantaneously with negligible RAM consumption.

### Next-Token Shift in Collation
Windows are sliced with length `max_seq_len + 1` (e.g., $1024 + 1 = 1025$ tokens). 
In [`src/llm/data_loading/collate.py`](file:///d:/GAN_AI/From%20Scratch/Bilingual_GPT_from_scratch/src/llm/data_loading/collate.py), each chunk is split into `input_ids` and `targets` via a 1-token shift:

$$\text{input\_ids} = \text{window}[:-1] \quad (\text{tokens } 0 \dots T-1)$$

$$\text{targets} = \text{window}[1:] \quad (\text{tokens } 1 \dots T)$$

Every real token position $t$ in `input_ids` is trained to predict the token at position $t$ in `targets`.

---

## 7. Training & Optimization Innovations

### Document-Balanced Cross-Entropy
Located in [`src/llm/training/trainer.py`](file:///d:/GAN_AI/From%20Scratch/Bilingual_GPT_from_scratch/src/llm/training/trainer.py):

Standard autoregressive language modeling computes the mean loss over all non-padded tokens in a batch. If a batch contains an 800-token article alongside three 50-token articles, the long article dominates 84% of the gradient update.

We implement **two-stage Document-Balanced Cross-Entropy**:
1. Compute the token loss per document/row, averaged only over the real (non-pad) tokens in that specific row.
2. Take the simple mean across the batch rows:

$$\mathcal{L}_{\text{batch}} = \frac{1}{B} \sum_{i=1}^B \left( \frac{1}{T_i} \sum_{j=1}^{T_i} -\log P(w_{i,j+1} \mid w_{i, \le j}) \right)$$

This guarantees every document exerts equal gradient influence, regardless of its sequence length.

---

### Selective Weight Decay (AdamW)
Weight decay penalizes large weights, but applying it to 1-dimensional parameters (LayerNorm gains $\gamma$, biases $\beta$) destabilizes normalization dynamics. 

Our optimizer constructor partitions model parameters:
- **Decay Group** (Weight Decay = 0.1): 2D+ tensors (Linear projection matrices, embedding matrices).
- **No-Decay Group** (Weight Decay = 0.0): 1D tensors (LayerNorm weights and all biases).
- **AdamW Betas**: Set to $\beta_1 = 0.9, \beta_2 = 0.95$ (tuned specifically for LLM autoregressive stability, matching GPT-3).

---

### Cosine Warmup Schedule
Located in [`src/llm/training/schedule.py`](file:///d:/GAN_AI/From%20Scratch/Bilingual_GPT_from_scratch/src/llm/training/schedule.py):
- **Linear Warmup**: Over the first `warmup_steps`, learning rate increases linearly from $0 \to \eta_{\max}$. This prevents large early gradients from destroying random initializations when AdamW running estimates are cold.
- **Cosine Annealing**: Decays smoothly down to a floor of `min_lr_ratio * learning_rate` (10% of peak):

$$\eta_t = \eta_{\min} + \frac{1}{2} (\eta_{\max} - \eta_{\min}) \left(1 + \cos\left(\pi \frac{t - t_{\text{warmup}}}{t_{\max} - t_{\text{warmup}}}\right)\right)$$

```
Learning Rate
     ▲
ηmax │       /‾‾‾\
     │      /     \
     │     /       \
     │    /         \
     │   /           \
ηmin │  /             \____
     └─┴───────┬───────────► Step
       0    warmup       max_steps
```

---

### Instruction Tuning & Prompt-Loss Masking
Located in [`src/llm/data_loading/chat.py`](file:///d:/GAN_AI/From%20Scratch/Bilingual_GPT_from_scratch/src/llm/data_loading/chat.py):

When fine-tuning on conversation or question-answering pairs, training the model to predict the *user's question* wastes model capacity and degrades assistant behavior.

We structure conversations using role markers and mask all prompt tokens with `pad_id=0`:

```
Tokens:  [BOS] [SYS] System... [USER] Question... [ASST] Answer... [EOS]
Labels:  [PAD] [PAD] [PAD] ... [PAD]  [PAD]   ... [ASST] Answer... [EOS]
                                                  ▲
                                         loss starts computing here
```

Because cross-entropy uses `ignore_index=pad_id`, zero gradient flows back from the prompt, forcing 100% of the optimization focus onto producing accurate assistant completions.

---

## 8. Step-by-Step Pipeline Guide

### Environment Setup

This project requires **Python 3.12+** and **PyTorch 2.0+**. Using [`uv`](https://github.com/astral-sh/uv) or a standard virtual environment:

```bash
# Clone the repository
git clone https://github.com/your-username/Bilingual_GPT_from_scratch.git
cd Bilingual_GPT_from_scratch

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install the package in editable mode with dependencies
pip install -e .
```

---

### Step 1: Download & Filter Raw Data

Downloads the raw bilingual shards from [`bakrianoo/jabarti-llm-dataset`](https://huggingface.co/datasets/bakrianoo/jabarti-llm-dataset) on Hugging Face Hub:

```bash
python src/utils/download_and_filter.py --max-shards 20 --tokenizer-docs 40000
```

- Downloads training and evaluation Parquet shards.
- Samples a balanced bilingual subset (`outputs/data/tokenizer_corpus.parquet`) for tokenizer training.
- Saves the conversational fine-tuning splits (`outputs/ft_train_filtered.parquet` and `outputs/ft_eval_filtered.parquet`).

---

### Step 2: Clean the Dataset Splits

Executes the heavy Type A offline cleanup pipeline:

```bash
python src/pipeline.py
```

- Strips citation templates (`استشهاد بويب`, `cite web`).
- Eliminates non-content reference and external link blocks.
- Drops sparse sections and normalizes MediaWiki headings into Markdown.
- Outputs cleaned datasets into `outputs/data/cleaned/`.

---

### Step 3: Train the Bilingual Tokenizer

Trains the 32,000-vocabulary BPE model with embedded Type B normalization:

```bash
python src/tokenizer/build_tokenizer.py
```

- Learns merge rules from `tokenizer_corpus.parquet`.
- Bakes Unicode NFC, Tashkeel removal, and Alef normalization directly into the pipeline.
- Configures `Metaspace` pre-tokenization.
- Writes serialized tokenizer to `outputs/tokenizer/tokenizer.json`.

---

### Step 4: Pack the Tokenized Corpus

Converts the clean textual datasets into flat, memory-mappable `uint16` binary files:

```bash
python src/tokenizer/pack_courpus.py
```

Optional classroom/smoke-test flag:
```bash
python src/tokenizer/pack_courpus.py --max-shards 2
```

Outputs generated in `outputs/`:
- `train.bin`: Flat array of token IDs with `[BOS]` and `[EOS]` document boundaries (~47.9M tokens).
- `eval.bin`: Evaluation token binary stream (~10.9M tokens).
- `meta.json`: Corpus provenance tracking (shuffling seed, document counts, vocabulary size).

---

### Step 5: Run Pretraining

Launch training using [`run_pretrain.py`](file:///d:/GAN_AI/From%20Scratch/Bilingual_GPT_from_scratch/run_pretrain.py):

#### Standard Pretraining Run (GPU)
```bash
python run_pretrain.py --epochs 2 --batch-size 16 --accumulation-steps 4 --lr 3e-4
```

#### CPU or Fast Smoke-Test Run
```bash
python run_pretrain.py --tiny --limit 1000 --batch-size 8 --device cpu
```

#### Resume from an Existing Checkpoint
```bash
python run_pretrain.py --resume checkpoints/train-phase_step4000.pt
```

#### Available CLI Arguments:
- `--tiny`: Use the lightweight 2-layer, 128-dim model for testing.
- `--epochs`: Number of full passes over the packed corpus (default: 1.0).
- `--steps`: Explicit optimizer steps (overrides `--epochs`).
- `--batch-size`: Per-device micro-batch size.
- `--accumulation-steps`: Gradient accumulation steps before calling `optimizer.step()`.
- `--lr`: Peak learning rate (default: `3e-4`).
- `--device`: Target compute device (`cuda` or `cpu`).
- `--save-every`: Checkpoint saving frequency in steps.
- `--eval-every`: Evaluation interval on `eval.bin` in steps.
- `--sample-every`: Interval to generate and display completions for bilingual validation prompts.

---

### Step 6: Text Generation & Sampling

Autoregressive inference can be executed directly in Python:

```python
import torch
from src.llm import GPT, ModelConfig, Tokenizer
from src.llm.config import GenerationConfig
from src.llm.generation.generate import generate

# 1. Load Tokenizer & Model
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = Tokenizer.from_file("outputs/tokenizer/tokenizer.json")
config = ModelConfig.main_model()

model = GPT(config)
checkpoint = torch.load("checkpoints/pretrain_final.pt", map_location=device)
model.load_state_dict(checkpoint["model"])
model.to(device)

# 2. Configure Sampling Parameters
gen_config = GenerationConfig(
    max_new_tokens=80,
    temperature=0.7,   # Lower = focused, Higher = creative
    top_k=50,          # Filter to top 50 candidates
    top_p=0.9,         # Nucleus sampling threshold
    use_cache=True     # Fast generation with KV caching
)

# 3. Generate Arabic Text
ar_prompt = "تقع مدينة القاهرة على ضفاف نهر"
print(generate(model, tokenizer, ar_prompt, gen_config, device=device))

# 4. Generate English Text
en_prompt = "The ancient Library of Alexandria was"
print(generate(model, tokenizer, en_prompt, gen_config, device=device))
```

---

## 9. Exploratory Jupyter Notebooks

The [`notebooks/`](file:///d:/GAN_AI/From%20Scratch/Bilingual_GPT_from_scratch/notebooks/) directory provides standalone, step-by-step educational notebooks demonstrating how each sub-component works in isolation:

1. [`01_text_cleaning.ipynb`](file:///d:/GAN_AI/From%20Scratch/Bilingual_GPT_from_scratch/notebooks/01_text_cleaning.ipynb): Interactive experimentation with Arabic regexes, citation template removal, and heading normalizers.
2. [`02_tokenizer_test.ipynb`](file:///d:/GAN_AI/From%20Scratch/Bilingual_GPT_from_scratch/notebooks/02_tokenizer_test.ipynb): Inspecting BPE splits, subword token IDs, Metaspace encoding, and special token round-tripping.
3. [`03_embedding_test.ipynb`](file:///d:/GAN_AI/From%20Scratch/Bilingual_GPT_from_scratch/notebooks/03_embedding_test.ipynb): Learnable token embeddings combined with 1D position embeddings and position offsets.
4. [`04_attention.ipynb`](file:///d:/GAN_AI/From%20Scratch/Bilingual_GPT_from_scratch/notebooks/04_attention.ipynb): Visualizing causal attention masks, dot-product scaling, and the step-by-step KV-cache tensor updates.
5. [`05_GPT_block.ipynb`](file:///d:/GAN_AI/From%20Scratch/Bilingual_GPT_from_scratch/notebooks/05_GPT_block.ipynb): Assembling Pre-LN transformer blocks, LayerNorm verification, and residual tensor summation.

---

## 10. Project Directory Layout

```
Bilingual_GPT_from_scratch/
├── notebooks/                      # Exploratory step-by-step research notebooks
│   ├── 01_text_cleaning.ipynb
│   ├── 02_tokenizer_test.ipynb
│   ├── 03_embedding_test.ipynb
│   ├── 04_attention.ipynb
│   └── 05_GPT_block.ipynb
├── outputs/                        # Artifacts & dataset binaries (created by pipeline)
│   ├── data/
│   │   ├── cleaned/                # Post-Type A cleaned parquet files
│   │   └── tokenizer_corpus.parquet
│   ├── tokenizer/
│   │   └── tokenizer.json          # Serialized BPE tokenizer with Type B normalizer
│   ├── train.bin                   # Memory-mapped uint16 training token stream
│   ├── eval.bin                    # Memory-mapped uint16 evaluation token stream
│   └── meta.json                   # Dataset packing metadata & provenance
├── src/
│   ├── data_processing/
│   │   ├── data_cleaning.py        # Type A: Heavy offline regex & structural surgery
│   │   └── text_normalization.py   # Type B: Train/inference invariant Arabic normalizer
│   ├── llm/
│   │   ├── data_loading/
│   │   │   ├── chat.py             # ChatDataset & prompt loss masking formatting
│   │   │   ├── collate.py          # Dynamic padding & next-token shift collation
│   │   │   └── dataset.py          # Memory-mapped PackedDataset implementation
│   │   ├── generation/
│   │   │   ├── generate.py         # KV-cached autoregressive generation loop
│   │   │   └── sampling.py         # Pure functions: temperature, top-k, nucleus (top-p)
│   │   ├── model/
│   │   │   ├── attention.py        # MultiHeadAttention with causal mask & KV cache
│   │   │   ├── embedding.py        # Token + 1D Positional InputEmbeddings
│   │   │   ├── feedforward.py      # MLP with GELU (tanh approximation)
│   │   │   ├── GPT_block.py        # Pre-LN Transformer Block
│   │   │   └── GPT_model.py        # Full GPT Model with weight tying & residual scaling
│   │   ├── training/
│   │   │   ├── checkpoint.py       # Checkpoint save/load with config validation
│   │   │   ├── freeze.py           # Parameter freezing & fine-tuning utilities
│   │   │   ├── schedule.py         # Cosine learning rate schedule with linear warmup
│   │   │   ├── tracking.py         # Trackio experiment tracking wrapper
│   │   │   └── trainer.py          # Trainer with document-balanced cross-entropy
│   │   ├── config.py               # ModelConfig, TrainingConfig, GenerationConfig
│   │   └── tokenizer.py            # High-level Tokenizer wrapper class
│   ├── tokenizer/
│   │   ├── build_tokenizer.py      # BPE tokenizer training script
│   │   └── pack_courpus.py         # Flat uint16 binary corpus packing script
│   ├── utils/
│   │   └── download_and_filter.py  # Hugging Face Hub dataset download & split filter
│   └── pipeline.py                 # Pipeline runner for dataset cleaning
├── run_pretrain.py                 # CLI entry point to train the model
├── pyproject.toml                  # Package configuration and dependencies
└── README.md                       # Project documentation
```

---

## License

This project is licensed under the MIT License. Contributions, experiments, and extensions are welcome!
