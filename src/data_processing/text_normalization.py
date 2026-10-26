"""
│ TYPE B — lightweight normalization  (run EVERY time, train AND inference)  │
│   Character-level normalization that MUST be applied identically when we   │
│   train the tokenizer and when we later encode a user's prompt — otherwise │
│   the tokens won't line up. Because it must be identical, we express it    │
│   ONCE as a HuggingFace `normalizers.Sequence` and embed that same object  │
│   inside the tokenizer, so it literally travels with tokenizer.json.       │
│       → build_normalizer(), normalize_text()                               │
└──────────────────────────────────────────────────────────────────────────┘

The golden rule this split protects:
    Whatever normalization the tokenizer was TRAINED with must also run at
    INFERENCE. Type B is that shared, embeddable normalization. Type A is the
    one-off corpus cleanup that never runs at inference.
"""

import re
from tokenizers import Regex
from tokenizers.normalizers import NFC, Replace, Sequence


# ============================================================
# TYPE B — lightweight, embeddable normalization
# ============================================================
# We build the normalization as tokenizer components so the *exact same*
# transformation can be (a) embedded in the tokenizer and (b) reused as a
# plain Python function. There is only one definition, so they cannot drift.
#
# Arabic normalization decisions (each is a real trade-off):
#   • remove tashkeel (diacritics) + tatweel  → collapses spelling variants
#   • unify alef  أ إ آ → ا                    → fewer variants, small quality cost
#   • alef maqsura ى → ي                       → common, consistent
#   • we deliberately KEEP ta-marbuta (ة)      → dropping it would stop the model
#                                                from ever generating ة (bad output)


def build_normalizer():
    return Sequence(
        [
            NFC(),
            Replace(Regex(r"[\u064B-\u0652\u0640]"), ""),
            Replace("أ", "ا"),
            Replace("إ", "ا"),
            Replace("آ", "ا"),
            Replace("ى", "ي"),
            Replace(Regex(r"[ \t]+"), " "),  # collapse spaces/tabs (keep newlines)
        ]
    )


_NORMALIZER = build_normalizer()


def normalize_text(text):
    """
    Lightweight normalization for a single string.
    """

    return _NORMALIZER.normalize_str(text)
