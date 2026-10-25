"""

====================================================

This module holds ALL the text handling for the tokenizer chapter, split into
the two kinds of cleaning that every real LLM pipeline needs. Keeping them
separate is one of the main lessons of this chapter.

┌──────────────────────────────────────────────────────────────────────────┐
│ TYPE A — heavy preparation  (run ONCE, offline, on the raw corpus)         │
│   Structural surgery on messy Wikipedia dumps: delete leftover citation    │
│   templates, drop reference / "See also" sections, remove empty sections,  │
│   unify heading styles. This is destructive and slow, so we do it a single │
│   time to produce a clean training corpus.                                 │
│       → prepare_document(), clean_dataframe()                              │
├──────────────────────────────────────────────────────────────────────────┤
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

    return Sequence([
        NFC(),
        Replace(Regex( r"[\u064B-\u0652\u0640]"), "" ),
        Replace("أ", "ا"),
        Replace("إ", "ا"),
        Replace("آ", "ا"),
        Replace("ى", "ي"),   
        Replace(Regex(r"[ \t]+"), " "),  # collapse spaces/tabs (keep newlines)
    ])

_NORMALIZER = build_normalizer()

def normalize_text(text):
    """
    Lightweight normalization for a single string.
    """

    return _NORMALIZER.normalize_str(text)


# ============================================================
# TYPE A — heavy, one-time corpus preparation
# ============================================================

# --- Lexicon 1: leftover citation templates to delete in place ---------------
# In the raw Arabic dump, {{استشهاد ويب}} templates were stripped imperfectly,
# leaving the bare template *name* glued to the text (e.g. «...لبناناستشهاد ويب»).
# These strings are ONLY ever template remnants, so deleting them is safe.
# English was cleaned properly upstream, but we cover its templates too.
_TEMPLATE_RE = re.compile(
    r"استشهاد\s*ب?(?:ويب|كتاب|دورية|خبر|مجلة|موسوعة|صحيفة)"   # AR cite templates
    r"|(?i:cite\s+(?:web|book|news|journal)|citation\s+needed)"  # EN cite templates
)

# --- Lexicon 2: navigation / reference sections to drop entirely -------------
# These sections are link and citation dumps with little language value.
# Titles are compared after light normalization (see _norm_title).
_NAV_SECTION_TITLES = {
    # Arabic
    "مراجع", "المراجع", "مراجع ومصادر", "راجع", "مصادر", "المصادر",
    "وصلات خارجيه", "روابط خارجيه", "وصلات", "انظر ايضا", "طالع ايضا",
    "هوامش", "ملاحظات", "للاستزاده", "قائمه المراجع", "معرض الصور", "بوابات",
    # English
    "references", "reference", "see also", "external links", "notes",
    "further reading", "bibliography", "sources", "citations",
    "notes and references", "footnotes",
}

# MediaWiki heading «== title ==» (any level 2-6).
_MEDIAWIKI_HEADING_RE = re.compile(r"(?m)^[ \t]*=+[ \t]*(.+?)[ \t]*=+[ \t]*$")
# Markdown heading «## title».
_MARKDOWN_HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]*(.+?)[ \t]*$")

# Category / portal link lines like «تصنيف: ...» or «بوابة: ...».
_CATEGORY_LINE_RE = re.compile(r"(?m)^[ \t]*(?:تصنيف|بوابة|Category|Portal)\s*:.*$")

# A section body with fewer than this many words is considered empty/sparse.
MIN_SECTION_WORDS = 8