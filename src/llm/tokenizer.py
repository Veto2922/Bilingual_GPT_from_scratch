from pathlib import Path
from tokenizers import Tokenizer as backingTokenizer


DEFAULT_TOKENIZER_PATH: Path = (
    Path(__file__).resolve().parent.parent.parent / "outputs" / "tokenizer" / "tokenizer.json"
)


class Tokenizer:
    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3
    SEP = 4
    AR = 5
    EN = 6
    SYS = 7
    USER = 8
    ASST = 9

    _SPECIAL_TOKENS: dict = {
        "[PAD]": PAD,
        "[UNK]": UNK,
        "[BOS]": BOS,
        "[EOS]": EOS,
        "[SEP]": SEP,
        "[AR]": AR,
        "[EN]": EN,
        "[SYS]": SYS,
        "[USER]": USER,
        "[ASST]": ASST,
    }

    def __init__(self, tokenizer: backingTokenizer) -> None:
        self._tokenizer = tokenizer

    @classmethod
    def from_file(cls, path: Path = DEFAULT_TOKENIZER_PATH) -> "Tokenizer":
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {path}")

        tokenizer = backingTokenizer.from_file(str(path))
        return cls(tokenizer)

    def normalize(self, text: str) -> str:
        return self._tokenizer.normalizer.normalize_str(text)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def tokenize(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return self._tokenizer.encode(
            text, add_special_tokens=add_special_tokens
        ).tokens

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()
