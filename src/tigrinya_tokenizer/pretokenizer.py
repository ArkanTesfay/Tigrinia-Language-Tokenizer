"""Tigrinya-aware pre-tokenization.

Pre-tokens are **words** (whitespace-separated chunks) so BPE can merge Ge'ez
syllables into subwords. Punctuation and script boundaries split off into
their own pre-tokens. Ge'ez numerals are grouped.
"""

from __future__ import annotations

import unicodedata
from typing import List, Tuple

ETHIOPIC_RANGES = [
    (0x1200, 0x137F),
    (0x1380, 0x139F),
    (0x2D80, 0x2DDF),
    (0xAB00, 0xAB2F),
]

ETHIOPIC_PUNCT = {0x1361, 0x1362, 0x1363, 0x1364, 0x1365, 0x1366, 0x1367, 0x1368}
ETHIOPIC_DIGITS = set(range(0x1369, 0x137D))

# Regex pattern for HF `Split` pre-tokenizer (Rust regex syntax).
# Matches the pre-tokens to KEEP (used with invert=True, behavior="removed"
# so non-matched whitespace is discarded). Order matters:
#   1. Ge'ez numerals (grouped)
#   2. Ge'ez word chars (syllables, excluding punctuation and digits)
#   3. ASCII alphanumeric runs (Latin loanwords / numbers)
#   4. Any other single non-space char (catches punctuation individually)
TIGRINYA_PRETOKENIZER_PATTERN = (
    r"[\u1369-\u137C]+"
    r"|[\u1200-\u1360\u137D-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]+"
    r"|[A-Za-z0-9]+"
    r"|[^\s]"
)


def is_ethiopic(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in ETHIOPIC_RANGES)


def is_ethiopic_punct(cp: int) -> bool:
    return cp in ETHIOPIC_PUNCT


def is_ethiopic_digit(cp: int) -> bool:
    return cp in ETHIOPIC_DIGITS


def _is_ascii_alnum(cp: int) -> bool:
    ch = chr(cp)
    return cp < 128 and (ch.isalpha() or ch.isdigit())


def _is_punct(cp: int) -> bool:
    cat = unicodedata.category(chr(cp))
    return cat.startswith("P") or cat.startswith("S")


def split_word(word: str) -> List[str]:
    """Split a whitespace-free word into pre-tokens.

    Consecutive Ge'ez syllables form one pre-token (a "word" for BPE).
    Ge'ez digits are grouped. Latin alnum runs are grouped. Punctuation
    is split off individually. Script boundaries start a new pre-token.
    """
    if not word:
        return []

    out: List[str] = []
    buf: List[str] = []
    cur: str | None = None  # "ethiopic" | "latin" | "digit"

    def flush() -> None:
        if buf:
            out.append("".join(buf))
            buf.clear()

    for ch in word:
        cp = ord(ch)

        if is_ethiopic_punct(cp):
            flush(); out.append(ch); cur = None; continue

        if is_ethiopic_digit(cp):
            if cur != "digit":
                flush(); cur = "digit"
            buf.append(ch); continue

        if _is_punct(cp):
            flush(); out.append(ch); cur = None; continue

        if _is_ascii_alnum(cp):
            if cur != "latin":
                flush(); cur = "latin"
            buf.append(ch); continue

        if is_ethiopic(cp):
            if cur not in ("ethiopic", None):
                flush()
            cur = "ethiopic"
            buf.append(ch); continue

        # Any other character: treat as its own pre-token.
        flush(); out.append(ch); cur = None

    flush()
    return out


def pretokenize(text: str) -> List[str]:
    """Flat list of pre-tokens for the whole text (whitespace is a boundary)."""
    out: List[str] = []
    for ws_chunk in text.split():
        out.extend(split_word(ws_chunk))
    return out


def _split_with_offsets(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    """Return (token, (start, end)) pairs with offsets into `text`."""
    result: List[Tuple[str, Tuple[int, int]]] = []
    cursor = 0
    for ws_chunk in text.split():
        start = text.find(ws_chunk, cursor)
        if start == -1:
            start = cursor
        local = start
        for pretok in split_word(ws_chunk):
            result.append((pretok, (local, local + len(pretok))))
            local += len(pretok)
        cursor = local
    return result


class TigrinyaPreTokenizer:
    """HF-compatible custom pre-tokenizer.

    Usage::

        from tokenizers import Tokenizer
        from tokenizers.pre_tokenizers import PreTokenizer
        tok = Tokenizer(...)
        tok.pre_tokenizer = PreTokenizer.custom(TigrinyaPreTokenizer())
    """

    def pre_tokenize(self, pretokenized_string) -> None:
        """Called by HF with a PreTokenizedString; we must call .split(fn).

        The split callback receives (index, NormalizedString) and must return
        a list of NormalizedString slices. HF tracks offsets from the slices.
        """
        def split_fn(idx, normalized_string):
            text = str(normalized_string)
            return [normalized_string[s:e] for _, (s, e) in _split_with_offsets(text)]
        pretokenized_string.split(split_fn)

    # Direct Python use (tests, library).
    def __call__(self, text: str) -> List[Tuple[str, Tuple[int, int]]]:
        return _split_with_offsets(text)
