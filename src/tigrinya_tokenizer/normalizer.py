"""Eritrean Tigrinya (ትግርኛ) text normalization.

The working syllabary is the one documented for Eritrean Tigrinya
(https://en.wikipedia.org/wiki/Tigrinya_language#Writing_system):
ሀ and ሐ stay distinct, the glottal and pharyngeal rows stay distinct,
and the *s* / *ṣ* rows in use are ሰ and ጸ.

This normalizer does not rewrite any of those letters unless a merge option
is turned on. That way older spellings in a corpus are not destroyed.

Optional merges (off = leave the text alone):

1. **a-order pairs** — the glottal (አ) series vs the pharyngeal (ዐ) series.
   Both rows are in the Tigrinya syllabary:
     አ U+12A0  /  ዐ U+12D0     (GLOTTAL A   / PHARYNGEAL A)
     ኡ U+12A1  /  ዑ U+12D1
     ኢ U+12A2  /  ዒ U+12D2
     ኣ U+12A3  /  ዓ U+12D3
     ኤ U+12A4  /  ዔ U+12D4
     እ U+12A5  /  ዕ U+12D5
     ኦ U+12A6  /  ዖ U+12D6

2. **Consonant pairs** Eritrean spelling keeps distinct, including
   ሀ U+1200 / ሐ U+1210.

Other normalization:
- Unicode NFC composition (combining marks are common in OCR / web text).
- Whitespace canonicalization (Ge'ez punctuation is NOT turned into space).
- Stripping of zero-width characters (ZWJ, ZWNJ, BOM).
- Optional normalization of Latin digits to Ge'ez numerals (off by default).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

# Ge'ez punctuation that should be preserved as-is (not turned to ASCII).
ETHIOPIC_PUNCT = {
    "\u1361",  # ፡ word space
    "\u1362",  # ። full stop
    "\u1363",  # ፣ comma
    "\u1364",  # ፤ semicolon
    "\u1365",  # ፥ colon
    "\u1366",  # ፦ preface colon
    "\u1367",  # ፧ question mark
    "\u1368",  # ፨ paragraph separator
}

# Glottal (አ-series) -> Pharyngeal (ዐ-series) pairs.
# Eritrean Tigrinya keeps both rows.
A_ORDER_PAIRS = [
    ("\u12A0", "\u12D0"),  # አ / ዐ
    ("\u12A1", "\u12D1"),  # ኡ / ዑ
    ("\u12A2", "\u12D2"),  # ኢ / ዒ
    ("\u12A3", "\u12D3"),  # ኣ / ዓ
    ("\u12A4", "\u12D4"),  # ኤ / ዔ
    ("\u12A5", "\u12D5"),  # እ / ዕ
    ("\u12A6", "\u12D6"),  # ኦ / ዖ
]

# Historical consonant pairs that Eritrean Tigrinya keeps distinct.
# (left = the "plain" form, right = the historically distinct form)
CONSONANT_PAIRS = [
    ("\u1200", "\u1210"),  # ሀ (HA)  / ሐ (HHA)   — h / ḥ
    ("\u1230", "\u1220"),  # ሰ / U+1220
    ("\u1338", "\u1340"),  # ጸ / U+1340
]

# Zero-width / format characters we strip outright.
_STRIP_CHARS = {
    "\u200B",  # zero-width space
    "\u200C",  # zero-width non-joiner
    "\u200D",  # zero-width joiner
    "\uFEFF",  # BOM / zero-width no-break space
}

# Latin digit -> Ge'ez numeral. Ge'ez has no zero, so ASCII "0" is kept
# as-is. Used only when normalize_digits=True.
_LATIN_TO_ETHIOPIC_DIGIT = {
    "1": "\u1369",  # ፩ (one)
    "2": "\u136A",  # ፪ (two)
    "3": "\u136B",  # ፫ (three)
    "4": "\u136C",  # ፬ (four)
    "5": "\u136D",  # ፭ (five)
    "6": "\u136E",  # ፮ (six)
    "7": "\u136F",  # ፯ (seven)
    "8": "\u1370",  # ፰ (eight)
    "9": "\u1371",  # ፱ (nine)
}


@dataclass
class NormalizerConfig:
    """Configuration for Eritrean Tigrinya normalization.

    By default no letters are rewritten. Merge options are for noisy corpora
    and are off.
    """
    nfc: bool = True
    strip_zero_width: bool = True
    collapse_whitespace: bool = True

    # a-order (አ/ዐ) handling. OFF = Eritrean (preserve distinction).
    merge_a_orders: bool = False
    # When merging, which series to keep: "pharyngeal" (ዐ) or "glottal" (አ).
    a_order_canonical: str = "pharyngeal"

    # Mixed consonant spellings. Off leaves them as written.
    merge_consonants: bool = False
    # When merging consonants, which form to keep: "plain" or "marked".
    consonant_canonical: str = "plain"

    normalize_digits: bool = False  # convert ASCII 1-9 -> ፩..፱ (0 kept)
    lowercase_latin: bool = True


class TigrinyaNormalizer:
    """Normalize Eritrean Tigrinya text prior to pre-tokenization and BPE."""

    def __init__(self, config: NormalizerConfig | None = None) -> None:
        self.cfg = config or NormalizerConfig()

    def normalize(self, text: str) -> str:
        cfg = self.cfg

        if cfg.strip_zero_width:
            text = "".join(ch for ch in text if ch not in _STRIP_CHARS)

        if cfg.nfc:
            text = unicodedata.normalize("NFC", text)

        if cfg.merge_a_orders:
            text = self._merge_pairs(text, A_ORDER_PAIRS, cfg.a_order_canonical)

        if cfg.merge_consonants:
            text = self._merge_pairs(
                text, CONSONANT_PAIRS, cfg.consonant_canonical
            )

        if cfg.normalize_digits:
            text = "".join(_LATIN_TO_ETHIOPIC_DIGIT.get(ch, ch) for ch in text)

        if cfg.lowercase_latin:
            # Lowercase ASCII letters only; Ge'ez has no case.
            text = "".join(
                ch.lower() if "A" <= ch <= "Z" else ch for ch in text
            )

        if cfg.collapse_whitespace:
            # Collapse runs of ASCII whitespace to a single space, but preserve
            # Ge'ez punctuation (which is *not* whitespace).
            out_chars: list[str] = []
            prev_ws = False
            for ch in text:
                if ch in ETHIOPIC_PUNCT:
                    out_chars.append(ch)
                    prev_ws = False
                    continue
                if ch.isspace():
                    if not prev_ws:
                        out_chars.append(" ")
                        prev_ws = True
                    continue
                out_chars.append(ch)
                prev_ws = False
            text = "".join(out_chars)

        return text.strip()

    def _merge_pairs(self, text: str, pairs, canonical: str) -> str:
        """Collapse a list of (left, right) pairs according to `canonical`.

        For each pair, `canonical` selects which form survives:
        - "pharyngeal" / "marked"  -> keep the right form
        - "glottal"   / "plain"    -> keep the left form
        Any other value defaults to keeping the left form.
        """
        keep_right = canonical in ("pharyngeal", "marked")
        mapping = {}
        for left, right in pairs:
            keep = right if keep_right else left
            drop = left if keep_right else right
            mapping[drop] = keep
        return "".join(mapping.get(ch, ch) for ch in text)
