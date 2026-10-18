"""Tests for the Tigrinya tokenizer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tigrinya_tokenizer import (
    TigrinyaNormalizer,
    TigrinyaPreTokenizer,
    TigrinyaTokenizer,
)
from tigrinya_tokenizer.normalizer import NormalizerConfig
from tigrinya_tokenizer.trainer import TrainerConfig


# A small sample of Tigrinya sentences (Ge'ez script).
SAMPLE_CORPUS = [
    "ሰላም ኣብ ዓለም።",
    "እዚ ትግርኛ ቋንቋ እዩ።",
    "ትግርኛ ኣብ ኤርትራ ይዕዛሕ።",
    "ኣብ ኤርትራ ካብ ምዕባለ ኣመሓዳሪነት እናተኣማመነ ንነብር።",
    "ቋንቋ ትግርኛ ብግዕዛ ፊደላት ይኽፈት።",
    "ሰብ ኩሉ እናተወልዱ ነፃነትን ክብርን እዩ እናተወልዱ።",
    "ምስ ካልኦት ሰባት ብሕታውነትን ሰብነትን ክሕሉ ይግባእ።",
    "ምስራሕ ኣብ ትግርኛ ቋንቋ ኣገዳሲ እዩ።",
    "እዚ ቴክኖሎጂ ንትግርኛ ክሓግዝ እዩ።",
    "ትግርኛ ቋንቋ ጥሙር እዩ።",
]


@pytest.fixture
def corpus_file(tmp_path: Path) -> Path:
    p = tmp_path / "corpus.txt"
    p.write_text("\n".join(SAMPLE_CORPUS), encoding="utf-8")
    return p


# ----- normalizer ----------------------------------------------------------


def test_normalizer_preserves_ethiopic_punct():
    norm = TigrinyaNormalizer()
    out = norm.normalize("ሰላም ኣብ ዓለም።")
    assert "።" in out  # Ge'ez full stop preserved
    assert out.endswith("።")


def test_normalizer_strips_zero_width():
    norm = TigrinyaNormalizer()
    out = norm.normalize("ሰ\u200Bላ\u200Dም")
    assert "\u200B" not in out
    assert "\u200D" not in out
    assert out == "ሰላም"


def test_normalizer_collapses_whitespace():
    norm = TigrinyaNormalizer()
    out = norm.normalize("ሰላም    ኣብ\t\tዓለም")
    assert "  " not in out
    assert out == "ሰላም ኣብ ዓለም"


def test_normalizer_keeps_a_order_distinction_by_default():
    norm = TigrinyaNormalizer()
    # Eritrean Tigrinya preserves the አ (glottal) / ዐ (pharyngeal) distinction.
    out = norm.normalize("አዐ")
    assert "አ" in out
    assert "ዐ" in out


def test_normalizer_merges_a_orders_to_pharyngeal():
    cfg = NormalizerConfig(merge_a_orders=True, a_order_canonical="pharyngeal")
    norm = TigrinyaNormalizer(cfg)
    # When merging, pharyngeal (ዐ) survives; glottal (አ) is replaced by ዐ.
    out = norm.normalize("አዐ")
    assert "አ" not in out
    assert "ዐ" in out
    assert out == "ዐዐ"


def test_normalizer_merges_a_orders_to_glottal():
    cfg = NormalizerConfig(merge_a_orders=True, a_order_canonical="glottal")
    norm = TigrinyaNormalizer(cfg)
    out = norm.normalize("አዐ")
    assert "ዐ" not in out
    assert "አ" in out
    assert out == "አአ"


def test_normalizer_keeps_consonant_distinctions_by_default():
    # Eritrean Tigrinya keeps these consonant spellings distinct.
    norm = TigrinyaNormalizer()
    for a, b in [("ሀ", "ሐ"), ("ሰ", "ሠ"), ("ጸ", "ፀ")]:
        out = norm.normalize(a + b)
        assert a in out and b in out, f"{a}/{b} merged unexpectedly"


def test_normalizer_merges_consonants_to_plain():
    cfg = NormalizerConfig(merge_consonants=True, consonant_canonical="plain")
    norm = TigrinyaNormalizer(cfg)
    # The plain form of each pair is what survives.
    assert norm.normalize("ሐ") == "ሀ"
    assert norm.normalize("ሠ") == "ሰ"
    assert norm.normalize("ፀ") == "ጸ"


def test_normalizer_lowercases_latin():
    norm = TigrinyaNormalizer()
    out = norm.normalize("Hello WORLD 123")
    assert out == "hello world 123"


def test_normalizer_normalizes_digits_when_configured():
    cfg = NormalizerConfig(normalize_digits=True)
    norm = TigrinyaNormalizer(cfg)
    # Ge'ez ONE is U+1369, so 1->፩, 2->፪, 3->፫, 4->፬. Ge'ez has no zero,
    # so ASCII "0" is preserved.
    out = norm.normalize("1234")
    assert out == "\u1369\u136A\u136B\u136C"  # ፩፪፫፬
    # Zero is preserved (Ge'ez has no zero).
    assert norm.normalize("10") == "\u13690"


# ----- pre-tokenizer -------------------------------------------------------


def test_pretokenizer_keeps_word_as_one_token():
    # BPE merges happen WITHIN pre-tokens, so a word stays one pre-token.
    pt = TigrinyaPreTokenizer()
    res = pt("ሰላም")
    tokens = [t for t, _ in res]
    assert tokens == ["ሰላም"]


def test_pretokenizer_separates_ethiopic_punctuation():
    pt = TigrinyaPreTokenizer()
    res = pt("ሰላም።")
    tokens = [t for t, _ in res]
    assert tokens == ["ሰላም", "።"]


def test_pretokenizer_splits_at_script_boundary():
    pt = TigrinyaPreTokenizer()
    res = pt("ሰላምhello")
    tokens = [t for t, _ in res]
    assert tokens == ["ሰላም", "hello"]


def test_pretokenizer_groups_ethiopic_digits():
    pt = TigrinyaPreTokenizer()
    res = pt("፪፩፪፬")
    tokens = [t for t, _ in res]
    assert tokens == ["፪፩፪፬"]  # grouped as a single number


def test_pretokenizer_offsets_are_valid():
    pt = TigrinyaPreTokenizer()
    text = "ሰላም ።"
    res = pt(text)
    for tok, (s, e) in res:
        assert text[s:e] == tok


# ----- end-to-end train + encode + decode ----------------------------------


def test_train_encode_decode_roundtrip(corpus_file: Path, tmp_path: Path):
    out_path = tmp_path / "tokenizer.json"
    tok = TigrinyaTokenizer.train(
        corpus_path=corpus_file,
        vocab_size=500,
        output_path=out_path,
    )
    assert out_path.exists()
    assert tok.vocab_size > 0

    text = "ሰላም ኣብ ዓለም።"
    enc = tok.encode(text, add_special_tokens=False)
    assert len(enc.ids) > 0
    assert all(isinstance(i, int) for i in enc.ids)

    decoded = tok.decode(enc.ids)
    # Decode should reproduce the normalized input (without special tokens).
    assert "ሰላም" in decoded
    assert "ዓለም" in decoded


def test_train_small_vocab_representable(corpus_file: Path, tmp_path: Path):
    """Even with a tiny vocab, every Tigrinya syllable should be representable
    because we seed the initial alphabet with the Ge'ez syllabary."""
    out_path = tmp_path / "tokenizer_small.json"
    tok = TigrinyaTokenizer.train(
        corpus_path=corpus_file,
        vocab_size=300,
        output_path=out_path,
    )
    enc = tok.encode("ትግርኛ", add_special_tokens=False)
    # No <unk> should appear because all syllables are in the initial alphabet.
    unk_id = tok.token_to_id("<unk>")
    assert unk_id not in enc.ids


def test_load_saved_tokenizer(corpus_file: Path, tmp_path: Path):
    out_path = tmp_path / "tokenizer.json"
    tok = TigrinyaTokenizer.train(
        corpus_path=corpus_file,
        vocab_size=500,
        output_path=out_path,
    )
    reloaded = TigrinyaTokenizer.load(out_path)
    assert reloaded.vocab_size == tok.vocab_size


def test_batch_encode(corpus_file: Path, tmp_path: Path):
    out_path = tmp_path / "tokenizer.json"
    tok = TigrinyaTokenizer.train(
        corpus_path=corpus_file,
        vocab_size=500,
        output_path=out_path,
    )
    results = tok.encode_batch(["ሰላም", "ኣብ ዓለም"], add_special_tokens=False)
    assert len(results) == 2
    assert all(len(r.ids) > 0 for r in results)
