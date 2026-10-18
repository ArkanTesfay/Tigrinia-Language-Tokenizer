"""Command-line interface for the Tigrinya tokenizer.

Entry points (defined in pyproject.toml):

- `tigrinya-train`   : train a BPE tokenizer on a corpus
- `tigrinya-encode`  : encode text -> token ids
- `tigrinya-decode`  : decode ids -> text
- `tigrinya-stats`   : compute compression / coverage stats on a corpus
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import click

from .normalizer import NormalizerConfig
from .tokenizer import TigrinyaTokenizer
from .trainer import TrainerConfig


# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--corpus",
    "corpus",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to a Tigrinya text file or a directory of text files.",
)
@click.option(
    "--vocab-size",
    default=32000,
    show_default=True,
    type=int,
    help="BPE vocabulary size.",
)
@click.option(
    "--out",
    "output",
    default="tokenizer.json",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Where to save the trained tokenizer.json.",
)
@click.option(
    "--min-frequency",
    default=2,
    show_default=True,
    type=int,
    help="Minimum frequency for a pair to be considered for a merge.",
)
@click.option(
    "--byte-level",
    is_flag=True,
    default=False,
    help="Use byte-level BPE (default: char-level on Ge'ez syllables).",
)
@click.option(
    "--add-bos-eos",
    is_flag=True,
    default=False,
    help="Add [BOS]/[EOS] automatically during encoding.",
)
@click.option(
    "--merge-a-orders",
    is_flag=True,
    default=False,
    help=(
        "Collapse the አ/ዐ a-order pairs. "
        "OFF by default so Eritrean Tigrinya keeps the distinction."
    ),
)
@click.option(
    "--merge-consonants",
    is_flag=True,
    default=False,
    help=(
        "Collapse mixed consonant spellings. OFF by default so Eritrean "
        "Tigrinya keeps them distinct."
    ),
)
@click.option(
    "--normalize-digits",
    is_flag=True,
    default=False,
    help="Convert ASCII digits 1-9 to Ge'ez numerals ፩..፱ (0 is kept).",
)
def train(
    corpus: Path,
    vocab_size: int,
    output: Path,
    min_frequency: int,
    byte_level: bool,
    add_bos_eos: bool,
    merge_a_orders: bool,
    merge_consonants: bool,
    normalize_digits: bool,
) -> None:
    """Train an Eritrean Tigrinya BPE tokenizer."""
    trainer_cfg = TrainerConfig(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        byte_level=byte_level,
        add_bos_eos=add_bos_eos,
    )
    norm_cfg = NormalizerConfig(
        merge_a_orders=merge_a_orders,
        merge_consonants=merge_consonants,
        normalize_digits=normalize_digits,
    )
    tok = TigrinyaTokenizer.train(
        corpus_path=corpus,
        vocab_size=vocab_size,
        output_path=output,
        trainer_config=trainer_cfg,
        normalizer_config=norm_cfg,
    )
    click.echo(
        f"Trained Eritrean Tigrinya tokenizer: vocab_size={tok.vocab_size} -> {output}"
    )


# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--tokenizer",
    "tokenizer_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option("--text", "text", default=None, help="Text to encode.")
@click.option(
    "--file",
    "file",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="File to encode (one sentence per line).",
)
@click.option(
    "--no-special-tokens",
    is_flag=True,
    default=False,
    help="Do not add [BOS]/[EOS] (if configured).",
)
def encode(
    tokenizer_path: Path,
    text: Optional[str],
    file: Optional[Path],
    no_special_tokens: bool,
) -> None:
    """Encode Tigrinya text to token ids."""
    tok = TigrinyaTokenizer.load(tokenizer_path)
    add_special = not no_special_tokens

    if text:
        res = tok.encode(text, add_special_tokens=add_special)
        click.echo(json.dumps({"ids": res.ids, "tokens": res.tokens}, ensure_ascii=False))
    elif file:
        lines = [ln.rstrip("\n") for ln in file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        results = tok.encode_batch(lines, add_special_tokens=add_special)
        out = [{"text": ln, "ids": r.ids, "n_tokens": len(r.ids)} for ln, r in zip(lines, results)]
        total = sum(r["n_tokens"] for r in out)
        click.echo(
            json.dumps(
                {"lines": len(out), "total_tokens": total, "results": out},
                ensure_ascii=False,
            )
        )
    else:
        click.echo("Provide --text or --file.", err=True)
        sys.exit(2)


# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--tokenizer",
    "tokenizer_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--ids",
    "ids",
    required=True,
    type=int,
    multiple=True,
    help="Token ids to decode (repeatable).",
)
@click.option(
    "--keep-special-tokens",
    is_flag=True,
    default=False,
    help="Do not strip special tokens from the output.",
)
def decode(tokenizer_path: Path, ids: List[int], keep_special_tokens: bool) -> None:
    """Decode token ids back to Tigrinya text."""
    tok = TigrinyaTokenizer.load(tokenizer_path)
    text = tok.decode(list(ids), skip_special_tokens=not keep_special_tokens)
    click.echo(text)


# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--tokenizer",
    "tokenizer_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--corpus",
    "corpus",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Corpus file or directory to compute stats over.",
)
def stats(tokenizer_path: Path, corpus: Path) -> None:
    """Compute compression and coverage statistics on a corpus."""
    tok = TigrinyaTokenizer.load(tokenizer_path)

    files: List[Path]
    if corpus.is_dir():
        files = sorted(p for p in corpus.rglob("*") if p.is_file() and p.suffix.lower() in (".txt", ".tig", ".md"))
    else:
        files = [corpus]

    total_chars = 0
    total_tokens = 0
    total_words = 0
    unk_count = 0
    unk_id = tok.token_to_id("<unk>")

    for f in files:
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            total_chars += len(line)
            total_words += len(line.split())
            res = tok.encode(line, add_special_tokens=False)
            total_tokens += len(res.ids)
            if unk_id is not None:
                unk_count += res.ids.count(unk_id)

    if total_tokens == 0:
        click.echo("No text found.")
        return

    chars_per_token = total_chars / total_tokens
    tokens_per_word = total_tokens / max(total_words, 1)
    unk_rate = unk_count / total_tokens

    click.echo(
        json.dumps(
            {
                "files": len(files),
                "total_chars": total_chars,
                "total_words": total_words,
                "total_tokens": total_tokens,
                "chars_per_token": round(chars_per_token, 3),
                "tokens_per_word": round(tokens_per_word, 3),
                "unk_rate": round(unk_rate, 5),
                "vocab_size": tok.vocab_size,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
