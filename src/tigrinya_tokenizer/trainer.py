"""BPE training pipeline for Tigrinya.

Wraps HuggingFace `tokenizers` with:
- A Tigrinya normalizer (see `normalizer.py`)
- A Tigrinya pre-tokenizer (see `pretokenizer.py`)
- A BPE trainer with a special-tokens set appropriate for downstream LM use
- A post-processor that adds [BOS]/[EOS] when configured
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from .normalizer import NormalizerConfig, TigrinyaNormalizer
from .pretokenizer import TigrinyaPreTokenizer, TIGRINYA_PRETOKENIZER_PATTERN


@dataclass
class TrainerConfig:
    vocab_size: int = 32_000
    min_frequency: int = 2
    special_tokens: List[str] = field(
        default_factory=lambda: [
            "<pad>",
            "<unk>",
            "<bos>",
            "<eos>",
            "<mask>",
        ]
    )
    # Whether to add [BOS]/[EOS] automatically during encoding.
    add_bos_eos: bool = False
    # Whether to use a byte-level BPE (False = char-level on Ge'ez syllables,
    # which is what we want for Tigrinya).
    byte_level: bool = False
    # Show progress bar during training.
    show_progress: bool = True
    # Number of parallel processes for training.
    num_processes: Optional[int] = None
    # Initial alphabet is the set of characters guaranteed to be in the vocab.
    # For Tigrinya we include the Ge'ez syllabary + ASCII letters/digits +
    # Ge'ez punctuation so every Tigrinya word is representable.
    initial_alphabet: Optional[List[str]] = None


def _default_initial_alphabet() -> List[str]:
    """All Ge'ez syllables + ASCII alphanumerics + Ge'ez punctuation."""
    alphabet: List[str] = []
    # Ge'ez syllables: U+1200..U+137F, plus the supplement and extended ranges.
    for cp in range(0x1200, 0x1380):
        alphabet.append(chr(cp))
    for cp in range(0x1380, 0x13A0):
        alphabet.append(chr(cp))
    for cp in range(0x2D80, 0x2DE0):
        alphabet.append(chr(cp))
    # Extended Ge'ez range used by some Tigrinya writing.
    for cp in range(0xAB00, 0xAB30):
        alphabet.append(chr(cp))
    # Ge'ez punctuation.
    for cp in (0x1361, 0x1362, 0x1363, 0x1364, 0x1365, 0x1366, 0x1367, 0x1368):
        alphabet.append(chr(cp))
    # Ge'ez numerals.
    for cp in range(0x1369, 0x137D):
        alphabet.append(chr(cp))
    # ASCII letters + digits + common punctuation.
    for cp in range(0x20, 0x7F):
        alphabet.append(chr(cp))
    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for ch in alphabet:
        if ch not in seen:
            seen.add(ch)
            unique.append(ch)
    return unique


class TigrinyaTrainer:
    """Train a BPE tokenizer on a Tigrinya corpus."""

    def __init__(
        self,
        trainer_config: Optional[TrainerConfig] = None,
        normalizer_config: Optional[NormalizerConfig] = None,
    ) -> None:
        self.trainer_cfg = trainer_config or TrainerConfig()
        self.normalizer_cfg = normalizer_config or NormalizerConfig()
        self._normalizer = TigrinyaNormalizer(self.normalizer_cfg)
        self._pretokenizer = TigrinyaPreTokenizer()

    def build_tokenizer(self) -> Tokenizer:
        """Construct a fresh (untrained) Tokenizer wired for Tigrinya."""
        cfg = self.trainer_cfg

        if cfg.byte_level:
            model = BPE(unk_token="<unk>")
            tokenizer = Tokenizer(model)
            tokenizer.normalizer = normalizers.Sequence(
                [normalizers.NFC(), normalizers.Lowercase()]
            )
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
                add_prefix_space=False
            )
            tokenizer.decoder = decoders.ByteLevel()
        else:
            model = BPE(unk_token="<unk>")
            tokenizer = Tokenizer(model)
            # We apply our Tigrinya normalization in the training-data iterator
            # (and in the encode path) rather than via a custom HF normalizer,
            # so the tokenizer stays fully serializable. Pre-tokenization is
            # done with a Rust-native regex `Split` that keeps each Ge'ez word
            # as a single pre-token (so BPE can merge syllables into subwords),
            # splits off Ge'ez and ASCII punctuation, and groups Ge'ez
            # numerals.
            tokenizer.pre_tokenizer = pre_tokenizers.Split(
                Regex(TIGRINYA_PRETOKENIZER_PATTERN),
                behavior="removed",
                invert=True,
            )
            tokenizer.decoder = decoders.BPEDecoder()

        return tokenizer

    def train_from_files(
        self,
        files: Iterable[str | os.PathLike],
        output_path: str | os.PathLike,
    ) -> Tokenizer:
        """Train on a list of text files. Each file is read line-by-line."""
        files = [Path(f) for f in files]
        tokenizer = self.build_tokenizer()

        initial_alphabet = self.trainer_cfg.initial_alphabet
        if initial_alphabet is None:
            initial_alphabet = _default_initial_alphabet()

        trainer = BpeTrainer(
            vocab_size=self.trainer_cfg.vocab_size,
            min_frequency=self.trainer_cfg.min_frequency,
            special_tokens=self.trainer_cfg.special_tokens,
            initial_alphabet=initial_alphabet,
            show_progress=self.trainer_cfg.show_progress,
        )

        # We normalize the corpus on the fly so the BPE trainer sees clean text.
        def gen() -> Iterator[str]:
            for f in files:
                with open(f, "r", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        norm = self._normalizer.normalize(line)
                        if norm:
                            yield norm

        tokenizer.train_from_iterator(
            gen(),
            trainer=trainer,
            length=sum(_line_count(f) for f in files) or None,
        )

        # Optionally wire up a post-processor for [BOS]/[EOS].
        if self.trainer_cfg.add_bos_eos:
            from tokenizers.processors import TemplateProcessing

            bos_id = tokenizer.token_to_id("<bos>")
            eos_id = tokenizer.token_to_id("<eos>")
            if bos_id is not None and eos_id is not None:
                tokenizer.post_processor = TemplateProcessing(
                    single="<bos> $A <eos>",
                    special_tokens=[("<bos>", bos_id), ("<eos>", eos_id)],
                )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(out))
        return tokenizer

    def train_from_dir(
        self,
        corpus_dir: str | os.PathLike,
        output_path: str | os.PathLike,
        extensions: tuple[str, ...] = (".txt", ".tig", ".md"),
    ) -> Tokenizer:
        """Train on every file under `corpus_dir` with the given extensions."""
        root = Path(corpus_dir)
        if not root.exists():
            raise FileNotFoundError(f"Corpus directory not found: {root}")
        files = [
            p
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix.lower() in extensions
        ]
        if not files:
            raise ValueError(
                f"No files with extensions {extensions} found under {root}"
            )
        return self.train_from_files(files, output_path)

    def train(
        self,
        corpus_path: str | os.PathLike,
        output_path: str | os.PathLike,
    ) -> Tokenizer:
        """Convenience: train on either a file or a directory."""
        p = Path(corpus_path)
        if not p.exists():
            raise FileNotFoundError(f"Corpus not found: {p}")
        if p.is_dir():
            return self.train_from_dir(p, output_path)
        return self.train_from_files([p], output_path)


def _line_count(path: Path) -> int:
    try:
        with open(path, "rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0
