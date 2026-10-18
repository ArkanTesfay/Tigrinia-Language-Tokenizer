"""High-level Tigrinya tokenizer wrapper.

Loads a trained `tokenizer.json` and provides a friendly Python API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from tokenizers import Encoding, Tokenizer

from .normalizer import NormalizerConfig, TigrinyaNormalizer
from .trainer import TigrinyaTrainer, TrainerConfig


@dataclass
class EncodeResult:
    ids: List[int]
    tokens: List[str]
    attention_mask: List[int]

    @classmethod
    def from_encoding(cls, enc: Encoding) -> "EncodeResult":
        return cls(
            ids=list(enc.ids),
            tokens=list(enc.tokens),
            attention_mask=list(enc.attention_mask),
        )


class TigrinyaTokenizer:
    """A trained Tigrinya BPE tokenizer.

    Either load an existing `tokenizer.json` via `TigrinyaTokenizer.load(path)`
    or train a new one via `TigrinyaTokenizer.train(corpus, vocab_size, out)`.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        normalizer: Optional[TigrinyaNormalizer] = None,
    ) -> None:
        self._tok = tokenizer
        self._normalizer = normalizer or TigrinyaNormalizer()

    # ---- construction -------------------------------------------------

    @classmethod
    def load(
        cls,
        path: Union[str, os.PathLike],
        normalizer_config: Optional[NormalizerConfig] = None,
    ) -> "TigrinyaTokenizer":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {p}")
        tok = Tokenizer.from_file(str(p))
        return cls(tok, TigrinyaNormalizer(normalizer_config))

    @classmethod
    def train(
        cls,
        corpus_path: Union[str, os.PathLike],
        vocab_size: int = 32_000,
        output_path: Optional[Union[str, os.PathLike]] = None,
        trainer_config: Optional[TrainerConfig] = None,
        normalizer_config: Optional[NormalizerConfig] = None,
    ) -> "TigrinyaTokenizer":
        cfg = trainer_config or TrainerConfig(vocab_size=vocab_size)
        if cfg.vocab_size != vocab_size:
            cfg.vocab_size = vocab_size
        trainer = TigrinyaTrainer(cfg, normalizer_config)
        if output_path is None:
            output_path = "tokenizer.json"
        tok = trainer.train(corpus_path, output_path)
        return cls(tok, TigrinyaNormalizer(normalizer_config))

    # ---- properties ---------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def get_vocab(self) -> dict:
        return self._tok.get_vocab()

    def id_to_token(self, idx: int) -> Optional[str]:
        return self._tok.id_to_token(idx)

    def token_to_id(self, token: str) -> Optional[int]:
        return self._tok.token_to_id(token)

    # ---- encode / decode ---------------------------------------------

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: Optional[bool] = None,
    ) -> Union[EncodeResult, List[EncodeResult]]:
        if isinstance(text, str):
            text = self._normalizer.normalize(text)
            if add_special_tokens is None:
                add_special_tokens = True
            enc = self._tok.encode(text, add_special_tokens=add_special_tokens)
            return EncodeResult.from_encoding(enc)
        # Batch.
        if add_special_tokens is None:
            add_special_tokens = True
        normed = [self._normalizer.normalize(t) for t in text]
        encs = self._tok.encode_batch(
            normed, add_special_tokens=add_special_tokens
        )
        return [EncodeResult.from_encoding(e) for e in encs]

    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
    ) -> List[EncodeResult]:
        normed = [self._normalizer.normalize(t) for t in texts]
        return [
            EncodeResult.from_encoding(e)
            for e in self._tok.encode_batch(
                normed, add_special_tokens=add_special_tokens
            )
        ]

    def decode(
        self,
        ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = True,
    ) -> Union[str, List[str]]:
        if ids and isinstance(ids[0], list):
            return self._tok.decode_batch(
                ids, skip_special_tokens=skip_special_tokens
            )
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    # ---- save ---------------------------------------------------------

    def save(self, path: Union[str, os.PathLike]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(p))

    # ---- introspection -----------------------------------------------

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"TigrinyaTokenizer(vocab_size={self.vocab_size})"
