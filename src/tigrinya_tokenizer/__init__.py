"""Tigrinya (ትግርኛ) subword tokenizer.

A BPE tokenizer tuned for the Eritrean Tigrinya language, which uses the
Ge'ez script. See README.md for details.
"""

from .normalizer import TigrinyaNormalizer
from .pretokenizer import TigrinyaPreTokenizer
from .tokenizer import TigrinyaTokenizer
from .trainer import TigrinyaTrainer

__all__ = [
    "TigrinyaNormalizer",
    "TigrinyaPreTokenizer",
    "TigrinyaTokenizer",
    "TigrinyaTrainer",
]

__version__ = "0.1.0"
