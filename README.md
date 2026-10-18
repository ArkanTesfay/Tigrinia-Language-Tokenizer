<p align="center">
  <img src="SVG/Asset%201.svg" width="400" alt="Eritrean Tigrinya Tokenizer logo"/>
</p>

<h1 align="center">Eritrean Tigrinya Tokenizer (ትግርኛ)</h1>

A subword **Byte-Pair-Encoding (BPE)** tokenizer for **Eritrean Tigrinya** (ትግርኛ),
written in the Ge'ez script.

Built on top of HuggingFace [`tokenizers`](https://github.com/huggingface/tokenizers)
with **Eritrean Tigrinya orthographic conventions** baked in:

- **Keeps the glottal series** (አ ኡ ኢ ኣ ኤ እ ኦ) **distinct from the pharyngeal
  series** (ዐ ዑ ዒ ዓ ዔ ዕ ዖ), as in the [Tigrinya syllabary](https://en.wikipedia.org/wiki/Tigrinya_language#Writing_system).
  The first-order አ and ዐ are the redundant forms; everyday spelling uses ኣ and ዓ.
- **Uses the consonants in that chart.** ሀ and ሐ stay distinct. The working
  *s* is ሰ and the working *ṣ* is ጸ.
- Unicode NFC normalization tuned for Ge'ez text.
- Pre-tokenization that respects Ge'ez syllable boundaries and Ge'ez
  punctuation (`። ፤ ፥ ፦ ፧ ፨`).
- Coverage of the Ge'ez ranges Eritrean Tigrinya uses:
  - `U+1200–U+137F`
  - `U+1380–U+139F`
  - `U+2D80–U+2DDF`
  - `U+AB00–U+AB2F`
- Mixed-script handling (Eritrean Tigrinya text commonly contains Latin
  loanwords and Arabic numerals).

The merge options (`--merge-a-orders`, `--merge-consonants`) are **off by
default** and exist only as a fallback for very noisy corpora. Leave them off
for a proper Eritrean Tigrinya tokenizer.

## Installation

```bash
pip install -e .
```

## Quick start

### Train

```bash
# Single file
tigrinya-train --corpus /path/to/tigrinya.txt --vocab-size 32000 --out tokenizer.json

# Directory of .txt files
tigrinya-train --corpus /path/to/corpus_dir/ --vocab-size 32000 --out tokenizer.json
```

### Use

```bash
tigrinya-encode --tokenizer tokenizer.json --text "ሰላም ኣብ ዓለም"
tigrinya-decode --tokenizer tokenizer.json --ids 342 1208 55 12 880
tigrinya-stats  --tokenizer tokenizer.json --corpus /path/to/tigrinya.txt
```

### As a library

```python
from tigrinya_tokenizer import TigrinyaTokenizer

tok = TigrinyaTokenizer.train(
    corpus_path="data/tigrinya.txt",
    vocab_size=32000,
    output_path="tokenizer.json",
)

enc = tok.encode("ሰላም ኣብ ዓለም")
print(enc.ids, enc.tokens)

dec = tok.decode(enc.ids)
print(dec)
```

## Web playground

An interactive visualizer: type Tigrinya in the browser and see it split
into colored BPE tokens with their IDs, plus live stats (token count,
bytes, bytes/token, vocab size).

```bash
python web/server.py
# -> opens http://localhost:8000
```

No web framework required — it uses only the Python standard library plus
the already-installed `tokenizers` package. The server loads the committed
`tokenizer.json`; set `TOKENIZER_PATH` to point it at a different one.

## Project layout

```
src/tigrinya_tokenizer/
├── __init__.py        # public API
├── normalizer.py      # Eritrean Tigrinya normalization (a-order + consonants)
├── pretokenizer.py    # Ge'ez word + punctuation + numeral pre-tokenization
├── trainer.py         # BPE training pipeline (serializable)
├── tokenizer.py       # load / encode / decode wrapper
└── cli.py             # command-line entry points
tests/
└── test_tokenizer.py
```

## Eritrean Tigrinya orthographic notes

- Eritrean Tigrinya is written in the Ge'ez script, an abugida where each
  consonant has 7 syllabic orders (በ ቡ ቢ ባ ቤ ብ ቦ).
- The sentence terminator `።` (U+1362) is treated as a sentence boundary.
- The syllabary is the one on the [Tigrinya Wikipedia page](https://en.wikipedia.org/wiki/Tigrinya_language#Writing_system): seven orders ä u i a e ə o, including ሀ/ሐ, ሸ, ቐ, ኸ, and both the glottal and pharyngeal rows.
- Glottal and pharyngeal stay distinct (`ኣባይ` vs `ዓባይ`). `--merge-a-orders` collapses them.
- `--merge-consonants` can collapse mixed consonant spellings when a corpus still mixes them. Leave it off for Eritrean Tigrinya.

## Training corpus

The committed `tokenizer.json` was trained on a ~88 MB Tigrinya corpus
assembled from the OPUS parallel corpora (NLLB, CCAligned, tico-19,
wikimedia, translatewiki, Tatoeba) plus a small local seed corpus. Only
the Tigrinya side of each parallel corpus is used.

The full corpus is **not** committed (it's ~88 MB and regenerable). To
rebuild it locally:

```bash
python scripts/build_corpus.py
# -> writes data/tigrinya_corpus_full.txt
```

Then retrain the tokenizer on it:

```bash
tigrinya-train --corpus data/tigrinya_corpus_full.txt --vocab-size 32000 --out tokenizer.json
```

Reference quality (32k vocab, held-out 10k lines):

| Metric | Value |
|---|---|
| Compression | ~7.9 bytes/token |
| Fertility | ~13.6 tokens/line |
| `<unk>` rate | 0.00% |
| Full-corpus tokens | ~11.1M |

## License

MIT
