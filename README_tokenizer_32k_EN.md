# Kasia — byte-level BPE 32k (Polish)

Byte-level BPE trained on a **cleaned SpeakLeash corpus**, using the HuggingFace `tokenizers` library
(`BpeTrainer`). 256-byte base alphabet at IDs 0–255, **zero `<UNK>`, lossless round-trip**.

There is a single, production artifact: `tokenizer.json` — **32,768 slots** (32,568 BPE tokens + 200 special
tokens). The size is a multiple of 128 (alignment for GPU kernels). Intended use: pretraining
a ~125M Polish model on a single GPU.

## Parameters

| field | value |
|---|---|
| type | byte-level BPE (HF `tokenizers`, `tokenizer.json` v1.0 format) |
| vocab (total) | **32,768** |
| BPE tokens | 32,568 |
| merges | 32,312 |
| base alphabet | 256 bytes at IDs **0–255** |
| special tokens | 200, at IDs **32,568–32,767** |
| normalizer | `null` (NFC performed **in the corpus pipeline**, not in the tokenizer) |
| pre-tokenizer | `Split` (cl100k regex) → `ByteLevel(add_prefix_space=false)` |
| decoder | `ByteLevel` |
| `byte_fallback` / `unk_token` | `false` / `None` — unnecessary with a byte base |
| `dropout` | `None` |

**The byte alphabet at IDs 0–255 is enforced deliberately.** Passing `special_tokens` to `BpeTrainer` shifts the bytes
out of this range; special tokens are therefore injected **after training**, and `sanity_check` includes an assertion
guarding this invariant.

## Pre-tokenization

```
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
```

A **cl100k (GPT-4)**-style regex, with one deliberate decision: **`\p{N}{1,3}`**, i.e. at most 3 digits per
pre-token. Consequences measured on the artifact:

- **0** tokens with ≥4 digits; 412 pure digit strings, all of length 1–3 → `1234567` → `123|45|6|7`.
  Without this limit BPE learns arbitrary tokens like `2020`, and arithmetic becomes a lottery.
- **Case-insensitive** contractions/apostrophes (`(?i:...)`) — no inconsistencies with text written in all caps.
- Leading space attached to the word, **never** as a standalone token between words.
- Whitespace is mergeable — the vocab contains indentation tokens of length **1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 64**
  spaces, so code in the corpus does not explode into sequences of single spaces.
- Merges do not cross a newline in the other direction: `\n\n` is a single token.

## Special tokens (200, IDs 32,568–32,767)

14 named + 186 reserved slots (`<|reserved_0|>` … `<|reserved_185|>`):

| group | tokens |
|---|---|
| document | `<\|endoftext\|>`, `<\|begin_of_text\|>` |
| chat | `<\|im_start\|>`, `<\|im_end\|>`, `<\|system\|>`, `<\|user\|>`, `<\|assistant\|>` |
| FIM | `<\|fim_prefix\|>`, `<\|fim_middle\|>`, `<\|fim_suffix\|>` |
| tools | `<\|tool_call\|>`, `<\|tool_result\|>` |
| technical | `<\|pad\|>`, `<\|unk\|>` |

The block is reserved **up front** to avoid resizing the embedding matrix midway through the model's life.
All of them have `normalized: false` and `special: true` — the encoder will not produce them from raw user
text.

## Corpus

- **Source:** SpeakLeash, 23 datasets, **~2.40 GB** of text after cleaning.
- **Pipeline:** `pl_clean.py` (`pl-clean-v3`) — deterministic and versioned, intended to be run
  **identically** on the model's training data. Steps: NFC normalization → control-character removal →
  per-document language filter → line-level boilerplate filtering (with a minimum line-length threshold so as
  not to cut short headings such as "Wstęp" ["Introduction"]) → encoding-corruption detector → cross-deduplication
  against the held-out set.
- **Held-out:** ~9.5M characters, 2,125 documents — several dozen literary works + the Constitution of the Republic of Poland.
  Five SpeakLeash datasets were **excluded from training** to eliminate contamination.

## Results

- **Lossless round-trip** on all 2,125 held-out documents.
- **Vocab utilization: 98.4%** on the held-out (a thin tail of dead tokens → lower risk of glitch tokens).
- Token length distribution (bytes): mode in the 4–6 B range, tail up to 20 B for text; above that, only
  indentation tokens (24/32/64 B).

Control comparison against the class-project byte-level BPE 6,756 (no pre-tokenization, corpus ~9.4M characters:
2 books + an excerpt of the Constitution), on a shared held-out of 725 words / 4,978 characters across 6 registers
(prose, colloquial, journalism, legal, scientific, numbers+code):

| tokenizer | vocab | tok/word | tok/1000 chars | chars/tok |
|---|---:|---:|---:|---:|
| **this one (32k)** | 32,768 | **1.665** | **242.5** | **4.12** |
| class-project BPE 6.7k | 6,756 | 2.338 | 340.5 | 2.94 |

Difference: **−28.8% tokens** on the same text. Spread per register: from −13% (prose) to −40%
(scientific text) — the small tokenizer looks decent only on a domain close to its own corpus.

## Honest caveats

- The table above is a **small ad hoc held-out** (725 words) with the word definition `\S+`. It shows magnitude
  and trend; it is not a publishable measurement — a 1:1 comparison with another tokenizer requires a shared
  held-out and a shared word definition.
- Fertility is a **distributional** metric: it measures compression, not grammaticality or alignment with morpheme
  boundaries. For an inflected language, good tokens are often whole frequent inflected forms — morphologically
  "ugly", statistically worthwhile.
- **Model** quality (perplexity, downstream) is a separate axis and is not measured here.
- `normalizer: null` in the file **does not mean "no normalization"** — NFC is in the corpus pipeline. If the model's
  training data goes through a different preprocessing path, the mismatch will produce glitch tokens.

## Format and loading

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer.json")
ids = tok.encode("Zażółć gęślą jaźń.").ids
assert tok.decode(ids) == "Zażółć gęślą jaźń."
```

The file is in native HF format, so it also works via `PreTrainedTokenizerFast`:

```python
from transformers import PreTrainedTokenizerFast
tok = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
```

## Author

Kasia — Tokenizer Course, Slayer Labs.
