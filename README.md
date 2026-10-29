# Polish Byte-Level BPE Tokenizers: A Course Study (6,756) and a Production Artifact (32,768)

> Two byte-level BPE tokenizers for Polish, released together with the experimental record that led from one to the other.
> **`tokenizer-6756`**: a from-scratch BPE implementation compared with HuggingFace `tokenizers` under controlled conditions.
> **`tokenizer-32k`**: a production tokenizer intended for pretraining a ~125M-parameter Polish language model on a single GPU.

Author: Kasia, Tokenizer 
Documentation structure follows *Model Cards for Model Reporting* [11] and *Datasheets for Datasets* [12].

---

## Table of contents

1. [Summary](#1-summary): includes research problem (1.1), observations (1.2), conclusions (1.3) and their limits (1.4)
2. [Intended use and scope](#2-intended-use-and-scope)
3. [Background and design rationale](#3-background-and-design-rationale)
4. [Artifact A: `tokenizer-32k` (production)](#4-artifact-a-tokenizer-32k-production)
5. [Artifact B: `tokenizer-6756` (course study)](#5-artifact-b-tokenizer-6756-course-study)
6. [Training data (datasheet)](#6-training-data-datasheet)
7. [Evaluation](#7-evaluation)
8. [Failure analysis and lessons](#8-failure-analysis-and-lessons)
9. [Limitations, risks and ethical considerations](#9-limitations-risks-and-ethical-considerations)
10. [Usage](#10-usage)
11. [Reproducibility](#11-reproducibility)
12. [Citation and references](#12-citation-and-references)

---

## 1. Summary

| | `tokenizer-32k` | `tokenizer-6756` |
|---|---|---|
| Purpose | Production, LM pretraining | Methodology study, teaching |
| Algorithm | Byte-level BPE (HF `BpeTrainer`) | Byte-level BPE, custom Python implementation; HF variants as controls |
| Total vocabulary | **32,768** (256 bytes + 32,312 merges + 200 special) | **6,756** (256 bytes + 6,500 merges) |
| Pre-tokenization | cl100k-style regex, digits capped at 3 | None (custom); with/without regex (HF controls) |
| Special tokens | 200 (14 named + 186 reserved) | None |
| Training corpus | SpeakLeash, 23 datasets, ~2.40 GB after cleaning | 3 literary/legal texts, 9,363,020 characters |
| `<UNK>` | None (complete byte base) | None |
| Round-trip | Lossless on 2,125 held-out documents | Lossless on all tested inputs, incl. emoji and CJK |

Headline results:

* On a shared 725-word, six-register held-out, `tokenizer-32k` needs **28.8% fewer tokens** than `tokenizer-6756` (1.665 vs 2.338 tokens/word; 4.12 vs 2.94 chars/token). The gap ranges from −13% on prose to −40% on scientific text.
* In the controlled study, a from-scratch BPE came within **0.9%** of the HuggingFace implementation (836 vs 844 tokens), and pre-tokenization and normalization each moved fertility by under 1%, which is within the noise of a 371-word held-out.

### 1.1 Research problem

The course study (`tokenizer-6756`) was not an attempt to build the best possible tokenizer. It was designed to answer questions that cannot be asked by using an off-the-shelf library alone:

* **RQ1 (implementation validity).** Does a BPE written from scratch reach results comparable to a production library under identical conditions?
* **RQ2 (effect of pre-tokenization).** How large is the real effect of regex pre-tokenization on compression, in particular for text outside the training domain?
* **RQ3 (failure modes).** What errors occur when a tokenizer is built by hand, and what do they reveal about the algorithm?

Starting from a custom implementation, rather than from the library, means the comparison measures one design decision (presence of pre-tokenization) and not the difference between two black boxes.

The production study (`tokenizer-32k`) addresses a practical question: how much does a larger, regex-pre-tokenized, multi-register vocabulary reduce token counts for Polish compared with the small course tokenizer, and does the advantage grow with distance from the training domain?

**Pre-registered expectation.** Before the experiment, pre-tokenization was expected to give a clear improvement in generalization. This was **not confirmed** on the held-out used (see O2).

### 1.2 Summary of observations

| ID | Observation | Evidence | Confidence |
|---|---|---|---|
| O1 | The custom BPE matches the production library (RQ1). | 836 vs 844 tokens (0.9%) in the matching configuration; identical first merge `(197, 130) → 257 = ł` as an independently trained HF tokenizer. | Moderate: single corpus, 371-word held-out |
| O2 | Pre-tokenization improved fertility only marginally (RQ2). | 844 → 840 tokens (~0.5%). | Low: within noise |
| O3 | All five variants lie within a 1.3% range. | 833–844 tokens on 371 words. | Shows that this held-out cannot rank variants |
| O4 | The normalizer's residual effect is small. | 840 → 833 (0.8%) with the variable isolated; corpus had already been manually normalized. | Low to moderate |
| O5 | The boilerplate filter did not change held-out fertility. | 833 tokens in both runs although 155 tokens (2.3%) of the vocabularies differ. | High for this held-out; says nothing about data without PDF footers |
| O6 | Pre-tokenization changes *what* is learned, not how much it compresses. | Without it, tokens such as `,␣bo␣`, `␣się␣` span word boundaries; with it, punctuation is separate and the space is a word prefix. | Qualitative, consistent across the custom and HF runs |
| O7 | A larger vocabulary helps most out of domain (RQ2, production). | 32k vs 6,756: −28.8% tokens overall, from −13% (prose) to −40% (scientific text). | Moderate: ad hoc 725-word held-out, `\S+` word definition |
| O8 | Most serious defects were in the artifact, not the algorithm (RQ3). | Training was correct throughout; a one-line export bug made the vocabulary non-invertible. | High (reproduced and fixed, §8) |

### 1.3 Conclusions

1. **Byte-level guarantees no `<UNK>`.** A complete 256-byte base covers any text, at the cost of 68 entries for non-printable bytes that practically never occur.
2. **A correct algorithm is not sufficient; the artifact must be reversible.** The byte→character map used to serialize the vocabulary must be injective, otherwise no decoder exists.
3. **Pre-tokenization improves boundary quality, not compression.** It had a marginal effect on fertility but changed the learned tokens from phrase-level to reusable.
4. **In-domain compression can mislead.** Boilerplate creates long tokens that improve metrics on data similar to the training set and give nothing beyond it.
5. **Vocabulary size includes the byte base.** 6,756 = 256 + 6,500; analogously GPT-2: 50,257 = 256 + 50,000 + 1.
6. **Small-vocabulary tokenizers look acceptable only near their training domain.** The 6,756 tokenizer's gap to the 32k one is smallest on prose, which resembles its corpus, and largest on scientific text.

### 1.4 What these results do and do not show

The controlled-study differences (≤1.3%) are within the noise of a 371-word held-out, so the study supports O1 and O6 and the failure analysis, but **not** a ranking of the HF variants. The 32k-vs-6,756 comparison shows magnitude and trend, not a publishable measurement, and model-level quality (perplexity, downstream) is not measured. Next steps: an out-of-domain held-out where pre-tokenization should matter more, and robustness tests on numbers, code and text without diacritics.

## 2. Intended use and scope

**Primary use.** `tokenizer-32k` is intended for pretraining and fine-tuning Polish (and Polish-dominant, code-tolerant) causal language models.

**Secondary use.** `tokenizer-6756` and its documentation serve as a worked example of tokenizer construction and debugging: a reproducible comparison between a hand-written and a library BPE.

**Out of scope.**

* Multilingual use. The vocabulary is fitted to Polish; other languages will be heavily over-segmented, a known source of cross-lingual unfairness in tokenizers [8].
* Any claim about downstream *model* quality. Tokenizer metrics here are intrinsic (§7).
* Use of `tokenizer-6756` for production. It was trained on ~9.4M characters and generalizes poorly outside its domain (§7.2).

## 3. Background and design rationale

**Byte-level BPE.** BPE was introduced as a compression scheme by Gage [1] and adapted to subword segmentation for neural MT by Sennrich et al. [2]. Applying it to raw UTF-8 bytes, as in GPT-2 [3], gives a fixed 256-symbol base alphabet: every string is encodable, so no `<UNK>` is needed. Both tokenizers here follow that design and the GPT-2 byte→character mapping.

**Pre-tokenization.** Splitting text with a regex before merging prevents merges across word and punctuation boundaries [3]. `tokenizer-32k` uses a cl100k-style pattern with one deliberate change (§4.2). `tokenizer-6756` was designed to *isolate* this variable (§5, §7.1).

**Digit handling.** Unconstrained BPE learns arbitrary numeric tokens (e.g. `2020`), which makes digit boundaries inconsistent and hurts arithmetic [9]. Capping digit runs at three (`\p{N}{1,3}`) gives a consistent, right-to-left-agnostic grouping and is the same family of choices as in recent production tokenizers.

**Vocabulary size and hardware.** 32,768 is a multiple of 128, which aligns the embedding and output matrices with GPU kernel tile sizes. Vocabulary size is also a modelling decision: larger vocabularies shorten sequences but dilute embedding statistics for rare tokens [10]. For a ~125M model, 32k is a conventional compromise.

**Why fertility, and its limits.** Fertility (tokens per word) and chars/token are standard intrinsic metrics [4], but compression is only a proxy: it does not guarantee better downstream performance [5, 13] and says nothing about morphological alignment [6]. For an inflected language such as Polish, whole frequent inflected forms are often statistically efficient tokens even when morphologically "ugly".

## 4. Artifact A: `tokenizer-32k` (production)

A single file, `tokenizer.json` (HF `tokenizers` format v1.0).

### 4.1 Specification

| Field | Value |
|---|---|
| Type | Byte-level BPE, HF `tokenizers` `BpeTrainer` |
| Total vocabulary | 32,768 |
| BPE tokens | 32,568 |
| Merges | 32,312 |
| Base alphabet | 256 bytes at IDs 0–255 |
| Special tokens | 200, at IDs 32,568–32,767 |
| Normalizer | `null` (NFC is applied in the corpus pipeline, not in the tokenizer) |
| Pre-tokenizer | `Split` (cl100k-style regex) → `ByteLevel(add_prefix_space=false)` |
| Decoder | `ByteLevel` |
| `byte_fallback` / `unk_token` | `false` / `None` (unnecessary with a byte base) |
| `dropout` | `None` |

**Invariant: bytes occupy IDs 0–255.** Passing `special_tokens` to `BpeTrainer` shifts the byte alphabet out of this range. Special tokens are therefore injected *after* training, and a `sanity_check` asserts the invariant.

### 4.2 Pre-tokenization

```
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
```

Measured properties of the trained artifact:

* **Digits:** 0 tokens with ≥4 digits; 412 pure-digit tokens, all of length 1–3 (`1234567` → `123|45|6|7`).
* **Case-insensitive contractions** (`(?i:...)`): consistent behaviour on all-caps text.
* **Leading space** is attached to the following word and never appears as a standalone token between words.
* **Whitespace is mergeable:** indentation tokens of 1–8, 16, 24, 32 and 64 spaces exist, so code does not explode into single-space sequences.
* **`\n\n` is a single token;** merges do not otherwise cross newlines.

### 4.3 Special tokens (200)

14 named tokens and 186 reserved slots (`<|reserved_0|>` … `<|reserved_185|>`).

| Group | Tokens |
|---|---|
| Document | `<\|endoftext\|>`, `<\|begin_of_text\|>` |
| Chat | `<\|im_start\|>`, `<\|im_end\|>`, `<\|system\|>`, `<\|user\|>`, `<\|assistant\|>` |
| FIM | `<\|fim_prefix\|>`, `<\|fim_middle\|>`, `<\|fim_suffix\|>` |
| Tools | `<\|tool_call\|>`, `<\|tool_result\|>` |
| Technical | `<\|pad\|>`, `<\|unk\|>` |

The block is reserved up front to avoid resizing the embedding matrix mid-lifecycle. All have `normalized: false` and `special: true`; the encoder does not produce them from raw user text.

## 5. Artifact B: `tokenizer-6756` (course study)

A byte-level BPE written from scratch in Python, plus three HuggingFace controls trained on the identical corpus.

```text
wynik FINALNY.json             HF tokenizer (final)
wyniki.json                    custom tokenizer (own format)
wyniki/hf_bez_regex.json       HF ByteLevel, no pre-tokenization
wyniki/hf_z_regex.json         HF ByteLevel, with pre-tokenization
wyniki/hf_normalizowany.json   HF ByteLevel, pre-tokenization + normalizer
```

**Custom implementation:** 256 UTF-8 byte base (IDs 0–255); 6,500 merges from ID 256; token length cap 20 bytes; pair counting via `Counter`, one merge per round; `\n` kept, no lowercasing, no punctuation removal; **no pre-tokenization** (training on a continuous byte stream).

**HF controls:**

| Variant | Pre-tokenization | Normalizer |
|---|---|---|
| No regex | no | no |
| With regex | yes | no |
| With normalizer | yes | yes |

The normalizer applies NFC, unifies line endings to `\n`, removes soft hyphens, zero-width characters, BOM and control characters, and collapses repeated spaces and blank lines.

**Research questions.** (i) Does a from-scratch BPE match a production library under identical conditions? (ii) How large is the effect of pre-tokenization on out-of-domain compression? (iii) What errors arise when building a tokenizer by hand, and what do they reveal about the algorithm?

## 6. Training data (datasheet)

### 6.1 `tokenizer-32k` corpus

| Item | Description |
|---|---|
| **Motivation** | Build a Polish tokenizer matched to a planned ~125M-parameter pretraining run. |
| **Source** | SpeakLeash [14]: 23 datasets, ~2.40 GB of text after cleaning. Five SpeakLeash datasets were **excluded from training** to prevent contamination of the held-out. |
| **Preprocessing** | `pl_clean.py` (`pl-clean-v3`): deterministic and versioned. Steps: NFC normalization → control-character removal → per-document language filter → line-level boilerplate filtering (with a minimum line-length threshold so short headings such as "Wstęp" are not removed) → encoding-corruption detector → cross-deduplication against the held-out. |
| **Held-out** | ~9.5M characters, 2,125 documents: several dozen literary works plus the Constitution of the Republic of Poland. |
| **Intended consistency** | The same `pl_clean.py` version must be run **identically** on the model's training data (see §9). |

### 6.2 `tokenizer-6756` corpus

Three texts: *Tajemniczy ogród* (F. H. Burnett, *The Secret Garden*), *200 mil*, and an excerpt of the Constitution of the Republic of Poland. After cleaning: **9,363,020 characters**.

Cleaning: removal of every line occurring ≥30 times (PDF headers and footers, `PROG = 30`) together with blank lines; unification of quotation marks and dashes; collapsing of repeated spaces and triple line breaks.

Held-out: **2,696 characters / 371 words** from outside the training corpus, checked for non-overlap.

## 7. Evaluation

### 7.1 Controlled study (`tokenizer-6756`)

**Fairness conditions.** Identical training corpus for all variants, verified by character count (difference 0); identical total vocabulary (6,756); identical metric code and word splitting (`str.split()`); held-out verified as non-overlapping.

| Tokenizer | Corpus | Tokens | tokens/word | chars/token |
|---|---|--:|--:|--:|
| HF, regex + normalizer | filtered | **833** | **2.245** | **3.236** |
| HF, regex + normalizer | raw | 833 | 2.245 | 3.236 |
| Custom implementation | filtered | 836 | 2.253 | 3.225 |
| HF, regex | filtered | 840 | 2.264 | 3.210 |
| HF, no regex | filtered | 844 | 2.275 | 3.194 |

Findings:

1. **Implementation parity.** The custom BPE differs from HF in the matching configuration (no pre-tokenization) by 0.9% (836 vs 844). The residual difference is attributable to boundary handling: HF trains per line, while the custom version uses a continuous stream.
2. **Pre-tokenization** improved fertility marginally (844 → 840, ~0.5%), less than hypothesized.
3. **All variants lie within a 1.3% range.** With a 371-word held-out this is within measurement noise and supports no strong ranking.
4. **Normalization**, with the variable isolated (retrained on the filtered corpus), gave 0.8% (840 → 833). This is a *residual* effect, since the corpus had already been manually normalized.
5. **Boilerplate filtering** had no effect on held-out fertility (833 in both cases) although the vocabularies differ by 155 tokens (2.3%); those tokens are PDF-footer fragments with nothing to compress in the held-out.

**Qualitative inspection.** For *Zadzwoniłam do serwisu, bo laptop przestał się ładować.* ("I called the service centre because my laptop stopped charging."):

| Variant | Segmentation |
|---|---|
| No regex | `Za \| dzwo \| ni \| łam␣ \| do␣ \| ser \| wi \| su \| ,␣bo␣ \| la \| p \| to \| p \| ␣prze \| stał \| ␣się␣ \| ład \| ować \| .` |
| With regex | `Za \| dzwo \| ni \| łam \| ␣do \| ␣ser \| wi \| su \| , \| ␣bo \| ␣la \| p \| to \| p \| ␣prze \| stał \| ␣się \| ␣ła \| dować \| .` |

Without pre-tokenization, BPE spends merge budget on multi-word, punctuation-gluing tokens (`,␣bo␣`, `␣się␣`) that compress only phrases seen in training and lose value out of domain. With it, tokens respect word boundaries and the leading-space convention of GPT-2 [3].

**Convergence check.** The first non-trivial merge, bytes (197, 130) → ID 257 = `ł`, is identical in the custom implementation and in an independently trained HF Polish tokenizer.

**Training cost.** HF (Rust, multi-core, two variants together): 8.1 s wall-clock / 52.3 s CPU. Custom pure Python, single core: ~15,500 s (~4.3 h), a speedup of order 4000×, attributable to language, parallelism and training on counted pre-tokens, not to a different algorithm. Winning-pair frequency decays from 228,141 (round 0) through 1,689 (500), 778 (1000), 202 (3000), 87 (6000) to 79 (final).

### 7.2 Cross-tokenizer comparison (32k vs 6,756)

Shared held-out: 725 words / 4,978 characters across six registers (prose, colloquial, journalism, legal, scientific, numbers+code); word = `\S+`.

| Tokenizer | Vocab | tok/word | tok/1000 chars | chars/tok |
|---|--:|--:|--:|--:|
| **`tokenizer-32k`** | 32,768 | **1.665** | **242.5** | **4.12** |
| `tokenizer-6756` | 6,756 | 2.338 | 340.5 | 2.94 |

Difference: **−28.8% tokens**; per register from −13% (prose) to −40% (scientific). The small tokenizer looks adequate only near its training domain.

### 7.3 Intrinsic properties of `tokenizer-32k`

* **Lossless round-trip** on all 2,125 held-out documents.
* **Vocabulary utilization 98.4%** on the held-out (a thin tail of unused tokens lowers glitch-token risk [7]).
* **Token length (bytes):** mode 4–6 B; text tokens up to 20 B; longer tokens are indentation only (24/32/64 B).

### 7.4 Correctness tests for `tokenizer-6756`

| Test | Result |
|---|---|
| byte→char map: entries / injectivity | 256 / yes |
| byte → char → byte round-trip, 256 assertions | pass |
| reversibility of char representation, all 6,756 tokens | pass |
| unique values in the base of the saved JSON | 256 |
| `decode(encode(x)) == x` on Polish demo sentence, full held-out, `Zażółć gęślą jaźń`, `emoji: 🙂 i chiński: 中文`, digits and symbols | OK |

Emoji and Chinese never occur in the training data yet round-trip losslessly, confirming the no-`<UNK>` guarantee. Representations match the HF ByteLevel convention (ID 65 = `A`, 197 = `Å`, 130 = `Ĥ`, 257 = `ÅĤ`).

## 8. Failure analysis and lessons

Four defects were found and fixed in `tokenizer-6756`; they are documented because they are the transferable part of the study.

| # | Defect | Cause | Fix | Lesson |
|---|---|---|---|---|
| 1 | **Monster tokens** from PDF boilerplate (e.g. a phrase repeated ~211 times) | BPE picks the most frequent pair, so repeated boilerplate wins merge rounds | Drop lines occurring ≥30 times (`PROG = 30`) | Data quality precedes training; in-domain compression can be a false signal |
| 2 | **Broken byte base in JSON export** (IDs 128–255 all exported as the same character) | `bytes.decode("utf-8", errors="replace")` on lone continuation/lead bytes yields one replacement character; the byte→char map stopped being injective, so no decoder could exist | GPT-2-style reversible byte→char map; each byte passed through it individually; decoder added | Correct algorithm + broken serialization = broken artifact; `errors="replace"` in an export is silent information loss |
| 3 | **Vocabulary size convention** (6,500 vs 6,756) | Size must include base bytes, merges and special tokens | One formula everywhere; HF run with `vocab_size=6756` | Report the real embedding-matrix row count; base has 256 values (IDs 0–255), not 255 |
| 4 | **Corpus mismatch** (9,485,253 vs 9,363,020 chars, +1.3%) | Raw corpus used; reconstruction used a wrong threshold (`PROG = 100`) and skipped normalization | Restore `PROG = 30` → normalization; difference 0 | Character count acts as an experiment checksum; fair comparison needs numerically verified identical inputs |

*Non-defect:* different base IDs in the HF file (`'A'` = 32 vs 65) result from sorting the alphabet by code point of the substitute characters. What matters is completeness, injectivity and that encoder and decoder share one map.

## 9. Limitations, risks and ethical considerations

**Evaluation limitations.**

* The 32k-vs-6,756 table uses a small ad hoc held-out (725 words) and the `\S+` word definition. It shows magnitude and trend and is **not a publishable measurement**. A 1:1 comparison with another tokenizer needs a shared held-out and a shared word definition.
* The controlled study's held-out (371 words) is too small to treat differences of ~1% as significant.
* Fertility measures distributional compression, not grammaticality or morpheme alignment [6].
* **Model quality** (perplexity, downstream tasks) is not measured; intrinsic tokenizer metrics correlate imperfectly with it [5, 13].
* Not yet done for `tokenizer-6756`: vocabulary utilization and bytes/token, robustness tests (numbers, code, markdown, emoji, text without diacritics, ALL CAPS), vocabulary-size ablation (only 6,756 was tested), comparison with other people's tokenizers on a shared test set. The custom implementation has no pre-tokenization and no special tokens.

**Preprocessing coupling (important).** `normalizer: null` in `tokenizer-32k` does **not** mean "no normalization": NFC is applied in the corpus pipeline. If model training data follows a different preprocessing path, the mismatch will produce glitch tokens [7]. Run `pl-clean-v3` identically.

**Domain and language bias.** The vocabulary is optimized for Polish and, within the corpus, for the registers SpeakLeash covers. Other languages and rarely covered registers will be over-segmented, increasing cost and reducing effective context for those users [8].

**Data provenance.** The corpus aggregates web-scale and literary sources. Texts may contain personal data, offensive content and biases that the tokenizer inherits only indirectly (via which strings become tokens), but dataset licences and personal-data handling should be reviewed before redistribution.

## 10. Usage

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer.json")
ids = tok.encode("Zażółć gęślą jaźń.").ids
assert tok.decode(ids) == "Zażółć gęślą jaźń."
```

Via `transformers` [15]:

```python
from transformers import PreTrainedTokenizerFast
tok = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
```

Special tokens are never produced from raw user text; add them explicitly when building chat, FIM or tool-call sequences.

## 11. Reproducibility

* Training library: HuggingFace `tokenizers` `BpeTrainer` (production); custom Python for the course study.
* Corpus cleaning: `pl_clean.py`, version `pl-clean-v3`.
* Fairness controls in the course study: identical corpus (character-count checksum), identical vocabulary size, identical metric code.

## 12. Citation and references

If you use these artifacts, please cite this repository and the works your use depends on.

1. Gage, P. (1994). A new algorithm for data compression. *The C Users Journal*, 12(2).
2. Sennrich, R., Haddow, B., Birch, A. (2016). Neural machine translation of rare words with subword units. *ACL 2016*.
3. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI.
4. Rust, P., Pfeiffer, J., Vulić, I., Ruder, S., Gurevych, I. (2021). How good is your tokenizer? On the monolingual performance of multilingual language models. *ACL 2021*.
5. Schmidt, C. W., et al. (2024). Tokenization is more than compression. *EMNLP 2024*.
6. Bostrom, K., Durrett, G. (2020). Byte pair encoding is suboptimal for language model pretraining. *Findings of EMNLP 2020*.
7. Land, S., Bartolo, M. (2024). Fishing for Magikarp: Automatically detecting under-trained tokens in large language models. *EMNLP 2024*.
8. Petrov, A., La Malfa, E., Torr, P., Bibi, A. (2023). Language model tokenizers introduce unfairness between languages. *NeurIPS 2023*.
9. Singh, A. K., Strouse, D. J. (2024). Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs. *arXiv:2402.14903*.
10. Tao, C., et al. (2024). Scaling laws with vocabulary: larger models deserve larger vocabularies. *NeurIPS 2024*.
11. Mitchell, M., et al. (2019). Model cards for model reporting. *FAT\* 2019*.
12. Gebru, T., et al. (2021). Datasheets for datasets. *Communications of the ACM*, 64(12).
13. Ali, M., et al. (2024). Tokenizer choice for LLM training: negligible or crucial? *Findings of NAACL 2024*.
14. SpeakLeash project, https://speakleash.org.
15. Wolf, T., et al. (2020). Transformers: state-of-the-art natural language processing. *EMNLP 2020: System Demonstrations*.

