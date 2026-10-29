# Polish Byte-Level BPE — Custom Implementation and Comparison with HuggingFace

## Overview

The project artifact is a byte-level BPE implemented from scratch in Python,
trained on a corpus of Polish literature, together with a set of HuggingFace
tokenizers trained on the same corpus as a reference point.

```text

wynik FINALNY.json             — HF tokenizer
wyniki.json                    — custom tokenizer (own format)
wyniki/hf_bez_regex.json       — HF ByteLevel, no pre-tokenization
wyniki/hf_z_regex.json         — HF ByteLevel, with pre-tokenization
wyniki/hf_normalizowany.json   — HF ByteLevel, with pre-tokenization and normalizer
```

Best result on the held-out set:

```text
tokens/word = 2.245
chars/token = 3.236
vocab       = 6756
```

## Research question

The goal was not to obtain the best possible tokenizer, but to answer
questions that cannot be posed using only an off-the-shelf
library:

* does a BPE implementation written from scratch give results comparable to a
production library under identical conditions;
* how large is the real impact of pre-tokenization on compression of
out-of-domain text;
* what mistakes are made when building a tokenizer by hand, and what they
teach about the algorithm itself.

The starting point was the custom implementation, not the library — this way
the comparison measures a specific design decision (the presence of
pre-tokenization), rather than the difference between two unknown black boxes.

## Tokenizer design

The tokenizer uses a base byte vocabulary, so any input string can be
encoded without `<UNK>` tokens.

Custom implementation configuration:

* base: 256 UTF-8 bytes, IDs 0–255;
* merges: IDs from 256 upward, 6500 rules;
* total vocabulary size: 6756 (256 base + 6500 merges);
* token length limit: 20 bytes;
* pair counting via `Counter`, one merge rule per round;
* `\n` preserved in the corpus, no lowercasing, no punctuation removal;
* Polish diacritics preserved (at the byte level);
* no pre-tokenization — training on a continuous byte stream.

HuggingFace variants trained for comparison:

|variant|pre-tokenization|normalizer|
|-|-|-|
|no regex|no|no|
|with regex|yes|no|
|with normalizer|yes|yes|

The normalizer in the third variant includes: NFC, unifying line endings to
`\n`, removing soft hyphens, zero-width characters, BOM and control characters,
and collapsing multiple spaces and blank lines.

## Training data

A corpus made up of three texts: *Tajemniczy ogród* (*The Secret Garden*,
F.H. Burnett), *200 mil*, and an excerpt from the Constitution of the
Republic of Poland.

Cleaning before training:

* removal of lines repeated 30 or more times in the corpus (footers and
headers from PDF files), along with blank lines;
* unification of quotation marks and dashes;
* collapsing multiple spaces and triple line breaks.

Corpus size after cleaning: **9,363,020 characters**.

Held-out set: **2,696 characters, 371 words**, text from outside the training corpus.

## Evaluation protocol

Metrics:

* fertility (tokens/word) — lower is better;
* chars/token — higher is better;
* number of tokens on the entire held-out set.

Fairness conditions of the comparison:

* identical training corpus for all variants, consistency
verified by character count (difference of 0);
* identical total vocabulary size (6756) in all variants;
* identical metric code, the same word splitting (`str.split()`);
* held-out checked for absence of overlap with the corpus.

## Held-out results

|tokenizer|corpus|tokens|tokens/word|chars/token|
|-|-|-:|-:|-:|
|HF with regex + normalizer (wynik FINALNY.json)|filtered|**833**|**2.245**|**3.236**|
|HF with regex + normalizer|raw|833|2.245|3.236|
|Custom implementation|filtered|836|2.253|3.225|
|HF with regex|filtered|840|2.264|3.210|
|HF without regex|filtered|844|2.275|3.194|

Observations:

1. **The custom implementation matches the production library.** The difference
between the custom implementation and the HF variant in the same configuration
(no pre-tokenization) is 0.9% (836 vs 844). Two independent implementations of
the same algorithm on the same data give practically the same result, which
is empirical confirmation that the implementation is correct.
2. **Pre-tokenization improved fertility slightly** (844 → 840, approx. 0.5%),
clearly less than the prediction made before the experiment had
assumed.
3. **The spread across the whole field is 1.3%.** With a held-out of 371 words,
differences of this order fall within noise and do not justify
strong conclusions about the superiority of any particular variant.
4. **Normalization gave a 0.8% improvement after isolating the variable.** The
normalizer variant was retrained on the filtered corpus so that
the normalizer remained the only difference relative to the "HF with regex"
variant: 840 → 833. What is measured is the residual effect, because the corpus
had previously undergone manual normalization of quotation marks, dashes and
multiple spaces — leaving the HF normalizer with NFC and the removal of soft
hyphens and control characters.
5. **The boilerplate filter did not affect the held-out result.** The
normalizer variants trained on the raw and the filtered corpus gave an
identical count of 833 tokens, even though their vocabularies differ by 155
tokens (2.3%). The differing tokens are boilerplate fragments present
only in the raw variant; the held-out contains no PDF footers, so they
had nothing to compress. With pre-tokenization enabled, protection against
monster tokens works twice over — the data filter and the regex partially
perform the same task.

## Token quality

Aggregate metrics do not show what BPE actually learned, so the
segmentation of the same sentence by the variants with and without
pre-tokenization was compared.

Test sentence: *Zadzwoniłam do serwisu, bo laptop przestał się ładować.*
("I called the service center because my laptop stopped charging.")

|variant|split|
|-|-|
|no regex|`Za \| dzwo \| ni \| łam␣ \| do␣ \| ser \| wi \| su \| ,␣bo␣ \| la \| p \| to \| p \| ␣prze \| stał \| ␣się␣ \| ład \| ować \| .`|
|with regex|`Za \| dzwo \| ni \| łam \| ␣do \| ␣ser \| wi \| su \| , \| ␣bo \| ␣la \| p \| to \| p \| ␣prze \| stał \| ␣się \| ␣ła \| dować \| .`|

Conclusions:

* the variant without pre-tokenization learns **multi-word** tokens
that cross word boundaries and glue punctuation on: `,␣bo␣`, `␣się␣`,
`łam␣`, `do␣` — with the space attached at the end;
* the variant with pre-tokenization keeps punctuation separate and attaches the
space **at the front** of the word (`␣do`, `␣bo`), following the GPT-2 convention;
* an identical pattern of multi-word tokens appears in the custom
implementation (`", bo "`, `" się "`, `" dopiero "`), which confirms that it is a
consequence of the lack of pre-tokenization, not an implementation bug;
* multi-word tokens compress only phrases known from the training
corpus, so their value drops as the domain distance of the test text
increases.

Convergence with the reference tokenizer: the first non-trivial merge in
the custom implementation is the byte pair (197, 130) → ID 257, i.e. the letter `ł`.
An identical merge under an identical ID appears in a HuggingFace tokenizer
trained independently by another person on a Polish corpus. The statistics of
the language shine through different implementations.

## Performance

|implementation|training time|
|-|-|
|HF (Rust, multi-core, two variants combined)|8.1 s wall-clock / 52.3 s CPU|
|Custom implementation (pure Python, single-core)|approx. 15,500 s (approx. 4.3 h)|

A speedup on the order of 4000× per training run. The ratio of CPU time to
wall-clock time in HF (52.3 / 8.1 ≈ 6.5) shows that the library parallelized
the work. The performance difference comes from the implementation language,
parallelization and algorithmic optimization (training on counted pre-tokens
instead of the full stream), not from a difference in the BPE algorithm itself.

Frequency curve of the winning pairs in the custom implementation: round 0 — 228,141
occurrences, round 500 — 1,689, round 1000 — 778, round 3000 — 202, round 6000 —
87, final round — 79. The decline is steep at the beginning and very slow
in the later part of training.

## Artifact correctness

After the serialization fix, the custom implementation artifact passes the full
set of correctness tests:

|test|result|
|-|-|
|byte→char map: number of entries|256|
|byte→char map: injectivity (unique values)|yes|
|round-trip byte → char → byte, 256 assertions|passes|
|reversibility of the character representation for the whole vocabulary (6756 tokens)|passes|
|number of unique values in the base of the saved JSON|256|

Round-trip `decode(encode(x)) == x` on texts:

|text|result|
|-|-|
|demo sentence (Polish)|OK|
|full held-out|OK|
|`Zażółć gęślą jaźń`|OK|
|`emoji: 🙂 i chiński: 中文`|OK|
|digits and special characters|OK|

The emoji and Chinese case is significant: emoji and Chinese script do not
appear even once in the training corpus, yet they are encoded and decoded
losslessly. This is practical confirmation of the no-`<UNK>` guarantee that
follows from the complete byte base.

Consistency with the HuggingFace ByteLevel convention: in the saved vocabulary ID 65 is
`A`, ID 197 is `Å`, ID 130 is `Ĥ`, and ID 257 is `ÅĤ` — exactly the same
representations that appear in tokenizers exported by the library.

## Conclusions

1. **Byte-level guarantees no `<UNK>`.** The complete base of 256 bytes covers
every possible text, at the cost of 68 entries for non-printable bytes that
practically never occur.
2. **A correct algorithm is not enough — the artifact must be reversible.**
Serializing the vocabulary requires a byte→char map that is an injection;
without it no decoder exists, despite fully correct training.
3. **Pre-tokenization improves boundary quality, not compression.** Its effect on
fertility was marginal, but it changes the nature of the learned tokens from
phrase-level to reusable.
4. **In-domain compression can be misleading.** Boilerplate in the corpus generates long
tokens that improve metrics on data similar to the training data and give
nothing beyond it.
5. **The vocabulary size convention includes the base.** 6756 = 256 + 6500;
analogously for GPT-2: 50257 = 256 + 50000 + 1.

## Limitations

* The custom implementation has no pre-tokenization and no special tokens.
* Vocabulary utilization and bytes/token were not measured.
* No robustness tests were carried out (numbers, code, markdown, emoji, text
without diacritics, CAPS LOCK).
* No vocabulary-size ablation was performed — only 6756 was tested.
* The held-out contains 371 words, which is too small a sample to treat differences
of around 1% as significant.
* Results were not compared with other people's tokenizers on a shared test
set.
* The normalizer's effect was measured only as a residual effect, on a corpus
that had previously undergone manual normalization.

## Next steps

* Evaluation on a held-out from outside the literary domain, where the advantage of
pre-tokenization should be more pronounced.
* Robustness tests on numbers, code, and text without diacritics.

## Final artifact

The artifact submitted in the project is the custom byte-level BPE implementation
(vocab 6756) together with a set of three HuggingFace tokenizers serving as a
comparative control. The value of the submission lies in the comparison methodology:
equalized conditions, an isolated variable, and a documented error-diagnosis process.
