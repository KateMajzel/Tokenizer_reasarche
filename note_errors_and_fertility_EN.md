# Project note: four bugs and fertility results

Project: byte-level BPE trained from scratch on a corpus of Polish literature (9,363,020 characters after cleaning), vocab 6756 (256 base + 6500 merges), compared with HuggingFace tokenizers trained on an identical corpus and with an identical vocabulary size.

---

## Case 1: Monster tokens from PDF boilerplate

**Symptom.** The tokenizer was learning very long, multi-word tokens corresponding to footers and headers from PDF files (e.g. fragments of the phrase "Materiał ćwiczeniowy dla uczniów niepełnosprawnych…" ["Practice material for students with disabilities…"], repeated approx. 211 times in the corpus). In-domain compression looked great, but the tokens were useless outside the corpus.

**Cause.** Repeated boilerplate dominated the pair statistics. BPE always picks the most frequent pair, so phrases repeated hundreds of times won merge rounds even though they do not represent the language, only an artifact of the data source.

**Fix.** A line-frequency filter before training: removal of every line occurring in the corpus 30 or more times (`PROG = 30`), along with blank lines.

**Lesson.** Data quality comes before training. A good compression metric on the training data can be a false signal — the result of overfitting to junk, not of tokenizer quality.

---

## Case 2: Broken byte base in the JSON export (main bug)

**Symptom.** Instructor's remark: "you didn't map the base 256 vocab onto standard characters; up to 255 you should have only a single character each". In the exported JSON, entries with IDs 0–127 were correct (single ASCII characters), but all 128 entries with IDs 128–255 contained the same character — a space.

**Cause.** The export line:

```python
"vocab": {str(i): vocab[i].decode("utf-8", errors="replace") for i in sorted(vocab)}
```

A single byte in the 128–255 range is not a standalone UTF-8 character — it is a fragment of a multi-byte sequence (e.g. byte 197 is only the first half of the letter `ł` = bytes 197+130). Decoding such a byte on its own is bound to fail, and `errors="replace"`, instead of raising an error, substituted a replacement character — the same one for all 128 bytes. The byte→char map stopped being a one-to-one function (an injection), so recovering the bytes from the strings became mathematically impossible: no decoder exists.

**Important distinction.** The training algorithm was correct the entire time — the in-memory vocabulary (`vocab = {i: bytes([i]) …}`) contained the full set of 256 distinct bytes, and the merge rules matched the reference tokenizer (the first non-trivial merge `(197, 130) → ID 257 = ł` was identical to the one in a colleague's HF file). Only the output artifact was broken — a single line of serialization.

**Fix.** A reversible GPT-2 ByteLevel-style byte→char map: printable bytes (33–126, 161–172, 174–255) remain themselves; non-printable bytes (0–32, 127–160, 173 — soft hyphen) are assigned consecutive unique substitute characters from code point 256 upward (hence `Ġ` = space = chr(288)). The export line was replaced by passing each byte individually through this map. The fix was deployed and verified: the base of the saved JSON contains 256 unique entries, the round-trip passes for all 256 bytes and for all 6756 vocabulary tokens, and the representations ID 65 = `A`, ID 197 = `Å`, ID 130 = `Ĥ`, ID 257 = `ÅĤ` are consistent with the HuggingFace ByteLevel convention. The missing decoder was also added — the round-trip `decode(encode(x)) == x` passes on Polish texts, emoji and Chinese script.

**Lesson.** Correct algorithm + broken serialization = broken artifact. The vocabulary export must be an injection, otherwise decoding is impossible. `errors="replace"` in a data export is silent information loss.

---

## Case 3: Vocabulary size convention (6500 vs 6756)

**Symptom.** Instructor's remark: "vocab_size + base_vocab = real vocab". Risk of reporting the vocabulary size as 6500 (merges only), while the real vocabulary is 6756.

**Cause.** An unestablished convention: vocabulary size includes all tokens that have an ID — the byte base (256), the merge tokens (6500) and any special tokens. This is how the whole industry counts (GPT-2: 50257 = 256 + 50000 + 1) and how the `vocab_size` parameter in the HuggingFace library counts. The number of rows of the model's future embedding matrix equals this total size.

**Fix.** A single formula throughout the documentation: "Vocab size 6756: 256 byte-base tokens + 6500 tokens from merge rules". For the comparative training in HF, `vocab_size=6756` was set so that the total vocabulary sizes were equal.

**Lesson.** A related detail: the byte base has 256 values (IDs 0–255 inclusive), not 255 — a classic off-by-one error that showed up in the notes several times.

---

## Case 4: Corpus mismatch when reconstructing the experiment

**Symptom.** The corpus file prepared for HF had 9,485,253 characters instead of the expected 9,363,020 (+122,233 characters, +1.3%). Training HF on such a file would have been unfair — the comparison required identical text.

**Cause.** The raw, pre-cleaning corpus had ended up in the file. Diagnosis with a line counter confirmed the presence of boilerplate (lines repeated 210–211 times). The first attempt to reconstruct the filter missed the target by 2,349 characters, because the reconstruction assumed a threshold of `PROG = 100` and skipped the normalization step — in the original Colab code the threshold was `PROG = 30`, and the filter was followed by normalization (unifying quotation marks and dashes, collapsing multiple spaces and blank lines).

**Fix.** Recreating the full pipeline from the original script: filter `PROG = 30` → normalization. Result: 9,363,020 characters, a difference of 0 from the target — the corpus was reproduced to the character.

**Lesson.** The corpus character count works as a checksum for the experiment: it makes it possible to detect any discrepancy in the data before it distorts the results. A fair comparison requires identical inputs, verified numerically, not "by eye".

---

## False alarm (not a bug): different base IDs in the HF file

In the exported HF file, the base characters have different numbers than in the reference file (`'A'` = 32 instead of 65, `'Ġ'` = 220 instead of 32). This is not a bug: HF sorted the base alphabet by the code points of the substitute characters, not by byte values. Programmatic verification: 256 entries, all distinct (injection), each one character long, the set identical to the GPT-2 ByteLevel alphabet. The specific ID numbers carry no meaning — what matters is that the map is complete and injective, and that the encoder and decoder use the same map.

---

## Fertility results on held-out

Measurement conditions: shared held-out (2,696 characters, 371 words, text from outside the training corpus), identical training corpus (9,363,020 characters, matching to the character), identical total vocabulary size (6756), identical metric code (denominator: `len(held_out.split())`).

| tokenizer | tokens | tokens/word | chars/token |
|---|---|---|---|
| HF with regex + normalizer (filtered corpus) | **833** | **2.245** | **3.236** |
| HF with regex + normalizer (raw corpus) | 833 | 2.245 | 3.236 |
| My BPE (Colab, no pre-tokenization) | 836 | 2.253 | 3.225 |
| HF ByteLevel, with regex (pre-tokenization) | 840 | 2.264 | 3.210 |
| HF ByteLevel, no regex | 844 | 2.275 | 3.194 |

Interpretation:

**All variants fall within a 1.3% range of differences** (833–844 tokens). With a held-out of 371 words, differences of this order are within measurement noise and do not justify strong conclusions about the superiority of any variant.

**Effect of normalization, after isolating the variable.** The normalizer variant was retrained on the already filtered corpus so that the normalizer remained the only difference relative to the "HF with regex" variant. The 840 → 833 comparison gives a 0.8% improvement. Caveat: the corpus had previously undergone manual normalization (quotation marks, dashes, multiple spaces), so what is measured is the normalizer's residual effect — NFC, removal of soft hyphens and control characters.

**The boilerplate filter did not affect the held-out result.** Both normalizer variants — trained on the raw corpus and on the filtered one — gave an identical count of 833 tokens, even though their vocabularies differ by 155 tokens (2.3% of the vocabulary). The differing tokens are boilerplate fragments present only in the raw variant; since the held-out contains no PDF footers, these tokens had nothing to compress, and the shared 97.7% of the vocabulary did all the work. Conclusion: with pre-tokenization enabled, protection against monster tokens works twice over — the data filter and the regex partially perform the same task.

**Custom implementation ≈ production implementation.** The difference between my BPE and HF in the same configuration (no pre-tokenization) is 0.9% (836 vs 844). This is empirical proof that my own implementation of the algorithm is correct — two independent implementations of the same algorithm on the same data give practically the same result. The small remaining difference comes from different boundary handling: HF trains per line (merges do not cross `\n`), while my implementation works on a continuous byte stream.

**Pre-tokenization did not significantly improve fertility** (844 → 840, approx. 0.5%), contrary to the prediction of a clear improvement in generalization. Its real value lies elsewhere: a tokenizer without pre-tokenization spends part of its merge budget on multi-word tokens crossing word boundaries (e.g. `", bo "`, `" się "` — visible both in my vocabulary and in the HF no-regex variant), which compress only phrases known from training. Pre-tokenization forces tokens that do not cross word boundaries (punctuation kept separate, leading space attached to the word: `Ġdo`, `Ġbo`), i.e. reusable units with better morphological boundaries. On a held-out stylistically close to the corpus the effect is small; the difference should grow with the domain distance of the test text.

**Training time:** HF (Rust, multi-core training, on counted pre-tokens) trained both variants combined in 8.1 s of wall-clock time (52.3 s CPU). The performance difference relative to the custom pure-Python implementation comes from the language and algorithmic optimization, not from a difference in the BPE algorithm itself.
