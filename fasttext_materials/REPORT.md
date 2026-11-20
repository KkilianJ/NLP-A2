# Phase 1: Exploring fastText for IMDb sentiment

## Questions and hypotheses

1. How much of the IMDb training vocabulary occurs **directly** in the pretrained model, and how do punctuation, contractions, markup, and case change the answer? I expected punctuation separation to improve direct coverage, but not necessarily downstream accuracy.
2. When an IMDb token is absent from the stored vocabulary, what kinds of tokens are they, and how well do character n-grams preserve similarity under controlled spelling changes? I expected some robustness, with variation by word and edit.
3. Does the space group words by sentiment, or can positive and negative words also be close when they share contexts? I expected both effects.

These are exploratory hypotheses, not classifier results. The complete measured tables and figures are linked below.

## Data, model, and experimental boundary

The previous assignment used [mteb/imdb at revision `d05f0155`](https://huggingface.co/datasets/mteb/imdb/tree/d05f0155defa7991dad75bc68c5ccb6774b1fdc5). Its 24,904 published train and 24,678 published test rows were pooled and stratified with seed 42 into **34,707 training**, **7,437 validation**, and **7,438 held-out test** reviews. This reproduces that assignment's procedure and is **not the official IMDb benchmark split**. Source text and labels were checked against its recorded ordered fingerprint, and source labels were used to reconstruct the existing stratified partition. Only the resulting training texts entered the analyses; validation and held-out test reviews and labels were not used to choose or assess these probes. Pooled row IDs in the tables identify the sampled training reviews.

The measured model was the original `cc.en.300.bin`, 300 dimensions and 2,000,000 stored words, with SHA-256 `14c7167b130056944cbdc37b7451f055867fe9a4e3fed3bbc1ecc0e74f6763ca`. The [fastText model documentation](https://fasttext.cc/docs/en/crawl-vectors.html) describes training on **Common Crawl and Wikipedia**, using CBOW with position weights and character n-grams of length five. The loaded binary also reports `minn = maxn = 5`. The [subword paper](https://aclanthology.org/Q17-1010/) explains the general mechanism, while the [157-language model paper](https://aclanthology.org/L18-1550/) describes this pretrained model family. Those are established model properties; the numbers below are measurements on our IMDb training partition.

A direct match means `get_word_id(token) >= 0` for the exact string and case. An OOV word can instead be represented from its subword information, but a nonzero vector alone says nothing about whether its meaning is correct. The code measures zero-vector OOV cases rather than presuming every string can be represented usefully.

## 1. Tokenization and vocabulary coverage

### Method

Four variants process each training review:

- `whitespace`: Python whitespace splitting, preserving attached punctuation and markup.
- `regex`: Unicode letter/number sequences with internal straight or curly apostrophes retained; punctuation is separated and case preserved.
- `regex_br`: the same regex after replacing IMDb's `<br />`-style break tags with spaces.
- `regex_br_lower`: the preceding tokens lowercased **after** tokenization.

The two additional variations isolate an evident markup artifact and a common case-normalization decision. They are not a sweep for a downstream winner. Token-weighted coverage is direct-match occurrences divided by all occurrences; type-weighted coverage is direct-match distinct strings divided by all distinct strings.

### Results

| Variant | Token occurrences | Types | Direct token coverage | Direct type coverage |
|---|---:|---:|---:|---:|
| whitespace | 8,013,808 | 349,444 | 91.14% | 31.11% |
| regex | 9,814,658 | 120,099 | 99.31% | 81.70% |
| regex_br | 9,252,706 | 120,098 | 99.27% | 81.70% |
| regex_br_lower | 9,252,706 | 96,312 | 98.85% | 69.04% |

[Complete counts and OOV vector availability](results/full/coverage.csv) · [Coverage plot](results/full/coverage.png)

For `regex_br`, **67,262 token occurrences / 21,979 types** are OOV. Their vectors are nonzero for **65,162 occurrences / 21,963 types**, leaving **2,100 occurrences / 16 types** with zero vectors. The high type-level nonzero rate is a statement about vector availability, not semantic quality. Type coverage is much lower than token coverage because many different rare strings occur infrequently.

The token denominators are different across tokenizers. For example, regex turns punctuation and raw markup into separate, often directly stored tokens, so a larger percentage does not by itself establish a better text representation. Removing break tags eliminates 561,952 regex tokens attributable to those tags. It is therefore unsurprising that the raw-regex fraction can be slightly higher even though the extra tokens are markup.

Lowercasing is not uniformly beneficial here. Among tokens whose spelling changed, **43,256 occurrences** went from direct to OOV, versus **3,663** going from OOV to direct. Examples include `I've` → `i've` (4,380 occurrences, direct to OOV) and `Let's` → `let's` (438, OOV to direct); names such as `Fulci` → `fulci` also lose direct membership. The distinct-type denominator changes under lowercasing, so its type percentages are not a paired estimate. [Transition counts](results/full/capitalization_summary.csv) and [all changed-membership examples](results/full/capitalization_examples.csv) show the evidence.

### Real tokenization examples and Europarl alignment

From training review 24835, the substring `say, I was very impressed.` becomes `['say,', 'I', 'was', 'very', 'impressed.']` with whitespace, and `['say', ',', 'I', 'was', 'very', 'impressed', '.']` with regex. From review 19671, regex preserves `don't` as one token. A run of the [published English Europarl preprocessing tools](https://www.statmt.org/europarl/) on the same excerpt yields `don` and `'t` instead. In another real excerpt, the Europarl tool retains `comedy-drama-musical-spy`, while our regex splits the hyphens. The tool and regex both split much ordinary punctuation, but neither this small sample nor our regex is a full reproduction of fastText's historical preprocessing pipeline. The available fastText documentation names Europarl preprocessing but does not establish that our downloaded reference toolkit is its exact historical version.

Markup also matters: the actual review fragment `it.<br /><br />Madhavi` becomes punctuation and `br` tokens under raw regex; `regex_br` removes the two breaks and retains `it`, `.`, and `Madhavi`. The [five real before/after examples](results/full/tokenization_examples.csv) and [Europarl comparison](results/full/europarl_reference.csv) show the complete context and token lists. Neither raw coverage nor theoretical alignment decides the eventual classifier's preprocessing; assess alternatives using training folds and validation later.

## 2. OOV words and spelling

### Natural IMDb tokens

For the case-preserving `regex_br` variant, the [full OOV table](results/full/natural_oov_types.csv) records frequencies and vector norms. Candidate corrections are restricted to directly stored lowercase words occurring at least 20 times in the training reviews and within one deletion, insertion, substitution, or adjacent transposition. At most the first 1,500 frequent eligible OOV types were screened, and suggestions are ranked by training frequency. They are **guesses**, not automatically verified corrections.

The presence, spelling, frequency, and contexts of these examples were checked against the training texts; their authors' intended spellings cannot be independently verified:

| Actual OOV token | Frequency | Training-review context / interpretation |
|---|---:|---|
| `anacronistic` | 2 | The excerpt describes characters as “SLIGHTLY anacronistic”; `anachronistic` is a strong contextual guess, but still a guess. |
| `esqe` | 4 | Occurs in `jungle-esqe`; `esque` is a plausible correction. |
| `slowwwwww` | 2 | An expressive repeated-character form describing a slow pace, not necessarily a typo to normalize. |
| `Nazarin` | 26 | The reviewer refers to a movie by this name; this is a title/name candidate, not evidence of a spelling error. |
| `blax` | 2 | Appears near film references and may be genre shorthand. The algorithm suggests `blah`, illustrating a likely bad correction. |
| `film's` | 1,345 | A common possessive, not an error; retaining the apostrophe can produce OOV forms. |

See [natural examples with original review excerpts](results/full/natural_oov_examples.csv). These are heuristic examples, not an estimate of the prevalence of spelling errors. Proper names, slang, non-English terms, emphasis, possessives, and tokenization artifacts should not automatically be corrected.

### Controlled experiment

I selected 19 correctly spelled, directly stored words that occur in the training reviews, including positive and negative words. The configured `bad` is excluded because the procedure requires at least four characters. For each word, a seeded generator makes one interior deletion, insertion, substitution, adjacent transposition, and character repetition, preferring OOV forms. The insertion alphabet is `x/a/e` and substitution alphabet `x/z/a/e`; these are controlled perturbations, not a realistic distribution of human errors. The table also includes an identity control and a randomly paired, manually selected unrelated word control. Each group has 19 pairs.

| Comparison | Mean cosine | Median cosine |
|---|---:|---:|
| deletion | 0.269 | 0.267 |
| insertion | 0.134 | 0.126 |
| substitution | 0.127 | 0.140 |
| transposition | 0.226 | 0.190 |
| repetition | 0.202 | 0.190 |
| unrelated control | 0.101 | 0.094 |
| identity control | 1.000 | 1.000 |

[Every synthetic pair and its vocabulary status](results/full/spelling_pairs.csv) · [Summary](results/full/spelling_summary.csv) · [Distribution plot](results/full/spelling_robustness.png)

In this small, selected set, deletion had the highest mean and insertion/substitution were only modestly above the unrelated control. Six of the 95 synthetic corruptions were **already in the stored vocabulary** despite OOV preference. For example, `beautiful` → synthetic `beautful` was stored and scored 0.847; among the **15 OOV deletions**, the mean was 0.202, versus 0.521 for the four directly stored deletions. Thus the overall deletion mean should not be presented as OOV-only performance. Other examples are OOV `excellent` → `excellnt` (0.268) and OOV `boring` → `boriing` (0.330). These misspellings were generated for this experiment; their presence in IMDb was not asserted.

The nearest-neighbor check searched the first 100,000 stored model words, not the entire vocabulary. `boriing` retrieved `boring`, `pointless`, and `tedious` among its top five, whereas `excellnt` did not retrieve `excellent` in its top five and returned several web-like strings. [All selected neighbors and scores](results/full/nearest_neighbors.csv) show this contrast. The limited pool and occasional web-template artifacts constrain the conclusion.

Character n-grams make OOV vectors possible because familiar substrings can contribute learned information. They do not guarantee robustness: this particular model uses five-character n-grams, a small edit can disrupt many shared fragments, and an OOV word lacks its own stored word entry. These are plausible contributing mechanisms, not a causal ablation. Short strings, unusual spellings, hash collisions, and coincidental character overlap can yield weak, zero, or misleading representations. The small fixed perturbation sample cannot establish a general ranking of real-world error types.

## 3. Similarity, morphology, and sentiment

The probe set consists of six manually chosen pairs in each category, all words observed in the IMDb training texts. It is deliberately transparent, small, and not an independent statistical sample: some words recur across pairs. Similarities use the binary model's vectors, with no training on IMDb.

| Pair category | Pairs | Mean cosine | Example cosine |
|---|---:|---:|---|
| positive–positive | 6 | 0.709 | `amazing`–`fantastic`: 0.835 |
| negative–negative | 6 | 0.746 | `terrible`–`awful`: 0.868 |
| positive–negative | 6 | 0.508 | `good`–`bad`: 0.752 |
| unrelated controls | 6 | 0.061 | `boring`–`river`: 0.028 |
| morphological controls | 6 | 0.509 | `bored`–`boring`: 0.651 |

[Individual pairs, training frequencies, and direct-match status](results/full/semantic_pairs.csv) · [Summary](results/full/semantic_summary.csv) · [Plot](results/full/semantic_similarity.png)

The selected same-sentiment pairs are close on average, while the unrelated controls are much less similar. But cross-polarity pairs are not necessarily far apart: `good` and `bad` have cosine 0.752 and are each other's closest neighbors **within the 100,000-word candidate pool**. This is consistent with the idea that opposites can share contexts, though this probe does not prove the causal mechanism. Morphologically related `beautiful`/`beautifully` score 0.642, but `hope`/`hopeless` score 0.339, so shared form is not a reliable sentiment test either. Neighbor lists also show coherent words such as `terrible` → `horrible`, `awful`, and `horrendous`.

These examples show semantic/contextual relatedness and morphology alongside sentiment information. They do **not** prove that the model lacks polarity information or that a classifier cannot learn a useful decision boundary from nearby vectors. Review context, negation, and word order can matter independently of individual word-vector cosines.

## Limitations and implications for the next phase

- The preprocessing variants change the units counted; raw token coverage alone cannot rank them. The Europarl comparison covers a few excerpts and a public reference version, not a verified byte-for-byte reproduction of model training.
- OOV rates depend on dataset and tokenizer. Nonzero vectors, correction guesses, cosine proximity, and limited-pool neighbors are not evidence that meaning is preserved. The spelling and semantic samples are small, manually selected, and not estimates of real-world performance.
- The pretrained web corpus may overlap with IMDb material; this analysis cannot rule that out. The earlier project also pooled the published partitions, making official-benchmark comparisons inappropriate.
- In the next phase, compare a small number of justified preprocessing choices using training folds and the separate validation split. Preserve access to the binary's OOV representation rather than silently mapping every unseen word to one placeholder, and test whether break-tag handling helps. Do not use the held-out test to choose these settings.
- A later classifier may learn polarity despite `good` and `bad` being close; classification performance and error analysis must establish whether these embedding observations matter in practice.

The exact configuration, data/model hashes, environment, split fingerprint, runnable commands, and output descriptions are in [README.md](README.md) and [run metadata](results/full/run_metadata.json).
