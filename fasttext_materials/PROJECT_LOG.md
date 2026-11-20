# NLP 1 Assignment 2: complete chronological research record

Last documented: 20 November 2026. This is the single project-wide record of the work completed so far, in the order we did it: repository inspection and data reconstruction; fastText exploration; four embed-and-pool classifiers; 256- and 512-token LSTMs with a linear head; the 512-token LSTM with an MLP head; trainable pretrained embeddings in A2; random trainable embeddings in A3; half-data training in B1; 32- and 128-unit LSTMs in C1 and C3; lower- and higher-learning-rate runs D1 and D3; regularization runs E2 and E3; seed-123 repeats of the baseline and D3; and paired validation error analysis of the original LSTM against mean-pooling logistic regression. It combines methods, observations, interpretation, limitations, exact commands, and detailed results. The original [Phase 1](REPORT.md), [Phase 2](phase2/REPORT.md), and [Phase 3](phase3/REPORT.md) reports and all saved results are preserved; the older Phase 3 report predates the later follow-up runs. **All classifier results below are training or validation results, not final test performance.**

## Contents

1. [Initial inspection and data provenance](#initial-inspection-and-data-provenance)
2. [Pretrained model and environment](#pretrained-model-and-environment)
3. [Phase 1: embedding exploration](#phase-1-embedding-exploration)
4. [Phase 2: embed-and-pool classification](#phase-2-embed-and-pool-classification)
5. [Phase 3: 256- and 512-token linear-head LSTMs](#phase-3-linear-head-lstm-runs-256-then-512)
6. [Phase 3 follow-up: MLP classification head](#phase-3-follow-up-lstm-with-an-mlp-classification-head)
7. [Experiment A2: fine-tuned embedding lookup](#experiment-a2-fine-tuned-embedding-lookup)
8. [Experiment A3: random embedding initialization](#experiment-a3-random-embedding-initialization)
9. [Experiment B1: half-data training](#experiment-b1-half-data-training)
10. [Experiment C1: 32-unit LSTM](#experiment-c1-32-unit-lstm)
11. [Experiment C3: 128-unit LSTM](#experiment-c3-128-unit-lstm)
12. [Experiment D1: lower learning rate](#experiment-d1-lower-learning-rate)
13. [Experiment D3: higher learning rate](#experiment-d3-higher-learning-rate)
14. [Experiments E2 and E3: regularization](#experiments-e2-and-e3-regularization)
15. [Experiment F: seed-123 repeats](#experiment-f-seed-123-repeats)
16. [Project 1 reference](#project-1-reference)
17. [Experiment G1: paired validation error analysis](#experiment-g1-paired-validation-error-analysis)
18. [Overall observations](#overall-observations-after-all-completed-runs)
19. [Files, reproduction, and current boundary](#files-reproduction-and-current-boundary)
20. [Detailed results and complete epoch histories](#detailed-results-appendix)

## Project objective and scope

The assignment investigates pretrained English fastText embeddings for IMDb movie-review sentiment classification. The completed phases and subsequent follow-ups ask:

1. How do tokenization, direct vocabulary coverage, subword representations, spelling changes, and sentiment-related similarities behave on the IMDb training reviews?
2. With fixed pretrained embeddings, how do mean and max pooling compare when used with logistic regression and a small MLP?
3. How does a frozen-fastText LSTM compare with those baselines; what changed when the maximum retained sequence length increased from 256 to 512 tokens; and did a small nonlinear classification head improve on the 512-token linear head?
4. Does fine-tuning the pretrained token embedding lookup improve on the frozen 512-token LSTM, and how does it compare with randomly initialized trainable embeddings?
5. How does training on a stratified half of the training reviews compare with using the full training partition?
6. How do 32- and 128-unit LSTMs compare with the original 64-unit LSTM under the specified training settings?
7. How do Adam learning rates of 0.0003 and 0.003 compare with the original 0.001 under the specified epoch limit?
8. Do post-LSTM dropout or Adam weight decay improve validation performance and reduce the training–validation gap?
9. How much do the original LSTM and the 0.003 learning-rate alternative vary across training seeds?
10. Where do the LSTM and mean-pooling logistic regression disagree, and how do their validation results vary with review features?

Phase 1 was restricted to exploration: no classifier was trained. Phase 2 implemented four specified classifiers, which the user then ran. Phase 3 implemented a unidirectional LSTM: the user ran its linear head at 256 and 512 tokens, then ran the 512-token MLP head. A2 subsequently fine-tuned the pretrained token lookup; A3 trained a randomly initialized lookup using the same token IDs; B1 trained the frozen model on half of the project training reviews; C1 and C3 tested 32 and 128 hidden units; D1 and D3 tested lower and higher learning rates; E2 and E3 tested regularization; the baseline and D3 were repeated with seed 123; and G1 compared saved validation predictions. Each training implementation was completed before the user started its training command. No GRU, bidirectional LSTM, attention model, retraining of fastText's underlying character n-gram model, or held-out test evaluation has been performed as part of this work.

## Initial inspection and data provenance

The existing course repository contained `part1` and `part2` tutorials and experiments. The earlier `NLP1-Assignment1` project identified the actual dataset and prior split, but its documented local Hugging Face cache was absent from the copy available here. The pinned source files were restored into this Assignment 2 folder; the earlier project and the teammate's existing experiments were left untouched.

The source is [mteb/imdb revision `d05f0155defa7991dad75bc68c5ccb6774b1fdc5`](https://huggingface.co/datasets/mteb/imdb/tree/d05f0155defa7991dad75bc68c5ccb6774b1fdc5). Its published files contain 24,904 training-source and 24,678 test-source reviews. Assignment 1 **pooled those published files** and created its own stratified split with seed 42:

| Project partition | Reviews | Purpose so far |
|---|---:|---|
| Training | 34,707 | Phase 1 exploration and Phase 2/3 model fitting |
| Validation | 7,437 | Phase 2/3 checkpoint selection and model comparisons |
| Held-out test | 7,438 | Reserved; no final test evaluation so far |

Training contains 17,288 negative and 17,419 positive reviews; validation contains 3,705 negative and 3,732 positive reviews. Negative is label 0 and positive is label 1. The source's published `train` and `test` file names should not be confused with the resulting project-specific partitions.

For classification, **accuracy** is the fraction of correctly predicted reviews. For either class, **precision** is the fraction of its predicted reviews that are correct, and **recall** is the fraction of its actual reviews that were identified. A class's F1 is the harmonic mean of its precision and recall; **macro-F1** averages the two class F1 scores equally. Confusion matrices in this record use true labels as rows and predictions as columns: `[[true negative, false positive], [false negative, true positive]]`.

Because the published partitions were pooled, the resulting project test split is **not** the official IMDb benchmark test split. The program verifies the pinned source files and the earlier project's ordered text/label fingerprint before reconstructing the split. Source labels were needed to reproduce the existing stratification and fingerprint. After the split, Phase 1 analyzed only training texts; Phases 2 and 3 prepared training and validation inputs only and never evaluated the project test partition.

The ordered source text/label SHA-256 is `0ed0de756c57a4ac8868e7cc2a9970650e930d94cdb0347716d385c9e1e705be`. The ordered training-index SHA-256 is `3fe569c54add383b4b7249ea3465297678599f30dc8c0d48d43ee7186f09ec82`; the Phase 2 validation-index SHA-256 is `4eb93d6010100998cfc33aecf6cf1533ca6ffb845ce463d88d2139a0f0e7514b`. The [Phase 2 run metadata](phase2/results/seed-42_epochs-30/run_config.json) records these and the class mapping: negative = 0, positive = 1.

## Pretrained model and environment

The original `cc.en.300.bin` binary was downloaded once and kept in `models/`. Its uncompressed SHA-256 is `14c7167b130056944cbdc37b7451f055867fe9a4e3fed3bbc1ecc0e74f6763ca`. It has 300-dimensional vectors, 2,000,000 stored vocabulary words, and `minn = maxn = 5`. The [official model documentation](https://fasttext.cc/docs/en/crawl-vectors.html) describes training on Common Crawl **and Wikipedia**, using CBOW with position weights, character n-grams of length five, and English preprocessing based on the Europarl tools. The binary format can generate subword-based vectors for many OOV strings; the existence of a nonzero vector does not establish semantic quality.

Phase 1 used a separate Python 3.11 environment and pinned dependencies. Phase 2 was implemented in its own directory and environment, extending the Phase 1 lock with PyTorch 2.6.0. The completed run recorded Python 3.11.14, fastText 0.9.3, NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2, and PyTorch 2.6.0; the MLPs used Apple's MPS backend. The original binary was reused instead of downloaded again. Phase 3 and the later follow-up runs used the Phase 2 environment; all 14 documented LSTM runs used MPS. Fixed seeds do not promise bit-for-bit reproducibility across hardware or package versions.

## Phase 1: embedding exploration

The implemented [Phase 1 runner](run_phase1.py), [configuration](config.py), output tables, and three plots compare four preprocessing variants on the full training partition. The two additional variants isolate break-tag removal and lowercasing; they were not used to choose a classification winner.

The questions were whether punctuation-aware tokenization improved direct vocabulary coverage, whether character n-grams retained useful information for unfamiliar or deliberately misspelled words, and whether the embedding space reflected sentiment polarity as well as broader semantic relatedness. The initial expectations were improved coverage from punctuation separation, some but uneven spelling robustness, and high contextual similarity for at least some sentiment opposites. These were hypotheses, not assumed results.

`whitespace` splits only at whitespace; `regex` separates punctuation while keeping internal straight or curly apostrophes and original case; `regex_br` additionally removes HTML break tags; and `regex_br_lower` removes breaks and lowercases after tokenization. A **direct** match is an exact case-sensitive `get_word_id(token) >= 0`. Token coverage counts occurrences, while type coverage counts distinct strings. A token without a stored entry is OOV; a nonzero subword vector shows representability, not that the vector has a useful meaning.

| Tokenization / preprocessing | Token occurrences | Types | Direct token coverage | Direct type coverage |
|---|---:|---:|---:|---:|
| Whitespace | 8,013,808 | 349,444 | 91.14% | 31.11% |
| Regex, case and markup preserved | 9,814,658 | 120,099 | 99.31% | 81.70% |
| Regex, `<br />` breaks removed | 9,252,706 | 120,098 | 99.27% | 81.70% |
| Regex, breaks removed, lowercased | 9,252,706 | 96,312 | 98.85% | 69.04% |

These denominators differ: a regex that separates punctuation and raw markup creates extra tokens, so its direct-coverage percentage is not a classifier-performance result. For case-preserving regex with break tags removed, 67,262 token occurrences were absent from the stored vocabulary; 65,162 had nonzero OOV vectors and 2,100 had zero vectors. The corresponding type counts were 21,979 OOV, of which 21,963 were nonzero. Lowercasing changed 43,256 token occurrences from direct to OOV and 3,663 in the opposite direction; for example, `I've` → `i've` lost direct membership while `Let's` → `let's` gained it.

Real training-review examples were saved in [tokenization_examples.csv](results/full/tokenization_examples.csv). For the actual fragment `say, I was very impressed.`, whitespace produces `['say,', 'I', 'was', 'very', 'impressed.']`, while regex produces `['say', ',', 'I', 'was', 'very', 'impressed', '.']`. In a second review, the regex keeps `don't` together, whereas the inspected English Europarl reference tool splits it into `don` and `'t`. For `it.<br /><br />Madhavi`, raw regex retains tokens representing each tag while `regex_br` removes them. The regex also splits `comedy-drama-musical-spy`, while the reference tool retains the hyphenated expression. The [reference comparison](results/full/europarl_reference.csv) uses a public Europarl toolkit version, which is not proven to be the exact historical version used for fastText training. Coverage and alignment alone do not determine the best downstream tokenizer; later classifiers deliberately used the original `regex`, not the other variants.

For natural OOV examples, the code recorded token frequencies and original training-review excerpts. `slowwwwww` occurred twice as an expressive form, `Nazarin` occurred 26 times as a movie/name candidate, and `anacronistic` occurred twice with `anachronistic` as a strong but unverified contextual correction. Other suggestions are uncertain: `blax` was suggested as `blah` despite a film-related context, illustrating why unfamiliar words should not automatically be corrected. See the [natural examples](results/full/natural_oov_examples.csv) and [OOV frequency table](results/full/natural_oov_types.csv).

These natural examples were selected from the `regex_br` training vocabulary. Possible corrections were restricted to stored lowercase words occurring at least 20 times in training and within one deletion, insertion, substitution, or adjacent transposition; at most the first 1,500 eligible frequent OOV types were screened. The original tokens and excerpts were verified, but the authors' intended spellings were not. For example, `esqe` appeared four times, including the excerpt `jungle-esqe island`; `esque` is a plausible guess. `film's` appeared 1,345 times and is a normal possessive, not a spelling error. All saved examples and guesses are reproduced in the appendix.

The controlled spelling probe used 19 correctly spelled training-vocabulary words and one seeded artificial corruption per word for each of five edit types. The configured word `bad` was excluded because the generator requires at least four characters; insertion used the alphabet `x/a/e` and substitution `x/z/a/e`. Mean original/corruption cosine similarities were 0.269 for deletion, 0.134 for insertion, 0.127 for substitution, 0.226 for transposition, and 0.202 for repetition; the unrelated control mean was 0.101. Six of the 95 corruptions were already directly stored vocabulary words, and the mean among the 15 **OOV deletions** was 0.202. These artificial corruptions were not presented as naturally occurring IMDb errors. In a nearest-neighbor search limited to the first 100,000 model words, `boriing` retrieved `boring`, `pointless`, and `tedious` among its top five, while `excellnt` did not retrieve `excellent` in its top five. See [spelling pairs](results/full/spelling_pairs.csv), [spelling chart](results/full/spelling_robustness.png), and [neighbors](results/full/nearest_neighbors.csv).

The identity control averaged 1.000. In this particular sample, directly stored synthetic `beautful` had cosine 0.847 with `beautiful`, whereas the OOV synthetic pairs `excellent`/`excellnt` and `boring`/`boriing` had cosines 0.268 and 0.330. The small selected sample and edit generator do not represent the distribution of natural typos. A character edit can disrupt multiple five-character n-grams, and shared character fragments can be misleading; zero vectors, weak neighbors, and hash collisions are possible limitations. These are explanations consistent with the mechanism, not tested causal ablations.

The manual semantic probe used six pairs per category. Mean cosine was 0.709 for positive–positive, 0.746 for negative–negative, 0.508 for positive–negative, 0.061 for unrelated controls, and 0.509 for morphological controls. The individual `good`–`bad` cosine was 0.752; `terrible`–`awful` was 0.868. Opposite-sentiment words can be related in context, but proximity does not prove that the embedding lacks usable sentiment information. This is a small, selected probe rather than a representative statistical sample. See the [individual pairs](results/full/semantic_pairs.csv) and [semantic chart](results/full/semantic_similarity.png).

Other individual examples include `amazing`–`fantastic` at 0.835, unrelated `boring`–`river` at 0.028, and morphologically related `bored`–`boring` at 0.651. Similar contexts offer a plausible reason that opposites such as `good` and `bad` are close, but the probe does not establish the cause. A downstream classifier can still learn a sentiment boundary from these embeddings; individual cosines alone cannot predict classification accuracy.

The full Phase 1 run was reproduced with matching key tables. Six checks passed, and the sampled tokenization and OOV excerpts were verified against their training reviews. The [Phase 1 report](REPORT.md) documents hypotheses, definitions, methods, examples, and limitations in detail.

## Phase 2: embed-and-pool classification

At the user's request, Phase 2 code was implemented without starting training or extracting IMDb embeddings. Eight synthetic checks covered pooling, zero vectors, the original regex, the probability threshold, cache identity, and graph regeneration from temporary synthetic metrics. Afterward, the user ran:

```bash
phase2/.venv/bin/python phase2/run.py --experiment all
```

The run reused the same original split and frozen fastText binary. All four models used the original Phase 1 **`regex` tokenizer with capitalization and HTML break tags preserved**. Each review produced both a mean-pooled and a max-pooled 300-dimensional vector. Zero vectors were excluded from both pools; empty/all-zero reviews would map to a zero vector. The run recorded 2,100 zero-vector training tokens and 333 validation tokens, with no empty or all-zero reviews. The pooled arrays were cached for reuse.

Logistic regression used a scaler fitted only on training features, `C=1.0`, `max_iter=1000`, no class weighting, and seed 42. Both MLPs used unscaled features and exactly `Linear(300,64) → ReLU → Linear(64,1)`, Adam with learning rate 0.001, batch size 64, BCE-with-logits loss, 30 epochs, no dropout or weight decay, and seed 42. MLP checkpoints were selected by the highest validation macro-F1. Both model families used a 0.5 probability threshold.

| Model | Pooling | Train accuracy | Validation accuracy | Validation macro-F1 | Best MLP epoch |
|---|---|---:|---:|---:|---:|
| Logistic regression | Mean | 0.8518 | 0.8407 | 0.8406 | — |
| Logistic regression | Max | 0.6727 | 0.6628 | 0.6626 | — |
| MLP (64) | Mean | 0.8497 | 0.8365 | 0.8365 | 28 |
| MLP (64) | Max | 0.6698 | 0.6605 | 0.6604 | 19 |

Validation precision and recall are given for each class below. Matrices are `[[TN, FP], [FN, TP]]`, with true classes as rows and predictions as columns.

| Experiment | Negative precision / recall | Positive precision / recall | Validation confusion matrix | Training time |
|---|---|---|---|---:|
| LR-Mean | 0.8439 / 0.8345 | 0.8375 / 0.8467 | `[[3092, 613], [572, 3160]]` | 0.69 s |
| LR-Max | 0.6670 / 0.6451 | 0.6588 / 0.6803 | `[[2390, 1315], [1193, 2539]]` | 0.26 s |
| MLP-Mean | 0.8379 / 0.8329 | 0.8351 / 0.8400 | `[[3086, 619], [597, 3135]]` | 26.71 s |
| MLP-Max | 0.6521 / 0.6826 | 0.6696 / 0.6385 | `[[2529, 1176], [1349, 2383]]` | 25.56 s |

In these runs, mean pooling exceeded max pooling by 17.79 percentage points of validation accuracy for LR and 17.60 points for the MLP. LR-Mean exceeded MLP-Mean by 0.42 percentage points on validation accuracy. This is a single split and seed, and LR used standardized inputs while the MLP did not, so the small LR/MLP difference cannot be attributed solely to classifier architecture. Both model comparisons and MLP checkpoint selection used validation; none of these results is a final test score.

MLP-Mean's best macro-F1 checkpoint was epoch 28. MLP-Max's best macro-F1 checkpoint was epoch 19 even though its lowest validation loss occurred at epoch 28 and its final-epoch validation accuracy was slightly higher: checkpoint selection followed the predefined macro-F1 criterion. The [Phase 2 report](phase2/REPORT.md) records every model's class precision/recall, confusion matrix, timing, warning, learning-curve interpretation, and caveats. The complete [comparison table](phase2/results/seed-42_epochs-30/comparison.csv) and [mean/max plot](phase2/results/seed-42_epochs-30/graphs/mean_vs_max.png) are saved with the run.

MLP-Mean's validation loss was 0.5361 at epoch 1 and 0.3707 at the selected epoch 28; its epoch-30 validation macro-F1 was 0.8334. MLP-Max selected epoch 19 with macro-F1 0.660350; its lowest validation loss was 0.6164 at epoch 28, while its final macro-F1 was 0.660196. The complete epoch histories for both models appear below. LR was fitted once; no epoch-based LR curves were invented.

Both LR fits recorded a SciPy deprecation warning about L-BFGS-B logging options; neither recorded a convergence warning, and their iteration counts were below the configured limit. The feature-preparation time was 36.39 seconds; LR fit times and MLP training times have different timing definitions, so they are not a controlled speed comparison.

## Phase 3: linear-head LSTM runs (256, then 512)

The Phase 3 implementation was first checked using synthetic sequences, a mocked fastText object, and small MPS forward/gradient checks; no IMDb training was started during implementation. The user then ran the initial 256-token experiment and a follow-up with `--max-sequence-length 512 --seed 42`. Both reused the original project split, the original case- and markup-preserving regex, and frozen 300-dimensional fastText embeddings. Each used a one-layer unidirectional LSTM with hidden size 64, a single `Linear(64,1)` classifier, Adam at 0.001, batch size 32, BCE-with-logits, no dropout or weight decay, and at most 30 epochs. Early stopping used patience 5 and the highest validation macro-F1; ties kept the earlier checkpoint.

| Model | Train accuracy | Validation accuracy | Validation macro-F1 | Best epoch / stopped after |
|---|---:|---:|---:|---:|
| LSTM, first 256 tokens | 0.9120 | 0.8791 | 0.8790 | 13 / 18 |
| LSTM, first 512 tokens | 0.9451 | 0.9045 | 0.9045 | 17 / 22 |

For the 256-token limit, 38.80% of training reviews and 38.42% of validation reviews were longer than the limit; 29.94% and 29.67% of the respective token occurrences were discarded. For 512 tokens, those figures were 11.79% and 11.90% of reviews, and 9.71% and 9.70% of tokens. The 512-token model achieved 2.55 percentage points higher validation macro-F1 than the 256-token model, and 6.39 points higher than Phase 2's LR-Mean baseline. The 512 configuration followed the 256 run and both selected checkpoints using the same validation data, so these are exploratory single-seed validation comparisons, not independent final scores or proof of statistical significance.

The model families also differ in more than sequence order: Phase 2 pooled full reviews and excluded zero-vector tokens, while Phase 3 kept only the first 256 or 512 tokens and retained zero-vector positions. The higher LSTM validation score therefore does not isolate the effect of word order. Both LSTM loss histories show training loss continuing to decrease while validation loss rose later in training, a pattern consistent with overfitting; early stopping restored the best macro-F1 checkpoints.

The 256-token checkpoint's validation matrix was `[[3161, 544], [355, 3377]]`; the 512-token checkpoint's was `[[3324, 381], [329, 3403]]`, with actual labels as rows and predicted labels as columns. Training-loop times were 298.71 seconds for 18 epochs and 664.90 seconds for 22; these include per-epoch evaluations and checkpoint writes, but exclude preprocessing and final reevaluation. The [Phase 3 report](phase3/REPORT.md) includes both class-level results, full truncation counts, learning-curve interpretation, commands, and limitations. See the saved [256-token results](phase3/results/lstm-s42-h64-l256-b32-lr0.001-e30/) and [512-token results](phase3/results/lstm-s42-h64-l512-b32-lr0.001-e30/).

At 256 tokens the best checkpoint had negative precision/recall 0.8990/0.8532 and positive precision/recall 0.8613/0.9049. At 512 tokens these were 0.9099/0.8972 and 0.8993/0.9118. The 512-token model made 710 validation errors versus 899 for the 256-token model; the aggregate confusion matrices do not tell us which reviews the models both misclassified.

For 256 tokens, the selected epoch 13 had validation loss 0.2992, while the lowest validation loss was 0.2983 at epoch 10. Training loss then decreased from 0.2169 at epoch 13 to 0.1635 at epoch 18, while validation loss increased to 0.3230. For 512 tokens, the selected epoch 17 had validation loss 0.2645; minimum validation loss was 0.2553 at epoch 15. Training loss decreased from 0.1468 at epoch 17 to 0.1078 at epoch 22, while validation loss rose to 0.3124. These are patterns consistent with overfitting, and the different loss-minimum epochs reflect selection by macro-F1 rather than loss.

The unidirectional model consumes an ordered vector for each token, including individual fastText subword vectors for OOV tokens. Sequences are right-truncated to their first N tokens and dynamically padded in each batch. The output at each review's last valid token is used, so right padding does not determine the classification state. An empty review would receive a zero hidden state; neither full run had empty reviews. The frozen lookup kept actual zero-vector tokens in sequence, unlike Phase 2 pooling, which excluded them.

| Limit | Partition | Reviews over limit | Tokens discarded / all tokens | Token share discarded |
|---:|---|---:|---:|---:|
| 256 | Train | 13,466 / 34,707 (38.80%) | 2,938,399 / 9,814,658 | 29.94% |
| 256 | Validation | 2,857 / 7,437 (38.42%) | 621,532 / 2,094,907 | 29.67% |
| 512 | Train | 4,092 / 34,707 (11.79%) | 952,551 / 9,814,658 | 9.71% |
| 512 | Validation | 885 / 7,437 (11.90%) | 203,188 / 2,094,907 | 9.70% |

The 512-token limit retained 1,985,848 more training tokens and 418,344 more validation tokens than the 256-token limit. The corresponding frozen lookups contained 111,826 and 125,403 distinct retained tokens, respectively, excluding the padding ID. The lookup also included vectors for validation-only strings, computed from the pretrained model; it did not fit a vocabulary cutoff, normalization statistic, or model parameter on validation data.

## Phase 3 follow-up: LSTM with an MLP classification head

After observing the 512-token linear-head result, the next question was: **does a small nonlinear head improve the LSTM's 0.9045 validation macro-F1?** The controlled implementation retained the existing tokenizer, split, 512-token sequence policy, frozen fastText table, one-layer unidirectional 64-unit LSTM, optimizer, seed, batch size, threshold, epoch limit, and early-stopping rule. It replaced only the classifier `Linear(64, 1)` with `Linear(64, 64) → ReLU → Linear(64, 1)`. The LSTM and head were optimized jointly with raw logits and `BCEWithLogitsLoss`; sigmoid was used for evaluation. Dropout and weight decay stayed at zero. The linear head has 65 trainable parameters; the MLP head has 4,225, an increase of 4,160. Changing that capacity and the resulting optimization trajectory is inherent in the head comparison.

Before training, 17 synthetic/read-only checks covered the MLP structure, LSTM/head gradients with frozen embeddings, padding, old and new checkpoint formats, cache/baseline identity, command-line safeguards, and plotting. They did not train on IMDb. The new result directory was separate from both earlier LSTM runs. The user then ran:

```bash
phase2/.venv/bin/python phase3/run.py --head mlp --head-hidden-size 64 --max-sequence-length 512
```

The run reused the 512-token sequence/vector cache `e76303b57c7e56e4879d739d`; its manifest matched the prior linear run's source, split, tokenizer, pretrained model, retained sequences, vectors, and file checksums. It used MPS, Adam at 0.001, batch size 32, seed 42, and patience 5. Sequence and truncation counts therefore match the 512-token row above. No embeddings or dataset were re-created and no new model was downloaded.

| Model, 512 tokens | Training accuracy | Training macro-F1 | Validation accuracy | Validation macro-F1 | Best epoch | Stopped after | Train − validation macro-F1 gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM + linear | 0.9451 | 0.9451 | **0.9045** | **0.9045** | 17 | 22 | 0.0405 |
| LSTM + MLP (64) | 0.9246 | 0.9246 | 0.8979 | 0.8979 | 11 | 16 | 0.0267 |

The MLP head's validation accuracy and macro-F1 were each about **0.66 percentage points lower** than the linear head's on this split. Their exact saved macro-F1 values were 0.897941 and 0.904521, respectively. The MLP had the smaller training–validation gap at its selected checkpoint, but it also had lower training and validation scores; the smaller gap alone does not prove that it generalizes better or diagnose how much it overfits.

For the MLP checkpoint, validation negative precision/recall was 0.8977/0.8974 and positive precision/recall was 0.8982/0.8984. Its matrix was `[[3325, 380], [379, 3353]]`: 759 errors, versus 710 for the linear checkpoint. Relative to the linear aggregate, the MLP had one fewer false positive and 50 more false negatives, for 49 more errors overall. This does not reveal whether the errors occur on the same reviews.

The MLP selected epoch 11, where training/validation loss was 0.1926/0.2583, and stopped after epoch 16 because none of the next five epochs strictly improved validation macro-F1. Epoch 11 also had the lowest validation loss in this run. By epoch 16, training loss had fallen to 0.1285 while validation loss had risen to 0.2787, and validation macro-F1 had fallen from 0.8979 to 0.8909. That divergence is consistent with overfitting after the selected epoch; it is not proof of its cause. Early losses also fluctuated, so the trajectory was not monotonic. Training-loop time was 487.42 seconds for 16 epochs, excluding preparation and final reevaluation. The 512-token linear run took 664.90 seconds for 22 epochs under the same timing definition; these totals reflect different numbers of epochs and are not a controlled speed comparison.

This is a closer comparison than LSTM versus pooling because the two 512-token runs share the exact cached input vectors and specified settings. Nevertheless, each is a **single run**, the heads have different capacities and training trajectories, and the same validation data helped select both checkpoints and the sequence-length follow-up. The observed 0.66-point difference is not a statistical significance test and does not establish that a linear head always wins. No held-out test set was evaluated.

The saved [MLP-head metrics](phase3/results/lstm-mlp64-s42-h64-l512-b32-lr0.001-e30/metrics.json), [per-epoch history](phase3/results/lstm-mlp64-s42-h64-l512-b32-lr0.001-e30/history.csv), [head comparison and gaps](phase3/results/lstm-mlp64-s42-h64-l512-b32-lr0.001-e30/head_comparison.csv), and [comparison graph](phase3/results/lstm-mlp64-s42-h64-l512-b32-lr0.001-e30/graphs/head_comparison.png) retain full numerical precision and recorded provenance.

## Experiment A2: fine-tuned embedding lookup

After implementing the separate [remaining-experiments suite](remaining_experiments/README.md), the user ran A2 to ask whether making the pretrained embedding lookup trainable improves the 512-token LSTM. The command was:

```bash
phase2/.venv/bin/python remaining_experiments/run.py --experiment A2
```

A2 initialized its lookup from the same cached 300-dimensional fastText vectors, including the original subword-derived vectors for OOV tokens, and trained individual lookup rows jointly with the LSTM and linear head. This **fine-tunes a token lookup table**; it does not retrain the original fastText model's word or character n-gram parameters, or modify its binary. Padding remains excluded from training. The lookup contains training and validation token types, as in the earlier Phase 3 cache; validation-only rows are not trained using validation labels.

The saved A2 and frozen-baseline configurations have the same project split, ordered training and validation IDs, tokenizer, 512-token cache identity and file hashes, seed 42, MPS device, 64-unit unidirectional LSTM, linear head, batch size 32, Adam learning rate 0.001, zero dropout and weight decay, threshold 0.5, maximum 30 epochs, and patience 5. The intended change is whether the pretrained lookup rows are trainable; this also changes the number of trainable parameters and the optimization trajectory. Both runs select the earliest checkpoint with the strictly highest validation macro-F1.

| 512-token LSTM | Training accuracy | Training macro-F1 | Validation accuracy | Validation macro-F1 | Best epoch | Stopped after | Train − validation macro-F1 gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen lookup, original baseline | 0.9451 | 0.9451 | **0.9045** | **0.9045** | 17 | 22 | 0.0405 |
| A2: trainable pretrained lookup | 0.9737 | 0.9737 | 0.8908 | 0.8907 | 2 | 7 | 0.0831 |

The exact saved validation macro-F1 values are **0.904521** for the frozen model and **0.890660** for A2: A2 is **1.39 percentage points lower** on this validation split. Its accuracy is 1.37 points lower. At the selected checkpoints, A2's training–validation macro-F1 gap is **8.31 percentage points**, versus **4.05 points** for the frozen baseline. A larger gap is consistent with greater overfitting in these runs, but the gap alone does not establish a cause or a general result across seeds.

At A2's selected epoch 2, training and validation loss were **0.0896 / 0.2700**. By epoch 7, training loss had fallen to **0.0022** while validation loss had risen to **0.6184**; training and validation macro-F1 were **0.9996 / 0.8748**. That divergence is consistent with overfitting after epoch 2. Validation macro-F1 did not strictly improve in epochs 3–7, so patience 5 stopped training and the epoch-2 checkpoint was restored. Training-loop time was **322.49 seconds**, including per-epoch evaluation and checkpoint writes but excluding cache loading and final reevaluation. The frozen model took 664.90 seconds over 22 epochs; different epoch counts and checkpoint sizes mean these totals are not a controlled speed comparison.

For A2, validation negative precision/recall was **0.8605 / 0.9320**, and positive precision/recall was **0.9264 / 0.8499**. Its confusion matrix was `[[3453, 252], [560, 3172]]` (true classes in rows; negative/positive predictions in columns), totaling **812 errors**. The frozen baseline's matrix was `[[3324, 381], [329, 3403]]`, totaling **710 errors**. A2 therefore had 129 fewer false positives but 231 more false negatives, for 102 more errors overall. These are aggregate counts; they do not identify which individual reviews changed between models.

This comparison uses one seed and a validation set already used for several choices. It does not prove that freezing always outperforms fine-tuning, or that a particular factor caused the difference. Further repetitions or a separately specified fine-tuning/regularization study would be new experiments. The currently configured E2 and E3 experiments regularize the **frozen** baseline, not A2, so they do not by themselves test whether regularization fixes this fine-tuned run. No held-out test evaluation was performed.

The [A2 metrics](remaining_experiments/results/A2/seed-42/metrics.json), [configuration](remaining_experiments/results/A2/seed-42/run_config.json), [epoch history](remaining_experiments/results/A2/seed-42/history.csv), [loss graph](remaining_experiments/results/A2/seed-42/graphs/loss_by_epoch.png), and [confusion matrix](remaining_experiments/results/A2/seed-42/graphs/confusion_matrix.png) preserve the recorded results. The complete seven-epoch history is reproduced in the appendix.

## Experiment A3: random embedding initialization

After A2, the user ran A3 to investigate how a randomly initialized trainable lookup compares with the pretrained trainable lookup and frozen baseline:

```bash
phase2/.venv/bin/python remaining_experiments/run.py --experiment A3
```

A3 reused the same cached **token IDs and ordered sequences**, project train/validation split, seed 42, MPS device, 512-token limit, 64-unit unidirectional LSTM, linear head, batch size 32, Adam learning rate 0.001, zero dropout/weight decay, and patience 5. Each token lookup row was initialized from a seeded normal distribution with mean 0 and standard deviation 0.05; the padding row remained zero and the lookup and LSTM were trained jointly. The random generator for the table was independent of the LSTM initialization.

Unlike A2, A3 did **not** use pretrained fastText vector values or character n-gram information. It used the same token-to-ID mapping, including distinct IDs for tokens that would be OOV in fastText, but their A3 vectors were random and learned only from training occurrences. Validation-only token types were present in the shared lookup, as in the existing cache, but were not fitted using validation labels. Thus A2 versus A3 compares pretrained versus random initialization with both lookups trainable; the accompanying difference in subword information is part of that comparison. The frozen baseline differs from A3 in **both** initialization and trainability.

| 512-token LSTM | Train macro-F1 | Validation accuracy | Validation macro-F1 | Best epoch | Stopped after | Train − validation macro-F1 gap |
|---|---:|---:|---:|---:|---:|---:|
| Frozen pretrained lookup | 0.9451 | 0.9045 | **0.9045** | 17 | 22 | 0.0405 |
| A2: trainable pretrained lookup | 0.9737 | 0.8908 | 0.8907 | 2 | 7 | 0.0831 |
| A3: random trainable lookup | 0.9981 | 0.8689 | 0.8689 | 7 | 12 | 0.1293 |

A3's exact validation macro-F1 was **0.868856**, compared with **0.890660** for A2 and **0.904521** for the frozen baseline. A2 exceeded A3 by **2.18 percentage points**; the frozen baseline exceeded A3 by **3.57 points**. In this one-seed comparison, pretrained initialization and its subword-derived representations were associated with better validation macro-F1 than random initialization when both lookups were trainable. This does not establish that pretraining will yield the same improvement across seeds, datasets, or training choices.

At the selected epoch 7, A3's training macro-F1 was **0.9981** against **0.8689** on validation, a **12.93-point** gap; training and validation loss were **0.0068 / 0.5835**. The large gap is consistent with overfitting in this run, but it does not identify its cause. The history was noisy: validation macro-F1 dropped to **0.7441** at epoch 6, then reached its maximum at epoch 7. Validation loss was lowest at epoch 3 (**0.3877**), not epoch 7 (**0.5835**), because the checkpoint-selection rule used macro-F1. Loss and macro-F1 measure different properties; a claim about calibration would require further analysis. After five epochs without a strict validation macro-F1 improvement, training stopped at epoch 12 and restored epoch 7. Training-loop time was **555.46 seconds** over 12 epochs, including per-epoch evaluation and checkpoint writes but excluding cache loading and final reevaluation. Different epoch counts prevent a direct total-time speed comparison with A2 or the frozen run.

At A3's best checkpoint, validation negative precision/recall was **0.8794 / 0.8540** and positive precision/recall was **0.8591 / 0.8837**. Its confusion matrix was `[[3164, 541], [434, 3298]]`, totaling **975 errors**; A2 had 812 and the frozen baseline 710. These aggregate counts do not show which reviews the models have in common. The results come from one seed and a repeatedly used validation split; no held-out test evaluation was performed.

The saved [A3 metrics](remaining_experiments/results/A3/seed-42/metrics.json), [configuration](remaining_experiments/results/A3/seed-42/run_config.json), [epoch history](remaining_experiments/results/A3/seed-42/history.csv), [loss graph](remaining_experiments/results/A3/seed-42/graphs/loss_by_epoch.png), and [confusion matrix](remaining_experiments/results/A3/seed-42/graphs/confusion_matrix.png) preserve the recorded evidence. The complete 12-epoch history appears in the appendix.

## Experiment B1: half-data training

After A3, the user ran B1 to investigate how the amount of labeled training data affects the frozen-fastText LSTM:

```bash
phase2/.venv/bin/python remaining_experiments/run.py --experiment B1
```

B1 trained on **17,353** reviews, selected by a seed-42 stratified 50% split of the existing 34,707-review project training partition. The saved training metrics show **8,644 negative** and **8,709 positive** examples in that subset; the selected review IDs and their fingerprint are saved with the run. The comparison baseline B2 is the existing frozen 512-token LSTM trained on all **34,707** project training reviews. Both models were evaluated on the same **7,437 validation reviews**.

The saved configurations match on the original project split and validation IDs, 512-token sequence/vector cache and fingerprints, original case-preserving regex and HTML behavior, frozen fastText lookup, seed 42, MPS device, 64-unit unidirectional LSTM, linear head, batch size 32, Adam learning rate 0.001, zero dropout/weight decay, threshold 0.5, maximum 30 epochs, and early-stopping patience 5. B1 changes the training-data fraction; the smaller dataset also means fewer optimization batches per epoch. Its training score is measured on its selected half, while B2's is measured on the full training set.

The shared frozen lookup was built earlier from token types in the full project training and validation partitions, including training reviews outside B1's selected half. Its vectors were supplied by the pretrained model; no labels from the unused half or validation set were used to fit the B1 classifier. Consequently, B1 studies the amount of **labeled training data**, not a setting in which only half the unlabeled text was available for constructing the token lookup.

| Frozen 512-token LSTM | Training reviews | Train macro-F1 | Validation accuracy | Validation macro-F1 | Best epoch | Stopped after | Train − validation macro-F1 gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1: stratified half | 17,353 | 0.9195 | 0.8922 | 0.8922 | 16 | 21 | 0.0273 |
| B2: full-data baseline | 34,707 | 0.9451 | **0.9045** | **0.9045** | 17 | 22 | 0.0405 |

The exact validation macro-F1 values were **0.892158** for B1 and **0.904521** for B2. The full-data run scored **1.24 percentage points higher** on macro-F1 and about **1.24 points higher** on accuracy. B1's selected-checkpoint training–validation macro-F1 gap was **2.73 points**, versus **4.05 points** for B2. Because the two training metrics use different sets of reviews and the models selected different epochs, the smaller gap does not by itself establish better generalization or quantify overfitting.

B1's best epoch was 16, with training/validation loss **0.2104 / 0.2716**; its validation loss was also lowest at that epoch. The validation macro-F1 then failed to improve strictly for five epochs, so training stopped after epoch 21 and restored epoch 16. At epoch 21 the training loss had fallen to **0.1805**, while validation loss had risen to **0.2939** and validation macro-F1 was **0.8817**. This later divergence is consistent with overfitting, although it does not identify a cause. The earlier history was noisy: validation macro-F1 fell from **0.7022** at epoch 3 to **0.3759** at epoch 4, before recovering. No cause for this fluctuation was established.

At B1's selected checkpoint, validation negative precision/recall was **0.8930 / 0.8901**, and positive precision/recall was **0.8913 / 0.8942**. Its confusion matrix was `[[3298, 407], [395, 3337]]`, with true classes as rows and predictions as columns: **802 validation errors**. The full-data model's matrix was `[[3324, 381], [329, 3403]]`, or **710 errors**. B1 had 26 more false positives and 66 more false negatives in aggregate, for 92 more errors; these counts do not reveal which individual reviews the models both missed.

B1's recorded training-loop time was **335.11 seconds** for 21 epochs, compared with **664.90 seconds** for 22 epochs for the full-data baseline. These times include per-epoch training/validation evaluation and checkpoint writes, but exclude cache loading and final reevaluation. B1 used 543 training batches per epoch versus 1,085 for the full run. The elapsed times describe these individual runs and are not a controlled speed benchmark.

The result is one stratified subset and one training seed, with repeated use of the validation set for checkpoint and model comparisons. The selected B1 checkpoint followed 16 × 543 = **8,688** training batches, versus 17 × 1,085 = **18,445** batches for the full-data checkpoint. It suggests that the full training set helped under this setup, but the comparison does not isolate data quantity from the number of optimization updates or establish the effect for other subsets, seeds, or budgets. No held-out test evaluation was performed. The [B1 metrics](remaining_experiments/results/B1/seed-42/metrics.json), [configuration](remaining_experiments/results/B1/seed-42/run_config.json), [epoch history](remaining_experiments/results/B1/seed-42/history.csv), [loss graph](remaining_experiments/results/B1/seed-42/graphs/loss_by_epoch.png), and [confusion matrix](remaining_experiments/results/B1/seed-42/graphs/confusion_matrix.png) preserve the measured evidence; all 21 epochs appear in the appendix.

## Experiment C1: 32-unit LSTM

After B1, the user ran C1 to compare a smaller recurrent hidden state with the existing 64-unit frozen-embedding baseline, C2:

```bash
phase2/.venv/bin/python remaining_experiments/run.py --experiment C1
```

C1 used a **32-unit**, one-layer unidirectional LSTM followed by `Linear(32, 1)`; the baseline used 64 units and `Linear(64, 1)`. Their saved configurations match on the full 34,707-review training partition, the 7,437 validation IDs, original regex and markup handling, 512-token sequence/vector cache and fingerprints, frozen pretrained fastText lookup, seed 42, MPS, batch size 32, Adam learning rate 0.001, zero dropout/weight decay, threshold 0.5, maximum 30 epochs, and early-stopping patience 5. Changing hidden size necessarily changes the recurrent and classifier parameter shapes and the optimization trajectory. C1 used the newer remaining-experiments training code, which reuses the original LSTM and evaluation utilities.

| Frozen 512-token LSTM | Train macro-F1 | Validation accuracy | Validation macro-F1 | Best epoch | Stopped after | Train − validation macro-F1 gap |
|---|---:|---:|---:|---:|---:|---:|
| C1: 32 hidden units | 0.8265 | 0.8167 | 0.8164 | 4 | 9 | 0.0101 |
| C2: 64 hidden units | 0.9451 | **0.9045** | **0.9045** | 17 | 22 | 0.0405 |

The exact validation macro-F1 was **0.816374** for C1 and **0.904521** for C2: the 64-unit run was **8.81 percentage points higher**. Its accuracy was **8.78 points higher**. C1's selected-checkpoint train–validation macro-F1 gap was only **1.01 points**, compared with **4.05 points** for C2, but C1's training and validation scores were both lower. A smaller gap alone does not mean a model generalizes better, and these observations do not establish whether limited capacity, optimization, the stopping rule, or another factor caused the difference.

C1's validation macro-F1 fluctuated substantially: it reached **0.8164** at epoch 4, fell to **0.6289** at epoch 8, then recovered to **0.8099** at epoch 9. Because epochs 5–9 did not strictly exceed the epoch-4 best, the specified patience-5 rule stopped the run after epoch 9 and restored epoch 4. Whether further training would have improved the score is unknown. C1's validation loss was also lowest at epoch 4 (**0.4317**). Its training-loop time was **251.05 seconds** over nine epochs, compared with **664.90 seconds** over 22 epochs for the baseline; different architectures and epoch counts mean these totals are not a controlled speed comparison.

At C1's best checkpoint, validation negative precision/recall was **0.8438 / 0.7757** and positive precision/recall was **0.7938 / 0.8574**. Its confusion matrix was `[[2874, 831], [532, 3200]]`, with true classes as rows and predictions as columns: **1,363 validation errors**. The 64-unit baseline's matrix was `[[3324, 381], [329, 3403]]`, or **710 errors**. C1 had 450 more false positives and 203 more false negatives in aggregate, for 653 more errors; these counts do not identify which individual reviews changed.

This is one seed and one validation split, already used for checkpoint selection and several model comparisons. It does not show that a 32-unit LSTM will always be worse under different seeds or optimization schedules. The 128-unit C3 configuration was run subsequently, as documented below. No held-out test evaluation was performed. The [C1 metrics](remaining_experiments/results/C1/seed-42/metrics.json), [configuration](remaining_experiments/results/C1/seed-42/run_config.json), [epoch history](remaining_experiments/results/C1/seed-42/history.csv), [loss graph](remaining_experiments/results/C1/seed-42/graphs/loss_by_epoch.png), and [confusion matrix](remaining_experiments/results/C1/seed-42/graphs/confusion_matrix.png) preserve the measured evidence; all nine epochs appear in the appendix.

## Experiment C3: 128-unit LSTM

After C1, the user ran C3 to complete the planned comparison of 32, 64, and 128 recurrent hidden units:

```bash
phase2/.venv/bin/python remaining_experiments/run.py --experiment C3
```

C3 used a **128-unit**, one-layer unidirectional LSTM followed by `Linear(128, 1)`. The saved configurations confirm that C1, C2, and C3 used the same 34,707-review training partition, 7,437 validation IDs, original case-preserving regex and markup behavior, 512-token sequence/vector cache and fingerprints, frozen pretrained fastText lookup, seed 42, MPS device, batch size 32, Adam learning rate 0.001, zero dropout/weight decay, threshold 0.5, maximum 30 epochs, and patience 5. Hidden size is the intended change; it necessarily changes parameter shapes and the optimization trajectory. C1 and C3 use the remaining-experiments training code; C2 is the earlier Phase 3 baseline, with the same original LSTM and evaluation utilities.

| Frozen 512-token LSTM | Train macro-F1 | Validation accuracy | Validation macro-F1 | Best epoch | Stopped after | Train − validation macro-F1 gap |
|---|---:|---:|---:|---:|---:|---:|
| C1: 32 hidden units | 0.8265 | 0.8167 | 0.8164 | 4 | 9 | 0.0101 |
| C2: 64 hidden units | 0.9451 | **0.9045** | **0.9045** | 17 | 22 | 0.0405 |
| C3: 128 hidden units | 0.9749 | 0.8961 | 0.8961 | 19 | 24 | 0.0788 |

C3's exact validation macro-F1 was **0.896054**, compared with **0.904521** for C2 and **0.816374** for C1. C3 was **0.85 percentage points below** C2 and **7.97 points above** C1 on this validation split. C3 fitted the training set more closely at its selected checkpoint—**0.9749** training macro-F1 versus **0.8961** validation, a **7.88-point** gap. The baseline's gap was **4.05 points**. The wider gap and subsequent loss divergence are consistent with overfitting in this run, but the gap alone does not establish its cause or a general ranking across seeds.

C3's best validation macro-F1 occurred at **epoch 19**, with training/validation loss **0.0777 / 0.3179**. Its lowest validation loss was earlier, at epoch 10 (**0.2631**), when validation macro-F1 was **0.8926**. Between epochs 10 and 19, training loss fell while validation loss rose, even as macro-F1 improved; the checkpoint was selected by macro-F1, not loss. These metrics assess different properties, and calibration was not analyzed. After five subsequent epochs without a strict macro-F1 improvement, the run stopped at epoch 24 and restored epoch 19. At epoch 24, training/validation loss was **0.0285 / 0.4440** and validation macro-F1 was **0.8857**.

At C3's best checkpoint, validation negative precision/recall was **0.8988 / 0.8918**, and positive precision/recall was **0.8934 / 0.9003**. Its confusion matrix was `[[3304, 401], [372, 3360]]`, with true classes as rows and predictions as columns: **773 validation errors**. C2's matrix was `[[3324, 381], [329, 3403]]`, or **710 errors**. C3 had 20 more false positives and 43 more false negatives in aggregate, for 63 more errors; these totals do not identify the individual reviews that changed.

C3's recorded training-loop time was **871.78 seconds** over 24 epochs, versus **664.90 seconds** over 22 epochs for C2 and **251.05 seconds** over nine epochs for C1. These timings include per-epoch evaluation and checkpoint writes but exclude cache loading and final reevaluation. Different architectures and epoch counts prevent a controlled total-time speed comparison.

On this one seed and repeatedly used validation split, 64 units had the highest macro-F1 of the three tested hidden sizes. The differences do not establish a universal ranking or rule out different outcomes under other seeds and optimization schedules; C1 in particular stopped after nine noisy epochs. No held-out test evaluation was performed. The [C3 metrics](remaining_experiments/results/C3/seed-42/metrics.json), [configuration](remaining_experiments/results/C3/seed-42/run_config.json), [epoch history](remaining_experiments/results/C3/seed-42/history.csv), [loss graph](remaining_experiments/results/C3/seed-42/graphs/loss_by_epoch.png), and [confusion matrix](remaining_experiments/results/C3/seed-42/graphs/confusion_matrix.png) preserve the evidence; all 24 epochs appear in the appendix.

## Experiment D1: lower learning rate

After C3, the user ran D1 to compare an Adam learning rate of **0.0003** with the existing 0.001 frozen-embedding LSTM baseline, D2:

```bash
phase2/.venv/bin/python remaining_experiments/run.py --experiment D1
```

The saved configurations match on the full 34,707-review training partition, the 7,437 validation IDs, original case-preserving regex and markup behavior, 512-token sequence/vector cache and fingerprints, frozen pretrained fastText lookup, seed 42, MPS device, 64-unit unidirectional LSTM with linear head, batch size 32, zero dropout/weight decay, threshold 0.5, maximum 30 epochs, and patience 5. The intended setting change is the Adam learning rate; it necessarily changes the optimization trajectory. D1 uses the remaining-experiments training code, which reuses the original LSTM and evaluation utilities; D2 is the earlier Phase 3 baseline.

| Frozen 512-token LSTM | Train macro-F1 | Validation accuracy | Validation macro-F1 | Best epoch | Epochs completed | Train − validation macro-F1 gap |
|---|---:|---:|---:|---:|---:|---:|
| D1: learning rate 0.0003 | 0.9141 | 0.8928 | 0.8928 | 30 | 30 | 0.0214 |
| D2: learning rate 0.001 | 0.9451 | **0.9045** | **0.9045** | 17 | 22 | 0.0405 |

The exact validation macro-F1 values were **0.892797** for D1 and **0.904521** for D2: D1 was **1.17 percentage points lower** on this validation split. Accuracy was also **1.17 points lower**. At their selected checkpoints, D1's train–validation macro-F1 gap was **2.14 points**, versus **4.05 points** for D2, but D1's training and validation scores were both lower. The smaller gap alone does not prove better generalization.

D1 reached the configured maximum of **30 epochs**; it did **not** stop early, and its highest validation macro-F1 was at the final epoch. At epoch 30, training/validation loss was **0.2208 / 0.2720**; that epoch also had the lowest validation loss recorded in the run. The validation history was noisy—for example, macro-F1 reached **0.8916** at epoch 26, fell to **0.8569** at epoch 27, and reached **0.8928** at epoch 30. Because the run ended at the prespecified limit while recording its best score, whether additional epochs would improve it is unknown. Changing the epoch budget would be a new experiment, not a result of this run.

At D1's selected checkpoint, validation negative precision/recall was **0.8759 / 0.9144**, and positive precision/recall was **0.9112 / 0.8714**. Its confusion matrix was `[[3388, 317], [480, 3252]]`, with true classes as rows and predictions as columns: **797 validation errors**. D2's matrix was `[[3324, 381], [329, 3403]]`, or **710 errors**. D1 had 64 fewer false positives but 151 more false negatives in aggregate, for 87 more errors; these totals do not identify which reviews changed.

D1's training-loop time was **914.53 seconds** over 30 epochs, versus **664.90 seconds** over 22 epochs for D2. Both timings include per-epoch train/validation evaluation and checkpoint writes but exclude cache loading and final reevaluation. The different numbers of completed epochs make these totals unsuitable as a controlled speed comparison.

This is one seed on a validation split already used for checkpoint selection and multiple model comparisons. The result does not establish that 0.001 is universally preferable or what a longer 0.0003 run would achieve. The higher-rate D3 configuration was run subsequently, as documented below. No held-out test evaluation was performed. The [D1 metrics](remaining_experiments/results/D1/seed-42/metrics.json), [configuration](remaining_experiments/results/D1/seed-42/run_config.json), [epoch history](remaining_experiments/results/D1/seed-42/history.csv), [loss graph](remaining_experiments/results/D1/seed-42/graphs/loss_by_epoch.png), and [confusion matrix](remaining_experiments/results/D1/seed-42/graphs/confusion_matrix.png) preserve the evidence; all 30 epochs appear in the appendix.

## Experiment D3: higher learning rate

After D1, the user ran D3 to test an Adam learning rate of **0.003**, comparing it with D1's 0.0003 and the original D2 baseline's 0.001:

```bash
phase2/.venv/bin/python remaining_experiments/run.py --experiment D3
```

The saved configurations identify the same 34,707 training and 7,437 validation reviews and the same 512-token sequence/vector cache and file fingerprints. The shared setup uses the original case-preserving regex tokenizer, frozen pretrained fastText lookup, seed 42, MPS, a 64-unit unidirectional LSTM and linear head, batch size 32, no dropout or weight decay, maximum 30 epochs, and patience 5. The intended setting changed is the Adam learning rate. D1 and D3 use the remaining-experiments training code; D2 is the earlier Phase 3 baseline. Changing the learning rate also changes the optimization trajectory.

| Frozen 512-token LSTM | Train macro-F1 | Validation accuracy | Validation macro-F1 | Best epoch | Epochs completed | Train − validation macro-F1 gap |
|---|---:|---:|---:|---:|---:|---:|
| D1: learning rate 0.0003 | 0.9141 | 0.8928 | 0.8928 | 30 | 30 | 0.0214 |
| D2: learning rate 0.001 | 0.9451 | **0.9045** | **0.9045** | 17 | 22 | 0.0405 |
| D3: learning rate 0.003 | 0.9345 | 0.9021 | 0.9021 | 5 | 10 | 0.0324 |

D3's exact validation macro-F1 was **0.902091**, compared with **0.904521** for D2 and **0.892797** for D1. On this validation split, D3 was **0.24 percentage points below** D2 and **0.93 points above** D1. D3's training–validation macro-F1 gap at the selected checkpoint was **3.24 points**; D2's was **4.05 points**, and D1's was **2.14 points**. A smaller gap by itself does not establish better generalization.

D3 selected **epoch 5** based on validation macro-F1; the same epoch also had its lowest recorded validation loss. After five subsequent epochs without a strict macro-F1 improvement, training stopped at epoch 10 and the epoch-5 checkpoint was restored. At epoch 5, training/validation loss was **0.1729 / 0.2487**, with macro-F1 **0.9345 / 0.9021**. By epoch 10, training loss had fallen to **0.0663** while validation loss had risen to **0.3396**, and macro-F1 was **0.9808 / 0.8866**. This divergence is consistent with overfitting after the selected checkpoint; it does not establish its cause.

At the best checkpoint, validation negative precision/recall was **0.9106 / 0.8910**, and positive precision/recall was **0.8940 / 0.9132**. The confusion matrix was `[[3301, 404], [324, 3408]]`, with true classes as rows and predictions as columns: **728 validation errors**. The original D2 baseline had **710** errors, with 23 fewer false positives and 5 more false negatives in aggregate. The matrices alone do not show which individual reviews changed.

D3's recorded training-loop time was **307.80 seconds** over 10 epochs, compared with **664.90 seconds** over 22 epochs for D2. Both timings include per-epoch train/validation evaluation and checkpoint writes but exclude cache loading and final reevaluation. They are not controlled speed comparisons, and elapsed time to the best epoch was not recorded.

These are single-seed results on a validation split already used for checkpoint selection and multiple model comparisons. D1 reached its 30-epoch limit while still recording its best score, so this comparison does not show how it would perform with a longer budget. The results do not establish a universal learning-rate ranking or statistical significance. No held-out test evaluation was performed. The [D3 metrics](remaining_experiments/results/D3/seed-42/metrics.json), [configuration](remaining_experiments/results/D3/seed-42/run_config.json), [epoch history](remaining_experiments/results/D3/seed-42/history.csv), [loss graph](remaining_experiments/results/D3/seed-42/graphs/loss_by_epoch.png), and [confusion matrix](remaining_experiments/results/D3/seed-42/graphs/confusion_matrix.png) preserve the evidence; all ten epochs appear in the appendix.

## Experiments E2 and E3: regularization

After the learning-rate experiments, the user launched E2 and E3 in separate terminal sessions with overlapping execution. The research question was whether regularization could improve validation performance or reduce the training–validation gap. Both use the original seed-42, 512-token frozen-fastText LSTM baseline's split, cached vectors, tokenization, 64-unit unidirectional encoder, linear head, Adam learning rate 0.001, batch size 32, maximum 30 epochs, and patience 5. The saved split and cache fingerprints match the baseline.

- **E1:** the existing baseline, without dropout or weight decay.
- **E2:** dropout probability 0.3 applied to the final valid LSTM representation before the linear layer during training. Dropout is disabled when training and validation metrics are evaluated. This is explicit post-LSTM dropout; PyTorch's between-layer LSTM dropout is not used for the one-layer encoder.
- **E3:** Adam weight decay 0.0001 on trainable parameters, with no dropout. This is Adam's weight-decay setting rather than AdamW.

| Seed-42 frozen 512-token LSTM | Train macro-F1 | Validation accuracy | Validation macro-F1 | Best epoch / completed | Train − validation macro-F1 gap |
|---|---:|---:|---:|---:|---:|
| E1: no regularization | 0.9451 | **0.9045** | **0.9045** | 17 / 22 | 0.0405 |
| E2: post-LSTM dropout 0.3 | 0.9538 | 0.8979 | 0.8979 | 17 / 22 | 0.0558 |
| E3: Adam weight decay 0.0001 | 0.9104 | 0.8946 | 0.8946 | 10 / 15 | 0.0158 |

E2's exact validation macro-F1 was **0.897940**, or **0.66 percentage points below** E1; E3's was **0.894580**, or **0.99 points below** E1. At the selected checkpoints, E2 had a **5.58-point** training–validation macro-F1 gap and E3 a **1.58-point** gap, compared with E1's **4.05 points**. E3's smaller gap came with lower training and validation scores; a smaller gap alone does not establish better generalization. E2's gap was larger in this run, despite the dropout intervention.

The checkpoint criterion was highest validation macro-F1, with the earliest epoch winning exact ties; five subsequent non-improving epochs triggered early stopping. E2's selected epoch was **17**, although its lowest validation loss occurred at epoch **10**, illustrating that selecting by macro-F1 and selecting by loss can produce different checkpoints. E3's selected epoch **10** also had its lowest recorded validation loss. E2 took **703.52 seconds** for 22 training-loop epochs and E3 **488.13 seconds** for 15; these concurrent runs shared resources and had different epoch counts, so their totals are not controlled speed comparisons.

At E2's selected checkpoint, negative precision/recall was **0.8985 / 0.8964**, positive precision/recall was **0.8974 / 0.8995**, and the validation confusion matrix was `[[3321, 384], [375, 3357]]` (**759 errors**). At E3's selected checkpoint, negative precision/recall was **0.8887 / 0.9012**, positive precision/recall was **0.9005 / 0.8880**, and the matrix was `[[3339, 366], [418, 3314]]` (**784 errors**). True labels are rows and predictions are columns. E1 made 710 validation errors; aggregate counts do not identify which reviews changed.

These are single-seed comparisons on a validation set used repeatedly for model development. They show the outcome for these particular dropout and weight-decay settings, not that regularization generally cannot help. No held-out test evaluation was performed. See the [E2 results](remaining_experiments/results/E2/seed-42/) and [E3 results](remaining_experiments/results/E3/seed-42/) for saved configurations, checkpoints, histories, metrics, and plots. The full per-epoch histories appear in the appendix.

## Experiment F: seed-123 repeats

Next, the user repeated the frozen 512-token baseline at learning rate 0.001 and D3 at 0.003 with seed **123**. Their execution overlapped. This varies the model-training seed while preserving the same project training/validation split and cached sequence/vector fingerprints. The original seed-42 baseline was trained using the earlier Phase 3 pipeline; the other runs use the remaining-experiments pipeline, which reuses the LSTM model and evaluation utilities. The selected alternative was identified using the seed-42 validation results, making this an exploratory follow-up.

| Seed | Baseline validation macro-F1, LR 0.001 | D3 validation macro-F1, LR 0.003 | D3 − baseline, percentage points |
|---:|---:|---:|---:|
| 42 | 0.904521 | 0.902091 | −0.243 |
| 123 | 0.899921 | 0.901842 | +0.192 |

The ordering reversed across the two seeds: D3 was **0.24 percentage points lower** at seed 42 and **0.19 points higher** at seed 123. Their two-seed validation macro-F1 averages are **0.902221** for the baseline and **0.901966** for D3; the corresponding sample standard deviations are **0.003253** and **0.000176**. The mean paired difference, D3 minus baseline, is **−0.025 percentage points**. With only two seeds, these standard deviations are unstable and do not establish that either model is generally superior. Both runs reuse the same validation split, so seed variability is not uncertainty across new data splits. Seed 2026 had not completed when this section was written.

At seed 123, the baseline selected epoch **11** and stopped after epoch **16**; its training/validation macro-F1 was **0.9234 / 0.8999**, with confusion matrix `[[3420, 285], [459, 3273]]`. D3 selected epoch **7** and stopped after epoch **12**; its training/validation macro-F1 was **0.9367 / 0.9018**, with confusion matrix `[[3357, 348], [382, 3350]]`. D3's lowest validation loss occurred at epoch 6, rather than its selected macro-F1 epoch 7. These are validation metrics, and no held-out test evaluation was performed.

The user launched the repeats with:

```bash
phase2/.venv/bin/python remaining_experiments/run.py --experiment F-base --seed 123
phase2/.venv/bin/python remaining_experiments/run.py --experiment F-alternative --alternative D3 --seed 123
```

The [baseline seed-123 results](remaining_experiments/results/F-base/seed-123/) and [D3 seed-123 results](remaining_experiments/results/D3/seed-123/) contain saved checkpoints, full-precision metrics, configurations, and graphs; the complete epoch histories are included below.

## Project 1 reference

Before G1, the user asked for the Project 1 outcome. The saved [Project 1 summary](../../NLP1-Assignment1/project_starter/PROJECT_SUMMARY.md) and [original validation scores](../../NLP1-Assignment1/project_starter/results/baseline/validation_scores.csv) identify the best original Project 1 model as BPE-token bigrams with a 64-unit MLP: validation macro-F1 **0.901155** and mean five-fold training cross-validation macro-F1 **0.903438**. The original baseline validation artifact does not record accuracy.

The 512-token fastText LSTM's validation macro-F1 was **0.904521**, about **0.34 percentage points** higher than that Project 1 validation score; mean-pooling logistic regression scored **0.840648**. These are different feature and classifier pipelines. Although the saved Project 1 setup describes the same pooled dataset, split proportions, seed, and counts, its historical result files do not record the ordered validation-ID fingerprints, so exact matching of the original evaluated reviews cannot be independently verified from those artifacts. The small difference does not by itself establish a meaningful improvement.

A separate Project 1 follow-up using 2,000-token BPE, unigram and bigram features, `min_df=2`, and a 64-unit MLP had mean five-fold training cross-validation macro-F1 **0.905027**. It was not evaluated on the separate validation partition, so its cross-validation score must not be directly compared with the LSTM's validation score. The Project 1 test partition was not evaluated. These values were read from existing results; no Project 1 or Project 2 training was started for this reference.

## Experiment G1: paired validation error analysis

After the regularization experiments and seed-123 repeats, the user ran a validation-only comparison of the saved seed-42 **512-token LSTM** and **mean-pooling logistic regression**:

```bash
phase2/.venv/bin/python remaining_experiments/error_analysis.py --left lstm512 --right lr-mean --output remaining_experiments/results/errors-lstm-vs-lr
```

This command loaded the saved models and aligned their predictions on the same **7,437** original validation review IDs; it did not train a model or evaluate the project test partition. The recomputed accuracy and macro-F1 of each model exactly matched its saved validation metrics. The analysis also reconstructed features from the original case-preserving regex tokens: review length, a surface-form negation indicator, direct-vocabulary OOV fractions on the whole review and on the 256/512-token prefixes, truncation flags, and model confidence. Here confidence is the larger of the predicted positive and negative probabilities; it is not evidence that the probabilities are calibrated. OOV means absent from fastText's stored vocabulary, not necessarily unrepresentable with subwords.

| Paired validation outcome | Reviews |
|---|---:|
| Both models correct | 5,975 |
| LSTM correct; logistic regression wrong | 752 |
| Logistic regression correct; LSTM wrong | 277 |
| Both models wrong | 433 |
| Total | 7,437 |

Overall validation macro-F1 was **0.9045** for the LSTM and **0.8406** for logistic regression. The 752 versus 277 disagreements explain the LSTM's net advantage in correct predictions, while showing that logistic regression still got some reviews right that the LSTM missed. The LSTM made 381 false positives and 329 false negatives; logistic regression made 613 and 572, respectively. At their selected checkpoints, the training–validation macro-F1 gaps were **4.05 percentage points** for the LSTM and **1.11 points** for logistic regression.

| Full review length in regex tokens | Reviews | LSTM macro-F1 | Logistic-regression macro-F1 |
|---|---:|---:|---:|
| 0–128 | 981 | 0.9184 | 0.8403 |
| 129–256 | 3,599 | 0.9005 | 0.8352 |
| 257–512 | 1,972 | 0.9102 | 0.8408 |
| Over 512 | 885 | 0.8914 | 0.8610 |

For the **885** reviews longer than 512 regex tokens, the LSTM's macro-F1 was **0.8914**, compared with **0.9063** on the 6,552 reviews of at most 512 tokens. The LSTM retained only the first 512 tokens; the logistic model pooled the full review. The LSTM–logistic macro-F1 difference was about **3.05 percentage points** for the over-512 group and **6.84 points** for the at-most-512 group. Length and truncation are associated here, but these groups can differ in other ways, so this alone does not establish truncation as the cause. The architectures, input lengths, handling of zero vectors, scaling, and optimization also differ; the overall model comparison does not isolate word order.

The surface negation indicator appeared in **6,510** reviews: LSTM/logistic macro-F1 was **0.9019 / 0.8391** in that group and **0.9134 / 0.8309** in the **927** reviews without the indicator. The groups differ in class balance, and a negation marker does not demonstrate that negation caused a particular error. For the **5,614** reviews with at most 1% direct-vocabulary OOV tokens, LSTM/logistic macro-F1 was **0.9043 / 0.8441**; for the **1,814** with over 1% to 5% OOV tokens, it was **0.9038 / 0.8271**. Only **9** reviews had more than 5% OOV tokens, too few for a reliable conclusion. These are descriptive subgroup comparisons, not causal tests of OOV handling.

In the **236** reviews where the LSTM's own confidence was between 0.50 and 0.60, its accuracy was **0.5424** and logistic regression's was **0.6229**. This subgroup was selected using the LSTM's confidence, so the result is not a general comparison of model calibration. All subgroup counts and metrics, including the 256/512-prefix OOV and both confidence groupings, are in the appendix and [subgroups.csv](remaining_experiments/results/errors-lstm-vs-lr/subgroups.csv).

Two illustrative review openings show disagreements without establishing why they occurred:

- Review **43835** has a positive label and includes “I'm surprised this wasn't worse.” The LSTM predicted positive (positive probability **0.9718**); logistic regression predicted negative (**0.4239**).
- Review **22492** has a positive label and begins “I enjoyed The Night Listener very much.” The LSTM predicted negative (positive probability **0.2442**); logistic regression predicted positive (**0.7222**).

These are short excerpts from longer reviews, not the complete inputs. The review labels and predictions support the stated disagreements; they do not establish why either model failed. The [paired predictions](remaining_experiments/results/errors-lstm-vs-lr/per_example.csv), [disagreements](remaining_experiments/results/errors-lstm-vs-lr/disagreements.csv), [summary](remaining_experiments/results/errors-lstm-vs-lr/summary.json), [length graph](remaining_experiments/results/errors-lstm-vs-lr/accuracy_by_length.png), and [paired-outcome graph](remaining_experiments/results/errors-lstm-vs-lr/paired_outcomes.png) preserve the evidence. This validation set has already been used for model selection, so subgroup observations are exploratory.

## Overall observations after all completed runs

| Chronological experiment | Training accuracy | Validation accuracy | Validation macro-F1 | Selected epoch |
|---|---:|---:|---:|---:|
| LR + mean pooling | 0.8518 | 0.8407 | 0.8406 | — |
| LR + max pooling | 0.6727 | 0.6628 | 0.6626 | — |
| 64-unit MLP + mean pooling | 0.8497 | 0.8365 | 0.8365 | 28 |
| 64-unit MLP + max pooling | 0.6698 | 0.6605 | 0.6604 | 19 |
| LSTM + linear, first 256 tokens | 0.9120 | 0.8791 | 0.8790 | 13 |
| LSTM + linear, first 512 tokens | 0.9451 | **0.9045** | **0.9045** | 17 |
| LSTM + 64-unit MLP, first 512 tokens | 0.9246 | 0.8979 | 0.8979 | 11 |
| LSTM + linear, first 512 tokens, trainable pretrained lookup (A2) | 0.9737 | 0.8908 | 0.8907 | 2 |
| LSTM + linear, first 512 tokens, random trainable lookup (A3) | 0.9981 | 0.8689 | 0.8689 | 7 |
| LSTM + linear, first 512 tokens, frozen lookup, half training data (B1) | 0.9195 | 0.8922 | 0.8922 | 16 |
| LSTM + linear, first 512 tokens, frozen lookup, 32 hidden units (C1) | 0.8266 | 0.8167 | 0.8164 | 4 |
| LSTM + linear, first 512 tokens, frozen lookup, 128 hidden units (C3) | 0.9749 | 0.8961 | 0.8961 | 19 |
| LSTM + linear, first 512 tokens, frozen lookup, learning rate 0.0003 (D1) | 0.9142 | 0.8928 | 0.8928 | 30 |
| LSTM + linear, first 512 tokens, frozen lookup, learning rate 0.003 (D3) | 0.9345 | 0.9021 | 0.9021 | 5 |
| LSTM + linear, first 512 tokens, dropout 0.3 (E2), seed 42 | 0.9538 | 0.8979 | 0.8979 | 17 |
| LSTM + linear, first 512 tokens, Adam weight decay 0.0001 (E3), seed 42 | 0.9104 | 0.8946 | 0.8946 | 10 |
| LSTM + linear, first 512 tokens, baseline, seed 123 | 0.9234 | 0.9000 | 0.8999 | 11 |
| LSTM + linear, first 512 tokens, learning rate 0.003 (D3), seed 123 | 0.9367 | 0.9018 | 0.9018 | 7 |

The best observed validation result remains the seed-42 512-token **frozen-embedding** linear-head LSTM with 64 hidden units and learning rate 0.001, trained on the full training partition. In this specific sequence of runs, mean pooling outperformed max pooling, increasing the LSTM prefix from 256 to 512 tokens improved validation macro-F1, and neither the later MLP head nor A2's trainable lookup improved the matched frozen 512-token result. A2's pretrained trainable lookup exceeded A3's random trainable lookup, B1's half-data frozen model scored below the full-data baseline, both the 32- and 128-unit runs scored below the 64-unit baseline, and both D1's 0.0003 and D3's 0.003 learning-rate runs scored below D2's 0.001 baseline at seed 42; D3 scored above D1. E2's dropout and E3's weight decay did not exceed the seed-42 baseline. At seed 123, D3 narrowly exceeded the baseline, reversing their seed-42 ordering. C1 also scored below the best pooled baseline, although that comparison changes several settings. These describe observations rather than universal rankings. The Phase 2/3 comparison also changes truncation, zero-vector handling, architecture, scaling, and optimization, so it cannot attribute the whole difference to word order. Pretraining-data overlap with the review corpus was not assessed.

The subsequent paired analysis found **752** validation reviews correct only for the LSTM and **277** correct only for logistic regression. The LSTM's performance differed across review-length groups, but the available comparison does not establish a causal role for truncation or any other single feature. The Project 1 bigram MLP's separate-validation macro-F1 of **0.9012** is a useful historical reference, with the split-fingerprint limitation noted above.

## Files, reproduction, and current boundary

- [Phase 1 guide](README.md), [research report](REPORT.md), and [results](results/full/) contain the original exploration and reproduction instructions.
- [Phase 2 guide](phase2/README.md), [research report](phase2/REPORT.md), [runner](phase2/run.py), and [results](phase2/results/seed-42_epochs-30/) contain the classifier implementation, commands, outputs, and interpretation.
- [Phase 3 guide](phase3/README.md) and [runner](phase3/run.py) describe both LSTM heads and reproduction. The earlier [Phase 3 report](phase3/REPORT.md) describes the two linear-head runs; this consolidated record adds the later MLP-head run.
- The [remaining-experiments guide](remaining_experiments/README.md), [run order](remaining_experiments/RUN_ORDER.md), [A2 outputs](remaining_experiments/results/A2/seed-42/), [A3 outputs](remaining_experiments/results/A3/seed-42/), [B1 outputs](remaining_experiments/results/B1/seed-42/), [C1 outputs](remaining_experiments/results/C1/seed-42/), [C3 outputs](remaining_experiments/results/C3/seed-42/), [D1 outputs](remaining_experiments/results/D1/seed-42/), and [D3 outputs](remaining_experiments/results/D3/seed-42/) document the later experiment suite and completed follow-up runs.
- The [E2 results](remaining_experiments/results/E2/seed-42/), [E3 results](remaining_experiments/results/E3/seed-42/), [baseline seed-123 results](remaining_experiments/results/F-base/seed-123/), and [D3 seed-123 results](remaining_experiments/results/D3/seed-123/) contain the regularization and repeat-run artifacts.
- The [G1 error-analysis results](remaining_experiments/results/errors-lstm-vs-lr/) contain paired validation predictions, descriptive subgroup tables, disagreements, and graphs.
- Phase 2 saves LR pipelines, MLP checkpoints, per-epoch histories, metrics in CSV/JSON, per-model charts, and overall comparison charts. Run `phase2/.venv/bin/python phase2/run.py --help` for options; the Phase 2 guide explains separate run directories and graph-only regeneration.
- Dataset files, the large binary, environments, caches, and Phase 2/3 and remaining-experiment results/models are locally ignored by Git. They exist in this workspace but must be explicitly preserved or exported if the project is shared.
- The validation split has now been used for model comparison, MLP and LSTM checkpoint selection, the comparison of two LSTM lengths, the later head comparison, A2, A3, B1, C1, C3, D1, D3, E2, E3, the seed-123 repeats, and the G1 error analysis. Keep the project test partition reserved until a final evaluation is explicitly planned. Any additional error analysis, repeated training, cross-validation, or change in preprocessing would be **new work**, not part of the results already reported here.

For the record, after entering `assignment 2`, the user created the Phase 2 environment and ran the classification experiments and validation analysis with the following commands, in order of launch. Some later runs overlapped. The Phase 1 exploration had already been completed. These historical commands are **not** executed by reading this document; their default result directories already exist and are protected against overwrite.

```bash
python3.11 -m venv phase2/.venv
uv pip install --python phase2/.venv/bin/python -r phase2/requirements.txt
phase2/.venv/bin/python phase2/run.py --experiment all
phase2/.venv/bin/python phase3/run.py
phase2/.venv/bin/python phase3/run.py --max-sequence-length 512 --seed 42
phase2/.venv/bin/python phase3/run.py --head mlp --head-hidden-size 64 --max-sequence-length 512
phase2/.venv/bin/python remaining_experiments/run.py --experiment A2
phase2/.venv/bin/python remaining_experiments/run.py --experiment A3
phase2/.venv/bin/python remaining_experiments/run.py --experiment B1
phase2/.venv/bin/python remaining_experiments/run.py --experiment C1
phase2/.venv/bin/python remaining_experiments/run.py --experiment C3
phase2/.venv/bin/python remaining_experiments/run.py --experiment D1
phase2/.venv/bin/python remaining_experiments/run.py --experiment D3
phase2/.venv/bin/python remaining_experiments/run.py --experiment E2
time phase2/.venv/bin/python remaining_experiments/run.py --experiment E3
phase2/.venv/bin/python remaining_experiments/run.py --experiment F-base --seed 123
phase2/.venv/bin/python remaining_experiments/run.py --experiment F-alternative --alternative D3 --seed 123
phase2/.venv/bin/python remaining_experiments/error_analysis.py --left lstm512 --right lr-mean --output remaining_experiments/results/errors-lstm-vs-lr
```

For a fresh repeat, choose a new `--run-dir`; rerunning training is a new experiment, and scores or stopping epochs may differ. The Phase 3 CLI defaults to 256 tokens for the linear head and 512 for the controlled MLP head. To regenerate the existing MLP-head graphs without retraining or reading the dataset:

```bash
phase2/.venv/bin/python phase3/run.py --head mlp --plots-only
```

The original Phase 1 [coverage](results/full/coverage.png), [spelling](results/full/spelling_robustness.png), and [similarity](results/full/semantic_similarity.png) plots and the Phase 2 [pooling comparison](phase2/results/seed-42_epochs-30/graphs/mean_vs_max.png) are saved. Each LSTM results directory contains loss, accuracy, macro-F1, and confusion-matrix graphs; the MLP-head directory also contains the direct head-comparison graph. In every final metrics file, the neural model's scores and confusion matrix correspond to its best validation macro-F1 checkpoint, not necessarily its final epoch.

## Detailed results appendix

The following tables reproduce the saved Phase 1 sample outputs and all logged neural-training epochs in chronological experiment order. They make the record readable in one place; the linked CSV/JSON artifacts retain full precision and additional metadata. Values are rounded to four decimals, and the underlying saved rows, sample provenance, and comparisons remain authoritative. The full OOV frequency file contains 21,979 types and is linked above rather than duplicating the entire frequency dataset here.

### A. All saved real-review tokenization examples

Review IDs refer to the pooled source rows assigned to the training partition. The four outputs are reproduced from the sampled excerpt; later classifiers used the second output, original case-preserving regex with markup retained.

**punctuation — training review 24835**

Original excerpt: just watched this movie and have to say, I was very impressed. It's very creepy

```text
whitespace:    ["just", "watched", "this", "movie", "and", "have", "to", "say,", "I", "was", "very", "impressed.", "It's", "very", "creepy"]
regex:         ["just", "watched", "this", "movie", "and", "have", "to", "say", ",", "I", "was", "very", "impressed", ".", "It's", "very", "creepy"]
regex_br:      ["just", "watched", "this", "movie", "and", "have", "to", "say", ",", "I", "was", "very", "impressed", ".", "It's", "very", "creepy"]
regex_br_lower:["just", "watched", "this", "movie", "and", "have", "to", "say", ",", "i", "was", "very", "impressed", ".", "it's", "very", "creepy"]
```

**contraction — training review 19671**

Original excerpt: here. Even Dyan Cannon - who I normally don't care for - is perfectly cast in a role

```text
whitespace:    ["here.", "Even", "Dyan", "Cannon", "-", "who", "I", "normally", "don't", "care", "for", "-", "is", "perfectly", "cast", "in", "a", "role"]
regex:         ["here", ".", "Even", "Dyan", "Cannon", "-", "who", "I", "normally", "don't", "care", "for", "-", "is", "perfectly", "cast", "in", "a", "role"]
regex_br:      ["here", ".", "Even", "Dyan", "Cannon", "-", "who", "I", "normally", "don't", "care", "for", "-", "is", "perfectly", "cast", "in", "a", "role"]
regex_br_lower:["here", ".", "even", "dyan", "cannon", "-", "who", "i", "normally", "don't", "care", "for", "-", "is", "perfectly", "cast", "in", "a", "role"]
```

**capitalization — training review 21340**

Original excerpt: the first season of Cosby is released on DVD someday. The episode with Hilton's eccentric

```text
whitespace:    ["the", "first", "season", "of", "Cosby", "is", "released", "on", "DVD", "someday.", "The", "episode", "with", "Hilton's", "eccentric"]
regex:         ["the", "first", "season", "of", "Cosby", "is", "released", "on", "DVD", "someday", ".", "The", "episode", "with", "Hilton's", "eccentric"]
regex_br:      ["the", "first", "season", "of", "Cosby", "is", "released", "on", "DVD", "someday", ".", "The", "episode", "with", "Hilton's", "eccentric"]
regex_br_lower:["the", "first", "season", "of", "cosby", "is", "released", "on", "dvd", "someday", ".", "the", "episode", "with", "hilton's", "eccentric"]
```

**html_break — training review 20293**

Original excerpt: irrespective of the crudeness attached to it.&lt;br /&gt;&lt;br /&gt;Madhavi (Konkan Sharma) is a journalist

```text
whitespace:    ["irrespective", "of", "the", "crudeness", "attached", "to", "it.<br", "/><br", "/>Madhavi", "(Konkan", "Sharma)", "is", "a", "journalist"]
regex:         ["irrespective", "of", "the", "crudeness", "attached", "to", "it", ".", "<", "br", "/", ">", "<", "br", "/", ">", "Madhavi", "(", "Konkan", "Sharma", ")", "is", "a", "journalist"]
regex_br:      ["irrespective", "of", "the", "crudeness", "attached", "to", "it", ".", "Madhavi", "(", "Konkan", "Sharma", ")", "is", "a", "journalist"]
regex_br_lower:["irrespective", "of", "the", "crudeness", "attached", "to", "it", ".", "madhavi", "(", "konkan", "sharma", ")", "is", "a", "journalist"]
```

**hyphenated_expression — training review 37230**

Original excerpt: Andrews, is reason enough to see this comedy-drama-musical-spy spoof. Julie is beautiful

```text
whitespace:    ["Andrews,", "is", "reason", "enough", "to", "see", "this", "comedy-drama-musical-spy", "spoof.", "Julie", "is", "beautiful"]
regex:         ["Andrews", ",", "is", "reason", "enough", "to", "see", "this", "comedy", "-", "drama", "-", "musical", "-", "spy", "spoof", ".", "Julie", "is", "beautiful"]
regex_br:      ["Andrews", ",", "is", "reason", "enough", "to", "see", "this", "comedy", "-", "drama", "-", "musical", "-", "spy", "spoof", ".", "Julie", "is", "beautiful"]
regex_br_lower:["andrews", ",", "is", "reason", "enough", "to", "see", "this", "comedy", "-", "drama", "-", "musical", "-", "spy", "spoof", ".", "julie", "is", "beautiful"]
```

### B. All saved natural OOV examples

These are observed training-review strings under case-preserving regex with break tags removed. Frequencies and excerpts are measured; **all proposed corrections are unverified guesses**. Names, slang, creative spellings, and possessives need not be errors. A dash means no correction was suggested.

| Token | Frequency | Review ID | Original excerpt | Suggested correction(s) |
|---|---:|---:|---|---|
| crocky | 4 | 12136 | crocodile special effects on a computer, all we'd need would be a cast of crocky fodder with Russ Meyer breasts and Ren Hoek pectoral implants.&lt;br | rocky &#124; cocky |
| esqe | 4 | 2271 | nearly killed by a mutated shark-man and trudging around a jungle-esqe island, there's nothing more cheerful than a middle-aged man reciting | esque |
| sammo | 4 | 13202 | sammo has to have a 10 out of 10 for this movie as it has everything. great | sammy &#124; sammi |
| bowm | 3 | 7358 | you, as long as you do something for me...' I can almost hear the bowm-chicka-bowm-bowm.&lt;br /&gt;&lt;br /&gt;When the parents view the footage shot | boom &#124; bowl |
| gadar | 3 | 35577 | but, the movie was a let down. specially for neone who has seen 'gadar'. inspite of the lead being played by sunny deol and the incessant | radar |
| kiger | 3 | 45602 | playboys famous playmates from the 70's like sexy tall blonde Susan kiger,Pamela jean Bryant,Lisa London,kc winkler and the late sexy Angela | tiger &#124; kier |
| remorsed | 3 | 41766 | documentary. A little bias but it hit hard on the government. I'm not very remorsed about the adults but the children............. it's very sad. Strong | remorse |
| actioneer | 2 | 34836 | again in their 3rd film -- YESTERDAY ONCE MORE -- directed by HK's actioneer Johnnie To...fans of To's action films will be disappointed to find | actioner |
| anacronistic | 2 | 44337 | brilliant, the plot is interesting, the characters - while SLIGHTLY anacronistic - are anacronistic not due to their intrinsic nature (there WERE | anachronistic |
| ardh | 2 | 22393 | napunsaktha, doosre palde mein paurush, aur teek tarazu ke kaante par, ardh satya ?". A rough translation - "The delicate balance of right &amp; | arch |
| bevare | 2 | 31742 | Dr. Alton also provides off screen narration. Lugosi keeps saying, bevare, bevare...take care, take care, as well as, some gibberish about | beware |
| blax | 2 | 25303 | try "Coffy" &amp; "Friday Foster" with Pam Grier. For a more realistic blax experience try "Black Heat". | blah |
| CATIII | 13 | 21914 | while. I'm a big fan of the "old-school" late 80s to mid 90s era CATIII films, and I had been hearing that that "style" of films is making | — |
| Froggg | 3 | 29158 | but instead this title only bores. First off, there is almost no Froggg in the entire movie which is the biggest disappointment. I have to | — |
| eeeb | 3 | 12518 | mickey by Christine McIntyre are superb; he gives his trademark sound (eeeb-eeeb-eeeb) and Miss McIntyre is noticeably trying not to laugh! Other | — |
| CNNN | 2 | 46397 | are watching. The Chaser crew who do the satirical newspaper and CNNN try something new by mixing live comedy, pre-recorded skits and political | — |
| III's | 2 | 18005 | pull-out-all-the-stops exciting fight scenes, and Howard Anderson III's gritty photography, this immensely enjoyable down'n'dirty exploitation | — |
| Masssacre | 2 | 21472 | fan this is a must see.&lt;br /&gt;&lt;br /&gt;Focusing on The Brian Jonestown Masssacre and The Dandy Warhols over the years is a brilliant way to show the | — |
| slowwwwww | 2 | 6264 | was ultra hammy, the plot was annoying, and the pace was SLOW, sooo slowwwwww. The whole time sitting in the theater i wanted the movie to end. | — |
| AAAAARGH | 1 | 26601 | who have been made for under 100.000 dollars and were still better. AAAAARGH! I count myself lucky that I didn't pay one penny to see this crap, | — |
| AAAAARRRRRRGGGGGGHHHHHH | 1 | 31045 | that, he was worse in this. Plus a dour soundtrack by Then Jericho AAAAARRRRRRGGGGGGHHHHHH. There is one film equally as bad as this i saw in the cinema Arthur | — |
| AAAAATCH | 1 | 24622 | NORMAL person would scream, he shouts something that sounds like "AAAAATCH-KAH!!" This movie you've popped in...isn't a normal movie. You've | — |
| AAAAHHHHHH | 1 | 36027 | extent that every time i've watched it I found my self screaming AAAAHHHHHH !! What a terrible gross !!..&lt;br /&gt;&lt;br /&gt;Cancel it please.. You've | — |
| AAAAWWWWWW | 1 | 41209 | this was so cool, and Larisa Oleynik was the first crush I ever had (AAAAWWWWWW). It was well written, funny-except when it it wanted to be serious, | — |
| film's | 1345 | 25839 | /&gt;Occasionally, Showtime gets laughs, but there simply aren't enough for the film's nearly two hour running time. Even worse, the really smart gags suggest | — |
| 80's | 633 | 18846 | just a baby) I later caught repeats of it on television in the late 80's, just when I was getting interested in the war and all of its aspects. | — |
| weren't | 628 | 5260 | and Michael Berrymore...just not in THIS movie...you can tell they weren't getting paid, or weren't getting paid much, because neither their | — |
| movie's | 567 | 26492 | shut it off b/c I am sorry but plot holes that are integral to a movie's plot make the movie nothing but garbage.&lt;br /&gt;&lt;br /&gt;The movie is | — |
| 70's | 476 | 22532 | crew discovers the body of a 30-foot yeti that resembles a hirsute 70's disco stud (complete with jumbo wavy afro) perfectly preserved in | — |
| Let's | 438 | 4579 | Frenchmen; and the Xavier character isn't going to sway them.&lt;br /&gt;&lt;br /&gt;Let's consider his stereotypical Frenchman attributes:&lt;br /&gt;&lt;br /&gt;1). Cynical | — |
| man's | 423 | 37013 | car, complete with the sound of the train traveling. In fact, the man's house is a train car, and he has a cassette of train sounds. The | — |
| character's | 385 | 6761 | disappointed with the ending of the film and the resolution of the character's relationships - Mikaele was the player who only messed around with | — |
| hadn't | 366 | 38956 | This is an excellent documentary about a story I hadn't heard about before. The first solo, non-stop sailing race around | — |
| director's | 362 | 43861 | start to finish. I saw it 3 times as I waited for my copy of the director's cut to arrive.&lt;br /&gt;&lt;br /&gt;Then I got the DC and watched it. Wow! | — |
| What's | 355 | 39117 | &lt;br /&gt;&lt;br /&gt;His partner prepares him for the moment by suggesting: "What's the worst that can happen? She says no."&lt;br /&gt;&lt;br /&gt;Armed with newfound | — |
| could've | 244 | 5925 | of this movie are people running up and down the same stairwell. I could've fixed the screenplay to this thing in half an hour- more cheese, | — |
| Nazarin | 26 | 41954 | critic, but when you have seen this and then you see a movie like "Nazarin", after you see this work you begin to be questioned about its author. | — |
| Hackenstein | 25 | 3449 | features an o.k. score and a not bad performance by David Muir as Dr. Hackenstein. The beginning and end credits show along with the most of the actors | — |
| Mraovich | 22 | 9120 | proclaimed "hollywood actor, director" and may i add beached whale Sam Mraovich, this film is legendarily terrible. Let me give you a hint of how | — |
| PolarisDiB | 21 | 46302 | that makes the movie pure visual pleasure to observe.&lt;br /&gt;&lt;br /&gt;--PolarisDiB | — |
| Grunnick | 20 | 49302 | time without saying anything. The love triangle is completed by Tom Grunnick. He is the slightly aloof ex-sportscaster who is the new reporter. | — |
| Protée | 18 | 23886 | English after class). The most obvious is the relationship between Protée and Aimée compared to the one between Protée and France.&lt;br /&gt;&lt;br | — |
| Hundstage | 15 | 18587 | sense of reality that never leaves the viewer throughout the movie. Hundstage is no exception. This effect is achieved not only by the depicted | — |
| Dumbland | 13 | 18085 | of a whiff of and are embarrassed to admit you enjoy.&lt;br /&gt;&lt;br /&gt;Dumbland distilled down beyond this is art. What can you do with aspects of | — |
| Gojoe | 13 | 16825 | One of the less widely lauded of recent Asian period action affairs Gojoe is an at first slow and often curious but overall pretty terrific | — |
| Interferencia | 13 | 26676 | Interferencia starts as unemployed Martin Sanders (Andres Bagg) hears something | — |
| Maléfique | 13 | 23515 | part about the film (unless he chose to include the baby). Anyway, Maléfique was a good way to get my respect back. . . it's a French film (obviously | — |
| Digicorp | 12 | 23289 | with his life and has decided to take a job as a company spy for Digicorp, a large technological corporation. He meets up with a recruitment | — |
| Holotik | 12 | 37829 | The impossibly sexy Rosie Holotik plays Charlotte Beale, a new nurse at one of those movie-type asylums | — |
| DEATHTRAP | 11 | 19670 | double and triple crosses.&lt;br /&gt;&lt;br /&gt;Sidney Lumet's comedy-thriller DEATHTRAP, as derived from Ira Levin's hit Broadway play, is a great example. | — |

### C. Every controlled spelling pair and control

The five edit categories are **synthetic corruptions**, not claims of natural IMDb misspellings. Original and comparison frequencies are training-token counts. “Direct” refers to the comparison string being stored in the pretrained vocabulary. Identity and unrelated rows are controls, not corruptions.

| Kind | Original | Comparison | Original frequency | Comparison frequency | Comparison direct? | Cosine |
|---|---|---|---:|---:|---|---:|
| deletion | excellent | excellnt | 2836 | 0 | no | 0.2676 |
| insertion | excellent | exacellent | 2836 | 0 | no | 0.1943 |
| substitution | excellent | excallent | 2836 | 0 | no | 0.2219 |
| transposition | excellent | excelelnt | 2836 | 0 | no | 0.3802 |
| repetition | excellent | exceellent | 2836 | 0 | no | 0.1852 |
| identity control | excellent | excellent | 2836 | 2836 | yes | 1.0000 |
| unrelated control | excellent | umbrella | 2836 | 16 | yes | 0.1231 |
| deletion | amazing | aazing | 1772 | 0 | no | 0.3651 |
| insertion | amazing | amaziang | 1772 | 0 | no | 0.1206 |
| substitution | amazing | amaziag | 1772 | 0 | no | 0.1334 |
| transposition | amazing | amazign | 1772 | 0 | no | 0.3272 |
| repetition | amazing | amazinng | 1772 | 0 | no | 0.2428 |
| identity control | amazing | amazing | 1772 | 1772 | yes | 1.0000 |
| unrelated control | amazing | river | 1772 | 347 | yes | 0.0846 |
| deletion | wonderful | wonderfl | 2246 | 0 | no | 0.1719 |
| insertion | wonderful | wondaerful | 2246 | 0 | no | 0.1164 |
| substitution | wonderful | woederful | 2246 | 0 | no | 0.1421 |
| transposition | wonderful | wondreful | 2246 | 0 | no | 0.2902 |
| repetition | wonderful | wonderrful | 2246 | 0 | no | 0.1449 |
| identity control | wonderful | wonderful | 2246 | 2246 | yes | 1.0000 |
| unrelated control | wonderful | planet | 2246 | 587 | yes | 0.1062 |
| deletion | brilliant | brlliant | 1627 | 0 | no | 0.4073 |
| insertion | brilliant | brillaiant | 1627 | 0 | no | 0.2216 |
| substitution | brilliant | brillaant | 1627 | 0 | no | 0.1692 |
| transposition | brilliant | brliliant | 1627 | 0 | no | 0.1140 |
| repetition | brilliant | briilliant | 1627 | 0 | no | 0.2032 |
| identity control | brilliant | brilliant | 1627 | 1627 | yes | 1.0000 |
| unrelated control | brilliant | bicycle | 1627 | 49 | yes | 0.0400 |
| deletion | enjoyable | enjoyabe | 1159 | 0 | no | 0.2783 |
| insertion | enjoyable | enjoyabale | 1159 | 0 | no | 0.1849 |
| substitution | enjoyable | enjoyaele | 1159 | 0 | no | 0.0374 |
| transposition | enjoyable | enjyoable | 1159 | 0 | no | 0.2770 |
| repetition | enjoyable | ennjoyable | 1159 | 0 | no | 0.2362 |
| identity control | enjoyable | enjoyable | 1159 | 1159 | yes | 1.0000 |
| unrelated control | enjoyable | planet | 1159 | 587 | yes | 0.0811 |
| deletion | beautiful | beautful | 2906 | 0 | yes | 0.8472 |
| insertion | beautiful | beauxtiful | 2906 | 0 | no | 0.2049 |
| substitution | beautiful | beaetiful | 2906 | 0 | no | 0.3228 |
| transposition | beautiful | beautiflu | 2906 | 0 | no | 0.1632 |
| repetition | beautiful | beautifuul | 2906 | 0 | no | 0.1747 |
| identity control | beautiful | beautiful | 2906 | 2906 | yes | 1.0000 |
| unrelated control | beautiful | umbrella | 2906 | 16 | yes | 0.1121 |
| deletion | great | gret | 12649 | 0 | yes | 0.6264 |
| insertion | great | grxeat | 12649 | 0 | no | 0.0640 |
| substitution | great | grzat | 12649 | 0 | no | 0.1543 |
| transposition | great | graet | 12649 | 0 | yes | 0.6299 |
| repetition | great | grreat | 12649 | 1 | yes | 0.5342 |
| identity control | great | great | 12649 | 12649 | yes | 1.0000 |
| unrelated control | great | river | 12649 | 347 | yes | 0.0937 |
| deletion | fantastic | fntastic | 1060 | 0 | no | 0.1526 |
| insertion | fantastic | fantasetic | 1060 | 0 | no | 0.0804 |
| substitution | fantastic | faneastic | 1060 | 0 | no | -0.0270 |
| transposition | fantastic | fnatastic | 1060 | 0 | no | 0.1036 |
| repetition | fantastic | fantasstic | 1060 | 0 | no | 0.1904 |
| identity control | fantastic | fantastic | 1060 | 1060 | yes | 1.0000 |
| unrelated control | fantastic | chemistry | 1060 | 648 | yes | 0.1325 |
| deletion | terrible | terrble | 2337 | 0 | no | 0.2669 |
| insertion | terrible | terribele | 2337 | 0 | no | 0.1799 |
| substitution | terrible | terrixle | 2337 | 0 | no | 0.0645 |
| transposition | terrible | terribel | 2337 | 0 | no | 0.2725 |
| repetition | terrible | terriblle | 2337 | 0 | no | 0.1832 |
| identity control | terrible | terrible | 2337 | 2337 | yes | 1.0000 |
| unrelated control | terrible | keyboard | 2337 | 42 | yes | 0.1418 |
| deletion | horrible | hrrible | 1769 | 0 | no | 0.1907 |
| insertion | horrible | horribxle | 1769 | 0 | no | 0.1793 |
| substitution | horrible | herrible | 1769 | 0 | no | 0.2464 |
| transposition | horrible | horrbile | 1769 | 0 | no | 0.3789 |
| repetition | horrible | horribble | 1769 | 0 | no | 0.2271 |
| identity control | horrible | horrible | 1769 | 1769 | yes | 1.0000 |
| unrelated control | horrible | river | 1769 | 347 | yes | 0.0522 |
| deletion | awful | awul | 2387 | 0 | no | 0.0229 |
| insertion | awful | aewful | 2387 | 0 | no | 0.0144 |
| substitution | awful | awzul | 2387 | 0 | no | 0.0832 |
| transposition | awful | afwul | 2387 | 0 | no | 0.2688 |
| repetition | awful | awwful | 2387 | 0 | no | 0.1678 |
| identity control | awful | awful | 2387 | 2387 | yes | 1.0000 |
| unrelated control | awful | planet | 2387 | 587 | yes | 0.1070 |
| deletion | boring | borng | 2553 | 0 | no | 0.1567 |
| insertion | boring | borixng | 2553 | 0 | no | 0.2141 |
| substitution | boring | boreng | 2553 | 0 | no | 0.1480 |
| transposition | boring | bornig | 2553 | 0 | no | 0.1261 |
| repetition | boring | boriing | 2553 | 0 | no | 0.3303 |
| identity control | boring | boring | 2553 | 2553 | yes | 1.0000 |
| unrelated control | boring | chemistry | 2553 | 648 | yes | 0.2060 |
| deletion | disappointing | disapponting | 588 | 0 | no | 0.3083 |
| insertion | disappointing | disappoxinting | 588 | 0 | no | 0.1156 |
| substitution | disappointing | disaepointing | 588 | 0 | no | 0.1712 |
| transposition | disappointing | diasppointing | 588 | 0 | no | 0.1896 |
| repetition | disappointing | disappointting | 588 | 0 | no | 0.2683 |
| identity control | disappointing | disappointing | 588 | 588 | yes | 1.0000 |
| unrelated control | disappointing | umbrella | 588 | 16 | yes | -0.0172 |
| deletion | dreadful | dreadfl | 326 | 0 | no | 0.1678 |
| insertion | dreadful | dreadfeul | 326 | 0 | no | 0.1259 |
| substitution | dreadful | dreadfel | 326 | 0 | no | 0.1965 |
| transposition | dreadful | deradful | 326 | 0 | no | 0.1296 |
| repetition | dreadful | dreaddful | 326 | 0 | no | 0.1928 |
| identity control | dreadful | dreadful | 326 | 326 | yes | 1.0000 |
| unrelated control | dreadful | chemistry | 326 | 648 | yes | 0.0738 |
| deletion | waste | wste | 1959 | 0 | no | 0.1538 |
| insertion | waste | waaste | 1959 | 0 | no | 0.0924 |
| substitution | waste | wxste | 1959 | 0 | no | -0.0661 |
| transposition | waste | wsate | 1959 | 0 | no | 0.0748 |
| repetition | waste | waaste | 1959 | 0 | no | 0.0924 |
| identity control | waste | waste | 1959 | 1959 | yes | 1.0000 |
| unrelated control | waste | bicycle | 1959 | 49 | yes | 0.0529 |
| deletion | acting | actig | 8903 | 0 | no | 0.0869 |
| insertion | acting | axcting | 8903 | 0 | no | 0.1317 |
| substitution | acting | acaing | 8903 | 0 | no | 0.1401 |
| transposition | acting | acitng | 8903 | 0 | no | 0.1792 |
| repetition | acting | actting | 8903 | 0 | no | 0.2749 |
| identity control | acting | acting | 8903 | 8903 | yes | 1.0000 |
| unrelated control | acting | bicycle | 8903 | 49 | yes | 0.0734 |
| deletion | cinema | cnema | 2018 | 0 | no | 0.0328 |
| insertion | cinema | cineama | 2018 | 0 | no | 0.1707 |
| substitution | cinema | cizema | 2018 | 0 | no | 0.0758 |
| transposition | cinema | cniema | 2018 | 0 | no | 0.0932 |
| repetition | cinema | ciinema | 2018 | 0 | no | 0.0746 |
| identity control | cinema | cinema | 2018 | 2018 | yes | 1.0000 |
| unrelated control | cinema | bicycle | 2018 | 49 | yes | 0.2115 |
| deletion | movie | moie | 60438 | 0 | yes | 0.3295 |
| insertion | movie | movxie | 60438 | 0 | no | 0.0665 |
| substitution | movie | movae | 60438 | 0 | no | 0.0574 |
| transposition | movie | mvoie | 60438 | 0 | no | 0.1923 |
| repetition | movie | movvie | 60438 | 0 | no | 0.0393 |
| identity control | movie | movie | 60438 | 60438 | yes | 1.0000 |
| unrelated control | movie | airport | 60438 | 150 | yes | 0.1691 |
| deletion | story | stry | 16068 | 0 | yes | 0.2800 |
| insertion | story | stoxry | 16068 | 0 | no | 0.0609 |
| substitution | story | seory | 16068 | 0 | no | 0.1378 |
| transposition | story | stoyr | 16068 | 0 | no | 0.1014 |
| repetition | story | sttory | 16068 | 0 | no | 0.0810 |
| identity control | story | story | 16068 | 16068 | yes | 1.0000 |
| unrelated control | story | bicycle | 16068 | 49 | yes | 0.0847 |

### D. Every semantic similarity pair

All pairs are manually selected training-vocabulary probes; frequencies are training-token counts. The same word can occur in more than one pair.

| Category | Word 1 | Word 2 | Training frequencies | Cosine |
|---|---|---|---|---:|
| positive-positive | excellent | wonderful | 2836 / 2246 | 0.6118 |
| positive-positive | amazing | fantastic | 1772 / 1060 | 0.8353 |
| positive-positive | good | great | 20658 / 12649 | 0.7426 |
| positive-positive | beautiful | lovely | 2906 / 596 | 0.7862 |
| positive-positive | enjoyable | entertaining | 1159 / 2033 | 0.6716 |
| positive-positive | brilliant | excellent | 1627 / 2836 | 0.6036 |
| negative-negative | terrible | awful | 2337 / 2387 | 0.8681 |
| negative-negative | horrible | awful | 1769 / 2387 | 0.8825 |
| negative-negative | bad | poor | 12936 / 2683 | 0.5761 |
| negative-negative | boring | dull | 2553 / 1107 | 0.7425 |
| negative-negative | disappointing | mediocre | 588 / 468 | 0.6068 |
| negative-negative | dreadful | horrible | 326 / 1769 | 0.7988 |
| positive-negative | good | bad | 20658 / 12936 | 0.7518 |
| positive-negative | excellent | terrible | 2836 / 2337 | 0.3854 |
| positive-negative | wonderful | awful | 2246 / 2387 | 0.4127 |
| positive-negative | enjoyable | boring | 1159 / 2553 | 0.4783 |
| positive-negative | beautiful | ugly | 2906 / 464 | 0.4063 |
| positive-negative | love | hate | 9035 / 1136 | 0.6165 |
| unrelated controls | excellent | potato | 2836 / 34 | 0.0440 |
| unrelated controls | terrible | planet | 2337 / 587 | 0.1541 |
| unrelated controls | wonderful | keyboard | 2246 / 42 | 0.0564 |
| unrelated controls | boring | river | 2553 / 347 | 0.0276 |
| unrelated controls | amazing | bicycle | 1772 / 49 | 0.0187 |
| unrelated controls | awful | umbrella | 2387 / 16 | 0.0627 |
| morphological controls | enjoy | enjoyable | 2484 / 1159 | 0.5415 |
| morphological controls | beautiful | beautifully | 2906 / 609 | 0.6420 |
| morphological controls | bored | boring | 738 / 2553 | 0.6507 |
| morphological controls | hope | hopeless | 2056 / 114 | 0.3391 |
| morphological controls | good | goodness | 20658 / 150 | 0.3571 |
| morphological controls | bad | badly | 12936 / 881 | 0.5231 |

### E. All saved nearest-neighbor probes

Five neighbors per query, in rank order, with cosine scores. The candidate pool was the first 100,000 stored model words, not the full model vocabulary; some queries are controlled synthetic spellings.

| Query | Query kind | Direct? | Ranked neighbors (cosine) |
|---|---|---|---|
| excellent | sentiment probe | yes | superb (0.8393), terrific (0.7652), fantastic (0.7519), excelent (0.7223), Excellent (0.7148) |
| wonderful | sentiment probe | yes | fantastic (0.8209), lovely (0.8132), fabulous (0.8088), marvelous (0.8079), amazing (0.7956) |
| good | sentiment probe | yes | bad (0.7518), great (0.7426), decent (0.7300), nice (0.7124), Good (0.6797) |
| bad | sentiment probe | yes | good (0.7518), terrible (0.7317), horrible (0.7189), lousy (0.7087), Bad (0.6730) |
| terrible | sentiment probe | yes | horrible (0.9434), awful (0.8681), horrendous (0.8550), dreadful (0.8397), horrid (0.8354) |
| boring | sentiment probe | yes | monotonous (0.7616), uninteresting (0.7565), dull (0.7425), bland (0.6908), tedious (0.6805) |
| excellnt | synthetic deletion | no | reserved.FlightsHotelsRound-TripOne-wayFromToCity1 (0.3220), StatesCanadaArgentinaAustraliaBelgiëBrasilDeutschlandEspañaFranceIndiaIrelandItaliaMéxicoNederlandÖsterreichPortugalSchweizSouth (0.3110), CLPFtHUFRMMYRkrNOKkrSEKCHFCHFkrDKKKčCZKRpIDRTRYAEDAEDSiteESEspañolMXMéxicoDEDeutscheMenuSign (0.2914), newVoteStoreCartEnglishArabicAzerbaijaniBengaliBulgarianChineseCzechDanishDutchEnglishFilipinoFrenchGermanGreekHebrewHungarianIndonesianItalianJapaneseKoreanNorwegianPolishPortugueseRomanianRussianSerbianSpanishSwedishThaiTurkishUkrainianVietnameseHelp (0.2904), helpful (0.2845) |
| exceellent | synthetic repetition | no | newVoteStoreCartEnglishArabicAzerbaijaniBengaliBulgarianChineseCzechDanishDutchEnglishFilipinoFrenchGermanGreekHebrewHungarianIndonesianItalianJapaneseKoreanNorwegianPolishPortugueseRomanianRussianSerbianSpanishSwedishThaiTurkishUkrainianVietnameseHelp (0.3466), RootVideoSploidPaleofutureio9ScienceReviewsField (0.2905), ChannelsBuddhistCatholicEvangelicalHinduJewishMormonMuslimNonreligiousPaganProgressive (0.2707), CLPFtHUFRMMYRkrNOKkrSEKCHFCHFkrDKKKčCZKRpIDRTRYAEDAEDSiteESEspañolMXMéxicoDEDeutsche (0.2582), optimal (0.2572) |
| terrble | synthetic deletion | no | CLPFtHUFRMMYRkrNOKkrSEKCHFCHFkrDKKKčCZKRpIDRTRYAEDAEDSiteESEspañolMXMéxicoDEDeutscheMenuSign (0.3379), 41.8 (0.3345), StatesCanadaArgentinaAustraliaBelgiëBrasilDeutschlandEspañaFranceIndiaIrelandItaliaMéxicoNederlandÖsterreichPortugalSchweizSouth (0.3277), UnterkunftObjekttypWohnungUnterkunft (0.3257), 38.6 (0.3226) |
| terriblle | synthetic repetition | no | reserved.FlightsHotelsRound-TripOne-wayFromToCity1 (0.2708), minimally (0.2494), MXNINRрубRUB (0.2481), page1 (0.2438), Painful (0.2395) |
| borng | synthetic deletion | no | born (0.3611), MDProfile (0.2900), Sat (0.2775), reserved.FlightsHotelsRound-TripOne-wayFromToCity1 (0.2751), Wed (0.2724) |
| boriing | synthetic repetition | no | boring (0.3303), pointless (0.2883), tedious (0.2819), monotonous (0.2734), mundane (0.2704) |

### F. Full neural-training histories

Each line is measured after that epoch in evaluation mode. The selected epoch is bold. The Phase 2 MLP logs did not record per-epoch training macro-F1; an em dash indicates a metric that was not logged, not a zero or an invented score. Phase 2 ran all 30 epochs; Phase 3 used patience-5 early stopping.

#### 1. Phase 2 — MLP with mean pooling

| Epoch | Train loss | Val loss | Train accuracy | Val accuracy | Train macro-F1 | Val macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5272 | 0.5361 | 0.7606 | 0.7477 | — | 0.7477 |
| 2 | 0.4554 | 0.4687 | 0.8012 | 0.7858 | — | 0.7858 |
| 3 | 0.4353 | 0.4525 | 0.8072 | 0.7951 | — | 0.7943 |
| 4 | 0.4120 | 0.4303 | 0.8188 | 0.8070 | — | 0.8069 |
| 5 | 0.4009 | 0.4207 | 0.8236 | 0.8107 | — | 0.8106 |
| 6 | 0.3917 | 0.4119 | 0.8282 | 0.8143 | — | 0.8142 |
| 7 | 0.3944 | 0.4138 | 0.8272 | 0.8132 | — | 0.8126 |
| 8 | 0.3820 | 0.4015 | 0.8330 | 0.8225 | — | 0.8225 |
| 9 | 0.3810 | 0.4036 | 0.8316 | 0.8169 | — | 0.8164 |
| 10 | 0.3725 | 0.3938 | 0.8394 | 0.8240 | — | 0.8239 |
| 11 | 0.3698 | 0.3919 | 0.8408 | 0.8213 | — | 0.8213 |
| 12 | 0.3659 | 0.3866 | 0.8416 | 0.8279 | — | 0.8278 |
| 13 | 0.3631 | 0.3847 | 0.8429 | 0.8280 | — | 0.8280 |
| 14 | 0.3633 | 0.3845 | 0.8409 | 0.8276 | — | 0.8275 |
| 15 | 0.3606 | 0.3825 | 0.8430 | 0.8282 | — | 0.8281 |
| 16 | 0.3613 | 0.3841 | 0.8438 | 0.8260 | — | 0.8260 |
| 17 | 0.3670 | 0.3909 | 0.8366 | 0.8271 | — | 0.8265 |
| 18 | 0.3556 | 0.3777 | 0.8462 | 0.8318 | — | 0.8318 |
| 19 | 0.3575 | 0.3798 | 0.8459 | 0.8295 | — | 0.8293 |
| 20 | 0.3553 | 0.3785 | 0.8437 | 0.8319 | — | 0.8318 |
| 21 | 0.3524 | 0.3751 | 0.8469 | 0.8343 | — | 0.8343 |
| 22 | 0.3525 | 0.3748 | 0.8470 | 0.8333 | — | 0.8332 |
| 23 | 0.3523 | 0.3769 | 0.8452 | 0.8331 | — | 0.8331 |
| 24 | 0.3494 | 0.3737 | 0.8488 | 0.8322 | — | 0.8322 |
| 25 | 0.3489 | 0.3738 | 0.8493 | 0.8343 | — | 0.8343 |
| 26 | 0.3477 | 0.3720 | 0.8498 | 0.8356 | — | 0.8355 |
| 27 | 0.3512 | 0.3745 | 0.8478 | 0.8350 | — | 0.8350 |
| **28** | 0.3464 | 0.3707 | 0.8497 | 0.8365 | — | 0.8365 |
| 29 | 0.3479 | 0.3731 | 0.8501 | 0.8347 | — | 0.8346 |
| 30 | 0.3469 | 0.3732 | 0.8489 | 0.8334 | — | 0.8334 |

#### 2. Phase 2 — MLP with max pooling

| Epoch | Train loss | Val loss | Train accuracy | Val accuracy | Train macro-F1 | Val macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6629 | 0.6669 | 0.6011 | 0.5962 | — | 0.5687 |
| 2 | 0.6405 | 0.6483 | 0.6338 | 0.6230 | — | 0.6184 |
| 3 | 0.6377 | 0.6467 | 0.6355 | 0.6232 | — | 0.6105 |
| 4 | 0.6394 | 0.6484 | 0.6353 | 0.6251 | — | 0.6134 |
| 5 | 0.6263 | 0.6366 | 0.6484 | 0.6392 | — | 0.6328 |
| 6 | 0.6291 | 0.6386 | 0.6466 | 0.6359 | — | 0.6296 |
| 7 | 0.6173 | 0.6281 | 0.6591 | 0.6517 | — | 0.6506 |
| 8 | 0.6154 | 0.6258 | 0.6620 | 0.6528 | — | 0.6527 |
| 9 | 0.6172 | 0.6284 | 0.6580 | 0.6513 | — | 0.6471 |
| 10 | 0.6139 | 0.6250 | 0.6633 | 0.6527 | — | 0.6521 |
| 11 | 0.6118 | 0.6227 | 0.6648 | 0.6556 | — | 0.6556 |
| 12 | 0.6188 | 0.6284 | 0.6573 | 0.6447 | — | 0.6406 |
| 13 | 0.6506 | 0.6629 | 0.6219 | 0.6136 | — | 0.5817 |
| 14 | 0.6124 | 0.6245 | 0.6628 | 0.6530 | — | 0.6502 |
| 15 | 0.6111 | 0.6220 | 0.6644 | 0.6556 | — | 0.6547 |
| 16 | 0.6084 | 0.6205 | 0.6693 | 0.6554 | — | 0.6551 |
| 17 | 0.6104 | 0.6231 | 0.6643 | 0.6536 | — | 0.6500 |
| 18 | 0.6088 | 0.6204 | 0.6676 | 0.6570 | — | 0.6564 |
| **19** | 0.6064 | 0.6189 | 0.6698 | 0.6605 | — | 0.6604 |
| 20 | 0.6063 | 0.6192 | 0.6709 | 0.6598 | — | 0.6588 |
| 21 | 0.6044 | 0.6183 | 0.6719 | 0.6597 | — | 0.6596 |
| 22 | 0.6125 | 0.6262 | 0.6620 | 0.6486 | — | 0.6420 |
| 23 | 0.6106 | 0.6250 | 0.6638 | 0.6495 | — | 0.6451 |
| 24 | 0.6055 | 0.6190 | 0.6712 | 0.6590 | — | 0.6581 |
| 25 | 0.6037 | 0.6178 | 0.6728 | 0.6602 | — | 0.6597 |
| 26 | 0.6167 | 0.6294 | 0.6569 | 0.6445 | — | 0.6359 |
| 27 | 0.6040 | 0.6184 | 0.6709 | 0.6613 | — | 0.6591 |
| 28 | 0.6018 | 0.6164 | 0.6731 | 0.6579 | — | 0.6568 |
| 29 | 0.6032 | 0.6173 | 0.6718 | 0.6601 | — | 0.6587 |
| 30 | 0.6020 | 0.6181 | 0.6722 | 0.6624 | — | 0.6602 |

#### 3. Phase 3 — LSTM + linear, 256 tokens

| Epoch | Train loss | Val loss | Train accuracy | Val accuracy | Train macro-F1 | Val macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6563 | 0.6568 | 0.6386 | 0.6316 | 0.6378 | 0.6305 |
| 2 | 0.6268 | 0.6311 | 0.6399 | 0.6375 | 0.6096 | 0.6078 |
| 3 | 0.5879 | 0.5907 | 0.7154 | 0.7132 | 0.7119 | 0.7091 |
| 4 | 0.4370 | 0.4504 | 0.7936 | 0.7866 | 0.7888 | 0.7814 |
| 5 | 0.3240 | 0.3427 | 0.8602 | 0.8495 | 0.8600 | 0.8495 |
| 6 | 0.2964 | 0.3234 | 0.8737 | 0.8635 | 0.8737 | 0.8635 |
| 7 | 0.2879 | 0.3179 | 0.8769 | 0.8672 | 0.8768 | 0.8671 |
| 8 | 0.2720 | 0.3093 | 0.8852 | 0.8685 | 0.8851 | 0.8684 |
| 9 | 0.2630 | 0.3133 | 0.8899 | 0.8727 | 0.8898 | 0.8725 |
| 10 | 0.2507 | 0.2983 | 0.8961 | 0.8743 | 0.8960 | 0.8743 |
| 11 | 0.2423 | 0.3002 | 0.8987 | 0.8728 | 0.8987 | 0.8727 |
| 12 | 0.2383 | 0.3105 | 0.9011 | 0.8744 | 0.9009 | 0.8741 |
| **13** | 0.2169 | 0.2992 | 0.9120 | 0.8791 | 0.9119 | 0.8790 |
| 14 | 0.2142 | 0.3116 | 0.9115 | 0.8737 | 0.9115 | 0.8735 |
| 15 | 0.1913 | 0.2997 | 0.9252 | 0.8780 | 0.9252 | 0.8780 |
| 16 | 0.1827 | 0.3020 | 0.9288 | 0.8790 | 0.9288 | 0.8790 |
| 17 | 0.1665 | 0.3252 | 0.9342 | 0.8762 | 0.9342 | 0.8761 |
| 18 | 0.1635 | 0.3230 | 0.9385 | 0.8688 | 0.9385 | 0.8688 |

#### 4. Phase 3 — LSTM + linear, 512 tokens

| Epoch | Train loss | Val loss | Train accuracy | Val accuracy | Train macro-F1 | Val macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6564 | 0.6609 | 0.6432 | 0.6361 | 0.6325 | 0.6258 |
| 2 | 0.5692 | 0.5755 | 0.7040 | 0.6928 | 0.7039 | 0.6926 |
| 3 | 0.5761 | 0.5883 | 0.6808 | 0.6710 | 0.6534 | 0.6408 |
| 4 | 0.5374 | 0.5482 | 0.7409 | 0.7355 | 0.7275 | 0.7215 |
| 5 | 0.3611 | 0.3855 | 0.8433 | 0.8327 | 0.8432 | 0.8327 |
| 6 | 0.3446 | 0.3643 | 0.8476 | 0.8356 | 0.8470 | 0.8350 |
| 7 | 0.3433 | 0.3611 | 0.8531 | 0.8436 | 0.8522 | 0.8427 |
| 8 | 0.2787 | 0.3022 | 0.8868 | 0.8778 | 0.8868 | 0.8778 |
| 9 | 0.2859 | 0.3279 | 0.8846 | 0.8720 | 0.8842 | 0.8714 |
| 10 | 0.2359 | 0.2735 | 0.9037 | 0.8942 | 0.9037 | 0.8942 |
| 11 | 0.2201 | 0.2664 | 0.9113 | 0.8949 | 0.9113 | 0.8948 |
| 12 | 0.2149 | 0.2685 | 0.9155 | 0.8961 | 0.9154 | 0.8959 |
| 13 | 0.2181 | 0.2820 | 0.9074 | 0.8829 | 0.9071 | 0.8822 |
| 14 | 0.2007 | 0.2716 | 0.9194 | 0.8907 | 0.9192 | 0.8904 |
| 15 | 0.1679 | 0.2553 | 0.9338 | 0.9044 | 0.9338 | 0.9044 |
| 16 | 0.1589 | 0.2558 | 0.9373 | 0.9028 | 0.9373 | 0.9027 |
| **17** | 0.1468 | 0.2645 | 0.9451 | 0.9045 | 0.9451 | 0.9045 |
| 18 | 0.1353 | 0.2606 | 0.9496 | 0.9004 | 0.9496 | 0.9003 |
| 19 | 0.1409 | 0.2911 | 0.9476 | 0.8992 | 0.9476 | 0.8991 |
| 20 | 0.1104 | 0.2738 | 0.9619 | 0.8967 | 0.9619 | 0.8967 |
| 21 | 0.0930 | 0.2960 | 0.9686 | 0.8970 | 0.9686 | 0.8970 |
| 22 | 0.1078 | 0.3124 | 0.9596 | 0.8865 | 0.9596 | 0.8864 |

#### 5. Phase 3 — LSTM + MLP, 512 tokens

| Epoch | Train loss | Val loss | Train accuracy | Val accuracy | Train macro-F1 | Val macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4866 | 0.4980 | 0.7674 | 0.7553 | 0.7633 | 0.7509 |
| 2 | 0.5644 | 0.5723 | 0.6952 | 0.6956 | 0.6717 | 0.6724 |
| 3 | 0.7629 | 0.7714 | 0.6224 | 0.6203 | 0.5683 | 0.5646 |
| 4 | 0.4253 | 0.4438 | 0.8137 | 0.8014 | 0.8131 | 0.8009 |
| 5 | 0.3844 | 0.3998 | 0.8300 | 0.8228 | 0.8284 | 0.8213 |
| 6 | 0.2962 | 0.3180 | 0.8754 | 0.8692 | 0.8752 | 0.8690 |
| 7 | 0.2837 | 0.3080 | 0.8831 | 0.8759 | 0.8827 | 0.8754 |
| 8 | 0.2358 | 0.2721 | 0.9040 | 0.8897 | 0.9040 | 0.8897 |
| 9 | 0.2319 | 0.2851 | 0.9067 | 0.8873 | 0.9066 | 0.8871 |
| 10 | 0.2140 | 0.2629 | 0.9150 | 0.8944 | 0.9150 | 0.8944 |
| **11** | 0.1926 | 0.2583 | 0.9246 | 0.8979 | 0.9246 | 0.8979 |
| 12 | 0.2010 | 0.2809 | 0.9207 | 0.8841 | 0.9205 | 0.8837 |
| 13 | 0.1778 | 0.2756 | 0.9315 | 0.8842 | 0.9315 | 0.8840 |
| 14 | 0.1532 | 0.2620 | 0.9440 | 0.8959 | 0.9440 | 0.8959 |
| 15 | 0.1382 | 0.2710 | 0.9499 | 0.8954 | 0.9499 | 0.8954 |
| 16 | 0.1285 | 0.2787 | 0.9552 | 0.8910 | 0.9552 | 0.8909 |

#### 6. Follow-up A2 — LSTM + trainable pretrained lookup, 512 tokens

| Epoch | Train loss | Val loss | Train accuracy | Val accuracy | Train macro-F1 | Val macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2295 | 0.3069 | 0.9215 | 0.8815 | 0.9215 | 0.8815 |
| **2** | 0.0896 | 0.2700 | 0.9737 | 0.8908 | 0.9737 | 0.8907 |
| 3 | 0.0294 | 0.3591 | 0.9927 | 0.8834 | 0.9927 | 0.8832 |
| 4 | 0.0117 | 0.3894 | 0.9979 | 0.8833 | 0.9979 | 0.8833 |
| 5 | 0.0061 | 0.4882 | 0.9992 | 0.8791 | 0.9992 | 0.8789 |
| 6 | 0.0024 | 0.5989 | 0.9996 | 0.8811 | 0.9996 | 0.8810 |
| 7 | 0.0022 | 0.6184 | 0.9996 | 0.8748 | 0.9996 | 0.8748 |

#### 7. Follow-up A3 — LSTM + random trainable lookup, 512 tokens

| Epoch | Train loss | Val loss | Train accuracy | Val accuracy | Train macro-F1 | Val macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3247 | 0.4246 | 0.8800 | 0.8134 | 0.8799 | 0.8132 |
| 2 | 0.2035 | 0.3940 | 0.9327 | 0.8291 | 0.9327 | 0.8290 |
| 3 | 0.1197 | 0.3877 | 0.9598 | 0.8524 | 0.9598 | 0.8519 |
| 4 | 0.0529 | 0.4107 | 0.9856 | 0.8643 | 0.9856 | 0.8642 |
| 5 | 0.0203 | 0.4960 | 0.9950 | 0.8505 | 0.9950 | 0.8503 |
| 6 | 0.2410 | 0.6338 | 0.8921 | 0.7463 | 0.8916 | 0.7441 |
| **7** | 0.0068 | 0.5835 | 0.9981 | 0.8689 | 0.9981 | 0.8689 |
| 8 | 0.0034 | 0.6632 | 0.9994 | 0.8618 | 0.9994 | 0.8618 |
| 9 | 0.0009 | 0.7612 | 0.9999 | 0.8634 | 0.9999 | 0.8633 |
| 10 | 0.0274 | 0.8162 | 0.9899 | 0.8361 | 0.9899 | 0.8350 |
| 11 | 0.0078 | 0.7287 | 0.9978 | 0.8443 | 0.9978 | 0.8443 |
| 12 | 0.0025 | 0.7771 | 0.9995 | 0.8526 | 0.9995 | 0.8526 |


#### 8. Follow-up B1 — LSTM + frozen lookup, half training data

| Epoch | Train loss | Val loss | Train accuracy | Val accuracy | Train macro-F1 | Val macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6672 | 0.6688 | 0.5743 | 0.5665 | 0.5331 | 0.5208 |
| 2 | 0.6164 | 0.6204 | 0.6711 | 0.6648 | 0.6664 | 0.6603 |
| 3 | 0.6024 | 0.6062 | 0.7157 | 0.7085 | 0.7094 | 0.7022 |
| 4 | 0.6789 | 0.6791 | 0.5189 | 0.5153 | 0.3790 | 0.3759 |
| 5 | 0.6716 | 0.6741 | 0.6357 | 0.6293 | 0.6357 | 0.6293 |
| 6 | 0.4680 | 0.4756 | 0.7930 | 0.7906 | 0.7929 | 0.7905 |
| 7 | 0.3845 | 0.3937 | 0.8351 | 0.8347 | 0.8347 | 0.8344 |
| 8 | 0.3391 | 0.3540 | 0.8574 | 0.8494 | 0.8573 | 0.8493 |
| 9 | 0.3163 | 0.3404 | 0.8648 | 0.8545 | 0.8644 | 0.8540 |
| 10 | 0.2957 | 0.3194 | 0.8830 | 0.8704 | 0.8830 | 0.8704 |
| 11 | 0.2923 | 0.3302 | 0.8769 | 0.8580 | 0.8762 | 0.8570 |
| 12 | 0.2517 | 0.2913 | 0.8999 | 0.8819 | 0.8999 | 0.8819 |
| 13 | 0.2410 | 0.2964 | 0.9038 | 0.8845 | 0.9038 | 0.8844 |
| 14 | 0.2332 | 0.2848 | 0.9086 | 0.8837 | 0.9086 | 0.8836 |
| 15 | 0.2155 | 0.2771 | 0.9156 | 0.8899 | 0.9156 | 0.8899 |
| **16** | 0.2104 | 0.2716 | 0.9195 | 0.8922 | 0.9195 | 0.8922 |
| 17 | 0.2054 | 0.2805 | 0.9193 | 0.8865 | 0.9193 | 0.8864 |
| 18 | 0.1999 | 0.2843 | 0.9212 | 0.8840 | 0.9212 | 0.8838 |
| 19 | 0.1805 | 0.2804 | 0.9303 | 0.8893 | 0.9303 | 0.8893 |
| 20 | 0.1890 | 0.2843 | 0.9308 | 0.8822 | 0.9307 | 0.8822 |
| 21 | 0.1805 | 0.2939 | 0.9316 | 0.8818 | 0.9315 | 0.8817 |

#### 9. Follow-up C1 — LSTM + frozen lookup, 32 hidden units

| Epoch | Train loss | Val loss | Train accuracy | Val accuracy | Train macro-F1 | Val macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6670 | 0.6706 | 0.5601 | 0.5575 | 0.4648 | 0.4600 |
| 2 | 0.4308 | 0.4456 | 0.8056 | 0.7935 | 0.8038 | 0.7916 |
| 3 | 0.4783 | 0.4951 | 0.7606 | 0.7506 | 0.7505 | 0.7396 |
| **4** | 0.4151 | 0.4317 | 0.8266 | 0.8167 | 0.8265 | 0.8164 |
| 5 | 0.4611 | 0.4712 | 0.7807 | 0.7769 | 0.7752 | 0.7715 |
| 6 | 0.5078 | 0.5191 | 0.7453 | 0.7409 | 0.7448 | 0.7404 |
| 7 | 0.5813 | 0.5921 | 0.6919 | 0.6824 | 0.6689 | 0.6576 |
| 8 | 0.6350 | 0.6421 | 0.6435 | 0.6317 | 0.6405 | 0.6289 |
| 9 | 0.4113 | 0.4339 | 0.8244 | 0.8109 | 0.8236 | 0.8099 |

#### 10. Follow-up C3 — LSTM + frozen lookup, 128 hidden units

| Epoch | Train loss | Val loss | Train accuracy | Val accuracy | Train macro-F1 | Val macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5325 | 0.5369 | 0.7357 | 0.7319 | 0.7291 | 0.7251 |
| 2 | 0.5753 | 0.5823 | 0.6885 | 0.6825 | 0.6752 | 0.6693 |
| 3 | 0.4368 | 0.4542 | 0.8070 | 0.7982 | 0.8059 | 0.7968 |
| 4 | 0.4883 | 0.5024 | 0.7796 | 0.7720 | 0.7734 | 0.7655 |
| 5 | 0.3181 | 0.3359 | 0.8647 | 0.8577 | 0.8647 | 0.8577 |
| 6 | 0.2934 | 0.3153 | 0.8786 | 0.8719 | 0.8785 | 0.8718 |
| 7 | 0.2729 | 0.2972 | 0.8864 | 0.8782 | 0.8861 | 0.8780 |
| 8 | 0.2452 | 0.2819 | 0.9004 | 0.8850 | 0.9004 | 0.8850 |
| 9 | 0.2433 | 0.2931 | 0.9015 | 0.8817 | 0.9013 | 0.8813 |
| 10 | 0.2166 | 0.2631 | 0.9150 | 0.8926 | 0.9150 | 0.8926 |
| 11 | 0.2017 | 0.2635 | 0.9203 | 0.8932 | 0.9203 | 0.8932 |
| 12 | 0.2087 | 0.2817 | 0.9161 | 0.8841 | 0.9159 | 0.8837 |
| 13 | 0.2010 | 0.2877 | 0.9178 | 0.8799 | 0.9175 | 0.8793 |
| 14 | 0.1675 | 0.2689 | 0.9364 | 0.8900 | 0.9364 | 0.8899 |
| 15 | 0.1455 | 0.2709 | 0.9458 | 0.8954 | 0.9458 | 0.8953 |
| 16 | 0.1358 | 0.2951 | 0.9488 | 0.8914 | 0.9488 | 0.8911 |
| 17 | 0.1233 | 0.3397 | 0.9534 | 0.8896 | 0.9533 | 0.8895 |
| 18 | 0.1013 | 0.3024 | 0.9665 | 0.8899 | 0.9665 | 0.8899 |
| **19** | 0.0777 | 0.3179 | 0.9749 | 0.8961 | 0.9749 | 0.8961 |
| 20 | 0.0607 | 0.3570 | 0.9815 | 0.8911 | 0.9815 | 0.8911 |
| 21 | 0.0521 | 0.3737 | 0.9848 | 0.8908 | 0.9848 | 0.8907 |
| 22 | 0.0469 | 0.3981 | 0.9850 | 0.8888 | 0.9850 | 0.8888 |
| 23 | 0.0337 | 0.4192 | 0.9902 | 0.8896 | 0.9902 | 0.8895 |
| 24 | 0.0285 | 0.4440 | 0.9933 | 0.8857 | 0.9933 | 0.8857 |

#### 11. Follow-up D1 — LSTM + frozen lookup, learning rate 0.0003

| Epoch | Train loss | Val loss | Train accuracy | Val accuracy | Train macro-F1 | Val macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5002 | 0.5154 | 0.7669 | 0.7568 | 0.7659 | 0.7558 |
| 2 | 0.5888 | 0.5931 | 0.7050 | 0.6985 | 0.6955 | 0.6887 |
| 3 | 0.4489 | 0.4700 | 0.7941 | 0.7807 | 0.7899 | 0.7760 |
| 4 | 0.3850 | 0.4094 | 0.8388 | 0.8261 | 0.8384 | 0.8258 |
| 5 | 0.3548 | 0.3777 | 0.8483 | 0.8369 | 0.8483 | 0.8369 |
| 6 | 0.3698 | 0.3914 | 0.8425 | 0.8294 | 0.8425 | 0.8294 |
| 7 | 0.3296 | 0.3534 | 0.8615 | 0.8518 | 0.8615 | 0.8518 |
| 8 | 0.3362 | 0.3605 | 0.8574 | 0.8454 | 0.8573 | 0.8453 |
| 9 | 0.3098 | 0.3413 | 0.8703 | 0.8599 | 0.8702 | 0.8597 |
| 10 | 0.4024 | 0.4260 | 0.8417 | 0.8307 | 0.8413 | 0.8304 |
| 11 | 0.4328 | 0.4699 | 0.8348 | 0.8224 | 0.8336 | 0.8211 |
| 12 | 0.3456 | 0.3694 | 0.8478 | 0.8381 | 0.8469 | 0.8371 |
| 13 | 0.3033 | 0.3364 | 0.8729 | 0.8600 | 0.8727 | 0.8596 |
| 14 | 0.2876 | 0.3198 | 0.8809 | 0.8682 | 0.8809 | 0.8682 |
| 15 | 0.2729 | 0.3063 | 0.8896 | 0.8758 | 0.8896 | 0.8758 |
| 16 | 0.2804 | 0.3163 | 0.8850 | 0.8697 | 0.8847 | 0.8693 |
| 17 | 0.2594 | 0.2944 | 0.8948 | 0.8803 | 0.8948 | 0.8803 |
| 18 | 0.2826 | 0.3144 | 0.8861 | 0.8731 | 0.8861 | 0.8731 |
| 19 | 0.2645 | 0.3003 | 0.8917 | 0.8794 | 0.8917 | 0.8794 |
| 20 | 0.2560 | 0.2983 | 0.8973 | 0.8794 | 0.8973 | 0.8793 |
| 21 | 0.2733 | 0.3037 | 0.8918 | 0.8786 | 0.8917 | 0.8784 |
| 22 | 0.2567 | 0.2941 | 0.8983 | 0.8856 | 0.8982 | 0.8855 |
| 23 | 0.2626 | 0.2982 | 0.8948 | 0.8823 | 0.8948 | 0.8823 |
| 24 | 0.2745 | 0.3112 | 0.8869 | 0.8706 | 0.8868 | 0.8705 |
| 25 | 0.2472 | 0.2881 | 0.8999 | 0.8818 | 0.8998 | 0.8817 |
| 26 | 0.2353 | 0.2786 | 0.9064 | 0.8916 | 0.9063 | 0.8916 |
| 27 | 0.3108 | 0.3629 | 0.8736 | 0.8569 | 0.8736 | 0.8569 |
| 28 | 0.2427 | 0.2921 | 0.9013 | 0.8853 | 0.9010 | 0.8851 |
| 29 | 0.2318 | 0.2789 | 0.9063 | 0.8860 | 0.9062 | 0.8858 |
| **30** | 0.2208 | 0.2720 | 0.9142 | 0.8928 | 0.9141 | 0.8928 |

#### 12. Follow-up D3 — LSTM + frozen lookup, learning rate 0.003

| Epoch | Train loss | Val loss | Train accuracy | Val accuracy | Train macro-F1 | Val macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2839 | 0.3008 | 0.8887 | 0.8798 | 0.8886 | 0.8797 |
| 2 | 0.2457 | 0.2718 | 0.9059 | 0.8946 | 0.9059 | 0.8946 |
| 3 | 0.2331 | 0.2740 | 0.9069 | 0.8911 | 0.9067 | 0.8909 |
| 4 | 0.2110 | 0.2722 | 0.9162 | 0.8908 | 0.9161 | 0.8907 |
| **5** | 0.1729 | 0.2487 | 0.9345 | 0.9021 | 0.9345 | 0.9021 |
| 6 | 0.1666 | 0.2569 | 0.9402 | 0.8985 | 0.9402 | 0.8985 |
| 7 | 0.1380 | 0.2700 | 0.9500 | 0.8965 | 0.9499 | 0.8963 |
| 8 | 0.1049 | 0.3078 | 0.9629 | 0.8947 | 0.9629 | 0.8947 |
| 9 | 0.1040 | 0.3324 | 0.9608 | 0.8857 | 0.9608 | 0.8854 |
| 10 | 0.0663 | 0.3396 | 0.9808 | 0.8866 | 0.9808 | 0.8866 |

#### 13. Follow-up E2 — LSTM + post-LSTM dropout 0.3

| Epoch | Train loss | Val loss | Train accuracy | Val accuracy | Train macro-F1 | Val macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6509 | 0.6645 | 0.6832 | 0.6773 | 0.6596 | 0.6526 |
| 2 | 0.4984 | 0.5065 | 0.7602 | 0.7533 | 0.7545 | 0.7476 |
| 3 | 0.6016 | 0.6072 | 0.6345 | 0.6364 | 0.5843 | 0.5868 |
| 4 | 0.3639 | 0.3760 | 0.8539 | 0.8510 | 0.8538 | 0.8510 |
| 5 | 0.2917 | 0.3110 | 0.8816 | 0.8766 | 0.8815 | 0.8765 |
| 6 | 0.2766 | 0.3016 | 0.8865 | 0.8772 | 0.8864 | 0.8771 |
| 7 | 0.2451 | 0.2756 | 0.9008 | 0.8910 | 0.9008 | 0.8909 |
| 8 | 0.2280 | 0.2677 | 0.9086 | 0.8908 | 0.9086 | 0.8908 |
| 9 | 0.2270 | 0.2822 | 0.9086 | 0.8918 | 0.9085 | 0.8915 |
| 10 | 0.2099 | 0.2613 | 0.9177 | 0.8962 | 0.9177 | 0.8962 |
| 11 | 0.1958 | 0.2709 | 0.9219 | 0.8967 | 0.9219 | 0.8967 |
| 12 | 0.1970 | 0.2772 | 0.9226 | 0.8912 | 0.9225 | 0.8909 |
| 13 | 0.1798 | 0.2775 | 0.9277 | 0.8901 | 0.9276 | 0.8899 |
| 14 | 0.1709 | 0.2724 | 0.9315 | 0.8892 | 0.9314 | 0.8889 |
| 15 | 0.1559 | 0.2868 | 0.9380 | 0.8950 | 0.9380 | 0.8949 |
| 16 | 0.1366 | 0.2675 | 0.9491 | 0.8978 | 0.9491 | 0.8978 |
| **17** | 0.1213 | 0.2979 | 0.9538 | 0.8979 | 0.9538 | 0.8979 |
| 18 | 0.1108 | 0.2950 | 0.9604 | 0.8936 | 0.9604 | 0.8936 |
| 19 | 0.1265 | 0.2796 | 0.9542 | 0.8969 | 0.9542 | 0.8969 |
| 20 | 0.1062 | 0.3045 | 0.9621 | 0.8957 | 0.9621 | 0.8957 |
| 21 | 0.1046 | 0.3072 | 0.9626 | 0.8900 | 0.9626 | 0.8900 |
| 22 | 0.1093 | 0.3614 | 0.9577 | 0.8876 | 0.9577 | 0.8875 |

#### 14. Follow-up E3 — LSTM + Adam weight decay 0.0001

| Epoch | Train loss | Val loss | Train accuracy | Val accuracy | Train macro-F1 | Val macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4882 | 0.4989 | 0.7840 | 0.7759 | 0.7840 | 0.7758 |
| 2 | 0.4560 | 0.4708 | 0.8057 | 0.7968 | 0.8056 | 0.7967 |
| 3 | 0.4934 | 0.5062 | 0.7588 | 0.7476 | 0.7540 | 0.7425 |
| 4 | 0.4650 | 0.4753 | 0.7797 | 0.7721 | 0.7745 | 0.7669 |
| 5 | 0.3100 | 0.3272 | 0.8707 | 0.8631 | 0.8706 | 0.8631 |
| 6 | 0.2859 | 0.3088 | 0.8806 | 0.8694 | 0.8805 | 0.8694 |
| 7 | 0.2778 | 0.3005 | 0.8843 | 0.8768 | 0.8840 | 0.8765 |
| 8 | 0.2469 | 0.2809 | 0.8997 | 0.8819 | 0.8996 | 0.8819 |
| 9 | 0.2437 | 0.2835 | 0.9027 | 0.8871 | 0.9026 | 0.8869 |
| **10** | 0.2276 | 0.2642 | 0.9104 | 0.8946 | 0.9104 | 0.8946 |
| 11 | 0.2282 | 0.2726 | 0.9090 | 0.8908 | 0.9089 | 0.8907 |
| 12 | 0.2188 | 0.2687 | 0.9133 | 0.8927 | 0.9132 | 0.8926 |
| 13 | 0.2149 | 0.2721 | 0.9139 | 0.8869 | 0.9138 | 0.8867 |
| 14 | 0.2149 | 0.2733 | 0.9139 | 0.8865 | 0.9138 | 0.8863 |
| 15 | 0.2002 | 0.2693 | 0.9217 | 0.8946 | 0.9217 | 0.8946 |

#### 15. Follow-up F — baseline LSTM, seed 123

| Epoch | Train loss | Val loss | Train accuracy | Val accuracy | Train macro-F1 | Val macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6360 | 0.6434 | 0.6482 | 0.6343 | 0.6482 | 0.6343 |
| 2 | 0.4530 | 0.4664 | 0.7994 | 0.7929 | 0.7994 | 0.7929 |
| 3 | 0.3937 | 0.4105 | 0.8332 | 0.8236 | 0.8332 | 0.8235 |
| 4 | 0.3462 | 0.3634 | 0.8507 | 0.8435 | 0.8507 | 0.8435 |
| 5 | 0.3202 | 0.3435 | 0.8686 | 0.8595 | 0.8684 | 0.8592 |
| 6 | 0.2782 | 0.3009 | 0.8860 | 0.8767 | 0.8859 | 0.8766 |
| 7 | 0.2643 | 0.2914 | 0.8902 | 0.8810 | 0.8901 | 0.8809 |
| 8 | 0.2374 | 0.2783 | 0.9031 | 0.8827 | 0.9030 | 0.8826 |
| 9 | 0.2253 | 0.2652 | 0.9074 | 0.8926 | 0.9073 | 0.8924 |
| 10 | 0.2215 | 0.2656 | 0.9081 | 0.8862 | 0.9079 | 0.8859 |
| **11** | 0.1955 | 0.2536 | 0.9234 | 0.9000 | 0.9234 | 0.8999 |
| 12 | 0.2010 | 0.2816 | 0.9184 | 0.8903 | 0.9182 | 0.8899 |
| 13 | 0.1731 | 0.2553 | 0.9333 | 0.8971 | 0.9333 | 0.8971 |
| 14 | 0.1603 | 0.2786 | 0.9391 | 0.8997 | 0.9391 | 0.8997 |
| 15 | 0.1438 | 0.2602 | 0.9489 | 0.8978 | 0.9489 | 0.8978 |
| 16 | 0.1541 | 0.2910 | 0.9406 | 0.8858 | 0.9406 | 0.8855 |

#### 16. Follow-up F — D3 learning rate 0.003, seed 123

| Epoch | Train loss | Val loss | Train accuracy | Val accuracy | Train macro-F1 | Val macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4493 | 0.4637 | 0.7983 | 0.7875 | 0.7982 | 0.7874 |
| 2 | 0.3350 | 0.3463 | 0.8620 | 0.8564 | 0.8612 | 0.8557 |
| 3 | 0.2442 | 0.2723 | 0.9018 | 0.8928 | 0.9018 | 0.8928 |
| 4 | 0.2203 | 0.2587 | 0.9120 | 0.8982 | 0.9120 | 0.8982 |
| 5 | 0.2131 | 0.2629 | 0.9142 | 0.8904 | 0.9141 | 0.8902 |
| 6 | 0.1863 | 0.2516 | 0.9275 | 0.8990 | 0.9275 | 0.8990 |
| **7** | 0.1645 | 0.2575 | 0.9367 | 0.9018 | 0.9367 | 0.9018 |
| 8 | 0.1528 | 0.2701 | 0.9421 | 0.8896 | 0.9420 | 0.8894 |
| 9 | 0.1211 | 0.2720 | 0.9580 | 0.8939 | 0.9580 | 0.8939 |
| 10 | 0.1001 | 0.2801 | 0.9685 | 0.8914 | 0.9685 | 0.8914 |
| 11 | 0.0803 | 0.3098 | 0.9752 | 0.8881 | 0.9752 | 0.8881 |
| 12 | 0.0604 | 0.3696 | 0.9798 | 0.8897 | 0.9798 | 0.8897 |

### G. Full paired validation subgroup results

LSTM is the left model and mean-pooling logistic regression is the right model. Group counts and metrics are rounded from the saved CSV; both sentiment classes occur in each listed group. A review can appear in one group for each feature, so rows from different groupings are not independent.

| Feature | Group | n | Negative / positive | Small group (<30) | LSTM accuracy | LSTM macro-F1 | LR accuracy | LR macro-F1 | LSTM only correct | LR only correct |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| length_bin | 0–128 | 981 | 435 / 546 | No | 0.9195 | 0.9184 | 0.8420 | 0.8403 | 106 | 30 |
| length_bin | 129–256 | 3599 | 1854 / 1745 | No | 0.9005 | 0.9005 | 0.8352 | 0.8352 | 368 | 133 |
| length_bin | 257–512 | 1972 | 981 / 991 | No | 0.9102 | 0.9102 | 0.8408 | 0.8408 | 200 | 63 |
| length_bin | >512 | 885 | 435 / 450 | No | 0.8915 | 0.8914 | 0.8610 | 0.8610 | 78 | 51 |
| negation_indicator | False | 927 | 323 / 604 | No | 0.9223 | 0.9134 | 0.8501 | 0.8309 | 91 | 24 |
| negation_indicator | True | 6510 | 3382 / 3128 | No | 0.9020 | 0.9019 | 0.8393 | 0.8391 | 661 | 253 |
| oov_bin | ≤1% | 5614 | 2880 / 2734 | No | 0.9043 | 0.9043 | 0.8441 | 0.8441 | 546 | 208 |
| oov_bin | >1–5% | 1814 | 822 / 992 | No | 0.9046 | 0.9038 | 0.8291 | 0.8271 | 206 | 69 |
| oov_bin | >5% | 9 | 3 / 6 | Yes | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| oov_bin_256 | ≤1% | 5560 | 2845 / 2715 | No | 0.9036 | 0.9036 | 0.8414 | 0.8413 | 552 | 206 |
| oov_bin_256 | >1–5% | 1868 | 857 / 1011 | No | 0.9069 | 0.9062 | 0.8378 | 0.8363 | 200 | 71 |
| oov_bin_256 | >5% | 9 | 3 / 6 | Yes | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| oov_bin_512 | ≤1% | 5639 | 2883 / 2756 | No | 0.9044 | 0.9044 | 0.8438 | 0.8437 | 551 | 209 |
| oov_bin_512 | >1–5% | 1789 | 819 / 970 | No | 0.9044 | 0.9037 | 0.8301 | 0.8284 | 201 | 68 |
| oov_bin_512 | >5% | 9 | 3 / 6 | Yes | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| truncated_256 | False | 4580 | 2289 / 2291 | No | 0.9046 | 0.9046 | 0.8367 | 0.8367 | 474 | 163 |
| truncated_256 | True | 2857 | 1416 / 1441 | No | 0.9044 | 0.9044 | 0.8470 | 0.8470 | 278 | 114 |
| truncated_512 | False | 6552 | 3270 / 3282 | No | 0.9063 | 0.9063 | 0.8379 | 0.8379 | 674 | 226 |
| truncated_512 | True | 885 | 435 / 450 | No | 0.8915 | 0.8914 | 0.8610 | 0.8610 | 78 | 51 |
| left_confidence_bin | 0.50–0.60 | 236 | 122 / 114 | No | 0.5424 | 0.5408 | 0.6229 | 0.6214 | 47 | 66 |
| left_confidence_bin | 0.60–0.80 | 687 | 377 / 310 | No | 0.6914 | 0.6871 | 0.6332 | 0.6294 | 139 | 99 |
| left_confidence_bin | 0.80–1.00 | 6514 | 3206 / 3308 | No | 0.9401 | 0.9401 | 0.8704 | 0.8704 | 566 | 112 |
| right_confidence_bin | 0.50–0.60 | 750 | 378 / 372 | No | 0.8040 | 0.8040 | 0.5520 | 0.5520 | 263 | 74 |
| right_confidence_bin | 0.60–0.80 | 1797 | 900 / 897 | No | 0.8481 | 0.8481 | 0.7190 | 0.7190 | 351 | 119 |
| right_confidence_bin | 0.80–1.00 | 4890 | 2427 / 2463 | No | 0.9407 | 0.9407 | 0.9297 | 0.9296 | 138 | 84 |
