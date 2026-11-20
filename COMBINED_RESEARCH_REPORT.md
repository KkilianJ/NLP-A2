# IMDb sentiment classification: what fastText and GloVe reveal

**Joint research report for the Project 2 blog and oral assessment**  
Evidence snapshot: 20 November 2026. The [artifact index](fasttext_materials/INDEX.md) explains the copied fastText materials and the existing GloVe files. Scores below are macro-F1 percentages unless another metric is named.

## Our question and the main result

We investigated whether pretrained word vectors can help classify IMDb reviews as positive or negative, and what happens when we change tokenization, pooling, sequence models, embedding training and data size. Our two tracks used **300-dimensional fastText vectors trained on Common Crawl and Wikipedia** and **GloVe 6B vectors**, usually 100-dimensional in the GloVe classifiers. We also revisited the sparse BPE systems from Project 1.

Codex helped with code.

Within the fastText track, a frozen, 512-token LSTM achieved **90.45% validation macro-F1** for seed 42; a full-review mean-pool logistic baseline achieved **84.06%**. In the GloVe track, fine-tuned mean pooling achieved **88.77% mean validation macro-F1 over three seeds**, while a fine-tuned BiGRU achieved **88.34%**. These are useful descriptions of the pipelines we ran, not a controlled ranking of the embedding families. They use different preprocessing, input limits, architectures, fitting data and validation policies. The GloVe track also has previously saved held-out test results; fastText has no saved test result.

The stronger shared lesson is methodological: representation is a pipeline, not just a vector file. Mean pooling was stronger than max pooling in both tracks' particular tests; retaining more tokens helped the tested recurrent configurations; and trainability, training size, optimization and checkpoint selection all affected the outcome. Small differences and best-of-many validation scores need caution.

## Dataset, split and evaluation boundary

Both classifier tracks use `mteb/imdb` at revision `d05f0155defa7991dad75bc68c5ccb6774b1fdc5`. Project 1 pooled the published train and test files and created a new stratified split using seed 42: **34,707 project-training**, **7,437 project-validation**, and **7,438 project-test** reviews. This project test set is **not** the official IMDb benchmark test split. Negative is label 0 and positive is label 1. Macro-F1 is the mean of the positive-class and negative-class F1 scores; accuracy is the fraction of correct predictions.

The saved classifier records agree on the ordered source text/label SHA-256, `0ed0de756c57a4ac8868e7cc2a9970650e930d94cdb0347716d385c9e1e705be`. We independently hashed the saved GloVe pooled row IDs and found the same ordered training hash, `3fe569c54add383b4b7249ea3465297678599f30dc8c0d48d43ee7186f09ec82`, and validation hash, `4eb93d6010100998cfc33aecf6cf1533ca6ffb845ce463d88d2139a0f0e7514b`, recorded in the fastText configuration. The saved final GloVe validation predictions and later fastText error-analysis predictions contain the same 7,437 IDs in the same order, with **zero label mismatches**. This establishes alignment of those saved validation records, not equivalence of the models or their inputs. We did not find saved IDs or a cross-track fingerprint with which to verify the project-test indices independently. See the [fastText run configuration](fasttext_materials/phase2/results/seed-42_epochs-30/run_config.json), [GloVe pooled manifest](meanpool_output/manifest.json), and [split audit](JOINT_EVIDENCE_AUDIT.json).

The fastText classifiers fit on all 34,707 project-training reviews, except the explicit half-data experiment, and used project validation repeatedly for checkpoint selection and comparisons. For the later GloVe studies, 31,236 reviews were used for fitting and 3,471 as an **inner validation** set; the 7,437 project-validation reviews were then used for further comparison. The GloVe test figures below are saved evaluations on the 7,438 project-test reviews. There is no saved fastText test evaluation. Moreover, three GloVe systems have already been evaluated on that project test set, so it is no longer an untouched test set for choosing among joint-project systems.

The initial GloVe coverage study used the **published source-training partition** of 24,904 reviews, rather than the 34,707-review project-training partition. With the same source snapshot, that published partition overlaps the later project validation and test partitions. The descriptive coverage and tokenizer exploration may therefore have exposed holdout text before the project split was respected; the coverage script does not pin or save a dataset revision for that particular run. It did not report classifier test scores, but this overlap limits how strongly we can call the project holdout untouched. See [GloVe coverage code](vocabulary.py), its [saved results](vocabulary_results.json), and the [fastText Step 01 protocol](fasttext_materials/step01/REPORT.md).

## Embeddings, tokenization and coverage

fastText's original `cc.en.300.bin` has 300-dimensional vectors and a stored vocabulary of two million words. It can construct vectors for many strings absent from that stored vocabulary using character subwords. **Direct vocabulary coverage** means exact stored-word membership; a nonzero constructed OOV vector is neither a direct match nor evidence that its meaning is useful. GloVe 6B supplies fixed word vectors, so a word absent from its vocabulary has no native fastText-style subword vector. The study tested separate ways to handle those misses.

| Descriptive coverage study | Population and preprocessing | Direct token coverage | Direct type coverage |
|---|---|---:|---:|
| fastText, whitespace | Project-training reviews; case and markup retained | 91.14% | 31.11% |
| fastText, regex | Same population; punctuation and markup tokens retained | 99.31% | 81.70% |
| GloVe, whitespace | Published source-training reviews; cleaned and lowercased | 85.33% | 24.43% |
| GloVe, regex, contractions intact | Same GloVe population; punctuation-only tokens omitted | 97.67% | 74.14% |
| GloVe, regex, selected negative contractions expanded | Same GloVe population, with changed token counts | 98.25% | 74.17% |

These rows have different populations, vocabularies, tokenization and denominators. In particular, the fastText regex counts punctuation and HTML-related tokens that the GloVe coverage analysis removes. **Do not subtract the percentages to claim one embedding has better comparable coverage.** Separating punctuation increased direct coverage within each study, but coverage is not a sentiment score. Expanding selected contractions reduced GloVe unmatched occurrences in its descriptive study; that tokenizer change was not tested as a classifier intervention. The new fastText three-way negation-expansion Step 01 was prepared but has **no saved run**. [fastText coverage](fasttext_materials/results/full/coverage.csv) · [GloVe coverage](vocabulary_results.json) · [pending Step 01](fasttext_materials/step01/REPORT.md).

For fastText regex after removing break tags, 67,262 token occurrences were directly OOV and 65,162 had nonzero constructed vectors. This is a representability count under the stated tokenization, not a quality measurement. The classifiers instead retained the original case- and markup-preserving regex. The GloVe classifiers generally decoded HTML entities, removed tags, lowercased, normalized apostrophes and retained contractions. In the final GloVe recurrent comparison, absent GloVe words were skipped before taking the first 256 **covered** tokens. The fastText LSTM retained the first 256 or 512 regex tokens, including punctuation, markup and OOV tokens; its pretrained lookup could contain constructed vectors and it retained zero-vector positions. Thus even a nominal limit of 256 does not give the same model input.

## What the word-vector probes can and cannot tell us

We used hand-picked cosine similarities and nearest neighbours to inspect the spaces. In fastText, `good`–`bad` had cosine **0.752**, while `terrible`–`awful` had **0.868**. In GloVe 100d, `overrated` retrieved `underrated` with cosine **0.809**. These examples show that sentiment opposites can be distributionally related; they do not prove that sentiment is absent or that a classifier cannot learn a useful decision boundary. The probes, candidate pools and vectors differ, and the words were selected for illustration. [fastText pairs](fasttext_materials/results/full/semantic_pairs.csv) · [fastText neighbours](fasttext_materials/results/full/nearest_neighbors.csv) · [GloVe probes](final_glove_study/semantic_probes.json).

The fastText spelling probe used 19 selected words and five synthetic edit types, one corruption per word and type. Mean cosine ranged from **0.127** for substitutions to **0.269** for deletions; the unrelated control mean was **0.101**. Some misspellings had useful neighbours, while others did not. This small artificial sample is not a natural-typo benchmark, and some corrupted strings were already stored words. [Spelling pairs](fasttext_materials/results/full/spelling_pairs.csv) · [spelling plot](fasttext_materials/results/full/spelling_robustness.png).

For GloVe, a character 3–5-gram ridge model was trained to reconstruct vectors of known words, with a held-out **word-type** split. It achieved mean cosine **0.3904**, versus **0.1559** for the fitting-word mean-vector baseline; RMSE was **0.467404** versus **0.467613**. This tests reconstruction direction for held-out known words, not sentiment performance for truly unknown words. Completed review vectors were saved, but no sentiment-classifier comparison using those ridge-completed representations was run. Adding OOV-only character bigrams to classifiers also produced no consistent validation gain and sometimes overfit. [Reconstruction results](embedding_output/embedding_results.json) · [GloVe OOV logistic comparison](oov_logistic_results.json) · [learning curves](oov_learning_curves.png).

## Pooling and recurrent models

Mean pooling averages word vectors; max pooling keeps the largest value in each dimension. Both discard token order. The fastText Phase 2 models pooled the **whole review**, omitted zero vectors and used frozen 300d vectors. Logistic regression standardized its inputs; the 64-unit MLP used unscaled inputs, Adam and a validation-selected checkpoint. Both used a 0.5 decision threshold.

| fastText model and pooling | Project-validation macro-F1 | Selection |
|---|---:|---|
| Logistic regression, mean | 84.06% | One fitted model |
| Logistic regression, max | 66.26% | One fitted model |
| MLP, mean | 83.65% | Best validation epoch 28 |
| MLP, max | 66.04% | Best validation epoch 19 |

Mean was roughly 18 percentage points higher than max for each specified fastText head. The two mean models were much closer; scaling and checkpoint rules also differ, so their small gap is not an architecture-only result. [Phase 2 metrics](fasttext_materials/phase2/results/seed-42_epochs-30/comparison.csv) · [pooling graph](fasttext_materials/phase2/results/seed-42_epochs-30/graphs/mean_vs_max.png).

In a separate seed-42 GloVe comparison using **trainable** embeddings and a 128-token limit, mean pooling scored **85.66%**, max pooling **83.58%**. This earlier setup included randomly initialized vectors for training-vocabulary GloVe misses, unlike the final GloVe 256-covered-token comparison. The result supports mean pooling for these runs, not a universal claim about max pooling or a comparable 18-point cross-embedding difference. [GloVe pooling results](mean_vs_max_pooling_results.json).

The fastText recurrent baseline used one unidirectional LSTM layer with 64 hidden units, frozen 300d vectors, a linear head, Adam at 0.001, batch size 32, and validation macro-F1 checkpoint selection. Increasing its prefix from 256 to 512 tokens raised the seed-42 validation score from **87.90% to 90.45%**. The fraction of validation reviews longer than the cap fell from **38.42% to 11.90%**. The second configuration followed the first and both used the same validation set for selection, so this is an exploratory within-track result. A 512-token MLP head scored **89.79%**, with no demonstrated improvement over the linear head. [256-token run](fasttext_materials/phase3/results/lstm-s42-h64-l256-b32-lr0.001-e30/metrics.json) · [512-token run](fasttext_materials/phase3/results/lstm-s42-h64-l512-b32-lr0.001-e30/metrics.json) · [head comparison](fasttext_materials/phase3/results/lstm-mlp64-s42-h64-l512-b32-lr0.001-e30/head_comparison.csv).

The final GloVe recurrent setup used a **bidirectional GRU**, 64 hidden units per direction, 100d vectors, 256 covered tokens, Adam at 0.001, batch size 128 and inner-validation selection. Its frozen version scored **86.87% ± 0.36 percentage points** across three seeds; a frozen 256-covered-token mean-pool logistic baseline scored **78.40%** in one fit. The optimized trainable mean-pool comparator was much stronger at **88.77% ± 0.12 pp**, versus **88.34% ± 0.23 pp** for fine-tuned BiGRU. The last two used different optimizers; the observed average difference is small. A single fine-tuned BiLSTM scored **86.86%**; it was not independently tuned, and equal hidden width does not equal equal parameter count. [GloVe recurrent runs](final_glove_study/recurrent_results.json) · [matched trainable mean pool](final_glove_study/matched_trainable_meanpool.json) · [BiLSTM](final_glove_study/bilstm_results.json).

The GloVe 50/100/200/300d frozen-pooling screen gave inner-validation macro-F1 of **76.15/80.35/83.12/83.95%**, respectively. The final recurrent study retained 100d for its internal comparison. It does not establish which GloVe dimension would be best for the recurrent model. [Dimension screen](final_glove_study/dimension_screen.json).

## Trainability, training size and overfitting

The following are within-track exploratory comparisons. For fastText, all rows use the 512-token LSTM and seed 42, unless stated otherwise. A2 updates the **token lookup** initialized from fastText; it does not retrain the underlying fastText character n-gram model. A3 uses a randomly initialized, trainable lookup. Its validation-only token rows do not receive training updates.

| fastText condition | Train macro-F1 | Project-validation macro-F1 | Best epoch |
|---|---:|---:|---:|
| Frozen pretrained lookup, full data | 94.51% | 90.45% | 17 |
| A2: trainable pretrained lookup, full data | 97.37% | 89.07% | 2 |
| A3: random trainable lookup, full data | 99.81% | 86.89% | 7 |
| B1: frozen lookup, half data (17,353 reviews) | 91.95% | 89.22% | 16 |

Here the trainable pretrained run fitted training data more closely but did not exceed the frozen run on validation; random trainable initialization scored lower still. These are single-seed observations, and the training and validation checkpoint scores are not independent because validation selected the checkpoint. In the GloVe BiGRU setup, by contrast, fine-tuning increased mean validation macro-F1 from **86.87% to 88.34%** across three paired seeds. The different outcomes concern different pipelines; they do not establish that one embedding family benefits more from fine-tuning. [fastText A2](fasttext_materials/remaining_experiments/results/A2/seed-42/metrics.json) · [A3](fasttext_materials/remaining_experiments/results/A3/seed-42/metrics.json) · [GloVe paired runs](final_glove_study/paired_summary.json).

Using half instead of all fitting reviews also lowered the observed score in both tracks: fastText **89.22% versus 90.45%** and frozen GloVe BiGRU **85.57% versus 86.87%** averaged over three seeds. The fastText full and half sets were 34,707 and 17,353; the GloVe sets were 31,236 and 15,618. The half-data conditions also received fewer optimizer updates for a given epoch budget, so data quantity is not the only changed exposure. [fastText B1 configuration](fasttext_materials/remaining_experiments/results/B1/seed-42/run_config.json) · [GloVe recurrent runs](final_glove_study/recurrent_results.json).

The fastText frozen 512-token baseline's training loss kept falling after its best validation checkpoint while validation loss increased; its selected train/validation macro-F1 gap was **4.05 pp**. The A2 gap was **8.31 pp** and A3's **12.93 pp**. This pattern is consistent with overfitting under these settings. In the GloVe fine-tuned BiGRU runs, online training macro-F1 approached 98% while inner validation declined after early peaks. Training metrics have different measurement conventions in the two tracks, so compare their trends rather than treating the gaps as a matched cross-track statistic. [fastText loss history](fasttext_materials/phase3/results/lstm-s42-h64-l512-b32-lr0.001-e30/history.csv) · [GloVe learning curves](final_glove_study/gru_learning_curves.png).

Other seed-42 fastText checks were 32 hidden units **81.64%**, 128 units **89.61%**, learning rate 0.0003 **89.28%**, learning rate 0.003 **90.21%**, dropout 0.3 **89.79%**, and weight decay 0.0001 **89.46%**, compared with the 64-unit, 0.001 baseline at **90.45%**. The low-rate run peaked at its 30-epoch cap; the 32-unit run stopped early after an unstable trajectory. These results do not establish general width, rate or regularization rankings. Configurations, checkpoints and complete histories are in the [fastText index](fasttext_materials/INDEX.md).

## Experimental variance and error analysis

The frozen fastText baseline and the 0.003 learning-rate variant have two saved seeds each. Their validation macro-F1 was **90.45% versus 90.21%** at seed 42 and **89.99% versus 90.18%** at seed 123: the ordering reversed. Two seeds do not reliably estimate training variance, and they share the same data split. The intended seed 2026 repeat is not present in the saved artifacts. The GloVe three-seed standard deviations likewise describe training variation on one fixed split, not variation across datasets or splits. [fastText baseline repeat](fasttext_materials/remaining_experiments/results/F-base/seed-123/metrics.json) · [fastText D3 repeat](fasttext_materials/remaining_experiments/results/D3/seed-123/metrics.json) · [GloVe paired summary](final_glove_study/paired_summary.json).

The saved fastText validation error analysis, completed after the initial version of the project log, pairs the 512-token LSTM and full-review mean-pool logistic regression by review ID. Both were correct on **5,975** reviews, only the LSTM on **752**, only logistic regression on **277**, and neither on **433**. For the **885** validation reviews longer than 512 fastText-regex tokens, LSTM macro-F1 was **89.14%**, versus **90.63%** on the 6,552 shorter reviews. Length, truncation and review content can all differ, so this does not prove truncation caused the gap. The logistic model also saw full reviews and used different scaling, zero handling and optimization; the paired comparison does not isolate word order. [FastText error summary](fasttext_materials/remaining_experiments/results/errors-lstm-vs-lr/summary.json) · [subgroups](fasttext_materials/remaining_experiments/results/errors-lstm-vs-lr/subgroups.csv) · [paired graph](fasttext_materials/remaining_experiments/results/errors-lstm-vs-lr/paired_outcomes.png).

For the GloVe seed-42 trainable comparison, only BiGRU was correct on **286** validation reviews and only MeanPool on **307**. A paired bootstrap over reviews gave **−0.94 to +0.35 pp** for BiGRU minus MeanPool macro-F1, for those fixed checkpoints. That interval is not a confidence interval over training seeds or model-selection decisions. Both scored lower on the defined long/truncated subgroup than on the short subgroup. Surface negation flags and short review excerpts can guide inspection but do not establish the cause of an error. [GloVe paired analysis](final_glove_study/trainable_baseline_error_analysis.json) · [three-model subgroup results](final_glove_study/bilstm_results.json).

## Previously saved held-out test results

These GloVe results were already present; this report did not evaluate the test set. Each uses seed 42 and an inner-selected checkpoint. Project-validation scores are included to make the split distinction explicit.

| GloVe system | Project-validation macro-F1 | Project-test macro-F1 | Project-test accuracy |
|---|---:|---:|---:|
| Fine-tuned MeanPool | 88.85% | 89.11% | 89.11% |
| Fine-tuned BiGRU | 88.57% | 88.76% | 88.76% |
| Fine-tuned BiLSTM | 86.86% | 86.68% | 86.69% |
| fastText systems | See validation tables | **Not evaluated in saved work** | **Not available** |

The BiLSTM test macro-F1 is **86.683%** in the underlying file; the original GloVe README rounds it to 86.69%, while ordinary two-decimal rounding gives **86.68%**. The separate test artifact was written after the validation-only BiLSTM file, whose `test_evaluated` field remains false; both files are retained as historical records. [MeanPool test](final_glove_study/matched_trainable_meanpool_test.json) · [BiGRU test](final_glove_study/recurrent_results.json) · [BiLSTM test](final_glove_study/bilstm_test_results.json).

## Relation to Project 1

Project 1 used byte-pair-encoded token features and classifiers rather than these dense word-vector pipelines. Its saved original baseline records **90.12% validation macro-F1** for BPE bigrams plus an MLP and **82.29%** for BPE counts plus a linear SVM; the bigram MLP's five-fold training CV mean was **90.34%**. Those historical systems used different feature construction and fitting procedures. The historical baseline configuration records source, seed and partition sizes, but does **not** preserve ordered dataset or validation-ID fingerprints, so exact identity of its scored validation rows cannot be independently proved from that result file alone. It is reasonable context for the same documented protocol, not a tightly matched, independently verified embedding comparison. [Project 1 saved validation table](../NLP1-Assignment1/project_starter/results/baseline/validation_scores.csv) · [Project 1 summary](../NLP1-Assignment1/project_starter/PROJECT_SUMMARY.md) · [historical configuration](../NLP1-Assignment1/project_starter/results/baseline/config.json).

## Limits and next evidence

- **No embedding-only experiment:** A direct fastText/GloVe test would require the same training and evaluation IDs, tokenizer, token handling, vector dimensionality, input budget, architecture, optimizer, fitting set and selection protocol. The existing models differ on several of these at once.
- **Validation selection:** fastText used the same project validation set for many choices and checkpoints; GloVe used inner validation but also compared many systems on project validation. The highest observed validation score is therefore susceptible to selection optimism. Seed repeats do not remove that effect.
- **Held-out boundary:** fastText has no saved final-test score. The project test split has already been used for several GloVe results, and initial GloVe descriptive coverage may overlap it. Any future final assessment should state that history and fix decisions before looking at further test outcomes.
- **Unfinished or absent evidence:** the new fastText negation-expansion coverage run, a matched cross-embedding comparison, further planned seed repeats, a classifier test of ridge-completed GloVe vectors, and a fastText held-out evaluation are not saved. There is no evidence here establishing pretraining-corpus overlap with IMDb or causal explanations for subgroup errors.
- **Reproducibility:** models and dataset remain in their original locations. The copied code has original relative defaults and some saved metadata contains absolute original paths; consult the [artifact index](fasttext_materials/INDEX.md) before reproducing anything. Checkpoints are executable serialized objects and should only be loaded from trusted sources.

## Oral-assessment talking points

1. **What was the question?** We tested how pretrained word vectors and classifier choices represent movie-review sentiment, then compared the resulting systems with Project 1's sparse BPE features.
2. **Why coverage first?** Tokenization determines which strings match a stored vocabulary. Report token and type denominators, and distinguish a fastText subword-generated OOV vector from exact coverage or semantic quality.
3. **Why cosine probes?** They provide interpretable examples of relatedness and spelling robustness, but selected pairs are not a downstream benchmark. Contextual neighbours can include sentiment opposites.
4. **Why pooling and recurrence?** Pooling is a simple order-insensitive baseline. LSTMs and GRUs process sequences, but our compared pipelines also changed input limits and other settings, so the scores do not isolate order alone.
5. **Why freeze or fine-tune?** Frozen embeddings limit the number of learned parameters; fine-tuning can adapt representations but can also fit training data too closely. Our two tracks illustrate both outcomes under different settings.
6. **How did we check reliability?** We saved histories, checkpoints, confusion matrices, paired validation predictions and a few seed repeats. The repeated validation choices, limited seeds, subgroup differences and previously used test set constrain conclusions.
7. **What would a fair next comparison require?** Match the data IDs, preprocessing, dimensions, token budget, model and selection policy first; lock the planned analysis before any further held-out evaluation.
