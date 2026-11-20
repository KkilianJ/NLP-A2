# Phase 2 research report: fastText embed-and-pool classification

Recorded on 20 November 2026. This report documents the four models the user ran with `phase2/.venv/bin/python phase2/run.py --experiment all`. It uses the saved run configuration, result files, and epoch histories, not newly trained models.

## Research questions

1. With the same frozen fastText embeddings and review tokenizer, how do mean and max pooling compare for IMDb sentiment classification?
2. For each pooling method, how does logistic regression compare with a one-hidden-layer MLP under the specified initial settings?
3. What do the per-class results and training histories reveal about these four specific runs?

These are validation-set comparisons from one partition and one training seed. They are not final test results, and they do not establish a general ranking across datasets, seeds, or preprocessing choices.

## Data and evaluation boundary

The dataset is the pinned [mteb/imdb revision `d05f0155`](https://huggingface.co/datasets/mteb/imdb/tree/d05f0155defa7991dad75bc68c5ccb6774b1fdc5), using the same split as Assignment 1 and Phase 1. Its published train and test files were pooled first, then stratified with seed 42 into **34,707 training**, **7,437 validation**, and **7,438 held-out test** reviews. Thus these are project-specific partitions, not the official IMDb benchmark split.

The training partition has 17,288 negative and 17,419 positive reviews. Validation has 3,705 negative and 3,732 positive reviews. Source labels were used to reproduce the original stratification and fingerprint; only the resulting training and validation reviews entered Phase 2 features, fitting, and evaluation. The held-out test partition was not evaluated or used to make experimental choices. The validation set was used both for these comparisons and for selecting MLP checkpoints, so it is no longer an untouched final evaluation set.

The [run configuration](results/seed-42_epochs-30/run_config.json) records the dataset revision, source fingerprint, train and validation index fingerprints, tokenizer, model hash, settings, and environment. The training-index fingerprint agrees with Phase 1.

## Representation and model setup

All models use the same original 300-dimensional `cc.en.300.bin` model. Its SHA-256 is `14c7167b130056944cbdc37b7451f055867fe9a4e3fed3bbc1ecc0e74f6763ca`. Embeddings were frozen; the binary's subword mechanism supplied vectors for OOV tokens when available. Every review was tokenized using the **unchanged Phase 1 `regex` tokenizer**: original capitalization and HTML break tags were retained. There was no HTML removal or lowercasing in these classifiers.

For each review, the program calculated both an elementwise mean and an elementwise maximum of its nonzero token vectors. It omitted zero-vector tokens from both operations; a review with no usable vectors would map to a zero vector. In this run, training had 9,814,658 tokens, including 2,100 zero-vector tokens; validation had 2,094,907 tokens, including 333 zero-vector tokens. Neither split contained an empty or all-zero review. Mean and max representations were cached and shared among classifiers.

Both pooling methods discard word order: two reviews with the same multiset of token vectors have the same pooled representation. Mean pooling reflects repeated occurrences in the average; repeating an already-present token cannot change a maximum. Max pooling can be influenced by individual extreme coordinate values. These properties help motivate hypotheses about the observed difference, but this experiment did not identify which tokens or coordinates caused it.

The models were:

| Configuration | Features and classifier | Training settings |
|---|---|---|
| LR-Mean | Mean vectors, training-fitted standardization, logistic regression | `C=1.0`, `max_iter=1000`, no class weighting, seed 42 |
| LR-Max | Max vectors, training-fitted standardization, logistic regression | Same logistic settings |
| MLP-Mean | Unscaled mean vectors, `Linear(300,64) → ReLU → Linear(64,1)` | Adam, learning rate 0.001, batch size 64, BCE-with-logits, 30 epochs, no dropout or weight decay, seed 42 |
| MLP-Max | Unscaled max vectors, the same MLP | Same MLP settings |

The logistic pipeline fitted its scaler **only on training vectors**. MLP training and validation losses and scores were computed after each epoch, in evaluation mode. A sigmoid and a 0.5 probability threshold were used for MLP inference; logistic probabilities used the same threshold. Each MLP's checkpoint was selected by the highest validation macro-F1, with the earliest epoch winning ties.

The LR/MLP comparison changes both classifier and feature scaling: LR used standardized features, while the MLP used unscaled features. Consequently, a difference between those two families cannot be attributed to architecture alone.

## Results

Scores below are proportions; multiply by 100 to express accuracy as a percentage. MLP values come from their respective best validation macro-F1 checkpoints, not necessarily their final epochs.

| Experiment | Training accuracy | Validation accuracy | Validation macro-F1 | Best MLP epoch |
|---|---:|---:|---:|---:|
| LR-Mean | 0.8518 | **0.8407** | **0.8406** | — |
| LR-Max | 0.6727 | 0.6628 | 0.6626 | — |
| MLP-Mean | 0.8497 | 0.8365 | 0.8365 | 28 |
| MLP-Max | 0.6698 | 0.6605 | 0.6604 | 19 |

[Exact comparison CSV](results/seed-42_epochs-30/comparison.csv) · [Accuracy chart](results/seed-42_epochs-30/graphs/validation_accuracy_comparison.png) · [Macro-F1 chart](results/seed-42_epochs-30/graphs/validation_macro_f1_comparison.png) · [Mean vs. max](results/seed-42_epochs-30/graphs/mean_vs_max.png)

Within this setup, mean pooling exceeded max pooling by **17.79 percentage points of validation accuracy** for logistic regression and **17.60 points** for the MLP. The corresponding macro-F1 differences were 17.80 and 17.61 points. This consistent gap is an observation about these representations, models, and one split; it does not by itself prove that max pooling is generally unsuitable or why it performed worse.

LR-Mean had the highest recorded validation accuracy and macro-F1. Its validation accuracy exceeded MLP-Mean by **0.42 percentage points**. That small difference has not been tested for statistical significance or stability across seeds. In addition to the scaling difference, the MLP checkpoint was selected from 30 validation evaluations while LR was fitted once. These results are useful for describing this run, not enough to declare a universally superior classifier.

### Precision, recall, and confusion matrices

Rows are actual classes and columns are predicted classes. Each matrix is `[[TN, FP], [FN, TP]]`, with negative = 0 and positive = 1.

| Experiment | Negative precision | Negative recall | Positive precision | Positive recall | Validation confusion matrix |
|---|---:|---:|---:|---:|---|
| LR-Mean | 0.8439 | 0.8345 | 0.8375 | 0.8467 | `[[3092, 613], [572, 3160]]` |
| LR-Max | 0.6670 | 0.6451 | 0.6588 | 0.6803 | `[[2390, 1315], [1193, 2539]]` |
| MLP-Mean | 0.8379 | 0.8329 | 0.8351 | 0.8400 | `[[3086, 619], [597, 3135]]` |
| MLP-Max | 0.6521 | 0.6826 | 0.6696 | 0.6385 | `[[2529, 1176], [1349, 2383]]` |

LR-Mean made 1,185 validation errors, compared with 1,216 for MLP-Mean, 2,508 for LR-Max, and 2,525 for MLP-Max. These totals do not tell us how many mistakes the models share. The MLP-Max checkpoint had lower positive recall than negative recall; this describes the fixed validation sample, not a claim about all future data.

### MLP training histories

- **MLP-Mean:** validation loss fell from 0.5361 at epoch 1 to its minimum of 0.3707 at epoch 28. At epoch 28, validation accuracy was 0.8365 and macro-F1 was 0.8365. At epoch 30, validation loss was 0.3732 and macro-F1 was 0.8334. The curves show some late variation, but a small train–validation gap alone does not diagnose the amount or cause of overfitting. [Loss](results/seed-42_epochs-30/mlp-mean/graphs/loss_by_epoch.png) · [Accuracy](results/seed-42_epochs-30/mlp-mean/graphs/accuracy_by_epoch.png) · [Macro-F1](results/seed-42_epochs-30/mlp-mean/graphs/macro_f1_by_epoch.png) · [Epoch data](results/seed-42_epochs-30/mlp-mean/epochs.csv).
- **MLP-Max:** the highest validation macro-F1 was 0.660350 at epoch 19, so that checkpoint was saved. Its lowest validation loss, 0.6164, was at epoch 28. The final epoch had validation accuracy 0.6624, slightly higher than the epoch-19 accuracy of 0.6605, but its macro-F1 (0.660196) was lower than epoch 19's. This is expected because checkpoint selection used macro-F1, not validation loss or accuracy. [Loss](results/seed-42_epochs-30/mlp-max/graphs/loss_by_epoch.png) · [Accuracy](results/seed-42_epochs-30/mlp-max/graphs/accuracy_by_epoch.png) · [Macro-F1](results/seed-42_epochs-30/mlp-max/graphs/macro_f1_by_epoch.png) · [Epoch data](results/seed-42_epochs-30/mlp-max/epochs.csv).

The saved PyTorch checkpoints were inspected without retraining: their selected epochs are 28 and 19, and their weight shapes match the stated 300 → 64 → 1 architecture.

## Timing and warnings

The saved feature-preparation time was **36.39 seconds**, including the first model fingerprint check, binary loading, feature extraction, and cache work. Reported model times were **0.69 s** for LR-Mean, **0.26 s** for LR-Max, **26.71 s** for MLP-Mean, and **25.56 s** for MLP-Max. These are machine-specific timings with different definitions: LR time includes pipeline fitting but excludes final evaluation; MLP time includes its training loop and per-epoch train/validation evaluation. They should not be treated as a controlled architecture-speed benchmark.

Both LR fits recorded a SciPy deprecation warning about the L-BFGS-B `disp` and `iprint` options. It is not a reported convergence failure. The saved solver iteration counts were 232 for LR-Mean and 64 for LR-Max, both below the 1,000-iteration cap; no convergence warning was recorded. The warning is relevant for future dependency updates.

The recorded environment was Python 3.11.14, fastText 0.9.3, NumPy 1.26.4, scikit-learn 1.5.2, and PyTorch 2.6.0 on Apple Silicon macOS; both MLPs used MPS. Identical seeds do not guarantee bit-for-bit equivalence across hardware, backends, or package versions.

## Interpretation and limitations

The strongest observed pattern is the roughly 18-point validation advantage of mean pooling for **both** specified classifiers. Averaging preserves information about token multiplicity, while an elementwise maximum records the largest value in each dimension; however, there was no token-contribution or representation analysis here, so an explanation involving punctuation, markup, review length, outliers, or a specific fastText property remains a hypothesis.

The two mean-pooling systems perform similarly on this validation split. Their small difference is confounded by different scaling and checkpoint-selection procedures. Both pooling methods discard order, so they cannot explicitly retain the position or scope of a negation such as “not good”; a later classifier may still learn useful correlations from the pooled features.

This is one fixed split and one seed, with four predefined configurations and no statistical comparison. MLP checkpoint selection and model comparisons both used validation, so repeat analyses or training-fold comparisons should precede strong comparative claims. The pretrained web corpus may overlap with IMDb text, which was not assessed. No held-out test evaluation, recurrent model, or hyperparameter sweep has been performed in Phase 2.

For a defensible next step, inspect errors and the contributors to max-pooled coordinates on training data, and assess promising choices through training-only cross-validation or additional appropriately planned repetitions. Keep the held-out project test partition reserved for a final evaluation after decisions are fixed.

## Reproduction and saved artifacts

See the [Phase 2 README](README.md) for environment setup, individual-model commands, cache reuse, and graph regeneration. For a new complete run without overwriting the existing one:

```bash
phase2/.venv/bin/python phase2/run.py --experiment all --run-dir phase2/results/reproduction
```

Run that command from the `assignment 2` directory. The original results remain in `phase2/results/seed-42_epochs-30/`: each LR folder contains a `model.joblib` pipeline, each MLP folder a `best_model.pt` checkpoint and `epochs.csv/json`, and each folder contains metrics and plots. The results directory, binary model, data files, environments, and caches are excluded by local `.gitignore` rules; they exist locally but will not be included in an ordinary Git commit. Preserve or explicitly export the needed artifacts when sharing the project.
