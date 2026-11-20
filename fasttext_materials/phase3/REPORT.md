# Phase 3 research report: fastText + LSTM sentiment classification

This report documents the two full runs performed by the user: the initial 256-token LSTM and a subsequent 512-token run. The figures are taken from the saved run configurations, statistics, histories, and best-checkpoint metrics. **All scores are validation scores, not held-out test scores.**

## Research questions

1. How does a unidirectional LSTM using frozen fastText token vectors compare on validation with the Phase 2 mean- and max-pooling baselines?
2. In this setup, what changes when the retained review prefix increases from 256 to 512 tokens?
3. How much review text is truncated, and what do the training histories show about the selected checkpoints?

Both experiments use one fixed split and one training seed. The 512-token run was performed after the 256-token run, and the same validation set was used to choose checkpoints and compare configurations. Results should therefore be treated as exploratory validation findings; they are not independent final evaluations.

## Data and experimental boundary

The runs use the same project split as Phases 1 and 2: **34,707 training**, **7,437 validation**, and **7,438 held-out test** reviews. Validation contains 3,705 negative and 3,732 positive reviews. Assignment 1 pooled the two published source partitions before making this stratified split with seed 42, so it is not the official IMDb benchmark split.

Source labels are read to reproduce the earlier stratification and fingerprint. Phase 3 sequence preparation, model fitting, checkpoint selection, and evaluation use the resulting training and validation partitions only. The project test partition was **not evaluated** in these runs. The data revision, fingerprints, model SHA-256, tokenizer, and environment are recorded in the [256-token configuration](results/lstm-s42-h64-l256-b32-lr0.001-e30/run_config.json) and [512-token configuration](results/lstm-s42-h64-l512-b32-lr0.001-e30/run_config.json); their train and validation index fingerprints match.

The uncompressed pretrained model SHA-256 is `14c7167b130056944cbdc37b7451f055867fe9a4e3fed3bbc1ecc0e74f6763ca`. Both runs reused the original 300-dimensional `cc.en.300.bin` binary. The original Phase 1 `regex` tokenization was applied without lowercasing or removing HTML break tags.

## Method

Each review's token sequence was kept in order. The program retained the **first N tokens** and discarded the remainder, where N was 256 or 512. The original fastText binary supplied a vector for each distinct token, including subword-based OOV vectors where available; every token received its own ID, with no generic OOV replacement. The embedding table was frozen. Unlike Phase 2 pooling, real tokens whose vectors were zero remained in the sequence as positions.

Review sequences were stored as compact token IDs and padded dynamically within a batch. For this unidirectional LSTM, the output at each review's final valid token was selected, so later right padding did not determine the classification state. Empty sequences would use a zero hidden state; neither full run encountered empty reviews.

Both models used the same settings except for the sequence limit:

| Setting | Value |
|---|---|
| Architecture | Frozen fastText (300) → one-layer unidirectional LSTM (hidden size 64) → `Linear(64, 1)` |
| Optimizer / learning rate | Adam / 0.001 |
| Training batch size | 32 |
| Loss / probability threshold | BCE-with-logits / 0.5 |
| Maximum epochs / early-stopping patience | 30 / 5 |
| Dropout / weight decay | 0 / 0 |
| Training seed / split seed | 42 / 42 |

Training and validation loss, accuracy, and macro-F1 were measured after each epoch. The checkpoint with the **strictly highest validation macro-F1** was saved; equal values favored the earlier epoch. Early stopping occurred after five consecutive epochs without a strict improvement. Final metrics and confusion matrices were recomputed after restoring the best checkpoint.

## Truncation and retained tokens

The review percentages and token-discard percentages measure different things: the first counts reviews exceeding the limit; the second counts the discarded tokens across all reviews in the partition. Tokens include punctuation and retained HTML-markup tokens produced by the specified regex.

| Limit | Partition | Reviews longer than limit | Tokens discarded / total tokens | Tokens discarded |
|---:|---|---:|---:|---:|
| 256 | Training | 13,466 / 34,707 (38.80%) | 2,938,399 / 9,814,658 | 29.94% |
| 256 | Validation | 2,857 / 7,437 (38.42%) | 621,532 / 2,094,907 | 29.67% |
| 512 | Training | 4,092 / 34,707 (11.79%) | 952,551 / 9,814,658 | 9.71% |
| 512 | Validation | 885 / 7,437 (11.90%) | 203,188 / 2,094,907 | 9.70% |

Moving from 256 to 512 retained an additional 1,985,848 training tokens and 418,344 validation tokens. The resulting token lookup contained 111,826 distinct retained tokens for the 256 run and 125,403 for the 512 run, excluding the dedicated padding entry. These lookup counts cover training and validation tokens; validation-only token vectors were cached from the pretrained model without fitting a vocabulary cutoff, normalization statistic, or model parameter on validation data.

[256-token statistics](results/lstm-s42-h64-l256-b32-lr0.001-e30/sequence_statistics.csv) · [512-token statistics](results/lstm-s42-h64-l512-b32-lr0.001-e30/sequence_statistics.csv)

## Results and Phase 2 baselines

All models below were assessed on the same project validation reviews. Accuracy and macro-F1 are proportions; the LSTM values use their best validation macro-F1 checkpoints.

| Model | Input / pooling | Train accuracy | Validation accuracy | Validation macro-F1 | Best epoch |
|---|---|---:|---:|---:|---:|
| Logistic regression | Mean, full review | 0.8518 | 0.8407 | 0.8406 | — |
| Logistic regression | Max, full review | 0.6727 | 0.6628 | 0.6626 | — |
| MLP (64) | Mean, full review | 0.8497 | 0.8365 | 0.8365 | 28 |
| MLP (64) | Max, full review | 0.6698 | 0.6605 | 0.6604 | 19 |
| LSTM (64) | First 256 tokens | 0.9120 | 0.8791 | 0.8790 | 13 |
| LSTM (64) | First 512 tokens | 0.9451 | **0.9045** | **0.9045** | 17 |

The 512-token LSTM achieved **2.54 percentage points higher validation accuracy** and **2.55 points higher validation macro-F1** than the 256-token LSTM. Its validation macro-F1 was **6.39 points higher** than the best Phase 2 baseline, LR-Mean; the 256-token LSTM was 3.84 points higher. These are observed differences on this validation split, not confidence intervals or evidence of statistical significance.

Both LSTMs use the same architecture and specified settings, but the longer run sees more text, follows a different optimization trajectory, and selected its best checkpoint at a different epoch. It is plausible that retaining more review content helped, but these runs do not isolate the cause or establish that 512 tokens will always be better. They are one seed and one split, with the follow-up and checkpoint choices informed by validation results.

The LSTM-versus-pooling comparison also changes more than word order: the LSTMs see at most the first 256 or 512 tokens, while Phase 2 pools the full review; Phase 2 excludes zero-vector tokens, while the LSTMs retain those positions. Architectures, scaling, optimization, and checkpoint-selection procedures differ. The higher LSTM score is consistent with sequential modeling being useful, but **does not prove that word order alone caused the improvement**.

[256 vs. Phase 2 chart](results/lstm-s42-h64-l256-b32-lr0.001-e30/graphs/phase2_comparison.png) · [512 vs. Phase 2 chart](results/lstm-s42-h64-l512-b32-lr0.001-e30/graphs/phase2_comparison.png) · [Phase 2 full report](../phase2/REPORT.md)

### Per-class metrics and confusion matrices

Rows are actual labels and columns are predictions. Matrices use `[[TN, FP], [FN, TP]]`, with negative = 0 and positive = 1.

| LSTM limit | Negative precision | Negative recall | Positive precision | Positive recall | Confusion matrix |
|---:|---:|---:|---:|---:|---|
| 256 | 0.8990 | 0.8532 | 0.8613 | 0.9049 | `[[3161, 544], [355, 3377]]` |
| 512 | 0.9099 | 0.8972 | 0.8993 | 0.9118 | `[[3324, 381], [329, 3403]]` |

The 256-token checkpoint made 899 validation errors; the 512-token checkpoint made 710. The latter is 189 fewer errors in total, but the saved aggregates do not show which errors the models share or which individual predictions changed. The 256-token model's negative recall was lower than its positive recall on this validation sample; the 512-token model's recalls were closer. These observations should not be generalized without further evaluation.

[256 metrics](results/lstm-s42-h64-l256-b32-lr0.001-e30/metrics.json) · [256 confusion matrix](results/lstm-s42-h64-l256-b32-lr0.001-e30/graphs/confusion_matrix.png) · [512 metrics](results/lstm-s42-h64-l512-b32-lr0.001-e30/metrics.json) · [512 confusion matrix](results/lstm-s42-h64-l512-b32-lr0.001-e30/graphs/confusion_matrix.png)

## Training histories and checkpoint selection

- **256 tokens:** the best validation macro-F1 was 0.8790 at epoch 13; early stopping occurred after epoch 18. Validation loss was lowest at epoch 10 (0.2983), not the selected epoch 13 (0.2992), because selection used macro-F1. Training loss fell from 0.2169 at epoch 13 to 0.1635 at epoch 18 while validation loss rose from 0.2992 to 0.3230. Training accuracy at the selected checkpoint was 0.9120 versus 0.8791 validation. [Loss](results/lstm-s42-h64-l256-b32-lr0.001-e30/graphs/loss_by_epoch.png) · [Accuracy](results/lstm-s42-h64-l256-b32-lr0.001-e30/graphs/accuracy_by_epoch.png) · [Macro-F1](results/lstm-s42-h64-l256-b32-lr0.001-e30/graphs/macro_f1_by_epoch.png) · [All epochs](results/lstm-s42-h64-l256-b32-lr0.001-e30/history.csv).
- **512 tokens:** the best validation macro-F1 was 0.9045 at epoch 17; early stopping occurred after epoch 22. Validation loss reached its minimum at epoch 15 (0.2553), while the selected epoch 17 had validation loss 0.2645. Training loss then fell from 0.1468 at epoch 17 to 0.1078 at epoch 22 while validation loss rose to 0.3124. Training accuracy at the selected checkpoint was 0.9451 versus 0.9045 validation. [Loss](results/lstm-s42-h64-l512-b32-lr0.001-e30/graphs/loss_by_epoch.png) · [Accuracy](results/lstm-s42-h64-l512-b32-lr0.001-e30/graphs/accuracy_by_epoch.png) · [Macro-F1](results/lstm-s42-h64-l512-b32-lr0.001-e30/graphs/macro_f1_by_epoch.png) · [All epochs](results/lstm-s42-h64-l512-b32-lr0.001-e30/history.csv).

The later divergence of training and validation losses is **consistent with overfitting**, though the histories are noisy and do not by themselves establish a cause. Early stopping restored the earlier macro-F1-selected checkpoints rather than using the final epochs. The saved checkpoint metadata and weight shapes were inspected and agree with epochs 13 and 17 and the stated 300 → 64 → 1 architecture.

## Timing and reproducibility

The 256-token run recorded **298.71 seconds** for 18 training epochs; the 512-token run recorded **664.90 seconds** for 22. Each training time includes per-epoch training/validation evaluation and checkpoint writes, but excludes sequence preparation and the final best-checkpoint reevaluation. Recorded sequence-preparation times were 12.10 and 7.40 seconds respectively; the later run reused the original binary's cached fingerprint, and filesystem caches may differ. These figures are machine-specific and the runs completed different epoch counts, so they are not a controlled speed benchmark.

Both runs recorded Apple Silicon macOS, MPS, Python 3.11.14, PyTorch 2.6.0, fastText 0.9.3, NumPy 1.26.4, pandas 2.2.3, and scikit-learn 1.5.2. A fixed seed aids reproduction, but exact numerical results can vary across devices and software versions.

The user ran these commands from `assignment 2`:

```bash
phase2/.venv/bin/python phase3/run.py
phase2/.venv/bin/python phase3/run.py --max-sequence-length 512 --seed 42
```

Existing nonempty output directories are protected from overwrite. For instructions on a new run, checkpoint loading, cache reuse, and graph-only regeneration, see the [Phase 3 README](README.md). The two result directories contain `best_model.pt`, history and metrics in CSV/JSON, sequence statistics, baseline snapshots, and plots. Results, model files, caches, dataset files, and virtual environments are ignored by local Git rules; preserve or export the required artifacts separately when sharing.

## Limits and next steps

The results are from one train/validation split and one seed per length, with repeated validation use for checkpoint selection and model comparison. The pretrained web corpus may overlap with IMDb reviews; that was not assessed. The comparison does not independently establish the benefit of word order or of retaining more text.

A later, explicitly planned investigation could compare repeated seeds or training-fold evaluations, inspect errors and review lengths, or use a matched truncation setting for a more controlled comparison. Those would be new experiments. Keep the project test partition untouched until the final system and evaluation procedure are fixed.
