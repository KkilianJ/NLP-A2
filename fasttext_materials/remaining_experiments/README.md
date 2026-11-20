# Remaining fastText experiments

This directory prepares individual experiments A–H. Nothing runs on import and there is no run-all command. Run every command from `assignment 2` with the existing `phase2/.venv` environment. [RUN_ORDER.md](RUN_ORDER.md) gives the suggested sequence. The existing `cc.en.300.bin`, Phase 2/3 code, artifacts, and caches are not modified.

The baseline is the existing seed-42, frozen-fastText, 512-token, 64-unit unidirectional LSTM with a linear head. The same original regex, case, HTML handling, token IDs, 300-dimensional vectors, validation reviews, batch size 32, Adam at 0.001, 30-epoch maximum, and patience 5 are used unless the experiment explicitly changes one item. The earliest epoch wins ties in validation macro-F1. The baseline was previously measured at validation macro-F1 0.904521; that is not a test score.

`run.py` verifies and reuses the saved Phase 3 **512-token sequence/vector cache**, including file checksums, split fingerprints, exact tokenizer, and pretrained model identity. It does not load the large binary when that cache is available. If it is missing, restore it from the earlier run; this suite will not silently regenerate or download it. Each non-baseline run gets its own directory and refuses to overwrite any existing directory. For a new repeat, add `--output` with a fresh path.

## Experiment commands

Each command is explicit and starts at most one model. A1/B2/C2/D2/E1 are read-only aliases for the existing baseline. Times depend on your Mac and early stopping; costs below are qualitative relative to one frozen 512-token LSTM run.

| ID | Research question / change | Existing baseline reusable? | Exact command | Output | Relative cost / limitation |
|---|---|---|---|---|---|
| A1 | Pretrained contribution: frozen lookup | Yes | `phase2/.venv/bin/python remaining_experiments/run.py --experiment A1` | Existing `phase3/results/lstm-s42-h64-l512-b32-lr0.001-e30/` | Read-only. |
| A2 | Fine-tune pretrained token lookup | A1 is reference | `phase2/.venv/bin/python remaining_experiments/run.py --experiment A2` | `remaining_experiments/results/A2/seed-42/` | Higher memory and checkpoint I/O; trains lookup rows, not fastText character n-grams. |
| A3 | Random trainable token lookup | A1 is reference | `phase2/.venv/bin/python remaining_experiments/run.py --experiment A3` | `remaining_experiments/results/A3/seed-42/` | Similar cost to A2; same token IDs but no pretrained or subword information. |
| B1 | 50% stratified training data | B2 is reference | `phase2/.venv/bin/python remaining_experiments/run.py --experiment B1` | `remaining_experiments/results/B1/seed-42/` | Roughly half as many training batches per epoch; same full validation and per-epoch validation cost. |
| B2 | 100% training data | Yes | `phase2/.venv/bin/python remaining_experiments/run.py --experiment B2` | Existing 512-token baseline | Read-only. |
| C1 | Hidden size 32 | C2 is reference | `phase2/.venv/bin/python remaining_experiments/run.py --experiment C1` | `remaining_experiments/results/C1/seed-42/` | Smaller recurrent state; cost still depends on sequence lengths and epochs. |
| C2 | Hidden size 64 | Yes | `phase2/.venv/bin/python remaining_experiments/run.py --experiment C2` | Existing 512-token baseline | Read-only. |
| C3 | Hidden size 128 | C2 is reference | `phase2/.venv/bin/python remaining_experiments/run.py --experiment C3` | `remaining_experiments/results/C3/seed-42/` | More recurrent parameters and memory. |
| D1 | Learning rate 0.0003 | D2 is reference | `phase2/.venv/bin/python remaining_experiments/run.py --experiment D1` | `remaining_experiments/results/D1/seed-42/` | Baseline per-epoch cost; stopping epoch may differ. |
| D2 | Learning rate 0.001 | Yes | `phase2/.venv/bin/python remaining_experiments/run.py --experiment D2` | Existing 512-token baseline | Read-only. |
| D3 | Learning rate 0.003 | D2 is reference | `phase2/.venv/bin/python remaining_experiments/run.py --experiment D3` | `remaining_experiments/results/D3/seed-42/` | Baseline per-epoch cost; optimization may be unstable. |
| E1 | No regularization | Yes | `phase2/.venv/bin/python remaining_experiments/run.py --experiment E1` | Existing 512-token baseline | Read-only. |
| E2 | Dropout 0.3 after final valid LSTM state | E1 is reference | `phase2/.venv/bin/python remaining_experiments/run.py --experiment E2` | `remaining_experiments/results/E2/seed-42/` | Similar per-epoch cost. This is explicit post-LSTM dropout, not ineffective single-layer recurrent dropout. |
| E3 | Adam weight decay 0.0001 | E1 is reference | `phase2/.venv/bin/python remaining_experiments/run.py --experiment E3` | `remaining_experiments/results/E3/seed-42/` | Similar per-epoch cost; Adam's L2-style weight decay applies to all trainable parameters. |
| F-base | Baseline variability, seed 123 | Seed 42 reused | `phase2/.venv/bin/python remaining_experiments/run.py --experiment F-base --seed 123` | `remaining_experiments/results/F-base/seed-123/` | One full LSTM run. Replace `123` with `2026` for the third seed; 42 reads the existing baseline. |
| F-alternative | Repeat one selected alternative, e.g. A2 | A2 seed 42 reused if already run | `phase2/.venv/bin/python remaining_experiments/run.py --experiment F-alternative --alternative A2 --seed 123` | `remaining_experiments/results/A2/seed-123/` | One run of the selected configuration; use `2026` for the third seed. Choose the alternative explicitly after reviewing results. |
| F-mean | Repeat mean-pooling LR | Phase 2 seed 42 reused | `phase2/.venv/bin/python remaining_experiments/run.py --experiment F-mean --seed 123` | `remaining_experiments/results/F-mean/seed-123/` | One LR fit on cached pooled features. LR may be effectively deterministic across seeds; this is not an LSTM repeat. |
| G1 | Paired validation errors: LSTM versus LR-Mean | Both saved models | `phase2/.venv/bin/python remaining_experiments/error_analysis.py --left lstm512 --right lr-mean --output remaining_experiments/results/errors-lstm-vs-lr` | `remaining_experiments/results/errors-lstm-vs-lr/` | Two inference passes; first analysis loads fastText for OOV features. LR pools full reviews, LSTM truncates. |
| G2 | Paired validation errors: 256 versus 512 tokens | Both saved models | `phase2/.venv/bin/python remaining_experiments/error_analysis.py --left lstm256 --right lstm512 --output remaining_experiments/results/errors-256-vs-512` | `remaining_experiments/results/errors-256-vs-512/` | Two inference passes; reused validation features if present. |
| G3 | Paired validation errors: frozen versus fine-tuned | Baseline and completed A2 | `phase2/.venv/bin/python remaining_experiments/error_analysis.py --left lstm512 --right A2 --output remaining_experiments/results/errors-frozen-vs-finetuned` | `remaining_experiments/results/errors-frozen-vs-finetuned/` | Requires A2; compares one selected checkpoint per model. |
| H1 | Compare Project 1 and Project 2 validation artifacts | Saved models and scores | `phase2/.venv/bin/python remaining_experiments/project1_comparison.py --output remaining_experiments/results/project1-comparison` | `remaining_experiments/results/project1-comparison/` | Low cost; Project 1's saved split fingerprints are missing, so exact comparability is unverified. |
| H2 | Final held-out evaluation of one locked model | Saved selected checkpoint | `phase2/.venv/bin/python remaining_experiments/final_test.py --run lstm512 --confirm-model-locked --output remaining_experiments/results/final-locked-lstm512` | `remaining_experiments/results/final-locked-lstm512/` | Test access: execute only after locking the choice. May load fastText for unseen tokens. |

`F-alternative` accepts `A2`, `A3`, `B1`, `C1`, `C3`, `D1`, `D3`, `E2`, or `E3`; it writes under that experiment's ID. All training experiments accept seeds 42, 123, or 2026. B1 always selects the **same** stratified half using subset seed 42 so changes in model seed do not change the subset. For 34,707 training reviews, the 50% selection uses 17,353; the chosen IDs and fingerprint are saved. Its training metric is calculated on that subset and should not be compared as if it used all training reviews.

## Embeddings, checkpoints, and reproducibility

The existing cache contains a compact token lookup for retained training **and validation** tokens, rather than a 2-million-word trainable table. Validation-only token rows are initialized without labels and receive no training examples; the original Phase 3 cache used this same lookup. OOV tokens have their own IDs and exact original fastText subword vectors, not a generic UNK. In A2 the individual lookup rows are trainable; this does **not** update fastText's underlying word or character-ngram parameters. In A3 the same IDs receive seeded independent `normal(0, 0.05)` vectors, so OOV token IDs exist but their vectors contain no pretrained subword information. Padding row 0 stays zero. The LSTM initialization is seeded separately from the random table generator.

Single-layer PyTorch recurrent dropout remains zero. E2 uses `Dropout(0.3)` on the last valid state before the original linear classifier and is disabled for validation. All other models have no dropout. For trainable lookup runs, the selected embedding weights are saved in `best_model.pt`; frozen runs reference the verified cache to avoid repeatedly copying its large table. Retain that cache and the checkpoint together. Temporary best-checkpoint files are replaced atomically within the new run directory. The original binary is never updated.

Every run records `run_config.json` (full specification, split and cache identity, chosen training-ID hash, code hashes, device, and software versions), `history.csv/json`, `metrics.csv/json`, `best_model.pt`, `training_review_ids.npy`, and `graphs/`. Metrics include per-class precision/recall, confusion matrix, train/validation scores, best epoch, timing, and gaps. Training loss and both training/validation metrics are measured after each epoch in evaluation mode. A fixed seed helps, but MPS and other hardware/software may not reproduce bit-for-bit.

## Variance and comparisons

After you have run the desired individual seeds, summarize the baseline and a chosen alternative without training:

```bash
phase2/.venv/bin/python remaining_experiments/variance.py --alternative A2 \
  --output remaining_experiments/results/variance-A2
```

Use `--include-mean` to include completed F-mean runs separately. The program lists missing seeds instead of inventing results, validates within-group settings and cache identities, and writes individual runs, sample mean/std (`ddof=1`), paired seed differences, and a graph. Only matching validation/cache fingerprints are paired. The historical Phase 2 pooled cache recorded its identity but not array checksums: the suite records current checksums for new F-mean runs, but cannot prove those bytes match the original historical execution. F-mean uses full-review pooled representations and is not an LSTM repeat.

The same validation split is used for many design choices and checkpoint selections, so choosing the strongest observed alternative can favor noise. Repeated training seeds measure initialization and optimization variability on this fixed split; their standard deviation does not measure uncertainty across new datasets or train/validation splits.

## Validation error analysis

The analysis loads trusted local checkpoints, predicts validation reviews, and aligns them by the original review IDs. It never fits a model. These are independent commands; use a fresh output directory for each:

```bash
phase2/.venv/bin/python remaining_experiments/error_analysis.py --left lstm512 --right lr-mean --output remaining_experiments/results/errors-lstm-vs-lr
phase2/.venv/bin/python remaining_experiments/error_analysis.py --left lstm256 --right lstm512 --output remaining_experiments/results/errors-256-vs-512
phase2/.venv/bin/python remaining_experiments/error_analysis.py --left lstm512 --right A2 --output remaining_experiments/results/errors-frozen-vs-finetuned
```

`--left` and `--right` also accept a completed run directory; `--variance-summary path/to/variance.json` attaches a saved variability report. The first analysis computes and caches validation-only length, truncation, original-regex negation indicators, direct-vocabulary OOV proportions for the full review and first 256/512 tokens, and short review excerpts. It loads the original binary once for OOV membership and can require several GB of RAM; subsequent analyses reuse the separately checksummed feature cache. `per_example.csv`, `errors.csv`, `disagreements.csv`, `subgroups.csv`, `summary.json`, and graphs include labels, probabilities, confidence, correct/incorrect positive and negative counts, paired accuracy and macro-F1, training/validation gaps, and subgroup sizes. The summary compares recomputed checkpoint macro-F1 with the saved score and flags a mismatch. Confidence is `max(p, 1-p)`, not a calibration guarantee. Negation is a surface heuristic, OOV means direct absence, and an excerpt is evidence to inspect—not proof of the cause of an error. Single-true-class macro-F1 is omitted; groups under 30 reviews are flagged. Phase 2 LR pools full reviews while LSTMs truncate them, so differences do not isolate word order.

## Project 1 and locked final test

Compare saved validation artifacts without reading IMDb or training:

```bash
phase2/.venv/bin/python remaining_experiments/project1_comparison.py --output remaining_experiments/results/project1-comparison
```

`--project1-dir` selects a different results directory, and `--project2-dir` adds a completed run. Project 1's saved baseline configuration in this workspace lacks ordered-source and validation-ID fingerprints. The comparison will mark its split **unverified**, even if source, seed, and counts appear to match. Missing accuracy is reported as missing; no score or exact comparability is invented.

The separate final-test command evaluates **one manually selected saved model**, only after you have locked selection. It requires an explicit acknowledgement and a fresh output path:

```bash
phase2/.venv/bin/python remaining_experiments/final_test.py --run lstm512 --confirm-model-locked --output remaining_experiments/results/final-locked-lstm512
```

Do not execute it until selection is finished. It reconstructs and verifies the original split, evaluates 7,438 held-out reviews once per invocation, and saves predictions, metrics, and a confusion matrix. For unseen test tokens, frozen/fine-tuned pretrained runs use the original binary's subword vector without fine-tuning that unseen row. Random-initialized runs use a deterministic per-token seeded normal vector; that test-only unseen-token rule differs from the sequential RNG used for the cached vocabulary and is recorded. LR-Mean pools the entire review using the Phase 2 zero-vector policy. Only load trusted local `joblib` and PyTorch checkpoints. Test scores must not be used to choose or tune models.

## Synthetic verification

```bash
phase2/.venv/bin/python -m unittest discover -s remaining_experiments -p 'test_suite.py' -v
```

Tests use synthetic tensors, fake checkpoints, and mocked dispatch. They do not fit a model, run IMDb, load the real binary, or access final-test examples. `run.py --help`, `error_analysis.py --help`, `variance.py --help`, `project1_comparison.py --help`, and `final_test.py --help` list the options.
