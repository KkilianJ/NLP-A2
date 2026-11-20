# Phase 2: fastText embed-and-pool classification

This code defines four experiments: `lr-mean`, `lr-max`, `mlp-mean`, and `mlp-max`. It does not run until you start a command. It uses the existing Phase 1 dataset and `cc.en.300.bin`; it does not download or duplicate either file. No test-set evaluation is implemented.

## Setup

The Phase 1 Python 3.11 environment contains the data-analysis packages and fastText, but **PyTorch is not installed**. To keep Phase 1's environment separate, create a Phase 2 environment from `assignment 2`:

```bash
cd "assignment 2"
python3.11 -m venv phase2/.venv
uv pip install --python phase2/.venv/bin/python -r phase2/requirements.txt
```

If you prefer pip, replace the last line with:

```bash
phase2/.venv/bin/python -m pip install -r phase2/requirements.txt
```

`requirements.txt` references the Phase 1 lock file and adds PyTorch 2.6.0. The latter is the additional package needed for the MLP; `torchvision` and `torchaudio` are not used. The same Phase 1 package versions are retained in this separate environment. If the dependencies are already installed in another compatible Python environment, use that environment's Python executable instead. Installing packages may require network access; the PyTorch package is large.

## Training commands

Run these from `assignment 2` **after** setup. To train all four:

```bash
phase2/.venv/bin/python phase2/run.py --experiment all
```

To train only both logistic models, or only both MLP models:

```bash
phase2/.venv/bin/python phase2/run.py --experiment lr
phase2/.venv/bin/python phase2/run.py --experiment mlp
```

To train individual experiments:

```bash
phase2/.venv/bin/python phase2/run.py --experiment lr-mean
phase2/.venv/bin/python phase2/run.py --experiment lr-max
phase2/.venv/bin/python phase2/run.py --experiment mlp-mean
phase2/.venv/bin/python phase2/run.py --experiment mlp-max
```

Individual commands with the same seed and epoch limit contribute to the same comparison. An existing experiment directory is never overwritten: the `all` command and its individual commands above are alternatives for the same default result group. To start a separate independent run, add e.g. `--run-dir phase2/results/my-second-run`. If an interrupted experiment left an incomplete directory, choose a new run directory or remove only that incomplete directory yourself.

Other options:

```bash
# The model seed changes, but the Phase 1 data split remains fixed at seed 42.
phase2/.venv/bin/python phase2/run.py --experiment all --seed 17 --epochs 20

# Feature caches are reused by default; this explicitly requests reuse.
phase2/.venv/bin/python phase2/run.py --experiment lr --reuse-cache

# Recompute pooled features in memory without reading or writing the pooled cache.
phase2/.venv/bin/python phase2/run.py --experiment lr --no-reuse-cache --run-dir phase2/results/uncached

# Recreate charts and comparison tables from saved results, without loading data,
# loading fastText, extracting embeddings, or training a model.
phase2/.venv/bin/python phase2/run.py --plots-only --run-dir phase2/results/seed-42_epochs-30
```

`--epochs` accepts 1–30. `--model /existing/path/cc.en.300.bin` and `--data-dir /existing/path` select existing inputs; the Phase 1 hashes are checked. Use `--help` for all flags. Plot-only mode needs the saved metric files and plotting dependencies, but no PyTorch, dataset, binary model, or pooled cache.

## Exact methodology

- `data.py` calls the original Phase 1 split loader and fingerprint checks, then reproduces its validation partition. It verifies the training indices against Phase 1 and records both training and validation ID fingerprints. The old project pooled the two published source files before creating its own split; the project test partition remains held out. Source labels are read to reconstruct the existing stratification, but held-out examples are never passed to feature extraction, fitting, or evaluation.
- `phase1.tokenize(text, "regex")` is called directly for every review. It preserves capitalization and the original regex handling of `<br />` markup. No HTML removal, lowercasing, or alternate tokenizer is used.
- The original binary fastText model is loaded once when a pooled cache must be built. `get_word_vector` obtains stored or subword-based OOV vectors. The embeddings are frozen. All zero-vector tokens are omitted from both the mean and maximum. An empty review or one with no nonzero vectors maps to a 300-dimensional zero vector; non-finite vectors are rejected. There is no padding. Both pools are computed together and cached for both classifiers; the cache records the dataset, split, model, tokenizer, and zero-vector policy.
- Logistic regression uses a scikit-learn pipeline, fitting `StandardScaler` and `LogisticRegression(C=1.0, max_iter=1000, class_weight=None, random_state=seed)` on training data only. The saved `model.joblib` contains the fitted scaler and classifier.
- Each PyTorch MLP is exactly `Linear(300,64) -> ReLU -> Linear(64,1)`, with Adam, learning rate 0.001, batch size 64, BCE-with-logits loss, no dropout or weight decay, and at most 30 epochs. It consumes the **unscaled** pooled features. It uses MPS when available, otherwise CPU. Seeds are set for Python, NumPy, PyTorch, and data-loader shuffling; GPU/backend differences can still prevent bit-for-bit repeatability.
- Both classifier families use the same probability threshold, `>= 0.5`, and the same metric function. Confusion matrices have true labels as rows and predicted labels as columns, ordered negative then positive: `[[TN, FP], [FN, TP]]`. Macro-F1 averages both classes. Per-epoch MLP losses and accuracies are measured after the epoch using the current model in evaluation mode, with batches of 512 for evaluation only (the training batch size remains 64). The earliest epoch with the highest validation macro-F1 is the saved checkpoint, and its train and validation metrics appear in the final table.
- Logistic training time covers the scaler/classifier fit. MLP training time covers the epoch loop and its per-epoch evaluations. Neither includes embedding extraction; the definitions differ and should be considered when comparing times.

## Files created by a run

For default settings, results are in `phase2/results/seed-42_epochs-30/`:

```text
run_config.json                  data/model/tokenizer provenance and settings
comparison.csv / comparison.json completed experiment summary
graphs/                          overall accuracy, macro-F1, and mean-vs-max charts
lr-mean/                         metrics.csv/json, model.joblib, graphs/
lr-max/                          metrics.csv/json, model.joblib, graphs/
mlp-mean/                        metrics.csv/json, epochs.csv/json, best_model.pt, graphs/
mlp-max/                         metrics.csv/json, epochs.csv/json, best_model.pt, graphs/
```

Per-model charts include the requested confusion matrices; MLP folders also have loss, accuracy and macro-F1 by epoch, and logistic folders have validation score bars. Each MLP line graph marks and labels the best epoch; all epoch values are in `epochs.csv` and `epochs.json`. Comparison plots and tables include completed experiments only until all four have finished.

The shared pooled vectors are saved under `phase2/.cache/embeddings/`; the model fingerprint cache is also inside `phase2/.cache/`. Their identity includes the model SHA-256, tokenizer source and pattern, source fingerprint, both split fingerprints, and pooling policy, so a changed input cannot silently reuse another cache. The binary hash is cached by file path, size, and modification time; remove `phase2/.cache/model_fingerprint.json` to force rehashing. The first extraction needs several GB of memory and can take substantial time; progress is printed every 500 reviews. Later classifier runs reuse the approximately 100 MB pooled cache. The per-token in-memory lookup cache is bounded.

`best_model.pt` contains the best state dict and architecture metadata. `model.joblib` and `.pt` files should be loaded only from sources you trust. Model and result directories are local and ignored by Git, as are caches and the separate environment.
