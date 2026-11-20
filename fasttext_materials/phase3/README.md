# Phase 3: frozen fastText + LSTM sentiment classification

This directory implements the unidirectional LSTM baseline and a controlled MLP-head comparison. It does **not** train or extract IMDb sequences until you run it. It reuses Phase 1's original case-preserving `regex` tokenizer, the original `cc.en.300.bin` binary, and the project split verified by the Phase 2 loader. It does not evaluate the held-out test partition or alter the Phase 1/2 files. The user has completed 256- and 512-token runs; their measured results and limitations are documented in the [Phase 3 report](REPORT.md).

## Why an LSTM?

Mean and max pooling compress the entire review into one vector without retaining token order. A unidirectional LSTM reads the token vectors in sequence and can learn a state that depends on earlier words. Its last **valid** hidden output summarizes the retained prefix and feeds a binary classification head. This permits sequential relationships to influence a prediction, although a single finite-sized state and the configured prefix length still limit what can be represented.

The original linear-head architecture is exactly:

```text
Phase 1 regex tokens -> frozen fastText (300) -> LSTM (300 -> 64)
                     -> last valid hidden output -> Linear(64, 1) -> logit
```

The new experiment replaces only the final linear classifier with
`Linear(64, 64) -> ReLU -> Linear(64, 1)`. It uses the same 64-dimensional LSTM
state and trains the LSTM and head jointly; fastText vectors remain frozen.

| Setting | Default |
|---|---|
| LSTM layers / direction | 1 / unidirectional |
| Embedding / hidden dimensions | 300 / 64 |
| Maximum sequence length | 256 tokens (linear); 512 tokens (MLP) |
| Batch size | 32 |
| Optimizer / learning rate | Adam / 0.001 |
| Loss | `BCEWithLogitsLoss` |
| Epochs / early-stopping patience | Up to 30 / 5 |
| Dropout / weight decay | 0 / 0 |
| Training seed / split seed | 42 / 42 |
| Probability threshold | 0.5 |

The embedding lookup is frozen; only the LSTM and chosen classification head are optimized. Sigmoid is used for evaluation/inference only; the loss receives raw logits. MPS is used when PyTorch reports it available, otherwise CPU. Seeds are set for Python, NumPy, PyTorch, and data-loader shuffling, although hardware/backend differences can prevent bit-for-bit repeatability.

## Order, padding, truncation, and OOV words

- The code calls the actual Phase 1 `tokenize(text, "regex")` function. Capitalization, punctuation handling, and HTML break tags are preserved exactly as in Phase 2. There is no lowercasing, break-tag removal, or alternate tokenizer.
- Each review is tokenized in full for length accounting, then **the first N tokens are retained**, where N is 256 by default. Remaining tokens at the end are discarded. After preparation, the terminal and `sequence_statistics.csv/json` report, separately for train and validation, how many reviews exceed the limit and the number and percentage of original tokens discarded. Percent discarded refers to **tokens**, not characters. The completed 256- and 512-token runs are documented in the [Phase 3 report](REPORT.md); each new run records its own statistics.
- Reviews are stored as compact, ordered integer IDs and offsets, not as a pre-padded `(reviews, 256, 300)` tensor. Each batch is padded only to its longest retained review. A frozen lookup table turns IDs into exact 300-dimensional fastText vectors.
- For the forward-only LSTM, the code selects the output at `length - 1` for each review, not `h_n` after right padding. Later right-padding cannot influence an earlier forward LSTM output. This is a correct masking approach for the specified **unidirectional** architecture and avoids relying on packed-sequence support on MPS. An empty review uses a zero hidden state, which the selected classification head converts to a logit; for the linear head, that logit is its bias.
- ID 0 is reserved for padding only. Every distinct retained token gets its own ID and `get_word_vector(token)` from the original binary. This includes OOV tokens with fastText subword vectors; there is no catch-all UNK substitution. A real token whose fastText vector is zero retains its own sequence position and ID. This differs from Phase 2, where zero-vector tokens were omitted from pooling.
- The lookup is assembled by traversing training tokens first, followed by validation tokens. Adding validation-only tokens merely caches their pretrained vectors: it does not fit a vocabulary cutoff, scaler, normalization statistic, or model parameter on validation data, and it does not change training token IDs. Validation labels are used only for evaluation and checkpoint selection.

The project uses the existing 34,707 / 7,437 / 7,438 train/validation/test split, reconstructed with split seed 42. As in Assignment 1, the published source files were pooled before that split, so it is not the official IMDb benchmark split. Source labels are read to reproduce the old stratification; held-out test examples never enter sequence preparation, fitting, or evaluation.

## Setup

The existing Phase 2 Python 3.11 environment already contains the required fastText, PyTorch, NumPy, pandas, scikit-learn, and plotting packages. No additional model or package download is required in this workspace. From the repository root:

```bash
cd "assignment 2"
phase2/.venv/bin/python phase3/run.py --help
```

If that environment is absent, create a separate environment and install the pinned requirements:

```bash
python3.11 -m venv phase3/.venv
uv pip install --python phase3/.venv/bin/python -r phase3/requirements.txt
```

In the commands below, replace `phase2/.venv/bin/python` with `phase3/.venv/bin/python` if using the new environment. The binary occupies about 6.7 GiB on disk and needs several GB of memory when constructing the lookup. It is not downloaded or modified by Phase 3.

## Commands

Run these from the `assignment 2` directory. The default full experiment is:

```bash
phase2/.venv/bin/python phase3/run.py
```

The controlled 512-token MLP-head experiment is started with:

```bash
phase2/.venv/bin/python phase3/run.py --head mlp --head-hidden-size 64
```

For `--head mlp`, the default sequence length is **512**; the existing linear default
remains **256**. The full MLP run fixes seed 42, LSTM hidden size 64, head hidden
size 64, batch size 32, learning rate 0.001, and maximum 30 epochs to match the
previous 512-token linear run. Conflicting values are rejected rather than
silently presented as a head-only comparison. The original linear head still
supports the existing CLI settings. To repeat the 512-token linear configuration
without overwriting the existing run:

```bash
phase2/.venv/bin/python phase3/run.py --head linear --max-sequence-length 512 \
  --run-dir phase3/results/lstm-linear-512-repeat
```

The MLP run verifies the existing linear baseline's split, tokenizer, binary,
training settings, and cached sequence/vector checksums. It reuses the existing
512-token cache when valid. The baseline's scores and best epoch are snapshotted
from its saved results; no MLP score exists until you run training.

A small smoke run selects 64 training and 32 validation reviews from the same project partitions. It still reads the pinned source files to reconstruct the split and, when its matching cache is absent, loads the full binary model, but only tokenizes and embeds the selected reviews. This command runs two epochs:

```bash
phase2/.venv/bin/python phase3/run.py --smoke --epochs 2
```

Smoke scores are for a tiny subset and must not be compared as if they came from the full validation set; its comparison chart is labeled accordingly. The smoke command has not been run as part of implementation.

To smoke-test the MLP head, use:

```bash
phase2/.venv/bin/python phase3/run.py --head mlp --smoke --epochs 2
```

To change the requested parameters for a new linear-head run:

```bash
phase2/.venv/bin/python phase3/run.py \
  --epochs 20 --learning-rate 0.0005 --batch-size 16 \
  --hidden-size 96 --max-sequence-length 192 --seed 17
```

`--epochs` accepts 1–30. The training seed changes initialization and batch order; the project split remains fixed at seed 42. `--model` and `--data-dir` point to existing copies of the original inputs, and `--baseline-csv` can select another Phase 2 comparison with matching metadata. `--run-dir` chooses a new output directory. Existing nonempty result directories are never overwritten; if a run is interrupted, choose a new directory or remove only your incomplete run.

To regenerate plots and the comparison table after the default run, without training, loading fastText, or reading the dataset:

```bash
phase2/.venv/bin/python phase3/run.py --plots-only \
  --run-dir phase3/results/lstm-s42-h64-l256-b32-lr0.001-e30
```

To regenerate the MLP-head plots after that run, without retraining:

```bash
phase2/.venv/bin/python phase3/run.py --head mlp --plots-only
```

Use `phase3/run.py --help` for all options.

## Checkpoint selection, evaluation, and comparison

Every epoch records training and validation loss, accuracy, and macro-F1. Training metrics are measured after the epoch in evaluation mode. A strictly higher validation macro-F1 replaces the best checkpoint; ties keep the earlier epoch. Training stops after **five consecutive epochs without a strict improvement**, or at the configured maximum. At completion the saved best weights are explicitly restored, then training and validation metrics and the validation confusion matrix are recalculated. Learning curves display all recorded epochs and mark the selected checkpoint; final metrics, the confusion matrix, and the Phase 2 comparison use the restored best checkpoint.

For an MLP-head run, `graphs/head_comparison.png` compares the restored best MLP
checkpoint against the saved 512-token linear-head checkpoint, showing validation
accuracy, macro-F1, and best epochs. `head_comparison.csv/json` also records both
training scores and train-minus-validation gaps. A smoke subset is marked as not
comparable to the full baseline. Validation has already been used to select the
512-token baseline and both checkpoints; a one-seed difference on this split is
exploratory, not a final test result or a significance claim.

The default Phase 2 baselines are loaded from the existing `phase2/results/seed-42_epochs-30/comparison.csv`. The associated Phase 2 metadata must match the original dataset, train/validation ID fingerprints, embedding binary, exact regex implementation, and 0.5 threshold. Their values are copied into a `baseline_snapshot.json` in each run, so plot regeneration does not need to reread Phase 2 results. The full comparison chart shows accuracy and macro-F1 for all four baselines and the LSTM.

Interpret that comparison cautiously: Phase 2 pools the **entire** review and drops zero-vector tokens, while this LSTM retains at most the configured prefix and keeps zero-vector tokens as positions. Phase 2 LR uses training-fitted standardization; Phase 2 MLP and this LSTM use their respective unscaled representations. Model architectures, checkpoint selection, and training procedures also differ. Thus a validation difference alone would not isolate the causal effect of word order. Validation has already been used for earlier comparisons and is also used for LSTM checkpoint selection; it is not a final test set.

## Outputs and caches

The default linear output directory remains `phase3/results/lstm-s42-h64-l256-b32-lr0.001-e30/`.
The MLP head uses the separate directory `phase3/results/lstm-mlp64-s42-h64-l512-b32-lr0.001-e30/`,
so it cannot overwrite either previous linear run:

```text
run_config.json                 split/model/cache identity, hyperparameters, environment
sequence_statistics.csv/json    review lengths, truncation, and discarded-token counts
baseline_snapshot.json          four saved Phase 2 validation baselines
linear_baseline_snapshot.json   saved 512-token linear reference (MLP runs)
head_comparison.csv/json        linear vs. MLP scores and gaps (MLP runs)
history.csv/json                train/validation loss, accuracy, and macro-F1 by epoch
metrics.csv/json                metrics recomputed from the best checkpoint
best_model.pt                   best trainable weights, frozen table, vocabulary, metadata
comparison.csv/json             best LSTM checkpoint and Phase 2 validation scores
graphs/                         curves, confusion, Phase 2 comparison, and head comparison PNGs
```

The full model binary is loaded once to build a compact cache under `phase3/.cache/sequences/`; subsequent matching runs verify and reuse it. Its key includes the source and split fingerprints, exact tokenizer implementation, model SHA-256, maximum length, smoke selection, and package versions. Cached sequence files, token mapping, and frozen vectors are individually checksummed. A model fingerprint is cached using file path, size, and modification time; remove `phase3/.cache/model_fingerprint.json` to force rehashing the original binary. The cache includes validation-only token vectors, but no held-out test reviews or labels.

A **completed** `best_model.pt` contains the selected LSTM/classifier weights plus the frozen token-vector table and mapping used for the run, so the original large binary is not copied into it. During an interrupted run, the interim checkpoint contains only the best trainable weights and references the cache. Both `phase3/results/` and `phase3/.cache/` are ignored by Git; preserve the artifacts separately when sharing. To handle genuinely new tokens not in a saved lookup at a later evaluation, obtain their vectors from the original binary rather than mapping them to UNK.

For a completed checkpoint, the architecture and trainable weights can be reconstructed without training:

```python
import sys
import torch
sys.path.insert(0, "phase3")
from sequence_model import model_from_checkpoint

checkpoint = torch.load("phase3/results/your-run/best_model.pt", map_location="cpu", weights_only=True)
model = model_from_checkpoint(checkpoint)  # old linear checkpoints and new heads
model.eval()
# checkpoint["tokens"] maps IDs to the exact token strings used in this run.
```

Only load checkpoints from sources you trust. The code does not implement a held-out test evaluation or trainable embeddings.

## Small code checks

The following unit tests use artificial sequences and a mocked fastText object; when the previous 512-token results are present, they also inspect their saved metadata. They do not read the real IMDb dataset, load the real pretrained model, or train a classifier:

```bash
phase2/.venv/bin/python -m unittest discover -s phase3 -p 'test_phase3.py' -v
```

They check sequence order and truncation, exact OOV mapping, zero vectors, dynamic padding, output shape, padding invariance, empty inputs, strict early-stopping ties, checkpoint compatibility, joint LSTM/head gradients, command-line safeguards, and regeneration of charts from temporary synthetic data. The unit suite ran on CPU without an optimizer or training. Before the MLP head was added, separate tiny synthetic checks of the linear model also passed on MPS; the user subsequently ran the two full MPS experiments documented in the [Phase 3 report](REPORT.md).
