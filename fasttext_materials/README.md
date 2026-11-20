# Assignment 2: fastText exploration, Phase 1

For the complete chronological record of **all phases and completed experiments**, including the latest LSTM-head comparison, see [PROJECT_LOG.md](PROJECT_LOG.md).

This folder contains a completed, training-only exploration of the original English fastText Common Crawl binary. Read [REPORT.md](REPORT.md) for the questions, actual results, interpretation, and limitations. The measured outputs are in [results/full](results/full/); [results/smoke](results/smoke/) is a much smaller check and is **not** the source of the report's figures.

## Data and split

Assignment 1's `project_starter/experiment.py` identifies the data as `mteb/imdb` revision `d05f0155defa7991dad75bc68c5ccb6774b1fdc5`. Its published train and test files were pooled and re-split using stratification and seed 42: **34,707 train / 7,437 validation / 7,438 held-out test**. That differs from the official IMDb split. The earlier checkout in this workspace did not contain its documented `.hf_cache`, so the pinned parquet files were restored here. The script checks each file's SHA-256 and Assignment 1's ordered text/label fingerprint before reproducing the old split.

All pooled source labels are used solely for the existing stratified split and the source fingerprint. After splitting, the experimental functions receive only the training texts and their pooled row IDs; neither validation nor held-out test reviews or labels are analyzed. A later classification phase should use the same split and preserve the test set until the final evaluation.

## Contents

- `config.py`: seed, model and dataset fingerprints, preprocessing/probe choices, and cache limits.
- `run_phase1.py`: one process loads the binary once and runs all three experiments.
- `compare_europarl.py`: optional comparison with the published Europarl tools on the saved real-review excerpts.
- `test_phase1.py`: checks tokenization, direct/OOV accounting, exact excerpts, edit operations, and the previous split.
- `requirements.txt`: pinned main dependencies; `requirements-lock.txt`: the complete versions installed for this run.
- `data/`, `models/`, `.cache/` and `.venv/`: local large inputs and caches; excluded from Git. The binary is stored once, not duplicated in each run.

## Run on this machine

From this folder:

```bash
.venv/bin/python -m unittest -v test_phase1
.venv/bin/python run_phase1.py --smoke --output results/my-smoke
.venv/bin/python run_phase1.py --output results/my-full-run
```

Each output directory must be new or empty, so existing results cannot be accidentally overwritten. The smoke run samples 200 training reviews and searches a 5,000-word neighbor pool. It still loads the full binary and therefore still requires several GB of memory. The full run searches the first 100,000 stored words; the approximately 115 MiB normalized neighbor matrix is cached in `.cache/` using the model fingerprint and pool size. Changing the pool size, seed, or probes is done in `config.py`. `--model /path/to/cc.en.300.bin` and `--data-dir /path/to/parquets` allow reuse of existing files without a second download. The original binary is checked against its SHA-256; a text `.vec` file is not a substitute because it cannot synthesize OOV vectors.

For a fresh environment, use Python 3.11 and the project's pip convention:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
```

The locked environment was generated on macOS ARM; other platforms may need compatible package builds. fastText may compile a small C++ extension. The model alone occupies about 6.7 GiB on disk; allow at least several additional GB of RAM for loading and analysis. The recorded run used a 48 GiB machine. Filenames and fingerprints appear in `results/full/run_metadata.json`. The model SHA-256 is cached after the first hash using file path, size, and modification time; remove `.cache/model_fingerprint.json` to force rehashing.

## Restore inputs if absent

These commands download the exact revision to `data/`. The two source files are both needed to reproduce the old *pooled* split; they are not the resulting project train and test partitions.

```bash
mkdir -p data models
curl --fail --location --retry 3 \
  'https://huggingface.co/datasets/mteb/imdb/resolve/d05f0155defa7991dad75bc68c5ccb6774b1fdc5/data/train-00000-of-00001.parquet' \
  -o data/train.parquet
curl --fail --location --retry 3 \
  'https://huggingface.co/datasets/mteb/imdb/resolve/d05f0155defa7991dad75bc68c5ccb6774b1fdc5/data/test-00000-of-00001.parquet' \
  -o data/test.parquet
```

Reuse an existing `cc.en.300.bin` via `--model` if available. Otherwise the following streams the **4.5 GB compressed** official download straight into one uncompressed binary; it does not keep a second compressed copy:

```bash
set -o pipefail
curl --fail --location --retry 3 \
  'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz' \
  | gzip -dc > models/cc.en.300.bin.part && \
  mv models/cc.en.300.bin.part models/cc.en.300.bin
```

Run the download only when the binary is absent. If interrupted, delete the incomplete `.part` before trying again. The script verifies the uncompressed binary and data fingerprints before analyzing reviews. The model is distributed under [CC BY-SA 3.0](https://fasttext.cc/docs/en/crawl-vectors.html).

## Optional Europarl comparison

The saved `europarl_reference.csv` was produced with the English tokenizer from the [published Europarl preprocessing archive](https://www.statmt.org/europarl/v7/tools.tgz). To reproduce it for a new run, Perl and network access are needed once:

```bash
mkdir -p .cache/europarl
curl --fail --location 'https://www.statmt.org/europarl/v7/tools.tgz' \
  | tar -xz -C .cache/europarl
.venv/bin/python compare_europarl.py \
  --tokenizer .cache/europarl/tools/tokenizer.perl \
  --examples results/my-full-run/tokenization_examples.csv \
  --output results/my-full-run/europarl_reference.csv
```

This compares short excerpts, not the complete corpus, and does not assert that the downloaded toolkit is the exact version used in fastText training.

## Output guide

`coverage.csv` gives exact vocabulary membership by token and by distinct type, plus the number of OOV tokens/types with zero or nonzero vectors. `capitalization_*.csv` isolates lowercasing changes. `tokenization_examples.csv` and `europarl_reference.csv` contain real, exact review substrings. `natural_oov_types.csv` is the complete OOV frequency table for the case-preserving regex with `<br>` removal, while `natural_oov_examples.csv` adds original contexts and **unverified** correction guesses. `spelling_pairs.csv` labels synthetic corruptions explicitly, including whether each happens to be in the pretrained vocabulary. `semantic_pairs.csv` contains every manual probe and its training frequency. Summary CSVs, `nearest_neighbors.csv`, and three PNG plots make the measurements easier to inspect.

Nearest neighbors are exact cosine neighbors **within the first 100,000 stored vocabulary words**, not within all two million. Candidate order follows the model's stored vocabulary order; the limited pool is a resource-conscious approximation and includes occasional web-like artifacts. No classification models or hyperparameter searches are implemented here.
