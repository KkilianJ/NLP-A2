# fastText materials and joint-project index

The joint narrative is [COMBINED_RESEARCH_REPORT.md](../COMBINED_RESEARCH_REPORT.md). This directory is a copy of the fastText work from `/Users/saevarbrekisnorrason/Desktop/nlp1-student/assignment 2`, preserving its original layout and local result files. The existing GloVe project files remain in the parent directory. Historical reports describe what was known when they were written; use the joint report for the combined test-set history and cross-track limitations.

## Where to look

| Material | Location | Contents and status |
|---|---|---|
| Consolidated fastText record | [PROJECT_LOG.md](PROJECT_LOG.md) | Chronology, methods, full histories, and the paired validation error analysis. |
| Phase 1 | [REPORT.md](REPORT.md), [README.md](README.md), [results/full](results/full/) | Tokenization, coverage, OOV, spelling, semantic probes, CSV tables and plots; `results/smoke/` contains the earlier small smoke output. |
| Phase 1 implementation | [run_phase1.py](run_phase1.py), [config.py](config.py), [test_phase1.py](test_phase1.py) | Runner, frozen choices, and synthetic checks. |
| Phase 2 | [phase2/REPORT.md](phase2/REPORT.md), [phase2/README.md](phase2/README.md), [phase2/results/seed-42_epochs-30](phase2/results/seed-42_epochs-30/) | Four pooling/classifier runs, comparison tables, LR `model.joblib` pipelines, MLP `best_model.pt` checkpoints, metrics, histories, configurations and graphs. |
| Phase 2 implementation | [phase2/run.py](phase2/run.py), [settings.py](phase2/settings.py), [training.py](phase2/training.py) | Classification, pooled feature preparation and training settings. |
| Phase 3 | [phase3/REPORT.md](phase3/REPORT.md), [phase3/README.md](phase3/README.md), [phase3/results](phase3/results/) | Completed 256-token and 512-token linear-head runs plus 512-token MLP-head run. Each folder contains a checkpoint, configuration, metrics, full history and graphs. The Phase 3 report predates the MLP follow-up; the project log includes it. |
| Phase 3 implementation | [phase3/run.py](phase3/run.py), [sequence_preparation.py](phase3/sequence_preparation.py), [sequence_model.py](phase3/sequence_model.py) | LSTM input, cache, model and training code. |
| Later experiments | [remaining_experiments/README.md](remaining_experiments/README.md), [RUN_ORDER.md](remaining_experiments/RUN_ORDER.md), [experiments.json](remaining_experiments/experiments.json), [results](remaining_experiments/results/) | Completed A2, A3, B1, C1, C3, D1, D3, E2 and E3 seed-42 runs; D3 and F-base seed-123 repeats. Each run folder includes saved metrics, configuration, history, graphs, checkpoint and training review IDs. |
| Newer validation error analysis | [summary.json](remaining_experiments/results/errors-lstm-vs-lr/summary.json), [subgroups.csv](remaining_experiments/results/errors-lstm-vs-lr/subgroups.csv), [paired predictions](remaining_experiments/results/errors-lstm-vs-lr/per_example.csv), [graphs](remaining_experiments/results/errors-lstm-vs-lr/) | Saved LSTM-versus-mean-pool analysis with matched validation IDs, disagreements, excerpts, subgroup tables and plots. This was saved after the version of the project log first read for this task; the original log was subsequently updated and recopied. |
| Prepared, not completed | [step01/REPORT.md](step01/REPORT.md), [step01/README.md](step01/README.md) | New fastText whitespace/regex/negation-expansion coverage study. No new Step 01 result was present in this snapshot. |
| Local integrity inventory | [SHA256SUMS.txt](SHA256SUMS.txt) | SHA-256 and relative path for each copied regular source file, excluding this index and the inventory itself. |

The result directories include files that were ignored by the original Git configuration: they are present on disk here even when `git status` does not list them individually. In particular, copied `phase2/.gitignore`, `phase3/.gitignore` and `remaining_experiments/.gitignore` ignore result directories and caches. **An ordinary Git commit will not automatically include all these artifacts.** If sharing through Git, inspect ignored files and deliberately force-add suitable artifacts, or share an archive; Git hosting may also impose file-size limits. Nothing has been committed or uploaded by this task.

## Code for completed Assignment 2 runs

The completed fastText runs used the following entry points and their local helpers, all already copied from the Assignment 2 folder:

- **Phase 1 exploration:** [run_phase1.py](run_phase1.py) and [config.py](config.py); [compare_europarl.py](compare_europarl.py) produced the saved reference-tokenizer comparison.
- **Phase 2 pooling classifiers:** [phase2/run.py](phase2/run.py), with `data.py`, `embeddings.py`, `metrics.py`, `phase1_imports.py`, `plots.py`, `settings.py` and `training.py` in the same folder, plus the Phase 1 code above.
- **Phase 3 LSTMs:** [phase3/run.py](phase3/run.py), with `baselines.py`, `experiment_settings.py`, `legacy.py`, `lstm_training.py`, `plots.py`, `sequence_dataset.py`, `sequence_model.py` and `sequence_preparation.py` in the same folder, plus the earlier-phase dependencies.
- **Completed follow-up LSTM runs and validation error analysis:** [remaining_experiments/run.py](remaining_experiments/run.py) and [error_analysis.py](remaining_experiments/error_analysis.py), with `bridge.py`, `data_access.py`, `inference.py`, `modeling.py`, `training.py`, `validation_features.py`, `workflow_config.py` and [experiments.json](remaining_experiments/experiments.json), plus the earlier-phase dependencies.

These files may also contain branches for other configurations. The preserved archive contains additional prepared code, including the unfinished Step 01 and final-test scripts; those scripts are not evidence of completed runs.

## Original model and dataset: references, not copies

Two symbolic links in this directory allow the copied code's usual relative paths to resolve to the original files on this computer:

- `data` → `/Users/saevarbrekisnorrason/Desktop/nlp1-student/assignment 2/data`, containing `train.parquet` and `test.parquet`.
- `models` → `/Users/saevarbrekisnorrason/Desktop/nlp1-student/assignment 2/models`, containing `cc.en.300.bin`.

The links do not duplicate those files and are **not write protection**: editing through them would edit the originals. They are machine-specific and will break if the original files move or the project is shared elsewhere. Repoint the links or use `--data-dir` and `--model` where the runner provides those options. Some later-experiment runners instead use the defaults in [workflow_config.py](remaining_experiments/workflow_config.py). The recorded binary SHA-256 is `14c7167b130056944cbdc37b7451f055867fe9a4e3fed3bbc1ecc0e74f6763ca`; the source Parquet SHA-256 values are recorded in [config.py](config.py). The published source filenames are inputs to the pooled project split, not the names of the final project partitions.

Virtual environments and OS/font caches were not copied. The original Phase 1 environment is `/Users/saevarbrekisnorrason/Desktop/nlp1-student/assignment 2/.venv`; the later experiments used `/Users/saevarbrekisnorrason/Desktop/nlp1-student/assignment 2/phase2/.venv`. Dependencies and saved versions are documented in [requirements-lock.txt](requirements-lock.txt), [phase2/requirements.txt](phase2/requirements.txt), and the individual run configurations. Recreating an environment may require package access and matching platform versions; hardware and software differences can change results.

## Copied analysis caches

The useful fastText-derived caches were copied, including ignored files:

- [.cache](.cache/) contains the saved 5,000- and 100,000-word neighbour indexes and model fingerprint metadata used by the embedding probes.
- [phase2/.cache/embeddings](phase2/.cache/embeddings/) contains the full-review mean and max feature arrays and their manifest.
- [phase3/.cache/sequences](phase3/.cache/sequences/) contains the 256- and 512-token sequence IDs, labels, offsets, vocabulary, derived embedding tables and manifests. These are derived feature caches, not copies of the source Parquet dataset or original fastText binary; some frozen-model checkpoints depend on the matching embedding table.
- [remaining_experiments/.cache/validation_features_v1](remaining_experiments/.cache/validation_features_v1/) contains the saved validation error-analysis features and identity metadata.

Original locations are the same relative paths under `/Users/saevarbrekisnorrason/Desktop/nlp1-student/assignment 2`. Saved run configurations may contain those **absolute original paths** even though cached files also exist in this copy. Cache identity checks, software versions and path-dependent fingerprint metadata may trigger validation or recomputation after a move. The caches can be regenerated from the original inputs, but doing so is new computation; this task did not regenerate them.

## Partner work and Project 1

The existing [GloVe README](../README.md) is the partner's original narrative. Its code is in the parent directory; [final_glove_study](../final_glove_study/), [embedding_output](../embedding_output/), [meanpool_output](../meanpool_output/), and the parent JSON/CSV/PNG files contain the saved evidence. The GloVe scripts refer to `glove.6B.zip` and Project 1's code. A search of the joint folder and Desktop to depth five did not find that GloVe archive; its location elsewhere is unknown. No GloVe model or dataset was downloaded or copied.

Project 1 remains at `/Users/saevarbrekisnorrason/Desktop/NLP1-Assignment1/project_starter`. The [historical scores](../../NLP1-Assignment1/project_starter/results/baseline/validation_scores.csv) and [summary](../../NLP1-Assignment1/project_starter/PROJECT_SUMMARY.md) are referenced in place. Its historical score file lacks its own saved ordered data and split fingerprints.

## Status and cautions

No fastText held-out test result, seed-2026 repeat, or new Step 01 result was found. The GloVe repository already contains held-out test results for three seed-42 systems; consult the joint report before treating that shared test partition as unused. Do not assume that a README's historical command is authorized to run now or safe to rerun into an existing output directory. Checkpoints should only be loaded from trusted sources.
