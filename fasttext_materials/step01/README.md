# Step 01: fastText tokenization and vocabulary coverage

This standalone experiment compares the original Phase 1 whitespace and regex tokenizers with one new regex-plus-negation variant. It does not modify the Phase 1–3 scripts, checkpoints, caches, reports, or results. The [research report](REPORT.md) separates previously observed Phase 1 findings from Step 01 results, which remain pending until you run the experiment and review them.

## What the experiment does

The default analyzes only the **34,707-review Project 1 training partition**, reconstructed with the existing Phase 1 loader and split seed 42. It counts every token, checks exact membership with `model.get_word_id(token) >= 0`, and uses `model.get_word_vector(token)` only for distinct OOV strings to determine whether a subword vector exists. One token-status cache is shared in memory across all three variants; the original 300-dimensional `.bin` is loaded only once. There is no classifier, fitting, test-set evaluation, or download.

The previous project pooled the two pinned `mteb/imdb` source files before splitting them. To reproduce that split, the existing loader reads source text and labels to verify the ordered dataset fingerprint and uses source labels for stratification; it returns only project training texts for this analysis. The project's validation and held-out test texts are not passed to tokenization, measurement, or example selection. These project partitions are not the official IMDb benchmark split.

The partner's workflow describes the pinned **24,904-row published source-train file**, which is a different population from the 34,707-review project training partition. The partner's implementation is not present in this repository, so this experiment reproduces the stated comparison with the explicit rules below; it does not claim byte-for-byte equivalence to their tokenizer.

## Tokenization and negation rules

| Variant | Definition |
|---|---|
| `whitespace` | Original Phase 1 Python whitespace splitting. |
| `regex` | Unchanged Phase 1 regex: separates punctuation, preserves case and internal straight/curly apostrophes, and retains HTML break-tag tokens. |
| `regex_negation` | Apply **the same regex**, then expand only listed complete negative contractions into an auxiliary and `not`. HTML and all other preprocessing remain unchanged. |

The full, fixed map is in [settings.py](settings.py). Examples include `don't` → `do not`, `isn't` → `is not`, `wasn't` → `was not`, `can't` → `can not`, and `won't` → `will not`. Straight and curly apostrophes are supported. Lowercase stays lowercase, initial-capital forms become e.g. `Don't` → `Do not`, and uppercase forms become e.g. `DON'T` → `DO NOT`. Unusual mixed-case forms are left unchanged. Ambiguous `ain't`, non-negative contractions such as `he'd`, multi-contractions such as `wouldn't've`, and forms not on the explicit list are left untouched. This is a conservative convention, not a general natural-language contraction resolver.

For each variant the output records token occurrences and types; direct occurrence- and type-weighted coverage; OOV occurrences and types and their proportions; and the fraction of OOV occurrences/types whose returned vector has finite components and at least one nonzero component. When there are no OOV occurrences, the conditional fraction is undefined and recorded as null/empty rather than zero. OOV means absent from the stored vocabulary **even if** fastText constructs a nonzero subword vector. A nonzero vector does not demonstrate semantic quality.

The original Phase 1 OOV calculation classified norms at or below `1e-12` as effectively zero. Step 01 tests whether any component is **exactly nonzero**, which follows this step's literal definition; exceptionally tiny vectors could therefore yield different OOV-vector counts. Direct membership uses the same definition in both phases. Tokenizers also have different denominators, so coverage alone cannot establish which is better for classification. The saved Phase 1 Europarl comparison is a small reference comparison, not proof of the exact historical fastText preprocessing pipeline.

## Run the project-training comparison

From the `assignment 2` directory, using the existing Phase 2 environment:

```bash
phase2/.venv/bin/python step01/run.py
```

This loads the original `models/cc.en.300.bin` once and can require several GB of memory. It uses the verified Phase 1 model fingerprint read-only when its file path, size, and modification time match; pass `--verify-model-hash` to force a fresh binary SHA-256. Both dataset files are checked against Phase 1's pinned hashes. The default output path is `step01/results/project-train-seed42/`. It must **not already exist**: this protects completed and interrupted runs. For a new repeat, choose a new output path:

```bash
phase2/.venv/bin/python step01/run.py --output step01/results/project-train-repeat
```

`--example-seed` controls reproducible example selection only; it does not change the project split. `--data-dir` and `--model` can point to existing copies of the pinned files. `--help` lists all options. The existing Phase 2 environment already includes the required fastText, NumPy, pandas, pyarrow, scikit-learn, and Matplotlib packages; no additional dependency is required if that environment is intact.

### Optional: published-source descriptive analysis

This is **not** the project training partition. The source file overlaps the project validation and test partitions because the published files were pooled before splitting. Its statistics and excerpts must not be used to select preprocessing, tune a model, or make other model-development decisions. It is available only as an explicitly acknowledged, separately labeled descriptive reproduction of the partner's source population:

```bash
phase2/.venv/bin/python step01/run.py --scope published-train-descriptive \
  --acknowledge-project-holdout-overlap
```

That command uses a distinct default directory, `step01/results/published-train-descriptive-seed42/`, reads the source `text` column only, checks the pinned file hash and 24,904-row count, and labels the output metadata and graph as descriptive. The ordinary project-training command above does not enable it. Do not combine the two populations or interpret them as equivalent splits.

## Saved outputs

- `coverage.csv` and `coverage.json`: all counts, direct coverage, OOV proportions, and conditional nonzero-vector fractions for A/B/C.
- `tokenization_examples.csv`: seeded review IDs, verbatim excerpts, matched text, and the actual token lists for all three variants. Categories are included only when a match exists.
- `negation_expansions.csv`: every actually expanded token string, its output, occurrence count, and number of affected reviews.
- `coverage.png`: direct token/type coverage and the percentage of OOV occurrences with nonzero vectors.
- `run_metadata.json`: scope and leakage warning where relevant, data and model fingerprints, tokenizer and experiment-code hashes, expansion map, vector definition, seed, timing, and environment.

Output directories and local Matplotlib cache are ignored by Git; preserve them separately if you share the repository. The command does **not** update [REPORT.md](REPORT.md) or the existing project reports. After you run it and share the results, the measured results, interpretation, and conclusions can be added to the research documentation.

## Synthetic checks

These unit tests use only artificial text, a mocked fastText model, and temporary files. They do not load the real model, access IMDb, or train a classifier:

```bash
phase2/.venv/bin/python -m unittest discover -s step01 -p 'test_step01.py' -v
```
