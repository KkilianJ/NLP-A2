# Step 01 research report: tokenization and fastText vocabulary

**Status: implementation prepared; the new Step 01 experiment has not been run.** All Step 01 results and conclusions below are pending. Existing Phase 1 findings are explicitly identified and are not new Step 01 measurements.

## 1. Research question

How does tokenization affect stored-vocabulary coverage and the availability of subword-based OOV representations for pretrained English fastText on IMDb reviews?

## 2. Hypotheses

- Separating punctuation may increase direct vocabulary coverage relative to whitespace splitting, because attached punctuation can turn a familiar word into a different string. This does not imply better sentiment classification.
- Expanding selected negative contractions might replace some OOV forms with familiar auxiliaries and `not`, but it changes token and type denominators and could also reduce exact coverage for some strings or casing. The result must be measured.
- Some OOV tokens will yield nonzero fastText subword vectors, but the proportion and the quality of their meanings cannot be assumed.

## 3. Dataset and preprocessing protocol

Default: the established seed-42 project training partition of **34,707 reviews**. The pinned published source files were pooled before the Project 1 stratified split into 34,707 training, 7,437 validation, and 7,438 held-out test reviews. The old split loader reads pooled texts and labels to verify its content fingerprint and reads labels to reproduce stratification, but only the resulting training texts enter tokenization and measurement. The project split is not the official IMDb benchmark split.

The partner's 24,904-row published source-train population is **not** the project training partition and overlaps project validation/test reviews. An opt-in, explicitly acknowledged `published-train-descriptive` scope exists for descriptive reproduction only; it must never be used to select preprocessing or models. It has a separate output directory, metadata, and labeled graph.

| Variant | Processing |
|---|---|
| A — whitespace | Original Phase 1 whitespace split. |
| B — regex | Original Phase 1 regex, preserving case, contractions, and HTML-break markup tokens. |
| C — regex + negation | Same B tokens, with only the explicitly listed negative contractions expanded. |

The complete expansion map and casing policy are in [settings.py](settings.py) and [README.md](README.md). Examples include `can't` → `can not` and `won't` → `will not`; curly apostrophes are accepted. Ambiguous `ain't`, multiply contracted forms, and unusual mixed-case forms remain unchanged. The partner's source code was not available in this repository, so exact implementation parity cannot be asserted.

## 4. Experimental methodology

The original `cc.en.300.bin` is loaded once. For each selected review, the program applies all three variants and counts token strings, caching each distinct string's status across variants. `get_word_id` determines direct membership; `get_word_vector` is consulted for OOV strings to check that all components are finite and whether any component is nonzero. No model parameters are trained, and no semantic quality is inferred from the presence of a vector. Seed 42 selects real review excerpts for before/after examples without using labels. The full explicit expansion map, tokenizer source hash, data/model fingerprints, runtime, and software versions are recorded.

The published fastText documentation associates its English preprocessing with Europarl tools. The [saved Phase 1 reference-tool comparison](../results/full/europarl_reference.csv) showed that its regex preserves `don't` while the inspected reference separates `don` and `'t`; the exact historical preprocessing version was not established. The new expansion is a different convention again, not a reproduction of that pipeline.

## 5. Metrics and definitions

- **Token occurrences:** every token emitted by a variant; **types:** distinct case-sensitive token strings.
- **Direct token coverage:** directly stored token occurrences divided by all token occurrences. **Direct type coverage:** directly stored types divided by all types.
- **OOV occurrence proportion:** occurrences not stored directly divided by all occurrences. OOV type counts and proportions are also saved.
- **OOV nonzero-vector fraction:** OOV occurrences for which the binary returns at least one nonzero component divided by all OOV occurrences. Its type-weighted counterpart is also saved. If there are no OOV occurrences, the conditional fraction is undefined, not zero.

The prior Phase 1 definition used vector norm `> 1e-12` as nonzero; this step uses the literal component-wise nonzero definition. These agree for ordinary vectors but may differ for exceptionally tiny ones. Both use exact, case-sensitive stored-vocabulary membership. A nonzero OOV vector is not a direct match and does not guarantee meaningful representation.

## 6. Configuration and reproducibility

The experiment, fixed negation map, seed, and default paths are in [settings.py](settings.py); exact commands and output descriptions are in [README.md](README.md). The default command is:

```bash
phase2/.venv/bin/python step01/run.py
```

The default output `step01/results/project-train-seed42/` must not exist before execution. Data checksums and the original binary fingerprint are verified; the latter may reuse Phase 1's read-only fingerprint cache for an unchanged file or be forced with `--verify-model-hash`. Run metadata captures which method was used. The script does not update this report automatically.

## 7. Results

### Existing Phase 1 findings — already observed, not new Step 01 results

The earlier full project-training exploration measured whitespace direct token/type coverage of **91.14% / 31.11%**, and original regex coverage of **99.31% / 81.70%**. That earlier regex retained capitalization and HTML markup. It found 67,262 OOV regex token occurrences; its norm-threshold rule counted 65,162 as having nonzero OOV vectors. The saved [Phase 1 coverage table](../results/full/coverage.csv) and [Phase 1 report](../REPORT.md) contain the evidence. These figures are context only; the new three-way run has not been executed and its literal nonzero definition differs slightly.

### New three-way Step 01 comparison — pending

| Variant | Occurrences | Types | Direct token coverage | Direct type coverage | OOV occurrence count / proportion | Nonzero fraction among OOV occurrences |
|---|---:|---:|---:|---:|---:|---:|
| Whitespace | Pending | Pending | Pending | Pending | Pending | Pending |
| Regex | Pending | Pending | Pending | Pending | Pending | Pending |
| Regex + negation | Pending | Pending | Pending | Pending | Pending | Pending |

Real review examples, observed expansion frequencies, and the graph will be recorded after the user executes the experiment. No new outcome is asserted here.

## 8. Interpretation

Pending the user's execution and saved results. Compare both occurrence- and type-weighted rates and inspect the actual transformed excerpts; the two weighting schemes answer different questions. Because tokenization and expansion change the units being counted, a raw coverage increase does not by itself prove semantic or downstream classification improvement.

## 9. Limitations

The negation list is deliberately incomplete and does not resolve context-dependent forms. The public Europarl comparison is limited, and pretrained web data could overlap IMDb reviews. OOV-vector availability is not semantic validation. The published-source descriptive scope overlaps project holdouts and is unsuitable for model-development choices. The project's test set remains reserved for final evaluation. Any later classifier comparison must be explicitly authorized and evaluated without using the test data to choose preprocessing.

## 10. Conclusions and implications

**Pending.** Once the user runs the project-training experiment and provides the outputs, document the measured tradeoffs and the limitations before deciding on any later stage. Existing Phase 1–3 results remain unchanged.
