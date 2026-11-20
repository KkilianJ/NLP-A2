# Suggested sequential run order

Run commands from `assignment 2`. Each starts at most one explicit run; do not execute the final-test command until model selection is locked. Existing result paths are protected.

1. **A2:** `phase2/.venv/bin/python remaining_experiments/run.py --experiment A2` — fine-tune the pretrained lookup and inspect its saved validation metrics, learning curves, and memory use.
2. **A3:** `phase2/.venv/bin/python remaining_experiments/run.py --experiment A3` — compare random initialization with A2 and the existing A1 baseline.
3. **B1:** `phase2/.venv/bin/python remaining_experiments/run.py --experiment B1` — compare the fixed stratified half to B2, the full-data baseline.
4. **C1 and C3:** run `phase2/.venv/bin/python remaining_experiments/run.py --experiment C1` and `phase2/.venv/bin/python remaining_experiments/run.py --experiment C3` separately to compare hidden sizes with C2.
5. **D1 and D3:** run `phase2/.venv/bin/python remaining_experiments/run.py --experiment D1` and `phase2/.venv/bin/python remaining_experiments/run.py --experiment D3` separately to compare learning rates with D2.
6. **E2 and E3:** run `phase2/.venv/bin/python remaining_experiments/run.py --experiment E2` and `phase2/.venv/bin/python remaining_experiments/run.py --experiment E3` separately to compare dropout and weight decay with E1.
7. Pick a relevant alternative from the completed validation runs. Run `phase2/.venv/bin/python remaining_experiments/run.py --experiment F-base --seed 123` and the same command with `--seed 2026` separately. For example, if A2 is selected, run `phase2/.venv/bin/python remaining_experiments/run.py --experiment F-alternative --alternative A2 --seed 123`, then the same command with `--seed 2026`. Their seed-42 runs are reused. Optionally repeat F-mean. Summarize completed runs with `variance.py` as shown in the README.
8. Run the validation error comparisons individually. They require the corresponding saved models and may build a separate validation-feature cache on first use.
9. Compare saved Project 1 and Project 2 validation artifacts, taking the Project 1 fingerprint limitation into account.
10. Only after selecting and locking one model, use the explicitly acknowledged final-test script if you decide to perform final evaluation.

The full exact commands, outputs, limitations, cache behavior, and analysis options are in [README.md](README.md).
