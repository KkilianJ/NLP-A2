"""Measure stored-vocabulary membership separately from OOV representability."""
import numpy as np


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def measure_coverage(model, counters, progress_every=10_000):
    """Return aggregate rows and cached (direct, nonzero_if_oov) token status."""
    vocabulary = sorted(set().union(*(set(counter) for counter in counters.values())))
    dimension = model.get_dimension()
    status = {}
    for i, token in enumerate(vocabulary, 1):
        direct = model.get_word_id(token) >= 0
        nonzero = None
        if not direct:
            vector = np.asarray(model.get_word_vector(token))
            if vector.shape != (dimension,) or not np.isfinite(vector).all():
                raise ValueError(f"Invalid fastText vector for token {token!r}")
            nonzero = bool(np.any(vector != 0))
        status[token] = (direct, nonzero)
        if progress_every and (i % progress_every == 0 or i == len(vocabulary)):
            print(f"  Checked {i:,}/{len(vocabulary):,} unique token strings", flush=True)

    rows = []
    for name, counter in counters.items():
        total = sum(counter.values())
        types = len(counter)
        direct_tokens = sum(n for token, n in counter.items() if status[token][0])
        direct_types = sum(status[token][0] for token in counter)
        oov_tokens = total - direct_tokens
        oov_types = types - direct_types
        nonzero_tokens = sum(n for token, n in counter.items() if status[token] == (False, True))
        nonzero_types = sum(status[token] == (False, True) for token in counter)
        rows.append({"variant": name, "token_occurrences": total, "token_types": types,
                     "direct_token_occurrences": direct_tokens, "direct_token_types": direct_types,
                     "direct_token_fraction": ratio(direct_tokens, total),
                     "direct_type_fraction": ratio(direct_types, types),
                     "oov_token_occurrences": oov_tokens, "oov_token_types": oov_types,
                     "oov_token_fraction": ratio(oov_tokens, total), "oov_type_fraction": ratio(oov_types, types),
                     "oov_nonzero_occurrences": nonzero_tokens, "oov_nonzero_types": nonzero_types,
                     "oov_zero_occurrences": oov_tokens - nonzero_tokens,
                     "oov_zero_types": oov_types - nonzero_types,
                     "oov_nonzero_occurrence_fraction": ratio(nonzero_tokens, oov_tokens),
                     "oov_nonzero_type_fraction": ratio(nonzero_types, oov_types)})
    return rows, status
