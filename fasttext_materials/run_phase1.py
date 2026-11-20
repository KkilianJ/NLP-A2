#!/usr/bin/env python3
"""Explore the pinned IMDb training split with the original fastText binary.

No classifier is trained and no held-out reviews are used for analysis. See README.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import string
import sys
from collections import Counter
from pathlib import Path

# Keep matplotlib's writable cache inside the assignment directory.
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config as cfg

WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*|[^\w\s]|_", re.UNICODE)
BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
ALPHA_RE = re.compile(r"^[a-z]{4,20}$")
REPEATED_RE = re.compile(r"([A-Za-z])\1{2,}", re.IGNORECASE)
TOKENIZERS = ("whitespace", "regex", "regex_br", "regex_br_lower")


def tokenize(text: str, variant: str) -> list[str]:
    if variant == "whitespace":
        return text.split()
    if variant not in TOKENIZERS:
        raise ValueError(f"Unknown tokenizer: {variant}")
    if variant in ("regex_br", "regex_br_lower"):
        text = BREAK_RE.sub(" ", text)
    words = WORD_RE.findall(text)
    return [w.lower() for w in words] if variant == "regex_br_lower" else words


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_sha256(rows) -> str:
    digest = hashlib.sha256()
    for row in rows:
        for value in row:
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def load_training_partition(data_dir: Path, smoke: bool):
    """Recreate assignment 1's split. Labels are used only to stratify it.

    The old project pooled both *published* splits before defining its own held-out
    split. Thus both source files are required to reconstruct that exact split.
    No held-out text or labels are returned to the experimental functions.
    """
    paths = [data_dir / name for name in ("train.parquet", "test.parquet")]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned IMDb source file: {path}; see README.md")
        actual = file_sha256(path)
        if actual != cfg.SOURCE_SHA256[path.name]:
            raise ValueError(f"Unexpected SHA-256 for {path}: {actual}. Refusing to change the split.")
    frames = [pd.read_parquet(path, columns=["text", "label_text"]) for path in paths]
    sizes = [len(frame) for frame in frames]
    df = pd.concat(frames, ignore_index=True)
    fingerprint = ordered_sha256(df.itertuples(index=False, name=None))
    if fingerprint != cfg.ORDERED_DATA_SHA256:
        raise ValueError(f"Dataset content/order differs from assignment 1: {fingerprint}")
    indices = np.arange(len(df))
    training, other = train_test_split(indices, test_size=0.30, stratify=df.label_text,
                                       random_state=cfg.SEED)
    validation, test = train_test_split(other, test_size=0.5,
                                        stratify=df.label_text.iloc[other], random_state=cfg.SEED)
    rng = np.random.default_rng(cfg.SEED)
    training, validation, test = (rng.permutation(x) for x in (training, validation, test))
    split_sizes = {"training": len(training), "validation": len(validation), "test": len(test)}
    # Hash IDs, not held-out labels, to make a later comparison of partitions possible.
    training_ids_sha256 = hashlib.sha256(training.astype("<i8").tobytes()).hexdigest()
    if smoke:
        sample = np.random.default_rng(cfg.SEED).choice(len(training),
                    size=min(cfg.SMOKE_REVIEWS, len(training)), replace=False)
        training = training[sample]
    records = [(int(i), df.at[int(i), "text"]) for i in training]
    selected_sha = ordered_sha256(records)
    del df, frames, validation, test, other
    return records, {"published_source_rows": dict(zip(("train", "test"), sizes)),
                     "project_split_sizes": split_sizes,
                     "full_training_indices_sha256_le_int64": training_ids_sha256,
                     "selected_training_rows": len(records),
                     "selected_review_ids_and_text_sha256": selected_sha,
                     "ordered_source_text_label_sha256": fingerprint}


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else float("nan")


def write_csv(path, rows, columns=None):
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return frame


def token_counts(records):
    counters = {name: Counter() for name in TOKENIZERS}
    for _, text in records:
        # Keep this explicit so the preprocessing decisions remain inspectable.
        for name in TOKENIZERS:
            counters[name].update(tokenize(text, name))
    return counters


def membership_and_coverage(model, counters):
    # Cache membership and OOV vector norms in memory across tokenizers.
    status = {}
    rows = []
    for name, counter in counters.items():
        direct_types = direct_tokens = nonzero_types = nonzero_tokens = 0
        for token, frequency in counter.items():
            if token not in status:
                word_id = model.get_word_id(token)
                # OOV vectors can in principle be zero; measure rather than assume.
                norm = float(np.linalg.norm(model.get_word_vector(token))) if word_id < 0 else None
                status[token] = (word_id >= 0, norm)
            direct, norm = status[token]
            if direct:
                direct_types += 1
                direct_tokens += frequency
            elif norm is not None and norm > 1e-12:
                nonzero_types += 1
                nonzero_tokens += frequency
        types, tokens = len(counter), sum(counter.values())
        rows.append({"tokenizer": name, "tokens": tokens, "types": types,
                     "direct_tokens": direct_tokens, "direct_types": direct_types,
                     "direct_token_fraction": direct_tokens / tokens,
                     "direct_type_fraction": direct_types / types,
                     "oov_tokens": tokens - direct_tokens, "oov_types": types - direct_types,
                     "oov_nonzero_tokens": nonzero_tokens, "oov_nonzero_types": nonzero_types,
                     "oov_zero_tokens": tokens - direct_tokens - nonzero_tokens,
                     "oov_zero_types": types - direct_types - nonzero_types})
        print(f"  {name}: {tokens:,} tokens, {types:,} types, "
              f"{100 * direct_tokens / tokens:.2f}% token / {100 * direct_types / types:.2f}% type direct", flush=True)
    return rows, status


def capitalization_transitions(counter, status):
    """Measure the isolated case-folding effect on tokens that actually change."""
    summary = Counter()
    examples = []
    for token, frequency in counter.items():
        lowered = token.lower()
        if token == lowered:
            continue
        was_direct, now_direct = status[token][0], status[lowered][0]
        outcome = ("direct" if was_direct else "oov") + " -> " + ("direct" if now_direct else "oov")
        summary[(outcome, "tokens")] += frequency
        summary[(outcome, "types")] += 1
        if was_direct != now_direct:
            examples.append({"original": token, "lowercase": lowered, "frequency": frequency,
                             "original_direct": was_direct, "lowercase_direct": now_direct,
                             "transition": outcome})
    rows = [{"transition": outcome, "tokens": summary[(outcome, "tokens")],
             "original_types": summary[(outcome, "types")]}
            for outcome in ("direct -> direct", "direct -> oov", "oov -> direct", "oov -> oov")]
    examples.sort(key=lambda row: (-row["frequency"], row["original"]))
    return rows, examples


def excerpt(text, start, end, radius=55):
    """Return an exact substring with boundaries expanded to whole whitespace tokens."""
    a, b = max(0, start - radius), min(len(text), end + radius)
    while a > 0 and not text[a - 1].isspace():
        a -= 1
    while b < len(text) and not text[b].isspace():
        b += 1
    return text[a:b]


def tokenization_examples(records):
    patterns = {
        "punctuation": re.compile(r"\w[,!?;:]\s"),
        "contraction": re.compile(r"\b[A-Za-z]+['’](?:t|s|re|ve|ll|d|m)\b", re.IGNORECASE),
        "capitalization": re.compile(r"\b[A-Z]{3,}\b"),
        "html_break": BREAK_RE,
        "hyphenated_expression": re.compile(r"\b[A-Za-z]+-[A-Za-z]+\b"),
    }
    order = list(records)
    random.Random(cfg.SEED).shuffle(order)
    rows = []
    used_reviews = set()
    for category, pattern in patterns.items():
        for review_id, text in order:
            if review_id in used_reviews:
                continue
            match = pattern.search(text)
            if match:
                used_reviews.add(review_id)
                sample = excerpt(text, match.start(), match.end(), radius=38)
                row = {"category": category, "review_id": review_id, "excerpt": sample}
                row.update({name: json.dumps(tokenize(sample, name), ensure_ascii=False) for name in TOKENIZERS})
                rows.append(row)
                break
    return rows


def edit_one(word):
    """All single insertion, deletion, substitution, or adjacent transposition edits."""
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    candidates = {a + b[1:] for a, b in splits if b}
    candidates.update(a + c + b for a, b in splits for c in string.ascii_lowercase)
    candidates.update(a + c + b[1:] for a, b in splits if b for c in string.ascii_lowercase)
    candidates.update(a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1)
    candidates.discard(word)
    return candidates


def spelling_suggestions(counter, lower_counts, status):
    # Only frequent in-domain, directly stored lowercase words are candidates.
    eligible = {w: n for w, n in lower_counts.items()
                if n >= cfg.MIN_CORRECTION_REVIEW_FREQUENCY and ALPHA_RE.fullmatch(w)
                and status[w][0]}
    candidates = [(w, n) for w, n in counter.most_common()
                  if ALPHA_RE.fullmatch(w) and not status[w][0]][:cfg.CORRECTION_OOV_LIMIT]
    suggestions = {}
    for word, _ in candidates:
        matches = sorted((w for w in edit_one(word) if w in eligible), key=lambda w: (-eligible[w], w))
        if matches:
            suggestions[word] = matches[:3]
    return suggestions, eligible


def natural_oov_tables(records, counter, lower_counts, status):
    suggestions, eligible = spelling_suggestions(counter, lower_counts, status)
    oov = sorted(((w, n) for w, n in counter.items() if not status[w][0]), key=lambda x: (-x[1], x[0]))
    rows = [{"token": w, "frequency": n, "oov_vector_norm": status[w][1],
             "suggestions_unverified": " | ".join(suggestions.get(w, [])),
             "suggestion_training_frequencies": " | ".join(str(eligible[x]) for x in suggestions.get(w, []))}
            for w, n in oov]
    # The entire OOV table is saved; the category examples are selected heuristically.
    category_predicates = {
        "possible_spelling_error": lambda w: w in suggestions,
        "repeated_character": lambda w: bool(REPEATED_RE.search(w)),
        "contraction": lambda w: "'" in w or "’" in w,
        "capitalized_name_or_title_candidate": lambda w: len(w) > 2 and w[0].isupper() and w.isalpha(),
        "other_alphabetic": lambda w: w.isalpha() and len(w) >= 7,
    }
    selected = {}
    for category, predicate in category_predicates.items():
        for word, _ in [item for item in oov if predicate(item[0])][:cfg.CONTEXT_EXAMPLES_PER_CATEGORY]:
            selected.setdefault(word, []).append(category)
    contexts = {}
    for review_id, text in records:
        # Use the same token definition and break handling; find exact spans in the original text.
        for match in WORD_RE.finditer(text):
            word = match.group()
            if word in selected and word not in contexts:
                contexts[word] = (review_id, excerpt(text, match.start(), match.end(), 65))
        if len(contexts) == len(selected):
            break
    examples = []
    for word, categories in selected.items():
        if word not in contexts:
            continue
        review_id, context = contexts[word]
        examples.append({"token": word, "frequency": counter[word], "heuristic_categories": " | ".join(categories),
                         "review_id": review_id, "excerpt": context,
                         "suggestions_unverified": " | ".join(suggestions.get(word, [])),
                         "suggestion_training_frequencies": " | ".join(str(eligible[x]) for x in suggestions.get(word, [])),
                         "note": "Original IMDb token; candidate corrections are unverified guesses."})
    return rows, examples


def typo_candidates(word, kind):
    positions = list(range(1, len(word) - 1))
    if kind == "deletion":
        values = [word[:i] + word[i + 1:] for i in positions]
    elif kind == "insertion":
        values = [word[:i] + c + word[i:] for i in positions for c in "xae"]
    elif kind == "substitution":
        values = [word[:i] + c + word[i + 1:] for i in positions for c in "xzae" if c != word[i]]
    elif kind == "transposition":
        values = [word[:i] + word[i + 1] + word[i] + word[i + 2:]
                  for i in positions if word[i] != word[i + 1]]
    elif kind == "repetition":
        values = [word[:i] + word[i] + word[i:] for i in positions]
    else:
        raise ValueError(kind)
    return list(dict.fromkeys(v for v in values if v != word))


def controlled_spelling(model, lower_counts):
    rng = random.Random(cfg.SEED)
    words = [w for w in cfg.SPELLING_WORDS if lower_counts[w] and model.get_word_id(w) >= 0 and len(w) >= 4]
    controls = [w for w in cfg.SPELLING_CONTROLS if lower_counts[w] and model.get_word_id(w) >= 0]
    if not controls:
        raise ValueError("No spelling controls occur in the selected training reviews; use a larger sample.")
    rows = []
    for word in words:
        orig = model.get_word_vector(word)
        for kind in ("deletion", "insertion", "substitution", "transposition", "repetition"):
            candidates = typo_candidates(word, kind)
            rng.shuffle(candidates)
            if not candidates:
                continue
            # Prefer an OOV perturbation; keep and flag an in-vocabulary one if necessary.
            corrupted = next((w for w in candidates if model.get_word_id(w) < 0), candidates[0])
            rows.append({"kind": kind, "original": word, "comparison": corrupted,
                         "original_training_frequency": lower_counts[word],
                         "comparison_is_direct_vocabulary": model.get_word_id(corrupted) >= 0,
                         "comparison_training_frequency": lower_counts[corrupted],
                         "cosine": cosine(orig, model.get_word_vector(corrupted)),
                         "provenance": "synthetic corruption"})
        control = rng.choice([c for c in controls if c != word])
        for kind, comp in (("identity control", word), ("unrelated control", control)):
            rows.append({"kind": kind, "original": word, "comparison": comp,
                         "original_training_frequency": lower_counts[word],
                         "comparison_is_direct_vocabulary": model.get_word_id(comp) >= 0,
                         "comparison_training_frequency": lower_counts[comp],
                         "cosine": cosine(orig, model.get_word_vector(comp)), "provenance": "control"})
    return rows


def semantic_probes(model, lower_counts):
    rows = []
    for category, pairs in cfg.SEMANTIC_PAIRS.items():
        for word1, word2 in pairs:
            present = bool(lower_counts[word1] and lower_counts[word2])
            rows.append({"category": category, "word1": word1, "word2": word2,
                         "training_frequency1": lower_counts[word1], "training_frequency2": lower_counts[word2],
                         "word1_direct": model.get_word_id(word1) >= 0, "word2_direct": model.get_word_id(word2) >= 0,
                         "both_in_selected_reviews": present,
                         "cosine": cosine(model.get_word_vector(word1), model.get_word_vector(word2)) if present else None})
    return rows


def group_summary(rows, group):
    df = pd.DataFrame(rows)
    return (df.groupby(group, sort=False, dropna=False)["cosine"]
            .agg(n="count", mean="mean", median="median", std="std", minimum="min", maximum="max")
            .reset_index().to_dict("records"))


def model_digest(model_path):
    cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = cfg.CACHE_DIR / "model_fingerprint.json"
    st = model_path.stat()
    info = {"path": str(model_path.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    if cache.exists():
        old = json.loads(cache.read_text(encoding="utf-8"))
        if all(old.get(key) == value for key, value in info.items()):
            return old["sha256"]
    print("Hashing the model once for provenance and cache invalidation...", flush=True)
    info["sha256"] = file_sha256(model_path)
    cache.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info["sha256"]


def neighbor_candidates(model, model_sha, count):
    cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    prefix = cfg.CACHE_DIR / f"neighbors-{model_sha[:16]}-{count}"
    matrix_path = prefix.with_suffix(".npy")
    words_path = prefix.with_suffix(".json")
    if matrix_path.exists() and words_path.exists():
        words = json.loads(words_path.read_text(encoding="utf-8"))
        matrix = np.load(matrix_path, mmap_mode="r")
        if matrix.shape == (len(words), model.get_dimension()) and len(words) == count:
            print(f"Reusing cached neighbor index ({len(words):,} candidates).", flush=True)
            return words, matrix
    print(f"Building neighbor index for first {count:,} stored model words...", flush=True)
    # get_words is frequency ordered in the official model. Avoid allocating its full input matrix.
    words = model.get_words()[:count]
    if len(words) != count:
        raise ValueError(f"The model has fewer than {count} stored words.")
    vectors = np.empty((len(words), model.get_dimension()), dtype=np.float32)
    for i, word in enumerate(words):
        vectors[i] = model.get_word_vector(word)
    norms = np.linalg.norm(vectors, axis=1)
    np.divide(vectors, norms[:, None], out=vectors, where=norms[:, None] > 0)
    vectors[norms == 0] = 0
    np.save(matrix_path, vectors)
    words_path.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
    return words, vectors


def nearest_neighbors(model, model_sha, spelling_rows, smoke):
    queries = [(w, "sentiment probe", w) for w in cfg.NEIGHBOR_WORDS]
    for row in spelling_rows:
        if row["original"] in cfg.TYPO_NEIGHBOR_BASES and row["kind"] in cfg.TYPO_NEIGHBOR_TYPES:
            queries.append((row["comparison"], f"synthetic {row['kind']}", row["original"]))
    count = cfg.SMOKE_NEIGHBOR_CANDIDATES if smoke else cfg.NEIGHBOR_CANDIDATES
    words, vectors = neighbor_candidates(model, model_sha, count)
    query_vecs = np.array([model.get_word_vector(q) for q, _, _ in queries], dtype=np.float32)
    norms = np.linalg.norm(query_vecs, axis=1)
    np.divide(query_vecs, norms[:, None], out=query_vecs, where=norms[:, None] > 0)
    # Batch queries, avoiding one full matrix scan per query and model.get_input_matrix().
    similarities = vectors @ query_vecs.T
    results = []
    for col, (query, kind, original) in enumerate(queries):
        scores = similarities[:, col].copy()
        if norms[col] == 0:
            continue
        # Sort only the best values; token provides a deterministic tie-break.
        candidates = np.argpartition(scores, -min(len(scores), cfg.NEIGHBORS_PER_QUERY + 1))[-(cfg.NEIGHBORS_PER_QUERY + 1):]
        ranked = sorted((int(i) for i in candidates if words[int(i)] != query),
                        key=lambda i: (-float(scores[i]), words[i]))[:cfg.NEIGHBORS_PER_QUERY]
        for rank, i in enumerate(ranked, 1):
            results.append({"query": query, "query_kind": kind, "original": original,
                            "query_direct": model.get_word_id(query) >= 0,
                            "rank": rank, "neighbor": words[i], "cosine": float(scores[i]),
                            "candidate_vocabulary_size": count})
    return results


def make_plots(output, coverage, spelling, semantic):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [r["tokenizer"] for r in coverage]
    x = np.arange(len(labels)); width = .35
    ax.bar(x - width/2, [100*r["direct_token_fraction"] for r in coverage], width, label="Token weighted")
    ax.bar(x + width/2, [100*r["direct_type_fraction"] for r in coverage], width, label="Type weighted")
    ax.set(xlabel="Preprocessing variant", ylabel="Direct vocabulary coverage (%)", ylim=(0, 105), xticks=x, xticklabels=labels)
    ax.legend(frameon=False); ax.set_title("IMDb training: exact stored-vocabulary coverage")
    fig.tight_layout(); fig.savefig(output / "coverage.png", dpi=170); plt.close(fig)

    df = pd.DataFrame(spelling)
    order = ["deletion", "insertion", "substitution", "transposition", "repetition", "unrelated control", "identity control"]
    groups = [df.loc[df.kind == kind, "cosine"].dropna().to_numpy() for kind in order]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(groups, tick_labels=order, showfliers=False)
    rng = np.random.default_rng(cfg.SEED)
    for i, values in enumerate(groups, 1):
        ax.scatter(i + rng.uniform(-.12, .12, len(values)), values, s=12, alpha=.55)
    ax.axhline(0, color="gray", linewidth=.7)
    ax.set(ylabel="Cosine similarity", title="Synthetic perturbations and controls")
    ax.tick_params(axis="x", rotation=25); fig.tight_layout(); fig.savefig(output / "spelling_robustness.png", dpi=170); plt.close(fig)

    df = pd.DataFrame(semantic).dropna(subset=["cosine"])
    categories = list(cfg.SEMANTIC_PAIRS)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for y, category in enumerate(categories):
        subset = df[df.category == category]
        ax.scatter(subset.cosine, np.full(len(subset), y), s=38, alpha=.7)
        if len(subset):
            ax.scatter([subset.cosine.mean()], [y], marker="D", color="black", s=35, zorder=3)
    ax.axvline(0, color="gray", linewidth=.7)
    ax.set(yticks=range(len(categories)), yticklabels=categories, xlabel="Cosine similarity", title="Hand-selected IMDb word pairs (diamonds: means)")
    ax.invert_yaxis(); fig.tight_layout(); fig.savefig(output / "semantic_similarity.png", dpi=170); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR)
    parser.add_argument("--model", type=Path, default=cfg.MODEL_PATH)
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--smoke", action="store_true", help=f"Sample {cfg.SMOKE_REVIEWS} training reviews and use a small neighbor index")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Missing fastText binary {args.model}; see README.md")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error(f"Output exists and is not empty: {args.output}; choose a new directory")
    args.output.mkdir(parents=True, exist_ok=True)
    records, data_info = load_training_partition(args.data_dir, args.smoke)
    print(f"Analyzing {len(records):,} training reviews; validation and held-out test are not analyzed.", flush=True)
    model_sha = model_digest(args.model)
    if model_sha != cfg.MODEL_SHA256:
        raise ValueError(f"Unexpected model SHA-256 {model_sha}; expected the original cc.en.300.bin (see README).")
    print("Loading fastText binary once (this may require several GB of RAM)...", flush=True)
    import fasttext
    model = fasttext.load_model(str(args.model))
    if model.get_dimension() != 300:
        raise ValueError("Expected a 300-dimensional fastText model.")
    print("Counting tokens and exact vocabulary membership...", flush=True)
    counts = token_counts(records)
    coverage, status = membership_and_coverage(model, counts)
    write_csv(args.output / "coverage.csv", coverage)
    case_summary, case_examples = capitalization_transitions(counts["regex_br"], status)
    write_csv(args.output / "capitalization_summary.csv", case_summary)
    write_csv(args.output / "capitalization_examples.csv", case_examples)
    write_csv(args.output / "tokenization_examples.csv", tokenization_examples(records))
    print("Finding natural OOV examples and unverified correction candidates...", flush=True)
    oov, contexts = natural_oov_tables(records, counts["regex_br"], counts["regex_br_lower"], status)
    write_csv(args.output / "natural_oov_types.csv", oov,
              ["token", "frequency", "oov_vector_norm", "suggestions_unverified", "suggestion_training_frequencies"])
    write_csv(args.output / "natural_oov_examples.csv", contexts,
              ["token", "frequency", "heuristic_categories", "review_id", "excerpt", "suggestions_unverified", "suggestion_training_frequencies", "note"])
    print("Running synthetic spelling and hand-selected semantic probes...", flush=True)
    spelling = controlled_spelling(model, counts["regex_br_lower"])
    semantic = semantic_probes(model, counts["regex_br_lower"])
    write_csv(args.output / "spelling_pairs.csv", spelling)
    write_csv(args.output / "spelling_summary.csv", group_summary(spelling, "kind"))
    write_csv(args.output / "semantic_pairs.csv", semantic)
    write_csv(args.output / "semantic_summary.csv", group_summary(semantic, "category"))
    print("Retrieving nearest neighbors in the explicitly limited candidate vocabulary...", flush=True)
    neighbors = nearest_neighbors(model, model_sha, spelling, args.smoke)
    write_csv(args.output / "nearest_neighbors.csv", neighbors)
    make_plots(args.output, coverage, spelling, semantic)
    metadata = {"seed": cfg.SEED, "smoke": args.smoke, "dataset_revision": cfg.DATASET_REVISION,
                "data": data_info, "model_file": str(args.model.resolve()), "model_sha256": model_sha,
                "model_dimensions": model.get_dimension(), "model_word_count": len(model.get_words()),
                "model_minn": model.f.getArgs().minn, "model_maxn": model.f.getArgs().maxn,
                "neighbor_candidate_limit": cfg.SMOKE_NEIGHBOR_CANDIDATES if args.smoke else cfg.NEIGHBOR_CANDIDATES,
                "correction_candidate_minimum_training_frequency": cfg.MIN_CORRECTION_REVIEW_FREQUENCY,
                "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                "pandas": pd.__version__, "fasttext_version": __import__("importlib.metadata", fromlist=["version"]).version("fasttext"),
                "notes": ["Source labels are used solely to reconstruct the assignment 1 stratified split and fingerprint.",
                          "Only the resulting training reviews are returned to the analyses.",
                          "The old project pooled published IMDb train/test before defining its own held-out test."]}
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Finished: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
