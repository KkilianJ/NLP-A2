"""Run the remaining GloVe experiments with inner-validation model selection."""
import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from compare_gru import Classifier, batches, evaluate
from complete_embeddings import DIMENSION, glove_lines, load_vectors
from prepare_meanpool import load_project1, load_reviews, tokenize

SEEDS = (42, 43, 44)
BATCH = 128
QUERIES = ("excellent", "awful", "boring", "overrated", "screenplay", "good", "bad")
NEGATIVE = re.compile(r"(?:\b(?:not|never|no|neither|nor|cannot)\b|n't\b)")


@dataclass(frozen=True)
class Configuration:
    hidden: int = 64
    limit: int = 128
    lr: float = 0.001
    frozen: bool = True


def save(path, content):
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def nearest_neighbors(path):
    words, vectors = [], []
    with glove_lines(path) as lines:
        for line in lines:
            fields = line.split(maxsplit=1)
            if not fields:
                continue
            vectors.append(np.fromstring(fields[1].decode("ascii"), sep=" ", dtype=np.float32))
            words.append(fields[0].decode("utf-8"))
    matrix = np.stack(vectors)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-12)
    lookup = {word: i for i, word in enumerate(words)}
    rows = {}
    for query in QUERIES:
        i = lookup[query]
        similarities = matrix @ matrix[i]
        similarities[i] = -np.inf
        top = np.argpartition(similarities, -10)[-10:]
        top = sorted(top, key=lambda j: (-similarities[j], words[j]))
        rows[query] = [{"word": words[j], "cosine": float(similarities[j])} for j in top]
    pairs = {
        f"{a}/{b}": float(matrix[lookup[a]] @ matrix[lookup[b]])
        for a, b in (("good", "bad"), ("excellent", "awful"), ("excellent", "great"),
                     ("boring", "dull"), ("good", "great"), ("bad", "terrible"))
    }
    return {"vocabulary_types": len(words), "neighbors": rows, "illustrative_pairs": pairs}


def predict_gru(model, sequences, labels, ids):
    model.eval()
    predicted, positions = [], []
    # batches sorts by length before padding; mirror that order for row-level alignment.
    with torch.inference_mode():
        for x, lengths, _ in batches(sequences, labels, ids, BATCH, "cpu"):
            predicted.extend((model(x, lengths) >= 0).numpy().tolist())
    for start in range(0, len(ids), BATCH * 16):
        positions.extend(sorted(ids[start : start + BATCH * 16], key=lambda i: len(sequences[i])))
    order = {int(i): prediction for i, prediction in zip(positions, predicted, strict=True)}
    return np.asarray([order[int(i)] for i in ids], dtype=np.int8)


def train_gru(config, weights, sequences, labels, fit, inner, seed, cap, model_factory=Classifier):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model_factory(weights.clone() if not config.frozen else weights, config.hidden)
    model.embedding.weight.requires_grad_(not config.frozen)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=config.lr)
    criterion = nn.BCEWithLogitsLoss()
    best, history = {}, []
    for epoch in range(1, cap + 1):
        model.train()
        loss_sum, correct = 0.0, np.zeros((2, 2), dtype=np.int64)
        for ids, lengths, target in batches(sequences, labels, fit, BATCH, "cpu", rng):
            optimizer.zero_grad()
            logits = model(ids, lengths)
            loss = criterion(logits, target.float())
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(target)
            cells = 2 * target.numpy() + (logits.detach().numpy() >= 0)
            correct += np.bincount(cells, minlength=4).reshape(2, 2)
        score = evaluate(model, batches(sequences, labels, inner, BATCH, "cpu"))
        row = {"epoch": epoch, "train_loss": loss_sum / len(fit),
               "online_train_macro_f1": float((2 * correct.diagonal() /
                                              (correct.sum(axis=0) + correct.sum(axis=1))).mean()),
               "inner_macro_f1": score}
        history.append(row)
        if not best or score > best["inner_macro_f1"]:
            best = row
            checkpoint = {key: value.detach().clone() for key, value in model.state_dict().items()
                          if not config.frozen or key != "embedding.weight"}
        if epoch - best["epoch"] >= 2:
            break
    model.load_state_dict(checkpoint, strict=not config.frozen)
    return model, {"best": best, "history": history, "trainable_parameters": sum(p.numel() for p in parameters)}


def scores(labels, predictions):
    return {"macro_f1": float(f1_score(labels, predictions, average="macro", labels=[0, 1])),
            "accuracy": float(accuracy_score(labels, predictions))}


def tune_meanpool(weights, sequences, labels, fit, inner):
    features = np.zeros((len(sequences), DIMENSION), dtype=np.float32)
    matrix = weights.numpy()
    for i, sequence in enumerate(sequences):
        if sequence:
            features[i] = matrix[sequence].mean(axis=0, dtype=np.float64)
    best, grid = None, []
    for c in (0.001, 0.01, 0.1, 1.0, 10.0):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000))
        model.fit(features[fit], labels[fit])
        row = {"C": c, **scores(labels[inner], model.predict(features[inner]))}
        grid.append(row)
        if best is None or row["macro_f1"] > best["macro_f1"]:
            best, selected = row, model
    assert selected[-1].n_iter_.max() < selected[-1].max_iter
    return selected.predict(features), {"selected": best, "grid": grid}


def error_analysis(features, labels, mean, gru, fit, validation, output):
    lengths, covered, negations, contrasts = features
    boundaries = np.quantile(lengths[fit], (1/3, 2/3)).astype(int)
    oov_rate = (lengths - covered) / lengths.clip(min=1)
    oov_cut = float(np.median(oov_rate[fit]))
    groups = {
        f"length <= {boundaries[0]}": lengths <= boundaries[0],
        f"length {boundaries[0]+1}–{boundaries[1]}": (lengths > boundaries[0]) & (lengths <= boundaries[1]),
        f"length > {boundaries[1]}": lengths > boundaries[1],
        "negation cue present": negations, "negation cue absent": ~negations,
        "contrast cue present": contrasts, "contrast cue absent": ~contrasts,
        "OOV rate at or below training median": oov_rate <= oov_cut,
        "OOV rate above training median": oov_rate > oov_cut,
        "truncated": covered > output["token_limit"], "not truncated": covered <= output["token_limit"],
    }
    results = {"training_length_tertiles": boundaries.tolist(), "training_median_oov_rate": oov_cut, "groups": {}}
    for name, mask in groups.items():
        ix = validation[mask[validation]]
        if len(ix):
            m, g = mean[ix] == labels[ix], gru[ix] == labels[ix]
            results["groups"][name] = {
                "n": len(ix), "meanpool": scores(labels[ix], mean[ix]), "gru": scores(labels[ix], gru[ix]),
                "only_meanpool_correct": int((m & ~g).sum()), "only_gru_correct": int((g & ~m).sum()),
            }
    return results


def run(args):
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.semantics_only:
        save(args.output / "semantic_probes.json", nearest_neighbors(args.glove))
        return

    project = load_project1(args.project1)
    frame, (train, validation, test), fingerprint = load_reviews(project, args.arrow_dir)
    row_ids = np.concatenate((train, validation, test))
    labels = frame.label.iloc[row_ids].map({"negative": 0, "positive": 1}).to_numpy(dtype=np.int64)
    fit, inner = train_test_split(np.arange(len(train)), test_size=.10, stratify=labels[:len(train)], random_state=42)
    half, _ = train_test_split(fit, train_size=.50, stratify=labels[fit], random_state=42)
    outer = np.arange(len(train), len(train) + len(validation))
    final_test = np.arange(len(train) + len(validation), len(row_ids))
    # Before locking, only prepare the fitting, inner, and outer validation texts.
    development = tokenize(frame.text.iloc[row_ids[:len(train) + len(validation)]])
    glove = load_vectors(args.glove, {word for doc in development for word in doc})
    word_ids = {word: i for i, word in enumerate(glove, 1)}
    weights = torch.zeros((len(glove) + 1, DIMENSION))
    weights[1:] = torch.from_numpy(np.stack(list(glove.values())))
    covered = [[word_ids[word] for word in doc if word in word_ids] for doc in development]
    coverage = np.array([len(doc) for doc in covered])
    base = Configuration()
    candidates = [base, replace(base, hidden=32), replace(base, lr=.0005),
                  replace(base, lr=.002), replace(base, limit=256)]
    result = {"fingerprint": fingerprint, "dataset_revision": project.IMDB_REVISION, "fitting_reviews": len(fit),
              "inner_validation_reviews": len(inner), "project1_validation_reviews": len(outer),
              "test_reviews": len(final_test), "seed_list": list(SEEDS), "inner_seed": 42,
              "screening_epoch_cap": 4, "confirmation_epoch_cap": 8, "patience": 2, "batch_size": BATCH,
              "optimizer": "Adam", "gradient_clip_norm": 1.0, "gn": "global L2 gradient norm",
              "vocabulary": "GloVe hits only; regex tokenizer keeps contractions",
              "screening": [], "runs": []}
    summary_path = args.output / "recurrent_results.json"
    for config in candidates:
        sequences = [doc[:config.limit] for doc in covered]
        _, record = train_gru(config, weights, sequences, labels, fit, inner, 42, 4)
        row = {"configuration": asdict(config), **record,
               "fitting_truncated_fraction": float((coverage[fit] > config.limit).mean())}
        result["screening"].append(row)
        save(summary_path, result)
        print("SCREEN", asdict(config), record["best"], flush=True)
    chosen = Configuration(**max(result["screening"], key=lambda row: row["best"]["inner_macro_f1"])["configuration"])
    result["selected_configuration"] = asdict(chosen)
    save(summary_path, result)
    sequences = [doc[:chosen.limit] for doc in covered]
    mean_pred, mean_results = tune_meanpool(weights, sequences, labels, fit, inner)
    result["meanpool"] = {**mean_results, "project1_validation": scores(labels[outer], mean_pred[outer])}

    saved = {}
    setups = (("frozen_full", chosen, fit), ("tuned_full", replace(chosen, frozen=False), fit),
              ("frozen_half", chosen, half))
    for name, config, subset in setups:
        for seed in SEEDS:
            model, record = train_gru(config, weights, sequences, labels, subset, inner, seed, 8)
            prediction = predict_gru(model, sequences, labels, outer)
            assert abs(evaluate(model, batches(sequences, labels, outer, BATCH, "cpu")) -
                       scores(labels[outer], prediction)["macro_f1"]) < 1e-12
            row = {"setup": name, "seed": seed, "fit_n": len(subset), "configuration": asdict(config),
                   **record, "project1_validation": scores(labels[outer], prediction)}
            result["runs"].append(row)
            if seed == 42 and name != "frozen_half":
                saved[name] = (model, prediction, record["best"]["inner_macro_f1"])
            save(summary_path, result)
            print("CONFIRM", name, seed, row["project1_validation"], flush=True)

    for name, _, _ in setups:
        values = [row["project1_validation"]["macro_f1"] for row in result["runs"] if row["setup"] == name]
        result.setdefault("validation_summary", {})[name] = {
            "macro_f1_mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))}
    locked_name = max(saved, key=lambda name: np.mean([
        row["best"]["inner_macro_f1"] for row in result["runs"] if row["setup"] == name]))
    locked, pred, _ = saved[locked_name]
    result["final_choice"] = {"setup": locked_name, "seed": 42, "token_limit": chosen.limit,
                              "rule": "highest mean inner validation across the three full-data runs; seed fixed to 42"}
    save(summary_path, result)

    text = frame.text.iloc[row_ids[:len(development)]]
    lengths = np.array([len(doc) for doc in development])
    negations = text.map(lambda s: bool(NEGATIVE.search(s.lower().replace("’", "'")))).to_numpy()
    contrasts = text.str.contains(r"\b(?:but|however|although|though|yet)\b", case=False, regex=True).to_numpy()
    pred_by_row = np.full(len(labels), -1, dtype=np.int8)
    pred_by_row[outer] = pred
    errors = error_analysis((lengths, coverage, negations, contrasts), labels, mean_pred, pred_by_row,
                            fit, outer, result["final_choice"])
    save(args.output / "error_analysis.json", errors)
    with (args.output / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("project1_row", "label", "meanpool", "gru", "tokens", "glove_hits", "truncated",
                         "negation", "contrast"))
        for i in outer:
            writer.writerow((row_ids[i], labels[i], mean_pred[i], pred_by_row[i], lengths[i], coverage[i],
                             int(coverage[i] > chosen.limit), int(negations[i]), int(contrasts[i])))

    # Keep original IDs and model weights. Test-only GloVe words use their original frozen values.
    test_docs = tokenize(frame.text.iloc[row_ids[final_test]])
    extra = load_vectors(args.glove, {word for doc in test_docs for word in doc} - word_ids.keys())
    if extra:
        first = len(word_ids) + 1
        word_ids.update({word: i for i, word in enumerate(extra, first)})
        new_weights = torch.cat((locked.embedding.weight.detach(), torch.from_numpy(np.stack(list(extra.values())))))
        locked.embedding = nn.Embedding.from_pretrained(new_weights, freeze=True, padding_idx=0)
    all_sequences = sequences + [[word_ids[word] for word in doc if word in word_ids][:chosen.limit]
                                 for doc in test_docs]
    test_pred = predict_gru(locked, all_sequences, labels, final_test)
    result["final_test"] = scores(labels[final_test], test_pred)
    save(summary_path, result)
    print("FINAL", result["final_choice"], result["final_test"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project1", type=Path, required=True)
    parser.add_argument("--glove", type=Path, required=True)
    parser.add_argument("--arrow-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("final_glove_study"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--semantics-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
