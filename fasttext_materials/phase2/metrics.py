"""Shared evaluation: class 0 negative, class 1 positive, threshold 0.5."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
import settings


def evaluate(y_true, probabilities):
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if truth.shape != probabilities.shape or not np.isfinite(probabilities).all():
        raise ValueError("Invalid evaluation labels or probabilities.")
    predicted = (probabilities >= settings.THRESHOLD).astype(np.int64)
    precision, recall, _, support = precision_recall_fscore_support(
        truth, predicted, labels=[0, 1], zero_division=0)
    return {"accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, labels=[0, 1], average="macro", zero_division=0)),
            "per_class": {
                name: {"precision": float(precision[i]), "recall": float(recall[i]), "support": int(support[i])}
                for i, name in enumerate(("negative", "positive"))},
            "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist()}
