"""Small synthetic checks only; these do not train models or load IMDb/fastText."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import settings
from embeddings import cache_identity, pool_vectors
from metrics import evaluate
from phase1_imports import phase1
from plots import regenerate


class PoolingTests(unittest.TestCase):
    def test_mean_max_and_zero_vectors(self):
        vectors = [np.array([-2., -4.]), np.zeros(2), np.array([-3., -1.])]
        mean, maximum, zero_count = pool_vectors(vectors, dimension=2)
        np.testing.assert_allclose(mean, [-2.5, -2.5])
        np.testing.assert_allclose(maximum, [-2., -1.])
        self.assertEqual(zero_count, 1)

    def test_empty_and_all_zero_reviews(self):
        for vectors in ([], [np.zeros(2), np.zeros(2)]):
            mean, maximum, zero_count = pool_vectors(vectors, dimension=2)
            np.testing.assert_array_equal(mean, [0., 0.])
            np.testing.assert_array_equal(maximum, [0., 0.])
            self.assertEqual(zero_count, len(vectors))

    def test_nonfinite_or_wrong_shape_are_rejected(self):
        with self.assertRaises(ValueError):
            pool_vectors([np.array([np.nan, 1.])], dimension=2)
        with self.assertRaises(ValueError):
            pool_vectors([np.zeros(3)], dimension=2)

    def test_phase1_regex_keeps_case_and_html(self):
        self.assertEqual(phase1.tokenize("Wow<br />Don't!", "regex"),
                         ["Wow", "<", "br", "/", ">", "Don't", "!"])

    def test_cache_identity_changes_with_order_and_model(self):
        train = SimpleNamespace(ids=np.array([1, 2]))
        val = SimpleNamespace(ids=np.array([3, 4]))
        info = {"ordered_source_sha256": "dataset"}
        first = cache_identity(train, val, info, "model-a")
        val.ids = np.array([4, 3])
        self.assertNotEqual(first, cache_identity(train, val, info, "model-a"))
        self.assertNotEqual(first, cache_identity(train, val, info, "model-b"))


class MetricTests(unittest.TestCase):
    def test_probability_threshold_and_class_order(self):
        result = evaluate([0, 1, 1, 0], [0.49, 0.5, 0.2, 0.8])
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["macro_f1"], 0.5)
        self.assertEqual(result["per_class"]["positive"]["recall"], 0.5)

    def test_nonfinite_probabilities_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([1], [np.nan])


class PlotTests(unittest.TestCase):
    def test_plot_regeneration_from_synthetic_saved_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            for name, (model, pool) in settings.EXPERIMENTS.items():
                directory = run_dir / name
                directory.mkdir()
                metrics = evaluate([0, 1, 1, 0], [0.1, 0.8, 0.7, 0.6])
                result = {"experiment": name, "model": model, "pooling": pool,
                          "training": metrics, "validation": metrics, "training_seconds": 1.0}
                if name.startswith("mlp-"):
                    result["best_epoch"] = 1
                    pd.DataFrame([{"epoch": 1, "training_loss": .6, "validation_loss": .7,
                                   "training_accuracy": .75, "validation_accuracy": .75,
                                   "validation_macro_f1": .7333}]).to_csv(directory / "epochs.csv", index=False)
                (directory / "metrics.json").write_text(json.dumps(result))
            regenerate(run_dir)
            self.assertTrue((run_dir / "graphs" / "mean_vs_max.png").is_file())
            self.assertTrue((run_dir / "mlp-mean" / "graphs" / "loss_by_epoch.png").is_file())
            self.assertTrue((run_dir / "lr-max" / "graphs" / "validation_scores.png").is_file())
            self.assertTrue(json.loads((run_dir / "comparison.json").read_text())["complete"])


if __name__ == "__main__":
    unittest.main()
