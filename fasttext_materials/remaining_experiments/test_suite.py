"""Small synthetic checks only: no real data, training, or final-test execution."""
from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn

import error_analysis
import final_test
import inference
from modeling import make_model, restore_model, training_positions
import project1_comparison
import run as runner
from data_access import reserve_output
from validation_features import has_negation, oov_statistics
import variance
import workflow_config as cfg


class ConfigTests(unittest.TestCase):
    def test_requested_variations_change_only_their_setting(self):
        base = cfg.specification("F-base")
        expected = {"A2": "embedding", "A3": "embedding", "B1": "train_fraction", "C1": "hidden_size",
                    "C3": "hidden_size", "D1": "learning_rate", "D3": "learning_rate",
                    "E2": "dropout", "E3": "weight_decay"}
        ignore = {"description", "experiment"}
        for name, key in expected.items():
            spec = cfg.specification(name)
            changed = {k for k in base if k not in ignore and base[k] != spec[k]}
            self.assertEqual(changed, {key}, name)
        self.assertEqual(cfg.specification("F-alternative", 123, "A2")["experiment"], "A2")
        with self.assertRaises(ValueError):
            cfg.specification("F-alternative", 42, "A1")

    def test_stratified_half_is_fixed_and_training_only(self):
        labels = np.array([0] * 50 + [1] * 50)
        a = training_positions(labels, .5)
        b = training_positions(labels, .5)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(len(a), 50)
        self.assertEqual(np.bincount(labels[a]).tolist(), [25, 25])
        self.assertEqual(len(set(a)), len(a))
        self.assertTrue(np.array_equal(training_positions(labels, 1.0), np.arange(100)))

    def test_output_existing_directory_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            path.mkdir(); (path / "prior.txt").write_text("keep")
            with self.assertRaises(FileExistsError):
                reserve_output(path)
            self.assertEqual((path / "prior.txt").read_text(), "keep")

    def test_runner_dispatches_a2_without_training_in_test(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "A2"
            fake_manifest = {"identity": {"max_length": 512}}
            with patch.object(runner.bridge.baselines, "load_linear_reference"), \
                 patch.object(runner, "sequence_cache", return_value=(Path("synthetic"), fake_manifest, {})), \
                 patch.object(runner.training, "train_lstm") as train, patch("builtins.print"):
                runner.main(["--experiment", "A2", "--output", str(out)])
            self.assertTrue(out.is_dir())
            self.assertEqual(train.call_args.args[0]["embedding"], "pretrained_trainable")
            train.assert_called_once()


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.weights = torch.randn(6, 300)
        self.weights[0].zero_()
        self.ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
        self.lengths = torch.tensor([2, 3])

    def test_frozen_and_trainable_lookup_gradients_and_padding(self):
        for group in ("F-base", "A2", "A3"):
            with self.subTest(group=group):
                torch.manual_seed(42)
                model = make_model(self.weights.clone(), cfg.specification(group))
                logits = model(self.ids, self.lengths)
                nn.BCEWithLogitsLoss()(logits, torch.tensor([1., 0.])).backward()
                self.assertEqual(logits.shape, (2,))
                self.assertIsNotNone(model.lstm.weight_ih_l0.grad)
                self.assertEqual(model.embedding.weight.requires_grad, group != "F-base")
                if group != "F-base":
                    self.assertTrue(torch.equal(model.embedding.weight.grad[0], torch.zeros(300)))
                    self.assertGreater(float(model.embedding.weight.grad[1:].abs().sum()), 0)
                else:
                    self.assertIsNone(model.embedding.weight.grad)

    def test_random_initialization_is_reproducible_and_lstm_initialization_matches(self):
        torch.manual_seed(42)
        base = make_model(self.weights.clone(), cfg.specification("F-base"))
        torch.manual_seed(42)
        a = make_model(self.weights.clone(), cfg.specification("A3"))
        torch.manual_seed(42)
        b = make_model(self.weights.clone(), cfg.specification("A3"))
        torch.testing.assert_close(a.lstm.weight_ih_l0, base.lstm.weight_ih_l0)
        torch.testing.assert_close(a.embedding.weight, b.embedding.weight)
        self.assertFalse(torch.equal(a.embedding.weight[1:], base.embedding.weight[1:]))
        self.assertTrue(torch.equal(a.embedding.weight[0], torch.zeros(300)))

    def test_dropout_is_post_lstm_and_disabled_in_eval(self):
        model = make_model(self.weights.clone(), cfg.specification("E2"))
        self.assertIsInstance(model.classifier[0], nn.Dropout)
        self.assertEqual(model.classifier[0].p, .3)
        self.assertEqual(model.lstm.dropout, 0)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(self.ids, self.lengths), model(self.ids, self.lengths))
            torch.testing.assert_close(model(self.ids[:1], self.lengths[:1]),
                                       model(torch.tensor([[1, 2, 0, 0]]), self.lengths[:1]))

    def test_checkpoint_roundtrip_including_finetuned_weights(self):
        for name in ("F-base", "A2", "A3", "E2"):
            torch.manual_seed(42)
            spec = cfg.specification(name)
            model = make_model(self.weights.clone(), spec).eval()
            checkpoint = {"specification": spec, "trainable_state_dict":
                          {k: v.clone() for k, v in model.state_dict().items() if k != "embedding.weight"}}
            if spec["embedding"] != "pretrained_frozen":
                with torch.no_grad():
                    model.embedding.weight[1].add_(.25)
                checkpoint["embedding_table"] = model.embedding.weight.detach().clone()
            restored = restore_model(checkpoint, self.weights).eval()
            with torch.no_grad():
                torch.testing.assert_close(model(self.ids, self.lengths), restored(self.ids, self.lengths))
        with self.assertRaisesRegex(ValueError, "must contain"):
            restore_model({"specification": cfg.specification("A2"), "trainable_state_dict": {}}, self.weights)

    def test_loading_synthetic_trainable_checkpoint_uses_cached_vocabulary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); cache = root / "cachekey"; cache.mkdir(); run_dir = root / "run"; run_dir.mkdir()
            np.save(cache / "embedding_table.npy", self.weights.numpy())
            spec = cfg.specification("A2")
            model = make_model(self.weights.clone(), spec).eval()
            checkpoint = {"format": "remaining-lstm-v1", "specification": spec, "cache_key": cache.name,
                          "tokens_sha256": "tokens", "embedding_table_sha256": "vectors",
                          "epoch": 1,
                          "embedding_table": model.embedding.weight.detach().clone(),
                          "trainable_state_dict": {k: v for k, v in model.state_dict().items() if k != "embedding.weight"}}
            torch.save(checkpoint, run_dir / "best_model.pt")
            manifest = {"tokens_sha256": "tokens", "embedding_table_sha256": "vectors", "files_sha256": {"a": "b"},
                        "identity": {"max_length": 512}}
            (run_dir / "run_config.json").write_text(json.dumps({"cache_files_sha256": {"a": "b"}, "specification": spec}))
            (run_dir / "metrics.json").write_text(json.dumps({"best_epoch": 1}))
            with patch.object(inference, "sequence_cache", return_value=(cache, manifest, {})):
                restored, _, _, policy, _ = inference.load_lstm(run_dir)
            self.assertEqual(policy, "pretrained_trainable")
            with torch.no_grad():
                torch.testing.assert_close(model(self.ids, self.lengths), restored(self.ids, self.lengths))

    def test_synthetic_checkpoint_predicts_aligned_validation_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); cache = root / "cachekey"; cache.mkdir(); run_dir = root / "run"; run_dir.mkdir()
            np.save(cache / "embedding_table.npy", self.weights.numpy())
            np.save(cache / "validation_ids.npy", np.array([1, 2, 3], dtype=np.int32))
            np.save(cache / "validation_offsets.npy", np.array([0, 2, 3], dtype=np.int64))
            np.save(cache / "validation_labels.npy", np.array([1, 0], dtype=np.int8))
            np.save(cache / "validation_review_ids.npy", np.array([41, 99], dtype=np.int64))
            spec = cfg.specification("A2")
            model = make_model(self.weights.clone(), spec).eval()
            checkpoint = {"format": "remaining-lstm-v1", "specification": spec, "cache_key": cache.name,
                          "tokens_sha256": "tokens", "embedding_table_sha256": "vectors", "epoch": 1,
                          "embedding_table": model.embedding.weight.detach().clone(),
                          "trainable_state_dict": {k: v for k, v in model.state_dict().items() if k != "embedding.weight"}}
            torch.save(checkpoint, run_dir / "best_model.pt")
            manifest = {"tokens_sha256": "tokens", "embedding_table_sha256": "vectors", "files_sha256": {"a": "b"},
                        "identity": {"max_length": 512}}
            (run_dir / "run_config.json").write_text(json.dumps({"cache_files_sha256": {"a": "b"}, "specification": spec}))
            (run_dir / "metrics.json").write_text(json.dumps({"best_epoch": 1}))
            with patch.object(inference, "sequence_cache", return_value=(cache, manifest, {})):
                result = inference.predict_validation(run_dir, device="cpu")
            self.assertEqual(result["ids"].tolist(), [41, 99])
            self.assertEqual(result["labels"].tolist(), [1, 0])
            self.assertEqual(result["probabilities"].shape, (2,))
            self.assertTrue(np.isfinite(result["probabilities"]).all())


class AnalysisTests(unittest.TestCase):
    def test_negation_heuristic(self):
        for token in ("not", "never", "can't", "WON’T", "wouldn't've"):
            self.assertTrue(has_negation(["x", token]))
        self.assertFalse(has_negation(["notice", "nothingness"]))
        stats = oov_statistics(["known"] * 256 + ["oov"] * 256 + ["known"], lambda token: token == "oov")
        self.assertEqual(stats["oov_occurrences"], 256)
        self.assertEqual(stats["oov_fraction_256"], 0)
        self.assertEqual(stats["oov_fraction_512"], .5)
        self.assertTrue(np.isnan(oov_statistics([], lambda token: False)["oov_fraction"]))

    def test_paired_ids_and_single_class_groups(self):
        left = {"ids": np.array([1, 2, 3]), "labels": np.array([0, 0, 1]), "probabilities": np.array([.1, .8, .7])}
        right = {"ids": np.array([1, 2, 3]), "labels": np.array([0, 0, 1]), "probabilities": np.array([.2, .2, .3])}
        features = pd.DataFrame({"review_id": [1, 2, 3], "label": [0, 0, 1], "token_length": [30, 90, 300],
                                 "negation_indicator": [False, False, True], "oov_fraction": [0., .1, .02],
                                 "truncated_256": [False, False, True], "truncated_512": [False, False, False]})
        frame = error_analysis.pair_predictions(left, right, features)
        self.assertEqual(frame.left_correct.tolist(), [True, False, True])
        self.assertEqual(frame.left_error_type.tolist(), ["correct", "false_positive", "correct"])
        self.assertEqual(frame.right_error_type.tolist(), ["correct", "correct", "false_negative"])
        groups = error_analysis.subgroup_metrics(frame)
        one_class = next(r for r in groups if r["grouping"] == "length_bin" and r["group"] == "0–128")
        self.assertIsNone(one_class["left_macro_f1"])
        self.assertEqual(one_class["negative_support"], 2)
        broken = {**right, "ids": np.array([2, 1, 3])}
        with self.assertRaisesRegex(ValueError, "identical ordered"):
            error_analysis.pair_predictions(left, broken, features)
        with tempfile.TemporaryDirectory() as temp:
            error_analysis.save_plots(groups, frame, Path(temp))
            self.assertTrue((Path(temp) / "accuracy_by_length.png").is_file())
            self.assertTrue((Path(temp) / "paired_outcomes.png").is_file())

    def test_variance_refuses_incompatible_repeats(self):
        def record(seed, vocab):
            return {"seed": seed, "validation_macro_f1": .8 + seed / 10000,
                    "spec": {"seed": seed, "experiment": "A2", "embedding": "pretrained_trainable"},
                    "context": {"split": "fixed", "cache_identity": "same", "cache_files_sha256": vocab,
                                "selected_training_review_ids_sha256": "same"}}
        rows = [record(42, "a"), record(123, "a")]
        self.assertEqual(variance.summarize("A2", rows)["n"], 2)
        self.assertGreater(variance.summarize("A2", rows)["validation_macro_f1_sample_std"], 0)
        rows[1] = record(123, "different")
        with self.assertRaisesRegex(ValueError, "refusing"):
            variance.summarize("A2", rows)

    def test_variance_and_project1_cli_write_only_synthetic_outputs(self):
        result = {"individual_runs": [{"group": "F-base", "seed": 42, "validation_macro_f1": .8}],
                  "summary": [{"group": "F-base", "n": 1, "validation_macro_f1_mean": .8,
                               "validation_macro_f1_sample_std": None}], "paired_differences": [], "missing": []}
        comparison = {"project1_split_status": "unverified_split_fingerprints",
                      "rows": [{"project": "Project 1", "model": "synthetic", "validation_macro_f1": .8}]}
        with tempfile.TemporaryDirectory() as temp, patch("builtins.print"):
            out1, out2 = Path(temp) / "variance", Path(temp) / "project1"
            with patch.object(variance, "aggregate", return_value=result):
                variance.main(["--alternative", "A2", "--output", str(out1)])
            with patch.object(project1_comparison, "compare", return_value=comparison):
                project1_comparison.main(["--output", str(out2)])
            self.assertTrue((out1 / "variance.png").is_file())
            self.assertTrue((out1 / "variance.json").is_file())
            self.assertTrue((out2 / "comparison.png").is_file())
            self.assertTrue((out2 / "comparison.csv").is_file())

    def test_project1_missing_fingerprints_are_unverified(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            (p / "config.json").write_text(json.dumps({"metric": "macro_f1", "seed": 42}))
            (p / "validation_scores.csv").write_text("representation,model,score\nbigrams,mlp,0.8\n")
            baseline = p / "baseline"; baseline.mkdir()
            (baseline / "run_config.json").write_text(json.dumps({"split": {"ordered_source_sha256": "a",
                                                          "validation_indices_sha256_le_int64": "b"}}))
            with patch.object(cfg, "BASELINE", baseline):
                result = project1_comparison.compare(p, [])
            self.assertEqual(result["project1_split_status"], "unverified_split_fingerprints")
            self.assertIsNone(result["rows"][0]["validation_accuracy"])

    def test_final_test_gate_and_random_unseen_vectors(self):
        with redirect_stderr(io.StringIO()), patch.object(final_test, "load_test_partition", side_effect=AssertionError("no data")):
            with self.assertRaises(SystemExit):
                final_test.main(["--run", "lstm512", "--output", "unused-final"])
        a = final_test.random_unseen_vector("synthetic", 42)
        b = final_test.random_unseen_vector("synthetic", 42)
        c = final_test.random_unseen_vector("synthetic", 123)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))
        train, val, test = final_test.reconstruct_indices(np.array([0, 1] * 50))
        self.assertEqual((len(train), len(val), len(test)), (70, 15, 15))
        self.assertEqual(len(set(train) | set(val) | set(test)), 100)
        self.assertFalse(set(train) & set(test))

    def test_final_inference_handles_synthetic_unseen_random_tokens(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)
            (cache / "tokens.json").write_text(json.dumps([None, "known"]))
            weights = torch.zeros((2, 300)); weights[1] = 1
            spec = cfg.specification("A3")
            model = make_model(weights, spec).eval()
            manifest = {"identity": {"max_length": 512}}
            checkpoint = {"specification": spec}
            with patch.object(final_test, "load_lstm", return_value=(model, cache, manifest, "random_trainable", checkpoint)), \
                 patch("builtins.print"):
                probabilities, info = final_test.lstm_probabilities(Path("unused"), ["known unseen", "unseen"],
                                                                      [1, 0], Path("unused"), "cpu")
            self.assertEqual(probabilities.shape, (2,))
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertEqual(info["new_token_types"], 1)

    def test_final_inference_unseen_pretrained_token_uses_binary_vector(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)
            (cache / "tokens.json").write_text(json.dumps([None, "known"]))
            weights = torch.zeros((2, 300)); weights[1] = 1
            model = make_model(weights, cfg.specification("F-base")).eval()
            manifest = {"identity": {"max_length": 512, "model_sha256": "synthetic"}}
            binary = SimpleNamespace(get_word_vector=lambda token: np.full(300, .25, dtype=np.float32))
            fasttext = SimpleNamespace(load_model=lambda path: binary)
            with patch.object(final_test, "load_lstm", return_value=(model, cache, manifest, "pretrained_frozen", {})), \
                 patch.object(final_test, "model_digest", return_value="synthetic"), \
                 patch.dict(sys.modules, {"fasttext": fasttext}), patch("builtins.print"):
                probabilities, info = final_test.lstm_probabilities(Path("unused"), ["known unseen", "unseen"],
                                                                      [1, 0], Path("unused"), "cpu")
            self.assertEqual(len(probabilities), 2)
            self.assertEqual(info["new_token_types"], 1)
            torch.testing.assert_close(model.embedding.weight[2], torch.full((300,), .25))

    def test_final_lr_pooling_uses_synthetic_binary_and_pipeline(self):
        binary = SimpleNamespace(get_word_vector=lambda token: np.ones(300, dtype=np.float32))
        fasttext = SimpleNamespace(load_model=lambda path: binary)
        pipeline = SimpleNamespace(classes_=np.array([0, 1]), predict_proba=lambda x: np.tile([.4, .6], (len(x), 1)))
        with patch.object(final_test, "model_digest"), patch.object(final_test, "validate_lr_run"), \
             patch.object(final_test.joblib, "load", return_value=pipeline), \
             patch.dict(sys.modules, {"fasttext": fasttext}), patch("builtins.print"):
            values, info = final_test.lr_probabilities(Path("unused"), ["known words", ""], Path("unused"))
        np.testing.assert_array_equal(values, [.6, .6])
        self.assertEqual(info["family"], "lr_mean_full_review")


if __name__ == "__main__":
    unittest.main()
