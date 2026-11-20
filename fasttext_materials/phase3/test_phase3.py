"""Synthetic unit checks only; no IMDb data, real fastText model, or training."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import experiment_settings as cfg
import legacy
from lstm_training import EarlyStopping, score_model
from sequence_dataset import collate_reviews, SequenceDataset
from sequence_model import SentimentLSTM, model_from_checkpoint
from sequence_preparation import encode_texts, prepare_sequences


class CLITests(unittest.TestCase):
    def test_head_result_directories_are_distinct(self):
        from run import default_run_dir
        args = SimpleNamespace(smoke=False, head="linear", head_hidden_size=64, seed=42,
                               hidden_size=64, max_sequence_length=512, batch_size=32,
                               learning_rate=0.001, epochs=30)
        linear = default_run_dir(args)
        args.head = "mlp"
        mlp = default_run_dir(args)
        self.assertEqual(linear.name, "lstm-s42-h64-l512-b32-lr0.001-e30")
        self.assertEqual(mlp.name, "lstm-mlp64-s42-h64-l512-b32-lr0.001-e30")
        self.assertNotEqual(linear, mlp)

    def test_conflicting_mlp_settings_fail_before_data_preparation(self):
        for extra in (("--max-sequence-length", "256"), ("--head-hidden-size", "32")):
            with self.subTest(extra=extra):
                result = subprocess.run([sys.executable, str(cfg.ROOT / "run.py"), "--head", "mlp", *extra],
                                        capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn("error:", result.stderr)
                self.assertNotIn("Reconstructing", result.stdout)


class SequenceTests(unittest.TestCase):
    def test_truncation_order_markup_and_statistics(self):
        token_to_id, tokens = {}, [None]
        flat, offsets, stats = encode_texts(["Wow<br />Don't!", ""], token_to_id, tokens,
                                            max_length=4, progress_every=0)
        self.assertEqual([tokens[i] for i in flat], ["Wow", "<", "br", "/"])
        self.assertEqual(offsets.tolist(), [0, 4, 4])
        self.assertEqual(stats["reviews_longer_than_limit"], 1)
        self.assertEqual(stats["tokens_before_truncation"], 7)
        self.assertEqual(stats["tokens_retained"], 4)
        self.assertEqual(stats["tokens_discarded"], 3)
        self.assertEqual(stats["empty_reviews"], 1)
        self.assertAlmostEqual(stats["reviews_truncated_percent"], 50.0)

    def test_dynamic_batch_padding_and_empty(self):
        ids, lengths, labels = collate_reviews([(torch.tensor([2, 3]), 1),
                                               (torch.tensor([4]), 0),
                                               (torch.tensor([], dtype=torch.long), 1)])
        self.assertEqual(ids.tolist(), [[2, 3], [4, 0], [0, 0]])
        self.assertEqual(lengths.tolist(), [2, 1, 0])
        self.assertEqual(labels.tolist(), [1., 0., 1.])
        all_empty, lengths, _ = collate_reviews([(torch.tensor([], dtype=torch.long), 0)])
        self.assertEqual(all_empty.shape, (1, 1))
        self.assertEqual(lengths.tolist(), [0])

    def test_distinct_oov_tokens_and_zero_vectors_survive_cache(self):
        class FakeFastText:
            def get_dimension(self): return 300
            def get_word_id(self, word): return 0 if word == "Known" else -1
            def get_word_vector(self, word):
                return np.zeros(300, dtype=np.float32) if word == "!" else np.full(300, sum(map(ord, word)), dtype=np.float32)
        fake = FakeFastText()
        calls = []
        def load_fake(path):
            calls.append(path)
            return fake
        train = SimpleNamespace(ids=np.array([11, 12]), texts=["Known Zyxq !", ""], labels=np.array([1, 0]))
        validation = SimpleNamespace(ids=np.array([17]), texts=["Zyxq Other"], labels=np.array([1]))
        with tempfile.TemporaryDirectory() as temp, patch.object(cfg, "CACHE_DIR", Path(temp)), \
             patch("sequence_preparation.verified_model_digest", return_value="synthetic-model"), \
             patch.dict(sys.modules, {"fasttext": SimpleNamespace(load_model=load_fake)}), \
             patch("builtins.print"):
            args = (train, validation, {"ordered_source_sha256": "synthetic-data"}, Path("unused"), 8, True)
            cache, manifest, reused = prepare_sequences(*args)
            self.assertFalse(reused)
            vocab = json.loads((cache / "tokens.json").read_text())
            matrix = np.load(cache / "embedding_table.npy")
            self.assertNotEqual(vocab.index("Known"), vocab.index("Zyxq"))
            np.testing.assert_array_equal(matrix[vocab.index("Zyxq")], fake.get_word_vector("Zyxq"))
            self.assertNotEqual(vocab.index("!"), 0)
            np.testing.assert_array_equal(matrix[vocab.index("!")], np.zeros(300))
            ds = SequenceDataset(cache, "train")
            self.assertEqual(ds[0][0].tolist(), [vocab.index("Known"), vocab.index("Zyxq"), vocab.index("!")])
            self.assertEqual(ds[1][0].numel(), 0)
            self.assertEqual(manifest["statistics"]["train"]["retained_zero_vector_token_occurrences"], 1)
            same, _, reused = prepare_sequences(*args)
            self.assertTrue(reused)
            self.assertEqual(cache, same)
            self.assertEqual(len(calls), 1)
            with (cache / "tokens.json").open("a") as f:
                f.write(" ")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                prepare_sequences(*args)


class ModelTests(unittest.TestCase):
    def test_frozen_embedding_output_shapes_and_padding_invariance(self):
        torch.manual_seed(8)
        weights = torch.randn(5, 300)
        weights[0] = 0
        model = SentimentLSTM(weights, hidden_size=8).eval()
        self.assertFalse(model.embedding.weight.requires_grad)
        self.assertTrue(all(p.requires_grad for p in model.lstm.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))
        self.assertEqual(tuple(model.classifier.weight.shape), (1, 8))
        with torch.no_grad():
            alone = model(torch.tensor([[1, 2]]), torch.tensor([2]))
            padded = model(torch.tensor([[1, 2, 0, 0]]), torch.tensor([2]))
            in_batch = model(torch.tensor([[1, 2, 0, 0], [2, 1, 3, 4], [0, 0, 0, 0]]),
                             torch.tensor([2, 4, 0]))
        self.assertEqual(tuple(in_batch.shape), (3,))
        torch.testing.assert_close(alone, padded, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(alone[0], in_batch[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(in_batch[2], model.classifier.bias[0])

    def test_mlp_head_structure_frozen_embeddings_and_joint_gradients(self):
        weights = torch.randn(5, 300); weights[0] = 0
        torch.manual_seed(42)
        linear = SentimentLSTM(weights.clone(), hidden_size=64, head="linear")
        torch.manual_seed(42)
        model = SentimentLSTM(weights.clone(), hidden_size=64, head="mlp", head_hidden_size=64)
        self.assertEqual([type(layer) for layer in model.classifier],
                         [torch.nn.Linear, torch.nn.ReLU, torch.nn.Linear])
        self.assertEqual(tuple(model.classifier[0].weight.shape), (64, 64))
        self.assertEqual(tuple(model.classifier[2].weight.shape), (1, 64))
        for name, value in linear.lstm.state_dict().items():
            torch.testing.assert_close(value, model.lstm.state_dict()[name], rtol=0, atol=0)
        self.assertFalse(model.embedding.weight.requires_grad)
        with torch.no_grad():
            alone = model(torch.tensor([[1, 2]]), torch.tensor([2]))
            padded = model(torch.tensor([[1, 2, 0, 0]]), torch.tensor([2]))
        torch.testing.assert_close(alone, padded, rtol=1e-5, atol=1e-6)
        # A synthetic gradient check, not a training step: no optimizer or parameter update.
        model(torch.tensor([[1, 2], [3, 4]]), torch.tensor([2, 2])).sum().backward()
        self.assertIsNone(model.embedding.weight.grad)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.lstm.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.classifier.parameters()))
        with self.assertRaises(ValueError):
            SentimentLSTM(weights, hidden_size=64, head="mlp", head_hidden_size=32)

    def test_old_and_new_checkpoint_formats_reconstruct(self):
        weights = torch.randn(4, 300); weights[0] = 0
        for head in ("linear", "mlp"):
            original = SentimentLSTM(weights.clone(), hidden_size=64, head=head).eval()
            checkpoint = {"embedding_table": weights, "hidden_size": 64,
                          "trainable_state_dict": {k: v.detach().clone() for k, v in original.state_dict().items()
                                                   if k != "embedding.weight"}}
            # Old linear checkpoints deliberately have no head metadata.
            if head == "mlp":
                checkpoint.update({"head": "mlp", "head_hidden_size": 64})
            restored = model_from_checkpoint(checkpoint).eval()
            self.assertEqual(restored.head, head)
            with torch.no_grad():
                ids, lengths = torch.tensor([[1, 2, 0]]), torch.tensor([2])
                torch.testing.assert_close(original(ids, lengths), restored(ids, lengths))

    def test_checkpoint_roundtrip_with_synthetic_untrained_model(self):
        torch.manual_seed(3)
        weights = torch.randn(3, 300); weights[0] = 0
        original = SentimentLSTM(weights, hidden_size=4).eval()
        checkpoint = {"embedding_table": weights, "tokens": [None, "Known", "OOV"],
                      "hidden_size": 4, "trainable_state_dict": {
                          k: v.detach().clone() for k, v in original.state_dict().items()
                          if k != "embedding.weight"}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.pt"
            torch.save(checkpoint, path)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            restored = SentimentLSTM(saved["embedding_table"], saved["hidden_size"]).eval()
            status = restored.load_state_dict(saved["trainable_state_dict"], strict=False)
            self.assertEqual(status.missing_keys, ["embedding.weight"])
            self.assertEqual(status.unexpected_keys, [])
            with torch.no_grad():
                token_ids, lengths = torch.tensor([[1, 2, 0]]), torch.tensor([2])
                torch.testing.assert_close(original(token_ids, lengths), restored(token_ids, lengths))
            self.assertEqual(saved["tokens"], [None, "Known", "OOV"])

    def test_synthetic_forward_evaluation_without_training(self):
        weights = torch.randn(4, 300); weights[0] = 0
        model = SentimentLSTM(weights, hidden_size=4)
        batch = (torch.tensor([[1, 2], [3, 0]]), torch.tensor([2, 1]), torch.tensor([0., 1.]))
        loss, scores = score_model(model, [batch], torch.device("cpu"))
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(scores["per_class"]["negative"]["support"], 1)
        self.assertEqual(scores["per_class"]["positive"]["support"], 1)
        self.assertEqual(sum(map(sum, scores["confusion_matrix"])), 2)

    def test_invalid_lengths_rejected(self):
        weights = torch.ones(2, 300); weights[0] = 0
        model = SentimentLSTM(weights, hidden_size=2)
        with self.assertRaises(ValueError):
            model(torch.tensor([[1, 0]]), torch.tensor([3]))

    def test_early_stopping_strict_ties_and_patience(self):
        stopper = EarlyStopping(5)
        self.assertTrue(stopper.update(.5, 1))
        self.assertTrue(stopper.update(.6, 2))
        for epoch in range(3, 7):
            self.assertFalse(stopper.update(.6, epoch))
            self.assertFalse(stopper.should_stop)
        self.assertFalse(stopper.update(.59, 7))
        self.assertTrue(stopper.should_stop)
        self.assertEqual(stopper.best_epoch, 2)
        self.assertTrue(stopper.update(.61, 8))
        self.assertFalse(stopper.should_stop)


class BaselineTests(unittest.TestCase):
    def test_split_and_tokenizer_are_verified(self):
        from baselines import load_baselines
        keys = {"training_indices_sha256_le_int64": "train", "validation_indices_sha256_le_int64": "val",
                "ordered_source_sha256": "dataset"}
        import hashlib, inspect
        metadata = {"split": keys, "experiment_settings": {"threshold": 0.5}, "features": {"identity": {
            "model_sha256": legacy.phase1_config.MODEL_SHA256,
            "tokenizer": "Phase 1 regex (case and HTML preserved)",
            "tokenizer_function_sha256": hashlib.sha256(inspect.getsource(legacy.phase1.tokenize).encode()).hexdigest(),
            "regex_pattern": legacy.phase1.WORD_RE.pattern}}}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            csv = directory / "comparison.csv"
            pd.DataFrame([{"experiment": name, "validation_accuracy": .8, "validation_macro_f1": .79}
                          for name in ("lr-mean", "lr-max", "mlp-mean", "mlp-max")]).to_csv(csv, index=False)
            (directory / "run_config.json").write_text(json.dumps(metadata))
            self.assertEqual(len(load_baselines(csv, keys)["rows"]), 4)
            with self.assertRaises(ValueError):
                load_baselines(csv, {**keys, "validation_indices_sha256_le_int64": "different"})


    @unittest.skipUnless(all((cfg.LINEAR_BASELINE_DIR / name).is_file() for name in
                             ("run_config.json", "metrics.json", "history.csv", "best_model.pt")),
                         "Saved 512-token linear baseline is not available")
    def test_linear_reference_requires_matching_sequence_cache(self):
        from baselines import load_linear_reference, snapshot_linear_reference
        reference = load_linear_reference(cfg.LINEAR_BASELINE_DIR)
        info = reference["config"]["split"]
        manifest = reference["config"]["sequence_cache"]["manifest"]
        snapshot = snapshot_linear_reference(reference, info, manifest)
        self.assertEqual(snapshot["best_epoch"], 17)
        self.assertAlmostEqual(snapshot["validation_macro_f1"], 0.9045206232864318)
        self.assertTrue(snapshot["same_full_data_and_cache"])
        changed = {**manifest, "files_sha256": {**manifest["files_sha256"], "train_ids.npy": "changed"}}
        with self.assertRaises(ValueError):
            snapshot_linear_reference(reference, info, changed)
        self.assertFalse(snapshot_linear_reference(reference, info, changed, smoke=True)["same_full_data_and_cache"])


class PlotTests(unittest.TestCase):
    def test_phase3_plot_module_is_not_shadowed_by_phase2(self):
        import plots
        self.assertEqual(Path(plots.__file__).resolve().parent, cfg.ROOT)

    def test_mlp_plot_includes_linear_baseline_and_best_epoch(self):
        from plots import regenerate
        with tempfile.TemporaryDirectory() as temp, patch("builtins.print"):
            out = Path(temp)
            metric = {"accuracy": .75, "macro_f1": .74, "confusion_matrix": [[3, 1], [1, 3]]}
            metrics = {"head": "mlp", "head_hidden_size": 64, "best_epoch": 2,
                       "hidden_size": 64, "max_sequence_length": 512,
                       "training": {"accuracy": .85, "macro_f1": .84}, "validation": metric}
            reference = {"model": "LSTM + Linear, 512 tokens", "validation_accuracy": .9,
                         "validation_macro_f1": .89, "best_epoch": 17, "train_accuracy": .95,
                         "train_macro_f1": .94, "training_validation_accuracy_gap": .05,
                         "training_validation_macro_f1_gap": .05, "same_full_data_and_cache": True,
                         "note": "synthetic"}
            baseline = {"rows": [{"experiment": x, "validation_accuracy": .7, "validation_macro_f1": .69}
                                 for x in ("lr-mean", "lr-max", "mlp-mean", "mlp-max")]}
            history = [{"epoch": i, "training_loss": 1/i, "validation_loss": 1/i,
                        "training_accuracy": .7, "validation_accuracy": .7,
                        "training_macro_f1": .69, "validation_macro_f1": .69} for i in (1, 2)]
            (out / "metrics.json").write_text(json.dumps(metrics))
            (out / "linear_baseline_snapshot.json").write_text(json.dumps(reference))
            (out / "baseline_snapshot.json").write_text(json.dumps(baseline))
            (out / "run_config.json").write_text(json.dumps({"smoke": False}))
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)
            regenerate(out)
            self.assertTrue((out / "graphs" / "head_comparison.png").is_file())
            compared = pd.read_csv(out / "head_comparison.csv")
            self.assertEqual(compared.best_epoch.tolist(), [17, 2])
            self.assertAlmostEqual(compared.loc[1, "training_validation_accuracy_gap"], .10)
            self.assertTrue(json.loads((out / "head_comparison.json").read_text())["comparable"])

    def test_regenerate_from_temporary_synthetic_metrics(self):
        from plots import regenerate
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            metrics = {"best_epoch": 2, "hidden_size": 64, "max_sequence_length": 256,
                       "validation": {"accuracy": .75, "macro_f1": .74, "confusion_matrix": [[3, 1], [1, 3]]}}
            baselines = {"rows": [{"experiment": x, "validation_accuracy": .7, "validation_macro_f1": .69}
                                  for x in ("lr-mean", "lr-max", "mlp-mean", "mlp-max")]}
            history = [{"epoch": i, "training_loss": 1/i, "validation_loss": 1/i,
                        "training_accuracy": .7, "validation_accuracy": .7,
                        "training_macro_f1": .69, "validation_macro_f1": .69} for i in (1, 2)]
            (out / "metrics.json").write_text(json.dumps(metrics))
            (out / "baseline_snapshot.json").write_text(json.dumps(baselines))
            (out / "run_config.json").write_text(json.dumps({"smoke": True}))
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)
            regenerate(out)
            for filename in ("loss_by_epoch.png", "accuracy_by_epoch.png", "macro_f1_by_epoch.png",
                             "confusion_matrix.png", "phase2_comparison.png"):
                self.assertTrue((out / "graphs" / filename).is_file())
            self.assertTrue(json.loads((out / "comparison.json").read_text())["smoke"])


if __name__ == "__main__":
    unittest.main()
