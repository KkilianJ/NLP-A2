"""Synthetic-only checks: no real IMDb data, pretrained binary, or training."""
from collections import Counter
from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from examples import expansion_rows, select_examples
from measurement import measure_coverage
from phase1_adapter import phase1, phase1_config
import run
import settings
from tokenization import count_records, expand_token, tokenize


class FakeFastText:
    def __init__(self, dimension=3):
        self.dimension = dimension
        self.vector_calls = []
        self.f = SimpleNamespace(getArgs=lambda: SimpleNamespace(minn=5, maxn=5))

    def get_dimension(self):
        return self.dimension

    def get_word_id(self, token):
        return 1 if token in {"known", "not"} else -1

    def get_word_vector(self, token):
        self.vector_calls.append(token)
        if token == "?":
            return np.zeros(self.dimension, dtype=np.float32)
        if token == "tiny":
            return np.full(self.dimension, 1e-30, dtype=np.float32)
        return np.ones(self.dimension, dtype=np.float32)


class TokenizationTests(unittest.TestCase):
    def test_original_variants_are_unchanged(self):
        text = "I can't!<br /> Don't stop."
        for name in ("whitespace", "regex"):
            self.assertEqual(tokenize(text, name), phase1.tokenize(text, name))
        self.assertEqual(tokenize(text, "regex"),
                         ["I", "can't", "!", "<", "br", "/", ">", "Don't", "stop", "."])
        self.assertEqual(tokenize(text, "regex_negation"),
                         ["I", "can", "not", "!", "<", "br", "/", ">", "Do", "not", "stop", "."])

    def test_common_negations_casing_and_apostrophes(self):
        expected = {"don't": ["do", "not"], "isn't": ["is", "not"], "wasn't": ["was", "not"],
                    "can't": ["can", "not"], "won't": ["will", "not"], "CAN’T": ["CAN", "NOT"],
                    "Don’t": ["Do", "not"], "Won't": ["Will", "not"]}
        for original, expanded in expected.items():
            with self.subTest(original=original):
                self.assertEqual(expand_token(original), expanded)

    def test_ambiguous_multiple_and_mixed_case_are_not_guessed(self):
        for token in ("ain't", "he'd", "wouldn't've", "Don't've", "dOn'T", "unknownn't", "cannot"):
            with self.subTest(token=token):
                self.assertEqual(expand_token(token), [token])
        self.assertEqual(tokenize("wouldn't've", "regex_negation"), ["wouldn't've"])
        with self.assertRaises(ValueError):
            tokenize("text", "unknown")

    def test_counts_and_expansion_review_counts(self):
        records = [(1, "don't don't like it"), (2, "Don't like it"), (3, "isn't good")]
        counters, occurrences, reviews = count_records(records, progress_every=0)
        self.assertEqual(occurrences, Counter({"don't": 2, "Don't": 1, "isn't": 1}))
        self.assertEqual(reviews, Counter({"don't": 1, "Don't": 1, "isn't": 1}))
        self.assertEqual(counters["regex_negation"]["not"], 4)
        self.assertEqual(sum(counters["regex_negation"].values()), sum(counters["regex"].values()) + 4)
        self.assertEqual(json.loads(expansion_rows(occurrences, reviews)[0]["expanded_tokens"]), ["do", "not"])

    def test_examples_are_deterministic_and_verbatim(self):
        records = [(100, "I don't like it, and it isn't good!<br />"),
                   (200, "Can't say I won't watch; it wasn't boring."), (300, "It ain't terrible.")]
        rows = select_examples(records, seed=42)
        self.assertEqual(rows, select_examples(records, seed=42))
        self.assertEqual({r["category"] for r in rows},
                         {"don't", "isn't", "wasn't", "can't", "won't", "ambiguous_unexpanded", "punctuation", "html_break"})
        texts = dict(records)
        for row in rows:
            self.assertIn(row["excerpt"], texts[row["review_id"]])
            for name in settings.VARIANTS:
                self.assertEqual(json.loads(row[name]), tokenize(row["excerpt"], name))


class CoverageTests(unittest.TestCase):
    def test_direct_membership_is_separate_from_nonzero_oov(self):
        model = FakeFastText()
        counters = {"a": Counter({"known": 3, "tiny": 2, "missing": 4, "?": 1}),
                    "b": Counter({"known": 2, "tiny": 1})}
        rows, status = measure_coverage(model, counters, progress_every=0)
        a, b = rows
        self.assertEqual((a["token_occurrences"], a["token_types"]), (10, 4))
        self.assertEqual((a["direct_token_occurrences"], a["direct_token_types"]), (3, 1))
        self.assertEqual((a["oov_token_occurrences"], a["oov_token_types"]), (7, 3))
        self.assertEqual((a["oov_nonzero_occurrences"], a["oov_zero_occurrences"]), (6, 1))
        self.assertAlmostEqual(a["oov_nonzero_occurrence_fraction"], 6 / 7)
        self.assertEqual(status["tiny"], (False, True))  # exact nonzero, even below the Phase 1 norm threshold
        self.assertNotIn("known", model.vector_calls)
        self.assertEqual(Counter(model.vector_calls), Counter({"tiny": 1, "missing": 1, "?": 1}))
        self.assertEqual(b["direct_token_occurrences"], 2)

    def test_empty_and_no_oov_denominators_are_undefined(self):
        rows, _ = measure_coverage(FakeFastText(), {"empty": Counter(), "direct": Counter({"known": 2})}, 0)
        self.assertIsNone(rows[0]["direct_token_fraction"])
        self.assertIsNone(rows[0]["oov_nonzero_occurrence_fraction"])
        self.assertEqual(rows[1]["direct_token_fraction"], 1.0)
        self.assertIsNone(rows[1]["oov_nonzero_occurrence_fraction"])

    def test_nonfinite_and_wrong_shape_vectors_are_rejected(self):
        model = FakeFastText()
        with patch.object(model, "get_word_vector", return_value=np.array([np.nan, 0, 0])):
            with self.assertRaisesRegex(ValueError, "Invalid fastText vector"):
                measure_coverage(model, {"a": Counter({"bad": 1})}, 0)
        with patch.object(model, "get_word_vector", return_value=np.array([1.0])):
            with self.assertRaisesRegex(ValueError, "Invalid fastText vector"):
                measure_coverage(model, {"a": Counter({"bad": 1})}, 0)


class ProtocolTests(unittest.TestCase):
    def test_project_scope_uses_existing_split_loader(self):
        records = [(5, "synthetic"), (7, "text")]
        info = {"full_training_indices_sha256_le_int64": settings.EXPECTED_TRAIN_IDS_SHA256}
        with patch.object(settings, "EXPECTED_TRAIN_COUNT", 2), \
             patch.object(phase1, "load_training_partition", return_value=(records, info)) as loader:
            got, meta = run.load_records(Path("unused"), "project-train")
        loader.assert_called_once_with(Path("unused"), smoke=False)
        self.assertEqual(got, records)
        self.assertFalse(meta["held_out_analysis"])

    def test_source_scope_reads_text_only_and_labels_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "train.parquet"
            source.write_bytes(b"synthetic file")
            with patch.object(settings, "EXPECTED_SOURCE_TRAIN_COUNT", 2), \
                 patch.object(phase1, "file_sha256", return_value=phase1_config.SOURCE_SHA256["train.parquet"]), \
                 patch.object(pd, "read_parquet", return_value=pd.DataFrame({"text": ["one", "two"]})) as reader:
                records, meta = run.load_records(Path(temp), "published-train-descriptive")
        reader.assert_called_once_with(source, columns=["text"])
        self.assertEqual(records, [(0, "one"), (1, "two")])
        self.assertTrue(meta["held_out_analysis"])
        self.assertIn("Do not use", meta["warning"])

    def test_optional_source_requires_ack_and_existing_outputs_are_protected(self):
        with redirect_stderr(io.StringIO()), patch.object(run, "load_records", side_effect=AssertionError("should not load")):
            with self.assertRaises(SystemExit) as error:
                run.main(["--scope", "published-train-descriptive"])
            self.assertEqual(error.exception.code, 2)
            with tempfile.TemporaryDirectory() as temp:
                existing = Path(temp) / "existing"
                existing.mkdir()
                marker = existing / "result.txt"
                marker.write_text("untouched")
                with self.assertRaises(SystemExit):
                    run.main(["--output", str(existing)])
                self.assertEqual(marker.read_text(), "untouched")

    def test_model_fingerprint_cache_is_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "synthetic.bin"
            model.write_bytes(b"synthetic model")
            sha = hashlib.sha256(model.read_bytes()).hexdigest()
            st = model.stat()
            cache = root / "model_fingerprint.json"
            cache.write_text(json.dumps({"path": str(model.resolve()), "size": st.st_size,
                                         "mtime_ns": st.st_mtime_ns, "sha256": sha}))
            before = cache.read_bytes()
            with patch.object(phase1_config, "CACHE_DIR", root), patch.object(phase1_config, "MODEL_SHA256", sha), \
                 patch.object(phase1, "file_sha256", side_effect=AssertionError("cache should be used")):
                got, method = run.model_sha256(model)
            self.assertEqual(got, sha)
            self.assertIn("read-only", method)
            self.assertEqual(cache.read_bytes(), before)

    def test_end_to_end_uses_only_synthetic_reviews_and_fake_model(self):
        records = [(1, "known don't!<br />"), (2, "Can't say it isn't known.")]
        fake = FakeFastText(dimension=300)
        fake_fasttext = SimpleNamespace(load_model=lambda path: fake)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "fake.bin"
            model.write_bytes(b"fake")
            output = root / "results"
            with patch.object(run, "load_records", return_value=(records, {"synthetic": True})), \
                 patch.object(run, "model_sha256", return_value=("synthetic-digest", "mock")), \
                 patch.dict(sys.modules, {"fasttext": fake_fasttext}), patch("builtins.print"):
                run.main(["--model", str(model), "--output", str(output)])
            self.assertEqual({p.name for p in output.iterdir()},
                             {"coverage.csv", "coverage.json", "tokenization_examples.csv", "negation_expansions.csv",
                              "run_metadata.json", "coverage.png"})
            data = json.loads((output / "coverage.json").read_text())
            self.assertEqual([r["variant"] for r in data], list(settings.VARIANTS))
            self.assertEqual(json.loads((output / "run_metadata.json").read_text())["scope"], "project-train")
            expansions = pd.read_csv(output / "negation_expansions.csv")
            self.assertEqual(set(expansions["original_token"]), {"don't", "Can't", "isn't"})
            examples = pd.read_csv(output / "tokenization_examples.csv")
            self.assertTrue(set(examples["review_id"]).issubset({1, 2}))


if __name__ == "__main__":
    unittest.main()
