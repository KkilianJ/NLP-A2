"""Checks for split integrity, tokenization, membership accounting, and edits."""
import unittest
from collections import Counter
import numpy as np
import config as cfg
import run_phase1 as exp


class FakeFastText:
    def get_word_id(self, word):
        return 7 if word == "known" else -1

    def get_word_vector(self, word):
        return np.array([0., 0.]) if word == "?" else np.array([1., 0.])


class MethodChecks(unittest.TestCase):
    def test_tokenization_and_html_are_separate_choices(self):
        text = "I can't!<br /> Movie-goers"
        self.assertEqual(exp.tokenize(text, "whitespace"), ["I", "can't!<br", "/>", "Movie-goers"])
        self.assertEqual(exp.tokenize(text, "regex"),
                         ["I", "can't", "!", "<", "br", "/", ">", "Movie", "-", "goers"])
        self.assertEqual(exp.tokenize(text, "regex_br"), ["I", "can't", "!", "Movie", "-", "goers"])
        self.assertEqual(exp.tokenize(text, "regex_br_lower"), ["i", "can't", "!", "movie", "-", "goers"])

    def test_direct_membership_and_nonzero_oov_are_distinct(self):
        rows, _ = exp.membership_and_coverage(FakeFastText(), {"example": Counter({"known": 3, "missing": 2, "?": 1})})
        row = rows[0]
        self.assertEqual((row["tokens"], row["types"]), (6, 3))
        self.assertEqual((row["direct_tokens"], row["direct_types"]), (3, 1))
        self.assertEqual((row["oov_nonzero_tokens"], row["oov_nonzero_types"]), (2, 1))
        self.assertEqual((row["oov_zero_tokens"], row["oov_zero_types"]), (1, 1))

    def test_capitalization_transition_counts_tokens_and_original_types(self):
        counter = Counter({"Name": 4, "WORD": 2, "same": 8})
        status = {"Name": (True, None), "name": (False, 1.),
                  "WORD": (False, 1.), "word": (True, None), "same": (True, None)}
        rows, examples = exp.capitalization_transitions(counter, status)
        by_label = {row["transition"]: row for row in rows}
        self.assertEqual(by_label["direct -> oov"]["tokens"], 4)
        self.assertEqual(by_label["oov -> direct"]["tokens"], 2)
        self.assertEqual(sum(row["original_types"] for row in rows), 2)
        self.assertEqual({row["original"] for row in examples}, {"Name", "WORD"})

    def test_excerpt_is_an_exact_whole_word_substring(self):
        text = "Here is an excellent film with vivid performances."
        i = text.index("excellent")
        sample = exp.excerpt(text, i, i + len("excellent"), radius=3)
        self.assertEqual(sample, "an excellent film")
        self.assertIn(sample, text)

    def test_single_edit_candidates_and_deterministic_perturbations(self):
        edits = exp.edit_one("film")
        self.assertTrue({"fil", "films", "fllm", "flim"}.issubset(edits))
        self.assertNotIn("film", edits)
        for kind in ("deletion", "insertion", "substitution", "transposition", "repetition"):
            candidates = exp.typo_candidates("excellent", kind)
            self.assertTrue(candidates)
            self.assertTrue(all(c in exp.edit_one("excellent") for c in candidates))
            self.assertEqual(candidates, exp.typo_candidates("excellent", kind))

    @unittest.skipUnless(all((cfg.DATA_DIR / name).is_file() for name in cfg.SOURCE_SHA256), "IMDb files absent")
    def test_previous_training_split_and_smoke_selection(self):
        records, info = exp.load_training_partition(cfg.DATA_DIR, smoke=True)
        self.assertEqual(info["project_split_sizes"], {"training": 34707, "validation": 7437, "test": 7438})
        self.assertEqual(info["full_training_indices_sha256_le_int64"],
                         "3fe569c54add383b4b7249ea3465297678599f30dc8c0d48d43ee7186f09ec82")
        self.assertEqual(len(records), cfg.SMOKE_REVIEWS)
        self.assertEqual(len({review_id for review_id, _ in records}), len(records))


if __name__ == "__main__":
    unittest.main()
