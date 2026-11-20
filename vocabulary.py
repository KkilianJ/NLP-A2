"""Compare IMDb tokenization schemes against a GloVe vocabulary."""
import argparse
from collections import Counter
from html import unescape
import json
from pathlib import Path
import re
from zipfile import ZipFile

TAG = re.compile(r"<[^>]+>")
TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)*|[^\w\s]", re.UNICODE)


def clean(text):
    return TAG.sub(" ", unescape(text)).lower().replace("’", "'")


def lexical_tokens(tokens):
    return [token for token in tokens if any(char.isalnum() for char in token)]


def expand_negation(tokens, expanded):
    words = []
    irregular = {"can't": "can", "won't": "will", "shan't": "shall"}
    for token in tokens:
        if token.endswith("n't") and len(token) > 3 and token != "ain't":
            words.extend((irregular.get(token, token[:-3]), "not"))
            expanded[token] += 1
        else:
            words.append(token)
    return words


def read_glove(path):
    if path.suffix == ".zip":
        with ZipFile(path) as archive:
            with archive.open("glove.6B.100d.txt") as lines:
                return {line.split(maxsplit=1)[0].decode("utf-8") for line in lines if line.strip()}
    with path.open("rb") as lines:
        return {line.split(maxsplit=1)[0].decode("utf-8") for line in lines if line.strip()}


def analyze(texts, glove):
    counters = {name: Counter() for name in ("whitespace", "regex", "regex_expand_negation")}
    expanded = Counter()
    n_reviews = 0
    example = None
    for text in texts:
        cleaned = clean(text)
        tokens = {
            "whitespace": lexical_tokens(cleaned.split()),
            "regex": lexical_tokens(TOKEN.findall(cleaned)),
        }
        tokens["regex_expand_negation"] = expand_negation(tokens["regex"], expanded)
        if example is None:
            example = {name: words[:30] for name, words in tokens.items()}
        for name, words in tokens.items():
            counters[name].update(words)
        n_reviews += 1

    schemes = {}
    for name, counts in counters.items():
        missing = Counter({word: count for word, count in counts.items() if word not in glove})
        n_tokens = counts.total()
        n_types = len(counts)
        covered_tokens = n_tokens - missing.total()
        covered_types = n_types - len(missing)
        schemes[name] = {
            "tokens": n_tokens,
            "types": n_types,
            "covered_tokens": covered_tokens,
            "covered_types": covered_types,
            "token_coverage_percent": 100 * covered_tokens / n_tokens if n_tokens else None,
            "type_coverage_percent": 100 * covered_types / n_types if n_types else None,
            "top_20_missing": [{"word": word, "count": count} for word, count in missing.most_common(20)],
        }
    return {
        "reviews": n_reviews,
        "example_first_30_tokens": example,
        "expanded_negation_occurrences": expanded.total(),
        "expanded_negation_by_form": dict(expanded.most_common()),
        "schemes": schemes,
    }


def main():
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glove", type=Path, required=True, help="GloVe 6B ZIP or extracted text file")
    parser.add_argument("--output", type=Path, default=Path("vocabulary_results.json"))
    args = parser.parse_args()
    if not args.glove.is_file():
        parser.error(f"GloVe file does not exist: {args.glove}")
    glove = read_glove(args.glove)
    dataset = load_dataset("mteb/imdb", split="train")
    result = analyze((row["text"] for row in dataset), glove)
    result.update(dataset="mteb/imdb", split="train", glove_file=args.glove.name, glove_types=len(glove))
    report = json.dumps(result, ensure_ascii=False, indent=2)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
