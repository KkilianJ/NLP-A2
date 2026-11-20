#!/usr/bin/env python3
"""Compare sampled IMDb excerpts with the published English Europarl tokenizer."""
import argparse
import csv
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True,
                        help="Path to the Europarl tools/tokenizer.perl (keep nonbreaking_prefixes next to it)")
    parser.add_argument("--examples", type=Path, required=True, help="A Phase 1 tokenization_examples.csv")
    parser.add_argument("--output", type=Path, required=True, help="New CSV file")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new file.")
    with args.examples.open(encoding="utf-8", newline="") as f:
        examples = list(csv.DictReader(f))
    rows = []
    for example in examples:
        result = subprocess.run(["perl", str(args.tokenizer.resolve()), "-l", "en", "-q"],
                                input=example["excerpt"] + "\n", text=True, capture_output=True, check=True)
        rows.append({"category": example["category"], "review_id": example["review_id"],
                     "excerpt": example["excerpt"], "europarl_tokens": json.dumps(result.stdout.split(), ensure_ascii=False),
                     "regex_tokens": example["regex"], "regex_br_tokens": example["regex_br"]})
    with args.output.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} observed Europarl comparisons to {args.output}")


if __name__ == "__main__":
    main()
