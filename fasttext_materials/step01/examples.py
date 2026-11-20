"""Select reproducible, verbatim review excerpts without looking at labels."""
import json
import random
import re

from phase1_adapter import phase1
import settings
from tokenization import expand_token, tokenize


def find_example(text, category):
    if category == "html_break":
        return phase1.BREAK_RE.search(text)
    if category == "punctuation":
        return re.search(r"[,!?;:]", text)
    for match in phase1.WORD_RE.finditer(text):
        token = match.group()
        key = token.replace("’", "'").lower()
        if category in settings.EXAMPLE_NEGATIONS and key == category and len(expand_token(token)) == 2:
            return match
        if category == "ambiguous_unexpanded" and key == "ain't":
            return match
    return None


def select_examples(records, seed=settings.SEED):
    order = list(records)
    random.Random(seed).shuffle(order)
    categories = (*settings.EXAMPLE_NEGATIONS, "ambiguous_unexpanded", "punctuation", "html_break")
    rows = []
    for category in categories:
        for review_id, text in order:
            match = find_example(text, category)
            if match is None:
                continue
            excerpt = phase1.excerpt(text, match.start(), match.end(), radius=settings.EXCERPT_RADIUS)
            if excerpt not in text:
                raise ValueError("The selected review excerpt is not verbatim.")
            row = {"category": category, "review_id": int(review_id),
                   "matched_text": match.group(), "excerpt": excerpt}
            row.update({variant: json.dumps(tokenize(excerpt, variant), ensure_ascii=False)
                        for variant in settings.VARIANTS})
            rows.append(row)
            break
    return rows


def expansion_rows(occurrences, reviews):
    return [{"original_token": token,
             "expanded_tokens": json.dumps(expand_token(token), ensure_ascii=False),
             "occurrences": occurrences[token], "reviews_with_token": reviews[token]}
            for token in sorted(occurrences, key=lambda item: (-occurrences[item], item))]
