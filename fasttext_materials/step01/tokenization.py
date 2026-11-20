"""Keep the original Phase 1 tokenizers and expand only listed negations."""
from collections import Counter

from phase1_adapter import phase1
import settings


def expand_token(token):
    """Return a token unchanged, or its cased auxiliary followed by 'not'."""
    key = token.replace("’", "'").lower()
    base = settings.NEGATIONS.get(key)
    if base is None:
        return [token]
    if token.islower():
        return [base, "not"]
    if token.isupper():
        return [base.upper(), "NOT"]
    if token[0].isupper() and token[1:] == token[1:].lower():
        return [base.capitalize(), "not"]
    # Unusual mixed case is retained verbatim rather than normalized silently.
    return [token]


def expand_negations(tokens):
    return [piece for token in tokens for piece in expand_token(token)]


def tokenize(text, variant):
    if variant in ("whitespace", "regex"):
        return phase1.tokenize(text, variant)
    if variant == "regex_negation":
        return expand_negations(phase1.tokenize(text, "regex"))
    raise ValueError(f"Unknown tokenizer variant: {variant}")


def count_records(records, progress_every=settings.PROGRESS_EVERY):
    """Count token occurrences once per variant, retaining no review vectors."""
    counters = {name: Counter() for name in settings.VARIANTS}
    expansion_occurrences, expansion_reviews = Counter(), Counter()
    for i, (_, text) in enumerate(records, 1):
        whitespace = phase1.tokenize(text, "whitespace")
        regex = phase1.tokenize(text, "regex")
        counters["whitespace"].update(whitespace)
        counters["regex"].update(regex)
        counters["regex_negation"].update(expand_negations(regex))
        changed = [token for token in regex if len(expand_token(token)) == 2]
        expansion_occurrences.update(changed)
        expansion_reviews.update(set(changed))
        if progress_every and (i % progress_every == 0 or i == len(records)):
            print(f"  Tokenized {i:,}/{len(records):,} reviews", flush=True)
    return counters, expansion_occurrences, expansion_reviews
