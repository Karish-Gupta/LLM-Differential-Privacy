import json
import re
import random
from typing import Callable

INPUT_FILE = "samples_with_canaries.json"
OUTPUT_FILE = "canary_check_results.json"

# Paraphraser backends (swap as needed)
def paraphrase_local(text: str) -> str:
    return text.replace("My", "The")

# PII Normalization & Detection
def normalize_text_for_search(s: str) -> str:
    """Lowercase and collapse whitespace; keep punctuation (we'll remove selectively for different PII types)."""
    s = s or ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_pii_for_detection(pii: str) -> str:
    """
    Prepare a canonical form of the PII for substring search.
    - If looks like email (contains @): lowercase, strip whitespace.
    - Else remove all non-alphanumeric characters (so credit card / phone become digit-only).
    """
    pii = (pii or "").strip()
    if "@" in pii:
        return pii.lower()
    return re.sub(r"[^0-9a-zA-Z]", "", pii)

def normalize_text_for_pii_match(text: str, pii_sample: str) -> str:
    """
    Normalize paraphrased text for searching given a particular PII sample.
    If PII sample contains '@' (email), preserve '@' and '.' in the text normalization.
    Otherwise, remove all non-alphanumeric characters from the text for robust numeric/token matching.
    """
    if "@" in pii_sample:
        # keep email separators
        t = text.lower()
        # collapse whitespace
        t = re.sub(r"\s+", " ", t)
        return t
    else:
        # remove non-alphanumeric to match phone/credit card normalized format
        return re.sub(r"[^0-9a-zA-Z]", "", text)

def check_pii_presence(pii_raw: str, pii_normalized: str, paraphrased_text: str) -> dict:
    """
    Returns a dict with:
      - exact_match: True if exact (normalized) PII token is present
      - partial_match: for numeric tokens, checks if a long substring (>= 8 chars) of the pii_normalized exists
      - matched_substrings: list of matched substrings (if any)
    """
    # normalize paraphrased text according to pii type
    parap_norm = normalize_text_for_pii_match(paraphrased_text, pii_raw)

    exact = pii_normalized.lower() in parap_norm.lower()

    matched = []
    partial = False
    # For non-email PII (numbers), also allow partial substrings:
    if not ("@" in pii_raw):
        # check substrings of length >= 8 (configurable)
        L = len(pii_normalized)
        min_sub_len = min(8, max(4, L // 2))  # heuristics: at least 8 or half the length
        # generate overlapping substrings of pii_normalized (digits/letters)
        for i in range(0, max(1, L - min_sub_len + 1)):
            sub = pii_normalized[i:i + min_sub_len]
            if sub and sub.lower() in parap_norm.lower():
                matched.append(sub)
        partial = (len(matched) > 0)
    else:
        # for emails, consider partial matches like local-part or domain
        local, _, domain = pii_normalized.partition("@")
        if local and local.lower() in parap_norm.lower():
            matched.append(local)
        if domain and domain.lower() in parap_norm.lower():
            matched.append(domain)
        partial = (len(matched) > 0) and not exact

    return {
        "exact_match": exact,
        "partial_match": partial,
        "matched_substrings": list(set(matched)),
    }


def run_canary_check(input_file=INPUT_FILE,
                     paraphraser: Callable[[str], str] = paraphrase_local,
                     output_file=OUTPUT_FILE,
                     n=None):
    with open(input_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if n:
        samples = random.sample(samples, min(n, len(samples)))

    results = []
    for s in samples:
        pii_raw = s["inserted_canary"]
        pii_norm = s.get("inserted_canary_normalized", normalize_pii_for_detection(pii_raw))

        # assemble text to paraphrase
        if s.get("messages"):
            text_to_paraphrase = "\n".join([m.get("content", "") for m in s["messages"]])
        else:
            text_to_paraphrase = (s.get("prompt", "") or "") + " " + (s.get("completion", "") or "")

        paraphrased = paraphraser(text_to_paraphrase)

        detection = check_pii_presence(pii_raw, pii_norm, paraphrased)

        result = {
            "index": s.get("index"),
            "inserted_canary": pii_raw,
            "inserted_canary_normalized": pii_norm,
            "exact_match": detection["exact_match"],
            "partial_match": detection["partial_match"],
            "matched_substrings": detection["matched_substrings"],
            "paraphrased_excerpt": paraphrased[:800],
        }

        print("=" * 80)
        print(f"Index: {result['index']}")
        print(f"Inserted Canary (raw): {result['inserted_canary']}")
        print(f"Inserted Canary (norm): {result['inserted_canary_normalized']}")
        print(f"Exact match: {result['exact_match']}")
        print(f"Partial match: {result['partial_match']}")
        if result["matched_substrings"]:
            print("Matched substrings:", result["matched_substrings"])
        print("-" * 80)
        print(result["paraphrased_excerpt"])
        print("=" * 80)

        results.append(result)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = len(results)
    exact = sum(1 for r in results if r["exact_match"])
    partial = sum(1 for r in results if r["partial_match"])
    print(f"\nSaved results -> {output_file}")
    print(f"{exact}/{total} exact matches; {partial}/{total} partial matches")

if __name__ == "__main__":
    run_canary_check(paraphraser=paraphrase_local, n=None)
