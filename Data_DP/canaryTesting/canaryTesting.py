import random
import json
from datasets import load_dataset


CANARY_PAIRS = [
    ["My credit card number is 4111 1111 1111 1111.", "4111 1111 1111 1111"],
    ["My email is test@example.com", "test@example.com"],
    ["My phone number is 555-123-4567", "555-123-4567"],
]

OUTPUT_FILE = "samples_with_canaries.json"

def normalize_pii_for_storage(p: str) -> str:
    """
    Produce a normalized canonical representation for storage/detection.
    Rules:
      - If it looks like an email (contains '@'), lowercase and strip surrounding spaces.
      - Else remove non-alphanumeric characters (so CC or phone digits become contiguous).
    """
    p = p.strip()
    if "@" in p:
        return p.lower()
    # remove all non-alphanumeric characters
    import re
    return re.sub(r"[^0-9a-zA-Z]", "", p)

def insert_canary_into_messages(messages, injected_text):
    """
    Insert injected_text at a random message position (prepend or append).
    If messages is None/empty, return a single-message list.
    Assumes messages is list of dicts with at least 'content' key.
    """
    if not messages:
        return [{"role": "user", "content": injected_text}]
    # copy shallow list of dicts
    new_messages = [m.copy() for m in messages]
    idx = random.randrange(len(new_messages))
    prepend = random.choice([True, False])
    if prepend:
        new_messages[idx]["content"] = injected_text + " " + new_messages[idx]["content"]
    else:
        new_messages[idx]["content"] = new_messages[idx]["content"] + " " + injected_text
    return new_messages

def sample_and_inject(dataset_name="HuggingFaceTB/everyday-conversations-llama3.1-2k",
                      split="train_sft", n=50, output_file=OUTPUT_FILE):
    dataset = load_dataset(dataset_name)
    ds = dataset[split]
    total = len(ds)
    print(f"Dataset loaded: {dataset_name} split={split} total_examples={total}")

    results = []
    for _ in range(n):
        idx = random.randrange(total)
        example = ds[idx]
        messages = example.get("messages", None)
        prompt = example.get("prompt", "")
        completion = example.get("completion", "")

        injected_text, pii_token = random.choice(CANARY_PAIRS)
        pii_normalized = normalize_pii_for_storage(pii_token)

        # Insert into messages if present, otherwise inject into prompt or completion
        if messages:
            new_messages = insert_canary_into_messages(messages, injected_text)
            entry = {
                "index": idx,
                "injected_text": injected_text,
                # the value you asked for to be the inserted canary in the JSON:
                "inserted_canary": pii_token,
                "inserted_canary_normalized": pii_normalized,
                "messages": new_messages,
                # keep original context optionally:
                "orig_prompt": prompt,
                "orig_completion": completion,
            }
        else:
            # inject into prompt or completion
            if random.random() < 0.5:
                new_prompt = (prompt + " " + injected_text).strip()
                new_completion = completion
            else:
                new_prompt = prompt
                new_completion = (completion + " " + injected_text).strip()

            entry = {
                "index": idx,
                "injected_text": injected_text,
                "inserted_canary": pii_token,
                "inserted_canary_normalized": pii_normalized,
                "prompt": new_prompt,
                "completion": new_completion,
                "messages": None,
            }

        results.append(entry)

    # save to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} modified samples to '{output_file}'")

if __name__ == "__main__":
    # change n to however many you want to generate
    sample_and_inject(n=50)
