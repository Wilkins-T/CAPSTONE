"""
Convert future stories + feature lists into X.npy/y.npy arrays.

Inputs:
  - drive-download-20260219T230708Z-1-001/data_stories_future/future_stories.jsonl
  - selected_features.json

Outputs:
  - drive-download-20260219T230708Z-1-001/data_future_arrays/X.npy
  - drive-download-20260219T230708Z-1-001/data_future_arrays/y.npy
  - drive-download-20260219T230708Z-1-001/data_future_arrays/README_arrays.md
"""

import json
import os
import time
import urllib.request
import sys

import numpy as np


BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "drive-download-20260219T230708Z-1-001",
)
IN_JSONL = os.path.join(BASE_DIR, "data_stories_future", "future_stories.jsonl")
FEATURES_JSON = os.path.join(BASE_DIR, "selected_features.json")
OUT_DIR = os.path.join(BASE_DIR, "data_future_arrays")
OUT_X = os.path.join(OUT_DIR, "X.npy")
OUT_Y = os.path.join(OUT_DIR, "y.npy")
OUT_README = os.path.join(OUT_DIR, "README_arrays.md")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

SLEEP_BETWEEN_REQUESTS = 0.0


def load_feature_maps(features_json_path: str):
    with open(features_json_path, "r") as f:
        name_to_idx = json.load(f)
    idx_to_name = {int(idx): name for name, idx in name_to_idx.items()}
    name_to_idx = {name: int(idx) for name, idx in name_to_idx.items()}
    return idx_to_name, name_to_idx


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _progress(n, total, elapsed_s, avg_tps):
    width = 30
    frac = 1.0 if total == 0 else n / total
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    pct = int(frac * 100)
    if n > 0 and elapsed_s > 0:
        rate = n / elapsed_s
        remaining = (total - n) / rate if rate > 0 else 0
    else:
        remaining = 0
    mins = int(remaining // 60)
    secs = int(remaining % 60)
    tps = f"{avg_tps:.1f}" if avg_tps is not None else "NA"
    sys.stdout.write(
        f"\r[{bar}] {n}/{total} ({pct}%) ETA {mins:02d}:{secs:02d} | avg tok/s {tps}"
    )
    sys.stdout.flush()


def ollama_generate(prompt: str):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 4096},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "").strip()


def _extract_json_object(text: str):
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    candidate = _extract_json_object(text)
    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return None


def build_prompt(rec):
    story = rec.get("future_story", "")
    features = rec.get("future_active_features", [])
    lines = [f"{f.get('index')} | {f.get('name')}" for f in features]
    feature_block = "\n".join(lines) if lines else "(no active features)"

    prompt = f"""
You are mapping a malware analysis story back to feature activations.

Story:
{story}

Candidate active features (index | name):
{feature_block}

Task:
Return the subset of candidate features that should be active based on the story.

Rules:
- Do not invent features not in the candidate list.
- Output ONLY valid JSON.

JSON schema:
{{
  "active_feature_indices": [int]
}}
"""
    return prompt.strip()


def validate_indices(idx_list, idx_to_name):
    out = []
    for idx in idx_list or []:
        try:
            idx = int(idx)
        except ValueError:
            continue
        if idx in idx_to_name:
            out.append(idx)
    # de-dup and sort for stable arrays
    return sorted(set(out))


def ensure_non_empty_indices(primary, fallback, idx_to_name):
    cleaned = validate_indices(primary, idx_to_name)
    if cleaned:
        return cleaned, None
    cleaned = validate_indices(fallback, idx_to_name)
    if cleaned:
        return cleaned, "empty_primary_indices_fallback_to_candidates"
    return [], "empty_indices_no_valid_fallback"


def _load_checkpoint(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_checkpoint(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def main():
    if not os.path.isfile(IN_JSONL):
        raise FileNotFoundError(f"Missing input: {IN_JSONL}")

    idx_to_name, _name_to_idx = load_feature_maps(FEATURES_JSON)
    n_features = len(idx_to_name)

    records = list(read_jsonl(IN_JSONL))
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUT_DIR, "checkpoint.json")
    ckpt = _load_checkpoint(ckpt_path) or {}

    n_records = len(records)
    if ckpt.get("n_records") and ckpt["n_records"] != n_records:
        raise RuntimeError("Checkpoint does not match current input size.")
    if ckpt.get("n_features") and ckpt["n_features"] != n_features:
        raise RuntimeError("Checkpoint does not match current feature size.")

    # Use memmap so we can resume safely
    if os.path.isfile(OUT_X) and os.path.isfile(OUT_Y):
        X = np.lib.format.open_memmap(OUT_X, mode="r+", dtype=np.int64, shape=(n_records, n_features))
        y = np.lib.format.open_memmap(OUT_Y, mode="r+", dtype=np.int64, shape=(n_records,))
    else:
        X = np.lib.format.open_memmap(OUT_X, mode="w+", dtype=np.int64, shape=(n_records, n_features))
        y = np.lib.format.open_memmap(OUT_Y, mode="w+", dtype=np.int64, shape=(n_records,))

    start_idx = int(ckpt.get("next_index", 0))

    t0 = time.time()
    total_chars = 0
    for i, rec in enumerate(records[start_idx:], start=start_idx):
        label = int(rec.get("label", 0))
        y[i] = label

        prompt = build_prompt(rec)
        raw = ""
        parsed = None
        error = None
        try:
            raw = ollama_generate(prompt)
            parsed = parse_json(raw)
        except Exception as exc:
            error = str(exc)
        total_chars += len(prompt) + len(raw)

        active_indices = []
        if parsed:
            active_indices = parsed.get("active_feature_indices", [])

        # fallback: use all candidate features
        candidates = rec.get("future_active_features", [])
        fallback_indices = [c.get("index") for c in candidates]
        active_indices, validate_note = ensure_non_empty_indices(
            active_indices, fallback_indices, idx_to_name
        )
        if not active_indices:
            # Allow empty if there were no candidate features.
            if not fallback_indices:
                validate_note = validate_note or "empty_candidates_allowed"
            else:
                error = error or "no_valid_indices_after_validation"
                raise RuntimeError(
                    f"No valid indices for row_index={rec.get('row_index')}"
                )

        X[i, active_indices] = 1

        if (i + 1) % 5 == 0 or (i + 1) == len(records):
            elapsed = time.time() - t0
            avg_tps = (total_chars / 4) / elapsed if elapsed > 0 else None
            _progress(i + 1, len(records), elapsed, avg_tps)
            _save_checkpoint(
                ckpt_path,
                {
                    "n_records": n_records,
                    "n_features": n_features,
                    "next_index": i + 1,
                },
            )
        if SLEEP_BETWEEN_REQUESTS:
            time.sleep(SLEEP_BETWEEN_REQUESTS)
    if len(records) > 0:
        sys.stdout.write("\n")

    # Ensure checkpoint marks completion
    _save_checkpoint(
        ckpt_path,
        {
            "n_records": n_records,
            "n_features": n_features,
            "next_index": n_records,
        },
    )

    with open(OUT_README, "w", encoding="utf-8") as f:
        f.write("# Future Arrays\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"- Input: {IN_JSONL}\n")
        f.write(f"- Output X: {OUT_X}\n")
        f.write(f"- Output y: {OUT_Y}\n")
        f.write(f"- Model: {OLLAMA_MODEL}\n")
        f.write(f"- Features: {n_features}\n")

    print(f"Wrote: {OUT_X}")
    print(f"Wrote: {OUT_Y}")
    print(f"Wrote: {OUT_README}")


if __name__ == "__main__":
    main()
