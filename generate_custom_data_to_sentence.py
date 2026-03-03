"""
Generate story-style explanations for each sample in X.npy/y.npy (2012 + 2014).

Inputs:
  - data_2012/X.npy, y.npy
  - data_2014/X.npy, y.npy
  - selected_features.json (optional, for human-readable names)

Outputs (default):
  - drive-download-20260219T230708Z-1-001/data_stories/stories.jsonl
  - drive-download-20260219T230708Z-1-001/data_stories/README_stories.md
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
DATA_2012 = os.path.join(BASE_DIR, "data_2012")
DATA_2014 = os.path.join(BASE_DIR, "data_2014")
FEATURES_JSON = os.path.join(BASE_DIR, "selected_features.json")
OUT_DIR = os.path.join(BASE_DIR, "data_stories")
OUT_JSONL = os.path.join(OUT_DIR, "stories.jsonl")
OUT_README = os.path.join(OUT_DIR, "README_stories.md")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

YEARS = [2012, 2014]
RNG_SEED = 42

# Prompt controls
# Set to None to include all active features in the prompt
MAX_FEATURES_IN_PROMPT = None
SLEEP_BETWEEN_REQUESTS = 0.0

# Sampling controls (set to None to process all)
MAX_SAMPLES_PER_YEAR = None
LIMIT_PER_CLASS = None
MAX_SAMPLES_TOTAL = None

PROB_SMOOTHING = 0.5  # Laplace smoothing to avoid 0/1 probabilities


def load_features(features_json_path: str, n_features: int):
    if not os.path.isfile(features_json_path):
        return [f"feature_{i}" for i in range(n_features)]
    with open(features_json_path, "r") as f:
        name_to_idx = json.load(f)
    idx_to_name = [f"feature_{i}" for i in range(n_features)]
    for name, idx in name_to_idx.items():
        try:
            idx = int(idx)
        except ValueError:
            continue
        if 0 <= idx < n_features:
            idx_to_name[idx] = name
    return idx_to_name


def load_xy(data_dir: str):
    X = np.load(os.path.join(data_dir, "X.npy"), allow_pickle=True)
    y = np.load(os.path.join(data_dir, "y.npy"), allow_pickle=True)
    if y.ndim > 1:
        y = y.ravel()
    return X.astype(np.int64), y.astype(np.int64)


def summarize_stats(X, y):
    benign = X[y == 0]
    malicious = X[y == 1]
    n_b = benign.shape[0]
    n_m = malicious.shape[0]
    count_b = benign.sum(axis=0)
    count_m = malicious.sum(axis=0)
    p_b = (count_b + PROB_SMOOTHING) / (n_b + 2 * PROB_SMOOTHING)
    p_m = (count_m + PROB_SMOOTHING) / (n_m + 2 * PROB_SMOOTHING)
    delta = p_m - p_b
    return {
        "p_b": p_b,
        "p_m": p_m,
        "delta": delta,
    }


def choose_indices(y, rng: np.random.Generator):
    idx = np.arange(len(y))
    if LIMIT_PER_CLASS is not None:
        picked = []
        for label in (0, 1):
            pool = np.where(y == label)[0]
            if LIMIT_PER_CLASS < pool.size:
                pool = rng.choice(pool, size=LIMIT_PER_CLASS, replace=False)
            picked.append(pool)
        idx = np.concatenate(picked)
    if MAX_SAMPLES_PER_YEAR is not None and MAX_SAMPLES_PER_YEAR < idx.size:
        idx = rng.choice(idx, size=MAX_SAMPLES_PER_YEAR, replace=False)
    if MAX_SAMPLES_TOTAL is not None and MAX_SAMPLES_TOTAL < idx.size:
        idx = rng.choice(idx, size=MAX_SAMPLES_TOTAL, replace=False)
    return np.sort(idx)


def select_prompt_features(row, idx_to_name, delta_rank, max_features):
    active = np.where(row == 1)[0]
    n_active = int(active.size)
    if n_active == 0:
        return [], n_active, []
    all_pairs = [(int(i), idx_to_name[int(i)]) for i in active]
    prompt_active = active
    if max_features is not None and active.size > max_features:
        prompt_active = sorted(active, key=lambda i: delta_rank[i], reverse=True)[:max_features]
    prompt_pairs = [(int(i), idx_to_name[int(i)]) for i in prompt_active]
    return prompt_pairs, n_active, all_pairs


def build_story_prompt(label, pairs):
    label_name = "malicious" if int(label) == 1 else "benign"
    lines = [f"{idx} | {name}" for idx, name in pairs]
    feature_block = "\n".join(lines) if lines else "(no active features)"

    prompt = f"""
You are analyzing an Android app sample using static binary features.

Ground truth label: {label_name}.

Active features (index | name):
{feature_block}

Task:
Write a story of whatever length is needed to accurately explain why this sample is likely {label_name}.

Rules:
- Use neutral, forensic language with uncertainty when appropriate.
- Avoid step-by-step instructions, code, or operational guidance.
- Mention what is being requested, what is being used, and what API calls or services are implied by the features when relevant.
- Do not list the features verbatim; synthesize them into a narrative.
- Output plain text only (no bullets, no JSON, no headings).
"""
    return prompt.strip()


def ollama_generate(prompt: str):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_ctx": 4096},
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


def count_existing_lines(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


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


def main():
    rng = np.random.default_rng(RNG_SEED)

    # Load data
    X12, y12 = load_xy(DATA_2012)
    X14, y14 = load_xy(DATA_2014)
    X_all = np.vstack([X12, X14])
    y_all = np.concatenate([y12, y14])

    n_features = X_all.shape[1]
    idx_to_name = load_features(FEATURES_JSON, n_features)
    stats = summarize_stats(X_all, y_all)
    delta_rank = np.abs(stats["delta"])

    # Build sample list in a stable order
    samples = []
    for year, (X, y) in ((2012, (X12, y12)), (2014, (X14, y14))):
        idx = choose_indices(y, rng)
        for i in idx:
            samples.append((year, int(i)))

    os.makedirs(OUT_DIR, exist_ok=True)
    start_at = count_existing_lines(OUT_JSONL)

    total = len(samples)
    if start_at >= total:
        print(f"Nothing to do. {OUT_JSONL} already has {start_at} lines.")
        return

    t0 = time.time()
    total_chars = 0
    with open(OUT_JSONL, "a", encoding="utf-8") as f:
        for n, (year, i) in enumerate(samples[start_at:], start=start_at + 1):
            X = X12 if year == 2012 else X14
            y = y12 if year == 2012 else y14
            row = X[i]
            label = int(y[i])
            pairs, n_active, all_pairs = select_prompt_features(
                row, idx_to_name, delta_rank, MAX_FEATURES_IN_PROMPT
            )

            prompt = build_story_prompt(label, pairs)
            story = ""
            error = None
            try:
                story = ollama_generate(prompt)
            except Exception as exc:
                error = str(exc)
            total_chars += len(prompt) + len(story)

            rec = {
                "year": int(year),
                "row_index": int(i),
                "label": int(label),
                "label_name": "malicious" if label == 1 else "benign",
                "n_active_features": int(n_active),
                "prompt_features": [
                    {"index": idx, "name": name} for idx, name in pairs
                ],
                "active_features": [
                    {"index": idx, "name": name} for idx, name in all_pairs
                ],
                "story": story,
                "error": error,
            }
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

            if n % 5 == 0 or n == total:
                elapsed = time.time() - t0
                avg_tps = (total_chars / 4) / elapsed if elapsed > 0 else None
                _progress(n, total, elapsed, avg_tps)
            if SLEEP_BETWEEN_REQUESTS:
                time.sleep(SLEEP_BETWEEN_REQUESTS)
    if total > 0:
        sys.stdout.write("\n")

    # README
    with open(OUT_README, "w", encoding="utf-8") as f:
        f.write("# Story Dataset\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Source Data\n")
        f.write(f"- Years: {', '.join(str(y) for y in YEARS)}\n")
        f.write(f"- Samples: 2012={X12.shape[0]}, 2014={X14.shape[0]}\n")
        f.write(f"- Features per sample: {X12.shape[1]}\n\n")
        f.write("## Generation Settings\n")
        f.write(f"- Model: {OLLAMA_MODEL}\n")
        f.write(f"- Max features in prompt: {MAX_FEATURES_IN_PROMPT}\n")
        f.write(f"- MAX_SAMPLES_PER_YEAR: {MAX_SAMPLES_PER_YEAR}\n")
        f.write(f"- LIMIT_PER_CLASS: {LIMIT_PER_CLASS}\n")
        f.write(f"- MAX_SAMPLES_TOTAL: {MAX_SAMPLES_TOTAL}\n")
        f.write(f"- Output: {OUT_JSONL}\n")

    print(f"Wrote: {OUT_JSONL}")
    print(f"Wrote: {OUT_README}")


if __name__ == "__main__":
    main()
