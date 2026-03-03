"""
Small end-to-end test: pick 1 benign + 1 malicious sample and run all stages.

This uses the existing pipeline logic and writes outputs under:
  drive-download-20260219T230708Z-1-001/test_pipeline_small/

Requires a running Ollama server at OLLAMA_URL.
"""

import json
import os
import sys
import time

import numpy as np

import generate_custom_data_to_sentence as s1
import generate_future_stories as s2
import generate_stories_to_array as s3


BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "drive-download-20260219T230708Z-1-001",
)
TEST_DIR = os.path.join(BASE_DIR, "test_pipeline_small")
STORIES_DIR = os.path.join(TEST_DIR, "data_stories")
FUTURE_DIR = os.path.join(TEST_DIR, "data_stories_future")
ARRAYS_DIR = os.path.join(TEST_DIR, "data_future_arrays")

STORIES_JSONL = os.path.join(STORIES_DIR, "stories.jsonl")
FUTURE_JSONL = os.path.join(FUTURE_DIR, "future_stories.jsonl")


def pick_one_each(X, y):
    idx_b = int(np.where(y == 0)[0][0])
    idx_m = int(np.where(y == 1)[0][0])
    return idx_b, idx_m


def main():
    os.makedirs(STORIES_DIR, exist_ok=True)
    os.makedirs(FUTURE_DIR, exist_ok=True)
    os.makedirs(ARRAYS_DIR, exist_ok=True)

    # Load data (2012 + 2014) and pick 1 benign + 1 malicious overall
    X12, y12 = s1.load_xy(s1.DATA_2012)
    X14, y14 = s1.load_xy(s1.DATA_2014)
    X_all = np.vstack([X12, X14])
    y_all = np.concatenate([y12, y14])

    n_features = X_all.shape[1]
    idx_to_name = s1.load_features(s1.FEATURES_JSON, n_features)
    stats = s1.summarize_stats(X_all, y_all)
    delta_rank = np.abs(stats["delta"])

    idx_b, idx_m = pick_one_each(X_all, y_all)
    picks = [idx_b, idx_m]

    # Generate stories.jsonl (2 samples)
    with open(STORIES_JSONL, "w", encoding="utf-8") as f:
        for idx in picks:
            row = X_all[idx]
            label = int(y_all[idx])
            prompt_pairs, n_active, all_pairs = s1.select_prompt_features(
                row, idx_to_name, delta_rank, s1.MAX_FEATURES_IN_PROMPT
            )
            prompt = s1.build_story_prompt(label, prompt_pairs)
            story = ""
            error = None
            try:
                story = s1.ollama_generate(prompt)
            except Exception as exc:
                error = str(exc)

            rec = {
                "year": 2012 if idx < X12.shape[0] else 2014,
                "row_index": int(idx if idx < X12.shape[0] else idx - X12.shape[0]),
                "label": int(label),
                "label_name": "malicious" if label == 1 else "benign",
                "n_active_features": int(n_active),
                "prompt_features": [
                    {"index": i, "name": name} for i, name in prompt_pairs
                ],
                "active_features": [
                    {"index": i, "name": name} for i, name in all_pairs
                ],
                "story": story,
                "error": error,
            }
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    # Stage 2: future stories (2016 + 2018 combined)
    s2.IN_JSONL = STORIES_JSONL
    s2.OUT_DIR = FUTURE_DIR
    s2.OUT_JSONL = FUTURE_JSONL
    s2.OUT_README = os.path.join(FUTURE_DIR, "README_future.md")
    s2.TARGET_YEARS = [2016, 2018]
    s2.main()

    # Stage 3: stories -> arrays
    s3.IN_JSONL = FUTURE_JSONL
    s3.OUT_DIR = ARRAYS_DIR
    s3.OUT_X = os.path.join(ARRAYS_DIR, "X.npy")
    s3.OUT_Y = os.path.join(ARRAYS_DIR, "y.npy")
    s3.OUT_README = os.path.join(ARRAYS_DIR, "README_arrays.md")
    s3.main()

    X_out = np.load(s3.OUT_X, allow_pickle=True)
    y_out = np.load(s3.OUT_Y, allow_pickle=True)
    print(f"Test complete. X_out: {X_out.shape}, y_out: {y_out.shape}")


if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    except Exception as exc:
        print(f"Test failed: {exc}")
        sys.exit(1)
    print(f"Elapsed: {time.time() - t0:.1f}s")
