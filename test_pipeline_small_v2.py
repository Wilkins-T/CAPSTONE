"""
Small end-to-end V2 test: build tiny story-only datasets, generate future stories,
convert to dense embeddings, and validate output shapes.

Default: 20 benign + 20 malicious per year (2012/2014/2016/2018) for stage-1 stories.
"""

import argparse
import json
import os
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np

import generate_year_stories_v2 as s1
import generate_future_stories_v2 as s2
import stories_to_embeddings_v2 as s3


BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "drive-download-20260219T230708Z-1-001",
)
TEST_DIR = os.path.join(BASE_DIR, "test_pipeline_small_v2")
STORIES_DIR = os.path.join(TEST_DIR, "data_stories_v2")
FUTURE_DIR = os.path.join(TEST_DIR, "data_stories_future_v2")
EMB_DIR = os.path.join(TEST_DIR, "data_story_embeddings_v2")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--per-class", type=int, default=20, help="Benign and malicious samples per year")
    p.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    return p.parse_args()


def run_module_main(main_fn, argv):
    old = sys.argv[:]
    try:
        sys.argv = [old[0], *argv]
        main_fn()
    finally:
        sys.argv = old


def _progress(n, total, elapsed, prefix=""):
    width = 30
    frac = 1.0 if total == 0 else n / total
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    eta = (total - n) / (n / elapsed) if n > 0 and elapsed > 0 else 0
    lead = f"{prefix} " if prefix else ""
    sys.stdout.write(f"\r{lead}[{bar}] {n}/{total} ETA {int(eta//60):02d}:{int(eta%60):02d}")
    sys.stdout.flush()


def pick_n_each(y, n_each):
    idx_b = np.where(y == 0)[0]
    idx_m = np.where(y == 1)[0]
    if len(idx_b) < n_each or len(idx_m) < n_each:
        raise RuntimeError(
            f"Not enough samples for per-class={n_each}: benign={len(idx_b)}, malicious={len(idx_m)}"
        )
    picked = np.concatenate([idx_b[:n_each], idx_m[:n_each]])
    return [int(i) for i in np.sort(picked)]


def build_small_stage1_stories(per_class):
    os.makedirs(STORIES_DIR, exist_ok=True)

    # names only used in prompt context; never saved as training features
    sample_x, _ = s1.load_xy(os.path.join(BASE_DIR, "data_2012"))
    idx_to_name = s1.load_features(s1.FEATURES_JSON, sample_x.shape[1])

    args = SimpleNamespace(
        temperature=0.25,
        num_ctx=4096,
        timeout_seconds=120,
        max_retries=3,
        retry_backoff_base=0.7,
        retry_backoff_factor=1.7,
        sleep_between_requests=0.0,
        min_story_chars=80,
    )

    for year in [2012, 2014, 2016, 2018]:
        X, y = s1.load_xy(os.path.join(BASE_DIR, f"data_{year}"))
        picks = pick_n_each(y, per_class)
        t_year = time.time()
        out = os.path.join(STORIES_DIR, f"stories_{year}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for i, row_idx in enumerate(picks, start=1):
                row = X[row_idx]
                label = int(y[row_idx])
                active_idx = np.where(row == 1)[0]
                active_pairs = [(int(i), idx_to_name[int(i)]) for i in active_idx]

                prompt = s1.build_prompt(label, active_pairs)
                prompt_hash = __import__("hashlib").sha256(prompt.encode("utf-8")).hexdigest()

                story = ""
                error = None
                attempts = 0
                pchars = 0
                rchars = 0
                backoff = 0.0
                for a in range(1, args.max_retries + 1):
                    attempts = a
                    pchars += len(prompt)
                    try:
                        story = s1.ollama_generate(prompt, args)
                    except Exception as exc:
                        error = str(exc)
                    rchars += len(story)
                    qf = s1._quality_flag(story, args.min_story_chars)
                    if error is None and qf == "ok":
                        break
                    if a < args.max_retries:
                        d = args.retry_backoff_base * (args.retry_backoff_factor ** (a - 1))
                        time.sleep(d)
                        backoff += d

                qf = s1._quality_flag(story, args.min_story_chars)
                rec = {
                    "year": year,
                    "row_index": int(row_idx),
                    "label": label,
                    "label_name": "malicious" if label == 1 else "benign",
                    "story": story,
                    "error": error,
                    "generation_attempts": attempts,
                    "quality_flag": qf,
                    "generation_meta": {
                        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "prompt_sha256": prompt_hash,
                        "prompt_chars": pchars,
                        "response_chars": rchars,
                        "retry_backoff_seconds": round(backoff, 4),
                        "model": s1.OLLAMA_MODEL,
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")
                _progress(i, len(picks), time.time() - t_year, prefix=f"test-s1-{year}")
        if picks:
            sys.stdout.write("\n")


def main(per_class: int, model_name: str):
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    os.makedirs(STORIES_DIR, exist_ok=True)
    os.makedirs(FUTURE_DIR, exist_ok=True)
    os.makedirs(EMB_DIR, exist_ok=True)
    t_all = time.time()
    stage_total = 4
    stage_done = 0
    _progress(stage_done, stage_total, 0.0, prefix="test-pipeline")

    # Stage 1 (custom tiny generation for test)
    build_small_stage1_stories(per_class)
    stage_done += 1
    _progress(stage_done, stage_total, time.time() - t_all, prefix="test-pipeline")

    # Stage 2 (module main with patched paths)
    s2.STORIES_DIR = STORIES_DIR
    s2.IN_PATHS = [
        os.path.join(STORIES_DIR, "stories_2012.jsonl"),
        os.path.join(STORIES_DIR, "stories_2014.jsonl"),
    ]
    s2.OUT_DIR = FUTURE_DIR
    s2.OUT_JSONL = os.path.join(FUTURE_DIR, "future_stories.jsonl")
    s2.OUT_QUARANTINE = os.path.join(FUTURE_DIR, "future_stories_quarantine.jsonl")
    s2.OUT_QUALITY = os.path.join(FUTURE_DIR, "future_stories_quality_report.json")
    s2.OUT_RESEARCH = os.path.join(FUTURE_DIR, "future_stories_research_report.json")
    s2.OUT_MANIFEST = os.path.join(FUTURE_DIR, "future_stories_run_manifest.json")
    s2.OUT_README = os.path.join(FUTURE_DIR, "README_future_v2.md")
    run_module_main(s2.main, ["--no-strict-gate"])
    stage_done += 1
    _progress(stage_done, stage_total, time.time() - t_all, prefix="test-pipeline")

    # Stage 3 (module main with patched paths)
    s3.STORIES_DIR = STORIES_DIR
    s3.FUTURE_STORIES = s2.OUT_JSONL
    s3.OUT_DIR = EMB_DIR
    s3.OUT_QUARANTINE = os.path.join(EMB_DIR, "embeddings_quarantine.jsonl")
    s3.OUT_QUALITY = os.path.join(EMB_DIR, "embeddings_quality_report.json")
    s3.OUT_RESEARCH = os.path.join(EMB_DIR, "embeddings_research_report.json")
    s3.OUT_MANIFEST = os.path.join(EMB_DIR, "embeddings_run_manifest.json")
    s3.OUT_README = os.path.join(EMB_DIR, "README_embeddings_v2.md")
    run_module_main(
        s3.main,
        ["--model-name", model_name, "--no-strict-gate"],
    )
    stage_done += 1
    _progress(stage_done, stage_total, time.time() - t_all, prefix="test-pipeline")

    # Validate outputs
    tags = ["data_2012", "data_2014", "data_2016", "data_2018", "data_future"]
    t_val = time.time()
    for i, tag in enumerate(tags, start=1):
        Xp = os.path.join(EMB_DIR, tag, "X.npy")
        yp = os.path.join(EMB_DIR, tag, "y.npy")
        X = np.load(Xp, allow_pickle=True)
        y = np.load(yp, allow_pickle=True)
        if y.ndim > 1:
            y = y.ravel()
        print(f"{tag}: X={X.shape}, y={y.shape}, benign={(y==0).sum()}, malicious={(y==1).sum()}")
        _progress(i, len(tags), time.time() - t_val, prefix="test-validate")
    if tags:
        sys.stdout.write("\n")
    stage_done += 1
    _progress(stage_done, stage_total, time.time() - t_all, prefix="test-pipeline")
    sys.stdout.write("\n")


if __name__ == "__main__":
    args = parse_args()
    t0 = time.time()
    try:
        main(args.per_class, args.model_name)
    except Exception as exc:
        print(f"Test failed: {exc}")
        sys.exit(1)
    print(f"Elapsed: {time.time() - t0:.1f}s")
