"""
Generate story-style explanations for each sample in X.npy/y.npy (2012 + 2014).

Inputs:
  - data_2012/X.npy, y.npy
  - data_2014/X.npy, y.npy
  - selected_features.json (optional, for human-readable names)

Outputs (default):
  - drive-download-20260219T230708Z-1-001/data_stories/stories.jsonl
  - drive-download-20260219T230708Z-1-001/data_stories/stories_quarantine.jsonl
  - drive-download-20260219T230708Z-1-001/data_stories/stories_quality_report.json
  - drive-download-20260219T230708Z-1-001/data_stories/stories_research_report.json
  - drive-download-20260219T230708Z-1-001/data_stories/stories_run_manifest.json
  - drive-download-20260219T230708Z-1-001/data_stories/README_stories.md
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

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
OUT_QUARANTINE = os.path.join(OUT_DIR, "stories_quarantine.jsonl")
OUT_QUALITY = os.path.join(OUT_DIR, "stories_quality_report.json")
OUT_RESEARCH = os.path.join(OUT_DIR, "stories_research_report.json")
OUT_MANIFEST = os.path.join(OUT_DIR, "stories_run_manifest.json")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

YEARS = [2012, 2014]
RNG_SEED = 42

# Prompt controls
# Set to None to include all active features in the prompt
MAX_FEATURES_IN_PROMPT = None

# Sampling controls (set to None to process all)
MAX_SAMPLES_PER_YEAR = None
LIMIT_PER_CLASS = None
MAX_SAMPLES_TOTAL = None

PROB_SMOOTHING = 0.5  # Laplace smoothing to avoid 0/1 probabilities

BEHAVIOR_KEYWORDS = {
    "permission",
    "api",
    "network",
    "request",
    "service",
    "intent",
    "receiver",
    "storage",
    "file",
    "background",
    "process",
    "connection",
    "sms",
    "location",
    "content",
    "provider",
}
MALICIOUS_HINTS = {"exfil", "payload", "command", "c2", "exploit", "abuse", "dropper", "evasion", "fraud"}
BENIGN_HINTS = {"user-facing", "normal", "legitimate", "utility", "benign", "no clear abuse"}
QUALITY_PROFILE = "research_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume by appending to existing stories.jsonl.")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--sleep-between-requests", type=float, default=0.0)
    parser.add_argument("--max-generation-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-base", type=float, default=0.7)
    parser.add_argument("--retry-backoff-factor", type=float, default=1.7)
    parser.add_argument("--min-story-chars", type=int, default=80)
    parser.add_argument("--max-empty-story-rate", type=float, default=0.01)
    parser.add_argument("--max-short-story-rate", type=float, default=0.02)
    parser.add_argument("--max-error-rate", type=float, default=0.01)
    parser.add_argument("--max-quarantine-rate", type=float, default=0.10)
    parser.add_argument("--max-duplicate-rate", type=float, default=0.10)
    parser.add_argument(
        "--strict-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail run when quality thresholds are violated (default: enabled).",
    )
    return parser.parse_args()


def load_features(features_json_path: str, n_features: int):
    if not os.path.isfile(features_json_path):
        return [f"feature_{i}" for i in range(n_features)]
    with open(features_json_path, "r", encoding="utf-8") as f:
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
    return {"p_b": p_b, "p_m": p_m, "delta": delta}


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
Write a forensic narrative that explains why this sample is likely {label_name}.

Rules:
- Use neutral, evidence-focused language with uncertainty when appropriate.
- Avoid step-by-step instructions, code, or operational guidance.
- Explicitly mention at least 2 concrete behaviors implied by the active features
  (for example permissions, APIs, network/service usage, storage, intents/receivers).
- Do not list the features verbatim; synthesize them into a coherent narrative.
- Avoid generic filler text and avoid markdown formatting.
- Output plain text only.
- Format with these plain-text tags to keep output consistently auditable:
  ASSESSMENT:
  BEHAVIOR:
  RATIONALE:
"""
    return prompt.strip()


def ollama_generate(prompt: str, args):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": args.temperature, "num_ctx": args.num_ctx},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=args.timeout_seconds) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "").strip()


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


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
    sys.stdout.write(f"\r[{bar}] {n}/{total} ({pct}%) ETA {mins:02d}:{secs:02d} | avg tok/s {tps}")
    sys.stdout.flush()


def _write_jsonl(path: str, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")


def _sha256_of_file(path: str):
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _git_info(repo_dir: str):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()
    except Exception:
        commit = None
    try:
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_dir, text=True).strip())
    except Exception:
        dirty = None
    return {"commit": commit, "dirty": dirty}


def _normalize_story_for_dup(story: str):
    return " ".join((story or "").lower().split())


def _story_has_behavioral_grounding(story: str):
    s = (story or "").lower()
    return any(k in s for k in BEHAVIOR_KEYWORDS)


def _semantic_mismatch(label: int, story: str):
    s = (story or "").lower()
    m_hits = sum(1 for k in MALICIOUS_HINTS if k in s)
    b_hits = sum(1 for k in BENIGN_HINTS if k in s)
    if label == 1 and b_hits >= 2 and m_hits == 0:
        return True
    if label == 0 and m_hits >= 2 and b_hits == 0:
        return True
    return False


def _evaluate_story_quality(story: str, label: int, min_story_chars: int):
    s = (story or "").strip()
    if not s:
        return "empty_story"
    if len(s) < min_story_chars:
        return "short_story"
    if "```" in s or s.startswith("#"):
        return "format_violation"
    if not _story_has_behavioral_grounding(s):
        return "missing_behavioral_grounding"
    if _semantic_mismatch(label, s):
        return "semantic_label_mismatch"
    return "ok"


def _generate_story_with_retries(prompt: str, label: int, args):
    last_story = ""
    last_error = None
    attempts = 0
    prompt_chars = 0
    response_chars = 0
    total_backoff_s = 0.0
    quality_flag = "empty_story"

    for attempt in range(1, args.max_generation_retries + 1):
        attempts = attempt
        prompt_chars += len(prompt)
        story = ""
        error = None
        try:
            story = ollama_generate(prompt, args)
        except Exception as exc:
            error = str(exc)
        response_chars += len(story)
        quality_flag = _evaluate_story_quality(story, label, args.min_story_chars)

        last_story = story
        last_error = error
        if error is None and quality_flag == "ok":
            break

        if attempt < args.max_generation_retries:
            delay = args.retry_backoff_base * (args.retry_backoff_factor ** (attempt - 1))
            time.sleep(delay)
            total_backoff_s += delay

    return {
        "story": last_story,
        "error": last_error,
        "attempts": attempts,
        "prompt_chars": prompt_chars,
        "response_chars": response_chars,
        "quality_flag": quality_flag,
        "retry_backoff_seconds": round(total_backoff_s, 4),
    }


def _group_quality_metrics(rows, quarantine_ids):
    groups = {}
    for rec in rows:
        g = f"year={rec.get('year')}|label={rec.get('label_name')}"
        if g not in groups:
            groups[g] = {
                "rows": 0,
                "empty_story": 0,
                "short_story": 0,
                "error": 0,
                "duplicate": 0,
                "semantic_label_mismatch": 0,
                "quarantined": 0,
            }
        m = groups[g]
        m["rows"] += 1
        qf = rec.get("quality_flag")
        if qf == "empty_story":
            m["empty_story"] += 1
        if qf == "short_story":
            m["short_story"] += 1
        if qf == "semantic_label_mismatch":
            m["semantic_label_mismatch"] += 1
        if qf == "duplicate_story":
            m["duplicate"] += 1
        if rec.get("error"):
            m["error"] += 1
        rec_id = (rec.get("year"), rec.get("row_index"))
        if rec_id in quarantine_ids:
            m["quarantined"] += 1

    for g, m in groups.items():
        n = m["rows"]
        m["rates"] = {
            "empty_story_rate": (m["empty_story"] / n) if n else 0.0,
            "short_story_rate": (m["short_story"] / n) if n else 0.0,
            "error_rate": (m["error"] / n) if n else 0.0,
            "duplicate_rate": (m["duplicate"] / n) if n else 0.0,
            "semantic_mismatch_rate": (m["semantic_label_mismatch"] / n) if n else 0.0,
            "quarantine_rate": (m["quarantined"] / n) if n else 0.0,
        }
    return groups


def _compute_quality_report(rows, quarantine_rows, args):
    n = len(rows)
    empty_story = 0
    short_story = 0
    error_rows = 0
    duplicate_rows = 0
    semantic_mismatch_rows = 0
    by_year = {}
    by_label = {}

    for rec in rows:
        year = str(rec.get("year"))
        by_year[year] = by_year.get(year, 0) + 1
        label = str(rec.get("label_name"))
        by_label[label] = by_label.get(label, 0) + 1

        qf = rec.get("quality_flag")
        if qf == "empty_story":
            empty_story += 1
        if qf == "short_story":
            short_story += 1
        if qf == "duplicate_story":
            duplicate_rows += 1
        if qf == "semantic_label_mismatch":
            semantic_mismatch_rows += 1
        if rec.get("error"):
            error_rows += 1

    def _rate(x):
        return float(x) / float(n) if n > 0 else 0.0

    quarantine_ids = {(r.get("year"), r.get("row_index")) for r in quarantine_rows}
    stratified = _group_quality_metrics(rows, quarantine_ids)

    rates = {
        "empty_story_rate": _rate(empty_story),
        "short_story_rate": _rate(short_story),
        "error_rate": _rate(error_rows),
        "quarantine_rate": _rate(len(quarantine_rows)),
        "duplicate_rate": _rate(duplicate_rows),
        "semantic_mismatch_rate": _rate(semantic_mismatch_rows),
    }

    thresholds = {
        "max_empty_story_rate": args.max_empty_story_rate,
        "max_short_story_rate": args.max_short_story_rate,
        "max_error_rate": args.max_error_rate,
        "max_quarantine_rate": args.max_quarantine_rate,
        "max_duplicate_rate": args.max_duplicate_rate,
    }

    violations = []
    if rates["empty_story_rate"] > args.max_empty_story_rate:
        violations.append("empty_story_rate_above_threshold")
    if rates["short_story_rate"] > args.max_short_story_rate:
        violations.append("short_story_rate_above_threshold")
    if rates["error_rate"] > args.max_error_rate:
        violations.append("error_rate_above_threshold")
    if rates["quarantine_rate"] > args.max_quarantine_rate:
        violations.append("quarantine_rate_above_threshold")
    if rates["duplicate_rate"] > args.max_duplicate_rate:
        violations.append("duplicate_rate_above_threshold")

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quality_profile": QUALITY_PROFILE,
        "thresholds": thresholds,
        "counts": {
            "rows": n,
            "empty_story_rows": empty_story,
            "short_story_rows": short_story,
            "error_rows": error_rows,
            "quarantine_rows": len(quarantine_rows),
            "duplicate_rows": duplicate_rows,
            "semantic_mismatch_rows": semantic_mismatch_rows,
            "rows_by_year": by_year,
            "rows_by_label": by_label,
        },
        "rates": rates,
        "stratified": stratified,
        "violations": violations,
        "passed": len(violations) == 0,
    }


def _compute_research_report(events, elapsed_seconds, rows_total, rows_generated_this_run):
    attempts = [e.get("attempts", 0) for e in events]
    prompt_chars = sum(e.get("prompt_chars", 0) for e in events)
    response_chars = sum(e.get("response_chars", 0) for e in events)
    total_tokens = (prompt_chars + response_chars) / 4.0
    avg_attempts = (sum(attempts) / len(attempts)) if attempts else 0.0
    med_attempts = statistics.median(attempts) if attempts else 0.0
    max_attempts = max(attempts) if attempts else 0
    elapsed = max(float(elapsed_seconds), 0.0)
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quality_profile": QUALITY_PROFILE,
        "model": OLLAMA_MODEL,
        "rows_total_output": int(rows_total),
        "rows_generated_this_run": int(rows_generated_this_run),
        "generation_stats": {
            "avg_attempts": float(avg_attempts),
            "median_attempts": float(med_attempts),
            "max_attempts": int(max_attempts),
            "events": len(events),
        },
        "efficiency": {
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60.0 if elapsed > 0 else 0.0,
            "estimated_prompt_tokens": prompt_chars / 4.0,
            "estimated_response_tokens": response_chars / 4.0,
            "estimated_total_tokens": total_tokens,
            "estimated_tokens_per_second": (total_tokens / elapsed) if elapsed > 0 else 0.0,
            "estimated_tokens_per_minute": (total_tokens * 60.0 / elapsed) if elapsed > 0 else 0.0,
            "rows_per_second": (rows_generated_this_run / elapsed) if elapsed > 0 else 0.0,
            "rows_per_minute": (rows_generated_this_run * 60.0 / elapsed) if elapsed > 0 else 0.0,
            "token_estimation_method": "chars_div_4_approximation",
        },
    }


def _write_manifest(args, samples, start_at, rows_generated_this_run):
    repo_root = os.path.dirname(os.path.abspath(__file__))
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": _git_info(repo_root),
        "mode": "resume" if args.resume else "fresh_run",
        "model": {
            "ollama_url": OLLAMA_URL,
            "ollama_model": OLLAMA_MODEL,
            "temperature": args.temperature,
            "num_ctx": args.num_ctx,
            "timeout_seconds": args.timeout_seconds,
        },
        "sampling": {
            "years": YEARS,
            "max_features_in_prompt": MAX_FEATURES_IN_PROMPT,
            "max_samples_per_year": MAX_SAMPLES_PER_YEAR,
            "limit_per_class": LIMIT_PER_CLASS,
            "max_samples_total": MAX_SAMPLES_TOTAL,
            "rng_seed": RNG_SEED,
            "samples_planned": len(samples),
            "start_at": int(start_at),
            "rows_generated_this_run": int(rows_generated_this_run),
        },
        "quality_config": {
            "quality_profile": QUALITY_PROFILE,
            "min_story_chars": args.min_story_chars,
            "max_generation_retries": args.max_generation_retries,
            "retry_backoff_base": args.retry_backoff_base,
            "retry_backoff_factor": args.retry_backoff_factor,
            "strict_gate": args.strict_gate,
            "max_empty_story_rate": args.max_empty_story_rate,
            "max_short_story_rate": args.max_short_story_rate,
            "max_error_rate": args.max_error_rate,
            "max_quarantine_rate": args.max_quarantine_rate,
            "max_duplicate_rate": args.max_duplicate_rate,
        },
        "inputs": {
            "data_2012_x": {"path": os.path.join(DATA_2012, "X.npy"), "sha256": _sha256_of_file(os.path.join(DATA_2012, "X.npy"))},
            "data_2012_y": {"path": os.path.join(DATA_2012, "y.npy"), "sha256": _sha256_of_file(os.path.join(DATA_2012, "y.npy"))},
            "data_2014_x": {"path": os.path.join(DATA_2014, "X.npy"), "sha256": _sha256_of_file(os.path.join(DATA_2014, "X.npy"))},
            "data_2014_y": {"path": os.path.join(DATA_2014, "y.npy"), "sha256": _sha256_of_file(os.path.join(DATA_2014, "y.npy"))},
            "selected_features": {"path": FEATURES_JSON, "sha256": _sha256_of_file(FEATURES_JSON)},
        },
        "outputs": {
            "stories_jsonl": {"path": OUT_JSONL, "sha256": _sha256_of_file(OUT_JSONL)},
            "quarantine_jsonl": {"path": OUT_QUARANTINE, "sha256": _sha256_of_file(OUT_QUARANTINE)},
            "quality_report": {"path": OUT_QUALITY, "sha256": _sha256_of_file(OUT_QUALITY)},
            "research_report": {"path": OUT_RESEARCH, "sha256": _sha256_of_file(OUT_RESEARCH)},
            "readme": {"path": OUT_README, "sha256": _sha256_of_file(OUT_README)},
        },
    }
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main():
    args = parse_args()
    run_started = time.time()
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

    if args.resume:
        start_at = count_existing_lines(OUT_JSONL)
        mode = "a"
    else:
        start_at = 0
        mode = "w"
        for p in [OUT_JSONL, OUT_QUARANTINE, OUT_QUALITY, OUT_RESEARCH, OUT_README, OUT_MANIFEST]:
            if os.path.isfile(p):
                os.remove(p)

    total = len(samples)
    if start_at >= total:
        print(f"Nothing to do. {OUT_JSONL} already has {start_at} lines.")
        all_rows = list(read_jsonl(OUT_JSONL)) if os.path.isfile(OUT_JSONL) else []
        quarantine_rows = []
        _write_jsonl(OUT_QUARANTINE, quarantine_rows)
        quality = _compute_quality_report(all_rows, quarantine_rows, args)
        with open(OUT_QUALITY, "w", encoding="utf-8") as f:
            json.dump(quality, f, indent=2)
        research = _compute_research_report(
            events=[],
            elapsed_seconds=(time.time() - run_started),
            rows_total=len(all_rows),
            rows_generated_this_run=0,
        )
        with open(OUT_RESEARCH, "w", encoding="utf-8") as f:
            json.dump(research, f, indent=2)
        _write_manifest(args, samples, start_at, rows_generated_this_run=0)
        return

    t0 = time.time()
    total_chars = 0
    generation_events = []
    quarantine_rows = []

    seen_norm = {}
    if args.resume and os.path.isfile(OUT_JSONL):
        for old in read_jsonl(OUT_JSONL):
            norm = _normalize_story_for_dup(old.get("story", ""))
            if norm:
                seen_norm[norm] = seen_norm.get(norm, 0) + 1

    with open(OUT_JSONL, mode, encoding="utf-8") as f:
        for n, (year, i) in enumerate(samples[start_at:], start=start_at + 1):
            X = X12 if year == 2012 else X14
            y = y12 if year == 2012 else y14
            row = X[i]
            label = int(y[i])
            pairs, n_active, all_pairs = select_prompt_features(row, idx_to_name, delta_rank, MAX_FEATURES_IN_PROMPT)

            prompt = build_story_prompt(label, pairs)
            gen = _generate_story_with_retries(prompt, label, args)
            story = gen["story"]
            error = gen["error"]
            attempts = gen["attempts"]
            pchars = gen["prompt_chars"]
            rchars = gen["response_chars"]
            quality_flag = gen["quality_flag"]

            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            norm_story = _normalize_story_for_dup(story)
            if norm_story and seen_norm.get(norm_story, 0) > 0:
                quality_flag = "duplicate_story"
            if norm_story:
                seen_norm[norm_story] = seen_norm.get(norm_story, 0) + 1

            total_chars += pchars + rchars
            generation_events.append(
                {
                    "year": int(year),
                    "label": int(label),
                    "attempts": int(attempts),
                    "prompt_chars": int(pchars),
                    "response_chars": int(rchars),
                    "quality_flag": quality_flag,
                }
            )

            rec = {
                "year": int(year),
                "row_index": int(i),
                "label": int(label),
                "label_name": "malicious" if label == 1 else "benign",
                "n_active_features": int(n_active),
                "prompt_features": [{"index": idx, "name": name} for idx, name in pairs],
                "active_features": [{"index": idx, "name": name} for idx, name in all_pairs],
                "story": story,
                "error": error,
                "generation_attempts": int(attempts),
                "quality_flag": quality_flag,
                "generation_meta": {
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "prompt_sha256": prompt_hash,
                    "prompt_chars": int(pchars),
                    "response_chars": int(rchars),
                    "model": OLLAMA_MODEL,
                    "temperature": args.temperature,
                    "num_ctx": args.num_ctx,
                    "timeout_seconds": args.timeout_seconds,
                    "retry_backoff_seconds": gen["retry_backoff_seconds"],
                },
            }
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

            if quality_flag != "ok" or error is not None:
                quarantine_rows.append(
                    {
                        "year": int(year),
                        "row_index": int(i),
                        "label": int(label),
                        "reason": quality_flag if quality_flag != "ok" else "generation_error",
                        "record": rec,
                    }
                )

            if n % 5 == 0 or n == total:
                elapsed = time.time() - t0
                avg_tps = (total_chars / 4) / elapsed if elapsed > 0 else None
                _progress(n, total, elapsed, avg_tps)
            if args.sleep_between_requests:
                time.sleep(args.sleep_between_requests)
    if total > 0:
        sys.stdout.write("\n")

    all_rows = list(read_jsonl(OUT_JSONL))
    _write_jsonl(OUT_QUARANTINE, quarantine_rows)

    quality = _compute_quality_report(all_rows, quarantine_rows, args)
    with open(OUT_QUALITY, "w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)

    elapsed = time.time() - run_started
    research = _compute_research_report(
        events=generation_events,
        elapsed_seconds=elapsed,
        rows_total=len(all_rows),
        rows_generated_this_run=max(0, total - start_at),
    )
    with open(OUT_RESEARCH, "w", encoding="utf-8") as f:
        json.dump(research, f, indent=2)

    with open(OUT_README, "w", encoding="utf-8") as f:
        f.write("# Story Dataset\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Source Data\n")
        f.write(f"- Years: {', '.join(str(y) for y in YEARS)}\n")
        f.write(f"- Samples: 2012={X12.shape[0]}, 2014={X14.shape[0]}\n")
        f.write(f"- Features per sample: {X12.shape[1]}\n\n")
        f.write("## Generation Settings\n")
        f.write(f"- Mode: {'resume' if args.resume else 'fresh-run'}\n")
        f.write(f"- Model: {OLLAMA_MODEL}\n")
        f.write(f"- Temperature: {args.temperature}\n")
        f.write(f"- Num ctx: {args.num_ctx}\n")
        f.write(f"- Timeout seconds: {args.timeout_seconds}\n")
        f.write(f"- Max features in prompt: {MAX_FEATURES_IN_PROMPT}\n")
        f.write(f"- MAX_SAMPLES_PER_YEAR: {MAX_SAMPLES_PER_YEAR}\n")
        f.write(f"- LIMIT_PER_CLASS: {LIMIT_PER_CLASS}\n")
        f.write(f"- MAX_SAMPLES_TOTAL: {MAX_SAMPLES_TOTAL}\n")
        f.write(f"- Min story chars: {args.min_story_chars}\n")
        f.write(f"- Max generation retries: {args.max_generation_retries}\n")
        f.write(f"- Strict gate: {args.strict_gate}\n")
        f.write(f"- Output: {OUT_JSONL}\n")
        f.write(f"- Quarantine: {OUT_QUARANTINE}\n")
        f.write(f"- Quality report: {OUT_QUALITY}\n")
        f.write(f"- Research report: {OUT_RESEARCH}\n")
        f.write(f"- Run manifest: {OUT_MANIFEST}\n")
        f.write(f"- Elapsed seconds: {research['efficiency']['elapsed_seconds']:.2f}\n")
        f.write(f"- Estimated total tokens: {research['efficiency']['estimated_total_tokens']:.1f}\n")
        f.write(f"- Estimated tokens/second: {research['efficiency']['estimated_tokens_per_second']:.2f}\n")

    _write_manifest(args, samples, start_at, rows_generated_this_run=max(0, total - start_at))

    print(f"Wrote: {OUT_JSONL}")
    print(f"Wrote: {OUT_QUARANTINE}")
    print(f"Wrote: {OUT_QUALITY}")
    print(f"Wrote: {OUT_RESEARCH}")
    print(f"Wrote: {OUT_MANIFEST}")
    print(f"Wrote: {OUT_README}")

    if args.strict_gate and not quality.get("passed", False):
        raise RuntimeError(
            f"Quality gate failed. See {OUT_QUALITY}. Violations: {', '.join(quality.get('violations', []))}"
        )


if __name__ == "__main__":
    main()
