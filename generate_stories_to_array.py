"""
Convert future stories + feature lists into X.npy/y.npy arrays.

Inputs:
  - drive-download-20260219T230708Z-1-001/data_stories_future/future_stories.jsonl
  - selected_features.json

Outputs:
  - drive-download-20260219T230708Z-1-001/data_future_arrays/X.npy
  - drive-download-20260219T230708Z-1-001/data_future_arrays/y.npy
  - drive-download-20260219T230708Z-1-001/data_future_arrays/arrays_quarantine.jsonl
  - drive-download-20260219T230708Z-1-001/data_future_arrays/arrays_quality_report.json
  - drive-download-20260219T230708Z-1-001/data_future_arrays/arrays_research_report.json
  - drive-download-20260219T230708Z-1-001/data_future_arrays/arrays_run_manifest.json
  - drive-download-20260219T230708Z-1-001/data_future_arrays/README_arrays.md
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
IN_JSONL = os.path.join(BASE_DIR, "data_stories_future", "future_stories.jsonl")
FEATURES_JSON = os.path.join(BASE_DIR, "selected_features.json")
OUT_DIR = os.path.join(BASE_DIR, "data_future_arrays")
OUT_X = os.path.join(OUT_DIR, "X.npy")
OUT_Y = os.path.join(OUT_DIR, "y.npy")
OUT_README = os.path.join(OUT_DIR, "README_arrays.md")
OUT_QUARANTINE = os.path.join(OUT_DIR, "arrays_quarantine.jsonl")
OUT_QUALITY = os.path.join(OUT_DIR, "arrays_quality_report.json")
OUT_RESEARCH = os.path.join(OUT_DIR, "arrays_research_report.json")
OUT_MANIFEST = os.path.join(OUT_DIR, "arrays_run_manifest.json")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
QUALITY_PROFILE = "research_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if present.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--sleep-between-requests", type=float, default=0.0)
    parser.add_argument("--max-index-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-base", type=float, default=0.5)
    parser.add_argument("--retry-backoff-factor", type=float, default=1.7)
    parser.add_argument("--max-fallback-rate", type=float, default=0.10)
    parser.add_argument("--max-parse-failed-rate", type=float, default=0.10)
    parser.add_argument("--max-error-rate", type=float, default=0.01)
    parser.add_argument("--max-zero-active-rate", type=float, default=0.05)
    parser.add_argument("--max-quarantine-rate", type=float, default=0.10)
    parser.add_argument(
        "--strict-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail run when quality thresholds are violated (default: enabled).",
    )
    return parser.parse_args()


def load_feature_maps(features_json_path: str):
    with open(features_json_path, "r", encoding="utf-8") as f:
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
    sys.stdout.write(f"\r[{bar}] {n}/{total} ({pct}%) ETA {mins:02d}:{secs:02d} | avg tok/s {tps}")
    sys.stdout.flush()


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
- Be conservative: only include a feature index if the story clearly supports it.
- Prefer precision over recall when uncertain.
- Do not invent features not in the candidate list.
- Output ONLY valid JSON (no prose, no markdown/code fences).

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


def _infer_indices_with_retries(prompt: str, args):
    attempts = 0
    prompt_chars = 0
    response_chars = 0
    parse_success_attempts = 0
    exception_attempts = 0
    total_backoff_s = 0.0
    last_raw = ""
    last_error = None
    parsed = None

    for attempt in range(1, args.max_index_retries + 1):
        attempts = attempt
        prompt_chars += len(prompt)
        raw = ""
        error = None
        local_parsed = None
        try:
            raw = ollama_generate(prompt, args)
            local_parsed = parse_json(raw)
            if local_parsed is not None:
                parse_success_attempts += 1
        except Exception as exc:
            error = str(exc)
            exception_attempts += 1
        response_chars += len(raw)
        last_raw = raw
        last_error = error
        parsed = local_parsed

        if error is None and local_parsed is not None:
            break

        if attempt < args.max_index_retries:
            delay = args.retry_backoff_base * (args.retry_backoff_factor ** (attempt - 1))
            time.sleep(delay)
            total_backoff_s += delay

    return {
        "attempts": attempts,
        "prompt_chars": prompt_chars,
        "response_chars": response_chars,
        "parse_success_attempts": parse_success_attempts,
        "exception_attempts": exception_attempts,
        "raw": last_raw,
        "raw_sha256": hashlib.sha256(last_raw.encode("utf-8")).hexdigest() if last_raw else None,
        "error": last_error,
        "parsed": parsed,
        "retry_backoff_seconds": round(total_backoff_s, 4),
    }


def _group_quality_metrics(process_rows, quarantine_ids):
    groups = {}
    for r in process_rows:
        g = f"year={r.get('year')}|label={r.get('label_name')}"
        if g not in groups:
            groups[g] = {
                "rows": 0,
                "fallback": 0,
                "parse_failed": 0,
                "error": 0,
                "zero_active": 0,
                "quarantined": 0,
            }
        m = groups[g]
        m["rows"] += 1
        if r.get("validate_note") == "empty_primary_indices_fallback_to_candidates":
            m["fallback"] += 1
        if not r.get("parse_success"):
            m["parse_failed"] += 1
        if r.get("error"):
            m["error"] += 1
        if int(r.get("n_active_indices", 0)) == 0:
            m["zero_active"] += 1
        rid = (r.get("year"), r.get("row_index"))
        if rid in quarantine_ids:
            m["quarantined"] += 1
    for m in groups.values():
        n = m["rows"]
        m["rates"] = {
            "fallback_rate": (m["fallback"] / n) if n else 0.0,
            "parse_failed_rate": (m["parse_failed"] / n) if n else 0.0,
            "error_rate": (m["error"] / n) if n else 0.0,
            "zero_active_rate": (m["zero_active"] / n) if n else 0.0,
            "quarantine_rate": (m["quarantined"] / n) if n else 0.0,
        }
    return groups


def _compute_quality_report(process_rows, quarantine_rows, args):
    n = len(process_rows)
    fallback_rows = sum(1 for r in process_rows if r.get("validate_note") == "empty_primary_indices_fallback_to_candidates")
    parse_fail_rows = sum(1 for r in process_rows if not r.get("parse_success"))
    error_rows = sum(1 for r in process_rows if r.get("error"))
    zero_rows = sum(1 for r in process_rows if int(r.get("n_active_indices", 0)) == 0)
    empty_candidates_rows = sum(1 for r in process_rows if r.get("validate_note") == "empty_candidates_allowed")

    by_year = {}
    by_label = {}
    for r in process_rows:
        y = str(r.get("year"))
        l = str(r.get("label_name"))
        by_year[y] = by_year.get(y, 0) + 1
        by_label[l] = by_label.get(l, 0) + 1

    def _rate(x):
        return float(x) / float(n) if n > 0 else 0.0

    rates = {
        "fallback_rate": _rate(fallback_rows),
        "parse_failed_rate": _rate(parse_fail_rows),
        "error_rate": _rate(error_rows),
        "zero_active_rate": _rate(zero_rows),
        "empty_candidates_rate": _rate(empty_candidates_rows),
        "quarantine_rate": _rate(len(quarantine_rows)),
    }

    thresholds = {
        "max_fallback_rate": args.max_fallback_rate,
        "max_parse_failed_rate": args.max_parse_failed_rate,
        "max_error_rate": args.max_error_rate,
        "max_zero_active_rate": args.max_zero_active_rate,
        "max_quarantine_rate": args.max_quarantine_rate,
    }

    violations = []
    if rates["fallback_rate"] > args.max_fallback_rate:
        violations.append("fallback_rate_above_threshold")
    if rates["parse_failed_rate"] > args.max_parse_failed_rate:
        violations.append("parse_failed_rate_above_threshold")
    if rates["error_rate"] > args.max_error_rate:
        violations.append("error_rate_above_threshold")
    if rates["zero_active_rate"] > args.max_zero_active_rate:
        violations.append("zero_active_rate_above_threshold")
    if rates["quarantine_rate"] > args.max_quarantine_rate:
        violations.append("quarantine_rate_above_threshold")

    quarantine_ids = {(q.get("year"), q.get("row_index")) for q in quarantine_rows}

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quality_profile": QUALITY_PROFILE,
        "thresholds": thresholds,
        "counts": {
            "rows_processed": n,
            "fallback_rows": fallback_rows,
            "parse_failed_rows": parse_fail_rows,
            "error_rows": error_rows,
            "zero_active_rows": zero_rows,
            "empty_candidates_rows": empty_candidates_rows,
            "quarantine_rows": len(quarantine_rows),
            "rows_by_year": by_year,
            "rows_by_label": by_label,
        },
        "rates": rates,
        "stratified": _group_quality_metrics(process_rows, quarantine_ids),
        "violations": violations,
        "passed": len(violations) == 0,
    }


def _compute_research_report(process_rows, elapsed_seconds):
    attempts = [int(r.get("attempts", 0)) for r in process_rows]
    prompt_chars = sum(int(r.get("prompt_chars", 0)) for r in process_rows)
    response_chars = sum(int(r.get("response_chars", 0)) for r in process_rows)
    estimated_total_tokens = (prompt_chars + response_chars) / 4.0
    n = len(process_rows)
    elapsed = max(float(elapsed_seconds), 0.0)

    hist = {}
    for a in attempts:
        k = str(a)
        hist[k] = hist.get(k, 0) + 1

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quality_profile": QUALITY_PROFILE,
        "model": OLLAMA_MODEL,
        "rows_processed": n,
        "inference_stats": {
            "avg_attempts": float(sum(attempts) / n) if n > 0 else 0.0,
            "median_attempts": float(statistics.median(attempts)) if attempts else 0.0,
            "max_attempts": int(max(attempts)) if attempts else 0,
            "attempts_histogram": hist,
            "parse_success_rows": int(sum(1 for r in process_rows if r.get("parse_success"))),
        },
        "efficiency": {
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60.0 if elapsed > 0 else 0.0,
            "estimated_prompt_tokens": prompt_chars / 4.0,
            "estimated_response_tokens": response_chars / 4.0,
            "estimated_total_tokens": estimated_total_tokens,
            "estimated_tokens_per_second": (estimated_total_tokens / elapsed) if elapsed > 0 else 0.0,
            "estimated_tokens_per_minute": (estimated_total_tokens * 60.0 / elapsed) if elapsed > 0 else 0.0,
            "rows_per_second": (n / elapsed) if elapsed > 0 else 0.0,
            "rows_per_minute": (n * 60.0 / elapsed) if elapsed > 0 else 0.0,
            "token_estimation_method": "chars_div_4_approximation",
        },
    }


def _write_manifest(args, n_records, start_idx, rows_processed_this_run):
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
        "runtime_config": {
            "quality_profile": QUALITY_PROFILE,
            "max_index_retries": args.max_index_retries,
            "retry_backoff_base": args.retry_backoff_base,
            "retry_backoff_factor": args.retry_backoff_factor,
            "sleep_between_requests": args.sleep_between_requests,
            "strict_gate": args.strict_gate,
            "max_fallback_rate": args.max_fallback_rate,
            "max_parse_failed_rate": args.max_parse_failed_rate,
            "max_error_rate": args.max_error_rate,
            "max_zero_active_rate": args.max_zero_active_rate,
            "max_quarantine_rate": args.max_quarantine_rate,
        },
        "rows": {
            "total_records": int(n_records),
            "start_index": int(start_idx),
            "processed_this_run": int(rows_processed_this_run),
        },
        "inputs": {
            "future_stories_jsonl": {"path": IN_JSONL, "sha256": _sha256_of_file(IN_JSONL)},
            "selected_features": {"path": FEATURES_JSON, "sha256": _sha256_of_file(FEATURES_JSON)},
        },
        "outputs": {
            "x_npy": {"path": OUT_X, "sha256": _sha256_of_file(OUT_X)},
            "y_npy": {"path": OUT_Y, "sha256": _sha256_of_file(OUT_Y)},
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

    if not os.path.isfile(IN_JSONL):
        raise FileNotFoundError(f"Missing input: {IN_JSONL}")

    idx_to_name, _name_to_idx = load_feature_maps(FEATURES_JSON)
    n_features = len(idx_to_name)
    records = list(read_jsonl(IN_JSONL))
    n_records = len(records)

    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUT_DIR, "checkpoint.json")

    if args.resume:
        ckpt = _load_checkpoint(ckpt_path) or {}
        if ckpt.get("n_records") and ckpt["n_records"] != n_records:
            raise RuntimeError("Checkpoint does not match current input size.")
        if ckpt.get("n_features") and ckpt["n_features"] != n_features:
            raise RuntimeError("Checkpoint does not match current feature size.")
        start_idx = int(ckpt.get("next_index", 0))
        if os.path.isfile(OUT_X) and os.path.isfile(OUT_Y):
            X = np.lib.format.open_memmap(OUT_X, mode="r+", dtype=np.int64, shape=(n_records, n_features))
            y = np.lib.format.open_memmap(OUT_Y, mode="r+", dtype=np.int64, shape=(n_records,))
        else:
            X = np.lib.format.open_memmap(OUT_X, mode="w+", dtype=np.int64, shape=(n_records, n_features))
            y = np.lib.format.open_memmap(OUT_Y, mode="w+", dtype=np.int64, shape=(n_records,))
    else:
        for p in [OUT_X, OUT_Y, OUT_QUARANTINE, OUT_QUALITY, OUT_RESEARCH, OUT_README, OUT_MANIFEST, ckpt_path]:
            if os.path.isfile(p):
                os.remove(p)
        start_idx = 0
        X = np.lib.format.open_memmap(OUT_X, mode="w+", dtype=np.int64, shape=(n_records, n_features))
        y = np.lib.format.open_memmap(OUT_Y, mode="w+", dtype=np.int64, shape=(n_records,))

    t0 = time.time()
    total_chars = 0
    process_rows = []
    quarantine_rows = []

    for i, rec in enumerate(records[start_idx:], start=start_idx):
        label = int(rec.get("label", 0))
        y[i] = label

        prompt = build_prompt(rec)
        inf = _infer_indices_with_retries(prompt, args)
        parsed = inf["parsed"]
        error = inf["error"]
        total_chars += inf["prompt_chars"] + inf["response_chars"]

        active_indices = []
        if isinstance(parsed, dict):
            active_indices = parsed.get("active_feature_indices", [])

        candidates = rec.get("future_active_features", [])
        fallback_indices = [c.get("index") for c in candidates]
        active_indices, validate_note = ensure_non_empty_indices(active_indices, fallback_indices, idx_to_name)

        row_reason = None
        if not active_indices:
            if not fallback_indices:
                validate_note = validate_note or "empty_candidates_allowed"
            else:
                error = error or "no_valid_indices_after_validation"
                row_reason = "no_valid_indices_after_validation"

        # Prevent stale 1s across reruns/resume overwrites.
        X[i, :] = 0
        X[i, active_indices] = 1

        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        parse_success = bool(parsed is not None)
        if not parse_success and row_reason is None:
            row_reason = "parse_failed"
        if validate_note == "empty_primary_indices_fallback_to_candidates" and row_reason is None:
            row_reason = "fallback_used"

        process_rec = {
            "output_row": int(i),
            "year": rec.get("year"),
            "row_index": rec.get("row_index"),
            "label": label,
            "label_name": rec.get("label_name", "malicious" if label == 1 else "benign"),
            "attempts": int(inf["attempts"]),
            "parse_success_attempts": int(inf["parse_success_attempts"]),
            "exception_attempts": int(inf["exception_attempts"]),
            "prompt_chars": int(inf["prompt_chars"]),
            "response_chars": int(inf["response_chars"]),
            "parse_success": parse_success,
            "validate_note": validate_note,
            "error": error,
            "n_candidate_indices": int(len(validate_indices(fallback_indices, idx_to_name))),
            "n_active_indices": int(len(active_indices)),
            "inference_meta": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "prompt_sha256": prompt_hash,
                "raw_sha256": inf["raw_sha256"],
                "model": OLLAMA_MODEL,
                "temperature": args.temperature,
                "num_ctx": args.num_ctx,
                "timeout_seconds": args.timeout_seconds,
                "retry_backoff_seconds": inf["retry_backoff_seconds"],
            },
        }
        process_rows.append(process_rec)

        if row_reason is not None or error is not None:
            quarantine_rows.append(
                {
                    "output_row": int(i),
                    "year": rec.get("year"),
                    "row_index": rec.get("row_index"),
                    "label": label,
                    "label_name": rec.get("label_name", "malicious" if label == 1 else "benign"),
                    "reason": row_reason if row_reason is not None else "error",
                    "record": process_rec,
                }
            )

        if (i + 1) % 5 == 0 or (i + 1) == len(records):
            elapsed = time.time() - t0
            avg_tps = (total_chars / 4) / elapsed if elapsed > 0 else None
            _progress(i + 1, len(records), elapsed, avg_tps)
            _save_checkpoint(
                ckpt_path,
                {"n_records": n_records, "n_features": n_features, "next_index": i + 1},
            )

        if args.sleep_between_requests:
            time.sleep(args.sleep_between_requests)

    if len(records) > 0:
        sys.stdout.write("\n")

    _save_checkpoint(ckpt_path, {"n_records": n_records, "n_features": n_features, "next_index": n_records})

    _write_jsonl(OUT_QUARANTINE, quarantine_rows)
    quality = _compute_quality_report(process_rows, quarantine_rows, args)
    with open(OUT_QUALITY, "w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)

    research = _compute_research_report(process_rows, time.time() - run_started)
    with open(OUT_RESEARCH, "w", encoding="utf-8") as f:
        json.dump(research, f, indent=2)

    with open(OUT_README, "w", encoding="utf-8") as f:
        f.write("# Future Arrays\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"- Mode: {'resume' if args.resume else 'fresh-run'}\n")
        f.write(f"- Input: {IN_JSONL}\n")
        f.write(f"- Output X: {OUT_X}\n")
        f.write(f"- Output y: {OUT_Y}\n")
        f.write(f"- Model: {OLLAMA_MODEL}\n")
        f.write(f"- Features: {n_features}\n")
        f.write(f"- Temperature: {args.temperature}\n")
        f.write(f"- Num ctx: {args.num_ctx}\n")
        f.write(f"- Timeout seconds: {args.timeout_seconds}\n")
        f.write(f"- Max index retries: {args.max_index_retries}\n")
        f.write(f"- Strict gate: {args.strict_gate}\n")
        f.write(f"- Quarantine: {OUT_QUARANTINE}\n")
        f.write(f"- Quality report: {OUT_QUALITY}\n")
        f.write(f"- Research report: {OUT_RESEARCH}\n")
        f.write(f"- Run manifest: {OUT_MANIFEST}\n")
        f.write(f"- Elapsed seconds: {research['efficiency']['elapsed_seconds']:.2f}\n")
        f.write(f"- Estimated total tokens: {research['efficiency']['estimated_total_tokens']:.1f}\n")
        f.write(f"- Estimated tokens/second: {research['efficiency']['estimated_tokens_per_second']:.2f}\n")

    _write_manifest(args, n_records=n_records, start_idx=start_idx, rows_processed_this_run=max(0, n_records - start_idx))

    print(f"Wrote: {OUT_X}")
    print(f"Wrote: {OUT_Y}")
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
