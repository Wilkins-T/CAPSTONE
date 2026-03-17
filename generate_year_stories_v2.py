"""
V2 Stage 1: Generate story-only datasets per year from X.npy/y.npy.

Inputs (per year):
  drive-download-20260219T230708Z-1-001/data_<year>/X.npy, y.npy

Outputs:
  drive-download-20260219T230708Z-1-001/data_stories_v2/stories_<year>.jsonl
  drive-download-20260219T230708Z-1-001/data_stories_v2/stories_quarantine.jsonl
  drive-download-20260219T230708Z-1-001/data_stories_v2/stories_quality_report.json
  drive-download-20260219T230708Z-1-001/data_stories_v2/stories_research_report.json
  drive-download-20260219T230708Z-1-001/data_stories_v2/stories_run_manifest.json
  drive-download-20260219T230708Z-1-001/data_stories_v2/README_stories_v2.md
"""

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request

import numpy as np


BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drive-download-20260219T230708Z-1-001")
FEATURES_JSON = os.path.join(BASE_DIR, "selected_features.json")
OUT_DIR = os.path.join(BASE_DIR, "data_stories_v2")
YEARS = [2012, 2014, 2016, 2018]

OUT_QUARANTINE = os.path.join(OUT_DIR, "stories_quarantine.jsonl")
OUT_QUALITY = os.path.join(OUT_DIR, "stories_quality_report.json")
OUT_RESEARCH = os.path.join(OUT_DIR, "stories_research_report.json")
OUT_MANIFEST = os.path.join(OUT_DIR, "stories_run_manifest.json")
OUT_README = os.path.join(OUT_DIR, "README_stories_v2.md")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
QUALITY_PROFILE = "research_v1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    p.add_argument("--temperature", type=float, default=0.25)
    p.add_argument("--num-ctx", type=int, default=4096)
    p.add_argument("--timeout-seconds", type=int, default=120)
    p.add_argument("--sleep-between-requests", type=float, default=0.0)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-backoff-base", type=float, default=0.7)
    p.add_argument("--retry-backoff-factor", type=float, default=1.7)
    p.add_argument("--min-story-chars", type=int, default=80)
    p.add_argument("--max-empty-story-rate", type=float, default=0.01)
    p.add_argument("--max-short-story-rate", type=float, default=0.02)
    p.add_argument("--max-error-rate", type=float, default=0.01)
    p.add_argument("--max-quarantine-rate", type=float, default=0.10)
    p.add_argument("--max-duplicate-rate", type=float, default=0.10)
    p.add_argument("--strict-gate", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _sha256_of_file(path: str):
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            c = f.read(1024 * 1024)
            if not c:
                break
            h.update(c)
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


def load_features(path: str, n_features: int):
    names = [f"feature_{i}" for i in range(n_features)]
    if not os.path.isfile(path):
        return names
    with open(path, "r", encoding="utf-8") as f:
        name_to_idx = json.load(f)
    for name, idx in name_to_idx.items():
        try:
            i = int(idx)
        except ValueError:
            continue
        if 0 <= i < n_features:
            names[i] = name
    return names


def load_xy(data_dir: str):
    X = np.load(os.path.join(data_dir, "X.npy"), allow_pickle=True)
    y = np.load(os.path.join(data_dir, "y.npy"), allow_pickle=True)
    if y.ndim > 1:
        y = y.ravel()
    return X.astype(np.int64), y.astype(np.int64)


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_rows(path):
    if not os.path.isfile(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


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


def build_prompt(label, active_pairs):
    feature_block = "\n".join(f"{i} | {name}" for i, name in active_pairs) if active_pairs else "(no active features)"
    return f"""
You are an Android app telemetry summarizer.

Observed active indicators (index | feature):
{feature_block}

Write a neutral, evidence-only summary of observable behavior.

Rules:
- Account for every listed active indicator in the final output.
- Use plain, factual wording. No dramatic or threatening tone.
- Focus on observable app behavior only (permissions, API usage, network patterns, execution style, data handling).
- Do not infer motive, attacker intent, victim impact, or campaign attribution.
- If uncertain, state uncertainty explicitly using neutral phrasing.
- Avoid instructions/code/operational steps.
- Do not use class labels or direct class terms (forbidden: malicious, benign).
- Avoid loaded security terms unless directly supported by listed indicators
  (forbidden by default: attacker, payload, compromise, weaponize, trojan, ransomware, spyware).
- In `BEHAVIOR:`, summarize observable behavior patterns implied by the indicators.
- In `RATIONALE:`, map each active indicator to the behavior statements it supports.
- Do not output markdown.
- Output plain text only with these tags:
  ASSESSMENT:
  BEHAVIOR:
  RATIONALE:
""".strip()


def _normalize_story(s: str):
    return " ".join((s or "").lower().split())


def _quality_flag(story: str, min_chars: int):
    t = (story or "").strip()
    if not t:
        return "empty_story"
    if len(t) < min_chars:
        return "short_story"
    if "ASSESSMENT:" not in t or "BEHAVIOR:" not in t or "RATIONALE:" not in t:
        return "missing_required_tags"
    if re.search(r"\b(malicious|benign)\b", t, flags=re.IGNORECASE):
        return "label_leak_terms"
    if re.search(r"\b(attacker|payload|compromise|weaponize|trojan|ransomware|spyware)\b", t, flags=re.IGNORECASE):
        return "loaded_security_tone"
    return "ok"


def _progress(n, total, elapsed):
    width = 30
    frac = 1.0 if total == 0 else n / total
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    eta = (total - n) / (n / elapsed) if n > 0 and elapsed > 0 else 0
    sys.stdout.write(f"\r[{bar}] {n}/{total} ETA {int(eta//60):02d}:{int(eta%60):02d}")
    sys.stdout.flush()


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def _compute_reports(all_rows, quarantine_rows, events, args, elapsed_s):
    n = len(all_rows)
    empty_rows = sum(1 for r in all_rows if r.get("quality_flag") == "empty_story")
    short_rows = sum(1 for r in all_rows if r.get("quality_flag") == "short_story")
    err_rows = sum(1 for r in all_rows if r.get("error"))
    dup_rows = sum(1 for r in all_rows if r.get("quality_flag") == "duplicate_story")

    by_year = {}
    by_label = {}
    for r in all_rows:
        y = str(r.get("year"))
        l = str(r.get("label_name"))
        by_year[y] = by_year.get(y, 0) + 1
        by_label[l] = by_label.get(l, 0) + 1

    def rate(x):
        return float(x) / float(n) if n else 0.0

    quality = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quality_profile": QUALITY_PROFILE,
        "thresholds": {
            "max_empty_story_rate": args.max_empty_story_rate,
            "max_short_story_rate": args.max_short_story_rate,
            "max_error_rate": args.max_error_rate,
            "max_quarantine_rate": args.max_quarantine_rate,
            "max_duplicate_rate": args.max_duplicate_rate,
        },
        "counts": {
            "rows": n,
            "empty_story_rows": empty_rows,
            "short_story_rows": short_rows,
            "error_rows": err_rows,
            "duplicate_rows": dup_rows,
            "quarantine_rows": len(quarantine_rows),
            "rows_by_year": by_year,
            "rows_by_label": by_label,
        },
        "rates": {
            "empty_story_rate": rate(empty_rows),
            "short_story_rate": rate(short_rows),
            "error_rate": rate(err_rows),
            "quarantine_rate": rate(len(quarantine_rows)),
            "duplicate_rate": rate(dup_rows),
        },
    }
    violations = []
    if quality["rates"]["empty_story_rate"] > args.max_empty_story_rate:
        violations.append("empty_story_rate_above_threshold")
    if quality["rates"]["short_story_rate"] > args.max_short_story_rate:
        violations.append("short_story_rate_above_threshold")
    if quality["rates"]["error_rate"] > args.max_error_rate:
        violations.append("error_rate_above_threshold")
    if quality["rates"]["quarantine_rate"] > args.max_quarantine_rate:
        violations.append("quarantine_rate_above_threshold")
    if quality["rates"]["duplicate_rate"] > args.max_duplicate_rate:
        violations.append("duplicate_rate_above_threshold")
    quality["violations"] = violations
    quality["passed"] = len(violations) == 0

    attempts = [int(e.get("attempts", 0)) for e in events]
    pchars = sum(int(e.get("prompt_chars", 0)) for e in events)
    rchars = sum(int(e.get("response_chars", 0)) for e in events)
    est_toks = (pchars + rchars) / 4.0

    research = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quality_profile": QUALITY_PROFILE,
        "model": OLLAMA_MODEL,
        "rows_total_output": n,
        "generation_stats": {
            "events": len(events),
            "avg_attempts": float(sum(attempts) / len(attempts)) if attempts else 0.0,
            "median_attempts": float(statistics.median(attempts)) if attempts else 0.0,
            "max_attempts": int(max(attempts)) if attempts else 0,
        },
        "efficiency": {
            "elapsed_seconds": float(elapsed_s),
            "elapsed_minutes": float(elapsed_s) / 60.0 if elapsed_s > 0 else 0.0,
            "estimated_prompt_tokens": pchars / 4.0,
            "estimated_response_tokens": rchars / 4.0,
            "estimated_total_tokens": est_toks,
            "estimated_tokens_per_second": est_toks / elapsed_s if elapsed_s > 0 else 0.0,
            "estimated_tokens_per_minute": est_toks * 60.0 / elapsed_s if elapsed_s > 0 else 0.0,
            "token_estimation_method": "chars_div_4_approximation",
        },
    }

    return quality, research


def main():
    args = parse_args()
    run_started = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    if not args.resume:
        for p in [OUT_QUARANTINE, OUT_QUALITY, OUT_RESEARCH, OUT_MANIFEST, OUT_README]:
            if os.path.isfile(p):
                os.remove(p)
        for year in YEARS:
            p = os.path.join(OUT_DIR, f"stories_{year}.jsonl")
            if os.path.isfile(p):
                os.remove(p)

    # feature names only for prompt construction; never written as dataset fields
    sample_x, _ = load_xy(os.path.join(BASE_DIR, "data_2012"))
    idx_to_name = load_features(FEATURES_JSON, sample_x.shape[1])

    outputs = []
    quarantine_rows = []
    events = []
    seen_norm = {}

    for year in YEARS:
        data_dir = os.path.join(BASE_DIR, f"data_{year}")
        out_path = os.path.join(OUT_DIR, f"stories_{year}.jsonl")
        X, y = load_xy(data_dir)
        n = len(y)
        start = count_rows(out_path) if args.resume else 0
        mode = "a" if args.resume else "w"

        t0 = time.time()
        with open(out_path, mode, encoding="utf-8") as f:
            for row_idx in range(start, n):
                row = X[row_idx]
                label = int(y[row_idx])
                active_idx = np.where(row == 1)[0]
                active_pairs = [(int(i), idx_to_name[int(i)]) for i in active_idx]

                prompt = build_prompt(label, active_pairs)
                prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                story = ""
                error = None
                attempts = 0
                pchars = 0
                rchars = 0
                backoff_s = 0.0

                for a in range(1, args.max_retries + 1):
                    attempts = a
                    pchars += len(prompt)
                    try:
                        story = ollama_generate(prompt, args)
                    except Exception as exc:
                        error = str(exc)
                    rchars += len(story)
                    qf = _quality_flag(story, args.min_story_chars)
                    if error is None and qf == "ok":
                        break
                    if a < args.max_retries:
                        delay = args.retry_backoff_base * (args.retry_backoff_factor ** (a - 1))
                        time.sleep(delay)
                        backoff_s += delay

                qf = _quality_flag(story, args.min_story_chars)
                norm = _normalize_story(story)
                if norm and seen_norm.get(norm, 0) > 0:
                    qf = "duplicate_story"
                if norm:
                    seen_norm[norm] = seen_norm.get(norm, 0) + 1

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
                        "retry_backoff_seconds": round(backoff_s, 4),
                        "model": OLLAMA_MODEL,
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")

                events.append({"attempts": attempts, "prompt_chars": pchars, "response_chars": rchars})
                if qf != "ok" or error is not None:
                    quarantine_rows.append(
                        {
                            "year": year,
                            "row_index": int(row_idx),
                            "label": label,
                            "reason": qf if qf != "ok" else "generation_error",
                            "record": rec,
                        }
                    )

                done = row_idx + 1
                if done % 5 == 0 or done == n:
                    _progress(done, n, time.time() - t0)
                if args.sleep_between_requests:
                    time.sleep(args.sleep_between_requests)

        if n > 0:
            sys.stdout.write("\n")
        outputs.append((year, out_path, n, start))
        print(f"Wrote {out_path} ({n} rows, start_at={start})")

    all_rows = []
    for year in YEARS:
        p = os.path.join(OUT_DIR, f"stories_{year}.jsonl")
        if os.path.isfile(p):
            all_rows.extend(iter_jsonl(p))

    _write_jsonl(OUT_QUARANTINE, quarantine_rows)
    quality, research = _compute_reports(all_rows, quarantine_rows, events, args, time.time() - run_started)
    with open(OUT_QUALITY, "w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)
    with open(OUT_RESEARCH, "w", encoding="utf-8") as f:
        json.dump(research, f, indent=2)

    repo_root = os.path.dirname(os.path.abspath(__file__))
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quality_profile": QUALITY_PROFILE,
        "git": _git_info(repo_root),
        "mode": "resume" if args.resume else "fresh_run",
        "model": {
            "ollama_url": OLLAMA_URL,
            "ollama_model": OLLAMA_MODEL,
            "temperature": args.temperature,
            "num_ctx": args.num_ctx,
            "timeout_seconds": args.timeout_seconds,
        },
        "quality_config": {
            "min_story_chars": args.min_story_chars,
            "max_retries": args.max_retries,
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
            "features_json": {"path": FEATURES_JSON, "sha256": _sha256_of_file(FEATURES_JSON)},
            **{
                f"data_{y}_x": {
                    "path": os.path.join(BASE_DIR, f"data_{y}", "X.npy"),
                    "sha256": _sha256_of_file(os.path.join(BASE_DIR, f"data_{y}", "X.npy")),
                }
                for y in YEARS
            },
            **{
                f"data_{y}_y": {
                    "path": os.path.join(BASE_DIR, f"data_{y}", "y.npy"),
                    "sha256": _sha256_of_file(os.path.join(BASE_DIR, f"data_{y}", "y.npy")),
                }
                for y in YEARS
            },
        },
        "outputs": {
            **{
                f"stories_{y}": {
                    "path": os.path.join(OUT_DIR, f"stories_{y}.jsonl"),
                    "sha256": _sha256_of_file(os.path.join(OUT_DIR, f"stories_{y}.jsonl")),
                }
                for y in YEARS
            },
            "quarantine": {"path": OUT_QUARANTINE, "sha256": _sha256_of_file(OUT_QUARANTINE)},
            "quality": {"path": OUT_QUALITY, "sha256": _sha256_of_file(OUT_QUALITY)},
            "research": {"path": OUT_RESEARCH, "sha256": _sha256_of_file(OUT_RESEARCH)},
            "readme": {"path": OUT_README, "sha256": _sha256_of_file(OUT_README)},
        },
    }
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(OUT_README, "w", encoding="utf-8") as f:
        f.write("# Story Datasets V2\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Quality profile: {QUALITY_PROFILE}\n")
        f.write(f"- Model: {OLLAMA_MODEL}\n")
        f.write(f"- Mode: {'resume' if args.resume else 'fresh-run'}\n")
        f.write(f"- Quality report: {OUT_QUALITY}\n")
        f.write(f"- Research report: {OUT_RESEARCH}\n")
        f.write(f"- Run manifest: {OUT_MANIFEST}\n")
        for year, p, n, s in outputs:
            f.write(f"- {year}: {p} ({n} rows, start_at={s})\n")

    print(f"Wrote {OUT_QUARANTINE}")
    print(f"Wrote {OUT_QUALITY}")
    print(f"Wrote {OUT_RESEARCH}")
    print(f"Wrote {OUT_MANIFEST}")
    print(f"Wrote {OUT_README}")

    if args.strict_gate and not quality.get("passed", False):
        raise RuntimeError(
            f"Quality gate failed. See {OUT_QUALITY}. Violations: {', '.join(quality.get('violations', []))}"
        )


if __name__ == "__main__":
    main()
