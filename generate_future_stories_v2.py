"""
V2 Stage 2: Generate unconstrained future stories from story-only inputs.

Inputs:
  drive-download-20260219T230708Z-1-001/data_stories_v2/stories_2012.jsonl
  drive-download-20260219T230708Z-1-001/data_stories_v2/stories_2014.jsonl

Outputs:
  drive-download-20260219T230708Z-1-001/data_stories_future_v2/future_stories.jsonl
  drive-download-20260219T230708Z-1-001/data_stories_future_v2/future_stories_quarantine.jsonl
  drive-download-20260219T230708Z-1-001/data_stories_future_v2/future_stories_quality_report.json
  drive-download-20260219T230708Z-1-001/data_stories_future_v2/future_stories_research_report.json
  drive-download-20260219T230708Z-1-001/data_stories_future_v2/future_stories_run_manifest.json
  drive-download-20260219T230708Z-1-001/data_stories_future_v2/README_future_v2.md
"""

import argparse
import ast
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request


BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drive-download-20260219T230708Z-1-001")
STORIES_DIR = os.path.join(BASE_DIR, "data_stories_v2")
IN_PATHS = [
    os.path.join(STORIES_DIR, "stories_2012.jsonl"),
    os.path.join(STORIES_DIR, "stories_2014.jsonl"),
]
OUT_DIR = os.path.join(BASE_DIR, "data_stories_future_v2")
OUT_JSONL = os.path.join(OUT_DIR, "future_stories.jsonl")
OUT_QUARANTINE = os.path.join(OUT_DIR, "future_stories_quarantine.jsonl")
OUT_QUALITY = os.path.join(OUT_DIR, "future_stories_quality_report.json")
OUT_RESEARCH = os.path.join(OUT_DIR, "future_stories_research_report.json")
OUT_MANIFEST = os.path.join(OUT_DIR, "future_stories_run_manifest.json")
OUT_README = os.path.join(OUT_DIR, "README_future_v2.md")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
TARGET_YEARS = [2016, 2018]
QUALITY_PROFILE = "research_v1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    p.add_argument("--temperature", type=float, default=0.35)
    p.add_argument("--num-ctx", type=int, default=4096)
    p.add_argument("--timeout-seconds", type=int, default=120)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-backoff-base", type=float, default=0.7)
    p.add_argument("--retry-backoff-factor", type=float, default=1.7)
    p.add_argument("--sleep-between-requests", type=float, default=0.0)
    p.add_argument("--min-story-chars", type=int, default=80)
    p.add_argument("--max-empty-story-rate", type=float, default=0.01)
    p.add_argument("--max-short-story-rate", type=float, default=0.02)
    p.add_argument("--max-error-rate", type=float, default=0.01)
    p.add_argument("--max-quarantine-rate", type=float, default=0.10)
    p.add_argument("--strict-gate", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def read_jsonl(path):
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


def ollama_generate(prompt, args):
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


def parse_json_relaxed(text: str):
    if not text:
        return None
    raw = text.strip()
    candidates = [raw]
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
        if stripped:
            candidates.insert(0, stripped)

    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            pass
        try:
            val = ast.literal_eval(cand)
            if isinstance(val, dict):
                return val
        except Exception:
            pass
    return None


def build_prompt(rec):
    years_str = ", ".join(str(y) for y in TARGET_YEARS)
    story = rec.get("story", "")
    return f"""
You are projecting an Android sample behavior into a future period ({years_str}).

Original analysis story:
{story}

Task:
Produce ONE future-facing story (single combined forecast, not year-split) describing how this sample would likely present in that period.

Rules:
- Forensic, neutral language; no operational guidance.
- Keep wording factual and non-dramatic.
- Focus on plausible observable shifts only (API usage, permission mix, infrastructure pattern, execution timing, packaging style).
- Do not infer motive, attacker intent, victim impact, or campaign attribution.
- Mark uncertainty explicitly with neutral phrasing such as "may" or "could".
- Keep class labels hidden; do not use direct class terms (forbidden: malicious, benign).
- Avoid loaded security terms unless directly supported by the source story
  (forbidden by default: attacker, payload, compromise, weaponize, trojan, ransomware, spyware).
- Do not include markdown/code fences.
- Output ONLY valid JSON.

JSON schema:
{{
  "future_story": "string",
  "changes_summary": "string"
}}
""".strip()


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


def _qflag(future_story: str, min_chars: int):
    s = (future_story or "").strip()
    if not s:
        return "empty_story"
    if len(s) < min_chars:
        return "short_story"
    if re.search(r"\b(malicious|benign)\b", s, flags=re.IGNORECASE):
        return "label_leak_terms"
    if re.search(r"\b(attacker|payload|compromise|weaponize|trojan|ransomware|spyware)\b", s, flags=re.IGNORECASE):
        return "loaded_security_tone"
    return "ok"


def main():
    args = parse_args()
    run_started = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    if not args.resume:
        for p in [OUT_JSONL, OUT_QUARANTINE, OUT_QUALITY, OUT_RESEARCH, OUT_MANIFEST, OUT_README]:
            if os.path.isfile(p):
                os.remove(p)

    source = []
    for p in IN_PATHS:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing input: {p}")
        source.extend(read_jsonl(p))

    start = count_rows(OUT_JSONL) if args.resume else 0
    mode = "a" if args.resume else "w"

    total = len(source)
    events = []
    quarantine = []
    t0 = time.time()

    with open(OUT_JSONL, mode, encoding="utf-8") as f:
        for i, rec in enumerate(source[start:], start=start):
            prompt = build_prompt(rec)
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            parsed = None
            raw = ""
            error = None
            attempts = 0
            pchars = 0
            rchars = 0
            backoff_s = 0.0

            for a in range(1, args.max_retries + 1):
                attempts = a
                pchars += len(prompt)
                try:
                    raw = ollama_generate(prompt, args)
                    parsed = parse_json_relaxed(raw)
                except Exception as exc:
                    error = str(exc)
                rchars += len(raw)
                if isinstance(parsed, dict) and str(parsed.get("future_story", "")).strip():
                    break
                if a < args.max_retries:
                    delay = args.retry_backoff_base * (args.retry_backoff_factor ** (a - 1))
                    time.sleep(delay)
                    backoff_s += delay

            future_story = ""
            changes_summary = ""
            if isinstance(parsed, dict):
                future_story = str(parsed.get("future_story", "")).strip()
                changes_summary = str(parsed.get("changes_summary", "")).strip()

            qf = _qflag(future_story, args.min_story_chars)
            out = {
                "year": rec.get("year"),
                "row_index": rec.get("row_index"),
                "label": rec.get("label"),
                "label_name": rec.get("label_name"),
                "target_years": TARGET_YEARS,
                "source_story": rec.get("story", ""),
                "future_story": future_story,
                "changes_summary": changes_summary,
                "raw_model_output": raw,
                "error": error,
                "generation_attempts": attempts,
                "quality_flag": qf,
                "generation_meta": {
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "prompt_sha256": prompt_hash,
                    "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest() if raw else None,
                    "prompt_chars": pchars,
                    "response_chars": rchars,
                    "retry_backoff_seconds": round(backoff_s, 4),
                    "model": OLLAMA_MODEL,
                },
            }
            f.write(json.dumps(out, ensure_ascii=True) + "\n")

            events.append({"attempts": attempts, "prompt_chars": pchars, "response_chars": rchars})
            if qf != "ok" or error is not None:
                quarantine.append(
                    {
                        "year": out.get("year"),
                        "row_index": out.get("row_index"),
                        "label": out.get("label"),
                        "reason": qf if qf != "ok" else "generation_error",
                        "record": out,
                    }
                )

            n = i + 1
            if n % 5 == 0 or n == total:
                _progress(n, total, time.time() - t0)
            if args.sleep_between_requests:
                time.sleep(args.sleep_between_requests)

    if total > 0:
        sys.stdout.write("\n")

    all_rows = list(read_jsonl(OUT_JSONL)) if os.path.isfile(OUT_JSONL) else []
    _write_jsonl(OUT_QUARANTINE, quarantine)

    n = len(all_rows)
    empty_rows = sum(1 for r in all_rows if r.get("quality_flag") == "empty_story")
    short_rows = sum(1 for r in all_rows if r.get("quality_flag") == "short_story")
    err_rows = sum(1 for r in all_rows if r.get("error"))

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
        },
        "counts": {
            "rows": n,
            "empty_story_rows": empty_rows,
            "short_story_rows": short_rows,
            "error_rows": err_rows,
            "quarantine_rows": len(quarantine),
        },
        "rates": {
            "empty_story_rate": rate(empty_rows),
            "short_story_rate": rate(short_rows),
            "error_rate": rate(err_rows),
            "quarantine_rate": rate(len(quarantine)),
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
    quality["violations"] = violations
    quality["passed"] = len(violations) == 0

    with open(OUT_QUALITY, "w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)

    attempts = [int(e.get("attempts", 0)) for e in events]
    pchars = sum(int(e.get("prompt_chars", 0)) for e in events)
    rchars = sum(int(e.get("response_chars", 0)) for e in events)
    est_toks = (pchars + rchars) / 4.0
    elapsed = time.time() - run_started
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
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60.0 if elapsed > 0 else 0.0,
            "estimated_prompt_tokens": pchars / 4.0,
            "estimated_response_tokens": rchars / 4.0,
            "estimated_total_tokens": est_toks,
            "estimated_tokens_per_second": est_toks / elapsed if elapsed > 0 else 0.0,
            "estimated_tokens_per_minute": est_toks * 60.0 / elapsed if elapsed > 0 else 0.0,
            "token_estimation_method": "chars_div_4_approximation",
        },
    }
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
        },
        "inputs": {f"stories_{i}": {"path": i, "sha256": _sha256_of_file(i)} for i in IN_PATHS},
        "outputs": {
            "future_stories": {"path": OUT_JSONL, "sha256": _sha256_of_file(OUT_JSONL)},
            "quarantine": {"path": OUT_QUARANTINE, "sha256": _sha256_of_file(OUT_QUARANTINE)},
            "quality": {"path": OUT_QUALITY, "sha256": _sha256_of_file(OUT_QUALITY)},
            "research": {"path": OUT_RESEARCH, "sha256": _sha256_of_file(OUT_RESEARCH)},
            "readme": {"path": OUT_README, "sha256": _sha256_of_file(OUT_README)},
        },
    }
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(OUT_README, "w", encoding="utf-8") as f:
        f.write("# Future Stories V2\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Quality profile: {QUALITY_PROFILE}\n")
        f.write(f"- Input files: {', '.join(IN_PATHS)}\n")
        f.write(f"- Output: {OUT_JSONL}\n")
        f.write(f"- Quarantine: {OUT_QUARANTINE}\n")
        f.write(f"- Quality report: {OUT_QUALITY}\n")
        f.write(f"- Research report: {OUT_RESEARCH}\n")
        f.write(f"- Run manifest: {OUT_MANIFEST}\n")
        f.write(f"- Model: {OLLAMA_MODEL}\n")
        f.write(f"- Resume: {args.resume}\n")

    print(f"Wrote {OUT_JSONL}")
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
