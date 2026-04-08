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
import concurrent.futures
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
    p.add_argument("--max-retries", type=int, default=1)
    p.add_argument("--retry-backoff-base", type=float, default=0.7)
    p.add_argument("--retry-backoff-factor", type=float, default=1.7)
    p.add_argument("--sleep-between-requests", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=2)
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

    # Models often wrap JSON in fenced blocks or add a short preamble first.
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL):
        stripped = m.group(1).strip()
        if stripped:
            candidates.insert(0, stripped)

    # Fall back to the first balanced JSON object embedded anywhere in the text.
    starts = [i for i, ch in enumerate(raw) if ch == "{"][:8]
    for start in starts:
        depth = 0
        in_str = False
        escaped = False
        for idx in range(start, len(raw)):
            ch = raw[idx]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    stripped = raw[start : idx + 1].strip()
                    if stripped:
                        candidates.insert(0, stripped)
                    break

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


def _extract_json_string_field(text: str, field_name: str):
    key = f'"{field_name}"'
    start = 0
    while True:
        idx = text.find(key, start)
        if idx == -1:
            return ""
        colon = text.find(":", idx + len(key))
        if colon == -1:
            return ""
        quote = text.find('"', colon + 1)
        if quote == -1:
            return ""

        buf = []
        i = quote + 1
        while i < len(text):
            ch = text[i]
            if ch == '"':
                tail = text[i + 1 :].lstrip()
                if tail.startswith(",") or tail.startswith("}"):
                    return "".join(buf).strip()
                buf.append(ch)
                i += 1
                continue
            if ch == "\\" and i + 1 < len(text):
                buf.append(ch)
                buf.append(text[i + 1])
                i += 2
                continue
            buf.append(ch)
            i += 1
        start = idx + len(key)


def _extract_schema_fields_relaxed(text: str):
    if not text:
        return None
    raw = text.strip()
    candidates = [raw]
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL):
        stripped = m.group(1).strip()
        if stripped:
            candidates.insert(0, stripped)

    for cand in candidates:
        future_story = _extract_json_string_field(cand, "future_story")
        changes_summary = _extract_json_string_field(cand, "changes_summary")
        if future_story or changes_summary:
            return {"future_story": future_story, "changes_summary": changes_summary}
    return None


def _normalize_section_tag(text: str, label: str):
    patterns = [
        rf"(?im)^\s*\*\*{label}\*\*\s*:?\s*$",
        rf"(?im)^\s*{label}\s*:?\s*$",
        rf"(?im)^\s*\"{label}\"\s*:?\s*$",
        rf"(?im)^\s*{label}\"\s*:?\s*$",
        rf"(?im)^\s*\"?{label}\"?\s*:\s*",
    ]
    out = text
    for pat in patterns:
        out = re.sub(pat, f"{label}:\n", out)
    return out


def _collapse_tag_repeats(text: str):
    out = text
    for label in ["ASSESSMENT", "BEHAVIOR", "RATIONALE"]:
        out = re.sub(rf"(?im)(?:{label}:\s*)+", f"{label}:\n", out)
    return out


def _render_section_value(value):
    if isinstance(value, list):
        parts = []
        for item in value:
            s = str(item).strip()
            if s:
                parts.append(s if s.startswith("*") else f"* {s}")
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            sv = str(v).strip()
            if sv:
                parts.append(f"* {k}: {sv}")
        return "\n".join(parts).strip()
    return str(value or "").strip()


def _coerce_story_text(value):
    if isinstance(value, dict):
        section_keys = ["ASSESSMENT", "BEHAVIOR", "RATIONALE"]
        if any(k in value for k in section_keys):
            parts = []
            for key in section_keys:
                if key in value:
                    rendered = _render_section_value(value.get(key))
                    if rendered:
                        parts.append(f"{key}:\n{rendered}")
            return "\n\n".join(parts).strip()
        return json.dumps(value, ensure_ascii=True)
    return str(value or "").strip()


def _normalize_story_format(story: str):
    s = str(story or "").strip()
    if not s:
        return ""
    s = s.replace("\r\n", "\n")
    s = _normalize_section_tag(s, "ASSESSMENT")
    s = _normalize_section_tag(s, "BEHAVIOR")
    s = _normalize_section_tag(s, "RATIONALE")
    s = _collapse_tag_repeats(s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def _archive_failed_outputs(attempt_tag: str):
    archive_root = os.path.join(OUT_DIR, "failed_quality_runs")
    os.makedirs(archive_root, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(archive_root, f"{stamp}_{attempt_tag}")
    os.makedirs(archive_dir, exist_ok=True)
    for path in [OUT_JSONL, OUT_QUARANTINE, OUT_QUALITY, OUT_RESEARCH, OUT_MANIFEST, OUT_README]:
        if os.path.isfile(path):
            dest = os.path.join(archive_dir, os.path.basename(path))
            with open(path, "rb") as src_f, open(dest, "wb") as dst_f:
                dst_f.write(src_f.read())
    return archive_dir


def build_prompt(rec):
    years_str = ", ".join(str(y) for y in TARGET_YEARS)
    story = rec.get("story", "")
    return f"""
You are rewriting an Android app telemetry summary for a later analysis period ({years_str}).

Original analysis story:
{story}

Task:
Produce ONE story for that later period using the same style as the original year stories.

Rules:
- Match the original year-story structure exactly inside `future_story`:
  ASSESSMENT:
  BEHAVIOR:
  RATIONALE:
- Write it as a direct evidence-style summary of how the sample presents in that later period.
- Do not narrate temporal change explicitly.
- Do not say things like "in the future", "would likely", "may continue", "could shift", "changed to", or "evolved".
- Do not include a forecast framing or compare against the source story.
- Keep wording factual, neutral, and non-dramatic.
- Focus on observable app behavior only (permissions, API usage, network patterns, execution style, data handling).
- Do not infer motive, attacker intent, victim impact, or campaign attribution.
- If uncertainty is necessary, state it sparingly and neutrally without discussing change over time.
- Keep class labels hidden; do not use direct class terms (forbidden: malicious, benign).
- Avoid loaded security terms unless directly supported by the source story
  (forbidden by default: attacker, payload, compromise, weaponize, trojan, ransomware, spyware).
- Do not output markdown.
- Output ONLY valid JSON.

JSON schema:
{{
  "future_story": "Plain text only. Must contain ASSESSMENT:, BEHAVIOR:, and RATIONALE: sections.",
  "changes_summary": ""
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


def _normalize_output_row(row, min_chars: int):
    out = dict(row)
    future_story = _normalize_story_format(_coerce_story_text(out.get("future_story", "")))
    changes_summary = str(out.get("changes_summary", "")).strip()
    if not future_story or not changes_summary:
        parsed = parse_json_relaxed(str(out.get("raw_model_output", "")))
        if not isinstance(parsed, dict):
            parsed = _extract_schema_fields_relaxed(str(out.get("raw_model_output", "")))
        if isinstance(parsed, dict):
            if not future_story:
                parsed_story = parsed.get("future_story", "")
                if not parsed_story and any(k in parsed for k in ["ASSESSMENT", "BEHAVIOR", "RATIONALE"]):
                    parsed_story = {k: parsed.get(k) for k in ["ASSESSMENT", "BEHAVIOR", "RATIONALE"] if k in parsed}
                future_story = _normalize_story_format(_coerce_story_text(parsed_story))
            if not changes_summary:
                changes_summary = str(parsed.get("changes_summary", "")).strip()
    out["future_story"] = _normalize_story_format(future_story)
    out["changes_summary"] = changes_summary
    out["quality_flag"] = _qflag(future_story, min_chars)
    return out


def _build_quarantine(rows):
    quarantine = []
    for out in rows:
        qf = out.get("quality_flag", "ok")
        if qf != "ok" or out.get("error") is not None:
            quarantine.append(
                {
                    "year": out.get("year"),
                    "row_index": out.get("row_index"),
                    "label": out.get("label"),
                    "reason": qf if qf != "ok" else "generation_error",
                    "record": out,
                }
            )
    return quarantine


def _generate_future_job(job):
    rec = job["rec"]
    args = job["args"]
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
            if not isinstance(parsed, dict):
                parsed = _extract_schema_fields_relaxed(raw)
            error = None
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
        parsed_story = parsed.get("future_story", "")
        if not parsed_story and any(k in parsed for k in ["ASSESSMENT", "BEHAVIOR", "RATIONALE"]):
            parsed_story = {k: parsed.get(k) for k in ["ASSESSMENT", "BEHAVIOR", "RATIONALE"] if k in parsed}
        future_story = _normalize_story_format(_coerce_story_text(parsed_story))
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
    event = {"attempts": attempts, "prompt_chars": pchars, "response_chars": rchars}
    return {"out": out, "event": event}


def _qflag(future_story: str, min_chars: int):
    s = (future_story or "").strip()
    if not s:
        return "empty_story"
    if len(s) < min_chars:
        return "short_story"
    if "ASSESSMENT:" not in s or "BEHAVIOR:" not in s or "RATIONALE:" not in s:
        return "missing_required_tags"
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

    existing_rows = []
    if args.resume and os.path.isfile(OUT_JSONL):
        existing_rows = [_normalize_output_row(r, args.min_story_chars) for r in read_jsonl(OUT_JSONL)]
        _write_jsonl(OUT_JSONL, existing_rows)

    start = len(existing_rows) if args.resume else 0
    mode = "a" if args.resume else "w"

    total = len(source)
    events = []
    t0 = time.time()
    pending_jobs = [{"rec": rec, "args": args} for rec in source[start:]]

    with open(OUT_JSONL, mode, encoding="utf-8") as f:
        workers = max(1, args.concurrency)
        if workers == 1:
            iterator = map(_generate_future_job, pending_jobs)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                iterator = ex.map(_generate_future_job, pending_jobs)
                for i, result in enumerate(iterator, start=start):
                    out = result["out"]
                    f.write(json.dumps(out, ensure_ascii=True) + "\n")
                    events.append(result["event"])
                    n = i + 1
                    if n % 5 == 0 or n == total:
                        _progress(n, total, time.time() - t0)
                    if args.sleep_between_requests:
                        time.sleep(args.sleep_between_requests)
            iterator = None

        if iterator is not None:
            for i, result in enumerate(iterator, start=start):
                out = result["out"]
                f.write(json.dumps(out, ensure_ascii=True) + "\n")
                events.append(result["event"])
                n = i + 1
                if n % 5 == 0 or n == total:
                    _progress(n, total, time.time() - t0)
                if args.sleep_between_requests:
                    time.sleep(args.sleep_between_requests)

    if total > 0:
        sys.stdout.write("\n")

    all_rows = list(read_jsonl(OUT_JSONL)) if os.path.isfile(OUT_JSONL) else []
    quarantine = _build_quarantine(all_rows)
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
        archive_dir = _archive_failed_outputs("strict_gate_failed")
        print(f"Archived failed-quality outputs to {archive_dir}")
        raise RuntimeError(
            f"Quality gate failed. See {OUT_QUALITY}. Violations: {', '.join(quality.get('violations', []))}"
        )


if __name__ == "__main__":
    main()
