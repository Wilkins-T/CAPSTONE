"""
Generate "future" stories from existing stories, with updated feature lists.

Modes:
- Default: full fresh generation into OUT_JSONL.
- Repair-only: rewrite only bad rows in an existing OUT_JSONL.

Inputs:
  - drive-download-20260219T230708Z-1-001/data_stories/stories.jsonl
  - selected_features.json

Outputs:
  - drive-download-20260219T230708Z-1-001/data_stories_future/future_stories.jsonl
  - drive-download-20260219T230708Z-1-001/data_stories_future/future_stories_quarantine.jsonl
  - drive-download-20260219T230708Z-1-001/data_stories_future/future_stories_quality_report.json
  - drive-download-20260219T230708Z-1-001/data_stories_future/future_stories_research_report.json
  - drive-download-20260219T230708Z-1-001/data_stories_future/README_future.md
"""

import argparse
import ast
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request


BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "drive-download-20260219T230708Z-1-001",
)
IN_JSONL = os.path.join(BASE_DIR, "data_stories", "stories.jsonl")
FEATURES_JSON = os.path.join(BASE_DIR, "selected_features.json")
OUT_DIR = os.path.join(BASE_DIR, "data_stories_future")
OUT_JSONL = os.path.join(OUT_DIR, "future_stories.jsonl")
OUT_README = os.path.join(OUT_DIR, "README_future.md")
OUT_QUARANTINE = os.path.join(OUT_DIR, "future_stories_quarantine.jsonl")
OUT_QUALITY = os.path.join(OUT_DIR, "future_stories_quality_report.json")
OUT_RESEARCH = os.path.join(OUT_DIR, "future_stories_research_report.json")
OUT_MANIFEST = os.path.join(OUT_DIR, "future_stories_run_manifest.json")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

TARGET_YEARS = [2016, 2018]
SLEEP_BETWEEN_REQUESTS = 0.0

# Harmonized threshold profile
QUALITY_PROFILE = "research_v1"
MIN_STORY_CHARS = 80
MIN_FEATURES_IF_ORIGINAL_NON_EMPTY = 2
MAX_REPAIR_RETRIES = 5
MAX_EMPTY_STORY_RATE = 0.01
MAX_SHORT_STORY_RATE = 0.02
MAX_FALLBACK_TO_ORIGINAL_RATE = 0.0
MAX_UNCHANGED_FEATURE_RATE = 0.10
MAX_ERROR_RATE = 0.01
MAX_QUARANTINE_RATE = 0.10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repair-only",
        action="store_true",
        help="Rewrite only rows with errors/fallback/empty story in existing output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show how many rows would be repaired; do not write changes.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume default generation from existing future_stories.jsonl.",
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


def ollama_generate(prompt: str):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.4, "num_ctx": 4096},
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


def _strip_code_fences(text: str):
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.splitlines()
    if not lines:
        return t
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_json_relaxed(text: str):
    if not text:
        return None

    candidates = []
    raw = text.strip()
    stripped = _strip_code_fences(raw)
    if stripped:
        candidates.append(stripped)

    obj = _extract_json_object(stripped or raw)
    if obj:
        candidates.append(obj)

    if raw and raw not in candidates:
        candidates.append(raw)

    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            pass
        try:
            val = ast.literal_eval(cand)
            if isinstance(val, (dict, list)):
                return val
        except Exception:
            pass
    return None


def build_prompt(rec, target_years):
    label_name = rec.get("label_name", "malicious" if rec.get("label", 0) == 1 else "benign")
    features = rec.get("active_features", [])
    lines = [f"{f.get('index')} | {f.get('name')}" for f in features]
    feature_block = "\n".join(lines) if lines else "(no active features)"
    story = rec.get("story", "")

    years_str = ", ".join(str(y) for y in target_years)
    prompt = f"""
You are a malware analyst looking ahead to years {years_str}.

Ground truth label (keep the same): {label_name}

Original story:
{story}

Original active features (index | name):
{feature_block}

Task:
1) Write ONE revised future story (single combined forecast) describing how this sample
   would likely appear across the later period ({years_str}).
2) Provide ONE updated list of active features that best match that single future story.

Rules:
- Use neutral, forensic language with uncertainty when appropriate.
- Avoid step-by-step instructions, code, or operational guidance.
- The revised story should be whatever length is needed to explain the behavior.
- Feature list must use exact feature names from the provided list.
- The future feature list should not be an unchanged copy of the original unless clearly justified in changes_summary.
- If you remove or add features, keep changes realistic and minimal.
- Do not invent features outside the provided list.
- Output ONLY valid JSON, no extra text, no markdown/code fences.
- Do NOT output year-specific keys like future_story_2016 or future_active_features_2018.

JSON schema:
{{
  "future_story": "string",
  "future_active_features": [{{"index": int, "name": "string"}}],
  "changes_summary": "string"
}}
"""
    return prompt.strip()


def validate_features(lst, idx_to_name, name_to_idx):
    cleaned = []
    for item in lst or []:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        name = item.get("name")
        if idx is not None:
            try:
                idx = int(idx)
            except ValueError:
                idx = None
        if name is not None:
            name = str(name)
        if idx is not None and idx in idx_to_name:
            canon = idx_to_name[idx]
            if name is None or name == canon:
                cleaned.append({"index": idx, "name": canon})
                continue
        if name is not None and name in name_to_idx:
            cleaned.append({"index": name_to_idx[name], "name": name})
            continue
    seen = set()
    out = []
    for item in cleaned:
        if item["index"] in seen:
            continue
        seen.add(item["index"])
        out.append(item)
    return out


def ensure_non_empty_features(future_features, fallback_features, idx_to_name, name_to_idx):
    cleaned = validate_features(future_features, idx_to_name, name_to_idx)
    if cleaned:
        return cleaned, None
    cleaned = validate_features(fallback_features, idx_to_name, name_to_idx)
    if cleaned:
        return cleaned, "empty_future_features_fallback_to_original"
    return [], "empty_future_features_no_valid_fallback"


def _coerce_story(story_value):
    if isinstance(story_value, str):
        return story_value.strip()
    if isinstance(story_value, list):
        parts = []
        for item in story_value:
            if isinstance(item, dict) and item.get("story"):
                parts.append(str(item.get("story")).strip())
            elif isinstance(item, str):
                parts.append(item.strip())
        return "\n\n".join([p for p in parts if p])
    if story_value is None:
        return ""
    return str(story_value).strip()


def _merge_year_specific_features(parsed):
    out = []
    for k, v in parsed.items():
        if not k.startswith("future_active_features_"):
            continue
        if isinstance(v, list):
            out.extend(v)
    return out


def _merge_year_specific_story(parsed):
    stories = []
    for k, v in parsed.items():
        if not k.startswith("future_story_"):
            continue
        if isinstance(v, str) and v.strip():
            stories.append(v.strip())
    return "\n\n".join(stories)


def normalize_parsed_output(parsed):
    if not isinstance(parsed, dict):
        return "", [], ""

    future_story = _coerce_story(parsed.get("future_story"))
    if not future_story:
        future_story = _merge_year_specific_story(parsed)

    future_features = parsed.get("future_active_features", None)
    if future_features is None:
        future_features = _merge_year_specific_features(parsed)
    if not isinstance(future_features, list):
        future_features = []

    changes_summary = parsed.get("changes_summary", "")
    if not isinstance(changes_summary, str):
        changes_summary = str(changes_summary)
    changes_summary = changes_summary.strip()

    return future_story, future_features, changes_summary


def allow_empty_if_original_empty(source_rec, idx_to_name, name_to_idx):
    return not validate_features(source_rec.get("active_features", []), idx_to_name, name_to_idx)


def build_output_record(source_rec, raw, parsed, error, idx_to_name, name_to_idx):
    future_story = ""
    future_features = []
    changes_summary = ""
    if parsed is not None:
        future_story, future_features, changes_summary = normalize_parsed_output(parsed)

    future_features, validate_note = ensure_non_empty_features(
        future_features, source_rec.get("active_features", []), idx_to_name, name_to_idx
    )

    if not future_features and not allow_empty_if_original_empty(source_rec, idx_to_name, name_to_idx):
        error = error or "no_valid_features_after_validation"
    elif not future_features:
        validate_note = validate_note or "empty_original_features_allowed"

    return {
        "year": source_rec.get("year"),
        "row_index": source_rec.get("row_index"),
        "label": source_rec.get("label"),
        "label_name": source_rec.get("label_name"),
        "target_years": TARGET_YEARS,
        "future_story": future_story,
        "future_active_features": future_features,
        "changes_summary": changes_summary,
        "raw_model_output": raw,
        "validation_note": validate_note,
        "error": error,
    }


def _feature_signature(lst, idx_to_name, name_to_idx):
    clean = validate_features(lst, idx_to_name, name_to_idx)
    return tuple(sorted(int(item["index"]) for item in clean))


def _same_features_as_original(source_rec, rec, idx_to_name, name_to_idx):
    src_sig = _feature_signature(source_rec.get("active_features", []), idx_to_name, name_to_idx)
    if not src_sig:
        return False
    rec_sig = _feature_signature(rec.get("future_active_features", []), idx_to_name, name_to_idx)
    return rec_sig == src_sig


def needs_repair(source_rec, rec, idx_to_name, name_to_idx):
    if not rec:
        return True
    if rec.get("error"):
        return True
    note = rec.get("validation_note")
    if note in {
        "empty_future_features_fallback_to_original",
        "empty_future_features_no_valid_fallback",
    }:
        return True
    story = str(rec.get("future_story", "")).strip()
    if not story:
        return True
    if len(story) < MIN_STORY_CHARS:
        return True
    src_non_empty = bool(_feature_signature(source_rec.get("active_features", []), idx_to_name, name_to_idx))
    n_future = len(validate_features(rec.get("future_active_features", []), idx_to_name, name_to_idx))
    if src_non_empty and n_future < MIN_FEATURES_IF_ORIGINAL_NON_EMPTY:
        return True
    if _same_features_as_original(source_rec, rec, idx_to_name, name_to_idx):
        return True
    return False


def _write_jsonl(path, records):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
    os.replace(tmp, path)


def _write_readme(input_path, output_path, mode, repaired_count=None, efficiency=None):
    with open(OUT_README, "w", encoding="utf-8") as f:
        f.write("# Future Story Dataset\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"- Input: {input_path}\n")
        f.write(f"- Output: {output_path}\n")
        f.write(f"- Model: {OLLAMA_MODEL}\n")
        f.write(f"- Target years: {', '.join(str(y) for y in TARGET_YEARS)}\n")
        f.write(f"- Mode: {mode}\n")
        if repaired_count is not None:
            f.write(f"- Rows repaired: {repaired_count}\n")
        if efficiency is not None:
            f.write(f"- Elapsed seconds: {efficiency.get('elapsed_seconds', 0.0):.2f}\n")
            f.write(f"- Estimated total tokens: {efficiency.get('estimated_total_tokens', 0.0):.1f}\n")
            f.write(f"- Estimated tokens/second: {efficiency.get('estimated_tokens_per_second', 0.0):.2f}\n")
        f.write(f"- Quarantine output: {OUT_QUARANTINE}\n")
        f.write(f"- Quality report: {OUT_QUALITY}\n")
        f.write(f"- Research report: {OUT_RESEARCH}\n")
        f.write(f"- Run manifest: {OUT_MANIFEST}\n")


def _write_quarantine(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


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


def _make_quarantine_row(source_rec, candidate, reason, attempts):
    return {
        "year": source_rec.get("year"),
        "row_index": source_rec.get("row_index"),
        "label": source_rec.get("label"),
        "reason": reason,
        "attempts": attempts,
        "candidate": candidate,
    }


def _generate_with_retries(source_rec, idx_to_name, name_to_idx, max_retries):
    total_chars = 0
    total_prompt_chars = 0
    total_raw_chars = 0
    parse_success_attempts = 0
    exception_attempts = 0
    last = None
    attempts = 0
    for attempt in range(1, max_retries + 1):
        attempts = attempt
        prompt = build_prompt(source_rec, TARGET_YEARS)
        total_prompt_chars += len(prompt)
        raw = ""
        parsed = None
        error = None
        try:
            raw = ollama_generate(prompt)
            parsed = parse_json_relaxed(raw)
            if parsed is not None:
                parse_success_attempts += 1
        except Exception as exc:
            error = str(exc)
            exception_attempts += 1
        total_raw_chars += len(raw)
        total_chars += len(prompt) + len(raw)
        candidate = build_output_record(source_rec, raw, parsed, error, idx_to_name, name_to_idx)
        last = candidate
        if not needs_repair(source_rec, candidate, idx_to_name, name_to_idx):
            meta = {
                "attempts": attempts,
                "prompt_chars": total_prompt_chars,
                "raw_chars": total_raw_chars,
                "parse_success_attempts": parse_success_attempts,
                "exception_attempts": exception_attempts,
            }
            return candidate, attempts, total_chars, meta

    if last is None:
        last = build_output_record(source_rec, "", None, "empty_generation_attempt", idx_to_name, name_to_idx)
    if not last.get("error"):
        last["error"] = "quarantined_unrepaired_after_retries"
    last["validation_note"] = "quarantined_unrepaired"
    meta = {
        "attempts": attempts,
        "prompt_chars": total_prompt_chars,
        "raw_chars": total_raw_chars,
        "parse_success_attempts": parse_success_attempts,
        "exception_attempts": exception_attempts,
    }
    return last, attempts, total_chars, meta


def _summarize_generation_events(events):
    if not events:
        return {
            "events": 0,
            "avg_attempts": 0.0,
            "max_attempts": 0,
            "parse_success_event_rate": 0.0,
            "avg_prompt_chars": 0.0,
            "avg_raw_chars": 0.0,
            "total_prompt_chars": 0,
            "total_raw_chars": 0,
            "attempts_histogram": {},
        }
    attempts = [int(e.get("attempts", 0)) for e in events]
    prompt_chars = [int(e.get("prompt_chars", 0)) for e in events]
    raw_chars = [int(e.get("raw_chars", 0)) for e in events]
    parse_success_events = sum(1 for e in events if int(e.get("parse_success_attempts", 0)) > 0)
    hist = {}
    for a in attempts:
        k = str(a)
        hist[k] = hist.get(k, 0) + 1
    return {
        "events": len(events),
        "avg_attempts": float(sum(attempts) / len(attempts)),
        "max_attempts": max(attempts),
        "parse_success_event_rate": float(parse_success_events / len(events)),
        "avg_prompt_chars": float(sum(prompt_chars) / len(events)),
        "avg_raw_chars": float(sum(raw_chars) / len(events)),
        "total_prompt_chars": int(sum(prompt_chars)),
        "total_raw_chars": int(sum(raw_chars)),
        "attempts_histogram": hist,
    }


def _percentile(vals, p):
    if not vals:
        return 0.0
    arr = sorted(vals)
    if len(arr) == 1:
        return float(arr[0])
    idx = int(round((len(arr) - 1) * p))
    idx = max(0, min(idx, len(arr) - 1))
    return float(arr[idx])


def _group_counts(rows, key):
    out = {}
    for r in rows:
        k = str(r.get(key))
        out[k] = out.get(k, 0) + 1
    return out


def _compute_research_report(mode, source_recs, out_recs, quarantine_rows, generation_events, idx_to_name, name_to_idx):
    checked = min(len(source_recs), len(out_recs))
    drift_jaccard = []
    add_counts = []
    remove_counts = []
    changed = 0
    unchanged = 0
    for i in range(checked):
        src = source_recs[i]
        rec = out_recs[i]
        src_set = set(_feature_signature(src.get("active_features", []), idx_to_name, name_to_idx))
        rec_set = set(_feature_signature(rec.get("future_active_features", []), idx_to_name, name_to_idx))
        inter = len(src_set & rec_set)
        union = len(src_set | rec_set)
        j = float(inter / union) if union > 0 else 1.0
        drift_jaccard.append(j)
        adds = len(rec_set - src_set)
        rems = len(src_set - rec_set)
        add_counts.append(adds)
        remove_counts.append(rems)
        if rec_set == src_set:
            unchanged += 1
        else:
            changed += 1

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quality_profile": QUALITY_PROFILE,
        "mode": mode,
        "model": OLLAMA_MODEL,
        "target_years": TARGET_YEARS,
        "source_rows": len(source_recs),
        "output_rows": len(out_recs),
        "quarantine_rows": len(quarantine_rows),
        "source_by_year": _group_counts(source_recs, "year"),
        "source_by_label": _group_counts(source_recs, "label_name"),
        "quarantine_by_year": _group_counts(quarantine_rows, "year"),
        "quarantine_by_label": _group_counts(quarantine_rows, "label"),
        "generation_stats": _summarize_generation_events(generation_events),
        "drift_stats": {
            "changed_feature_rows": changed,
            "unchanged_feature_rows": unchanged,
            "changed_rate": float(changed / checked) if checked > 0 else 0.0,
            "avg_jaccard_similarity": float(sum(drift_jaccard) / checked) if checked > 0 else 0.0,
            "median_jaccard_similarity": float(statistics.median(drift_jaccard)) if drift_jaccard else 0.0,
            "p25_jaccard_similarity": _percentile(drift_jaccard, 0.25),
            "p75_jaccard_similarity": _percentile(drift_jaccard, 0.75),
            "avg_added_features": float(sum(add_counts) / checked) if checked > 0 else 0.0,
            "avg_removed_features": float(sum(remove_counts) / checked) if checked > 0 else 0.0,
        },
    }
    return report


def _write_research_report(path, report):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def _compute_quality_report(source_recs, out_recs, quarantine_rows, idx_to_name, name_to_idx):
    total_source = len(source_recs)
    total_output = len(out_recs)
    checked = min(total_source, total_output)
    empty_story = 0
    short_story = 0
    fallback_to_original = 0
    unchanged_features = 0
    error_rows = 0

    for i in range(checked):
        src = source_recs[i]
        rec = out_recs[i]
        story = str(rec.get("future_story", "")).strip()
        if not story:
            empty_story += 1
        if story and len(story) < MIN_STORY_CHARS:
            short_story += 1
        note = rec.get("validation_note")
        if note == "empty_future_features_fallback_to_original":
            fallback_to_original += 1
        if _same_features_as_original(src, rec, idx_to_name, name_to_idx):
            unchanged_features += 1
        if rec.get("error"):
            error_rows += 1

    # Missing rows are always a hard failure.
    missing_rows = max(0, total_source - total_output)
    error_rows += missing_rows

    def _rate(n, d):
        return float(n) / float(d) if d > 0 else 0.0

    denom = checked if checked > 0 else 1
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quality_profile": QUALITY_PROFILE,
        "thresholds": {
            "max_empty_story_rate": MAX_EMPTY_STORY_RATE,
            "max_short_story_rate": MAX_SHORT_STORY_RATE,
            "max_fallback_to_original_rate": MAX_FALLBACK_TO_ORIGINAL_RATE,
            "max_unchanged_feature_rate": MAX_UNCHANGED_FEATURE_RATE,
            "max_error_rate": MAX_ERROR_RATE,
            "max_quarantine_rate": MAX_QUARANTINE_RATE,
        },
        "counts": {
            "total_source_rows": total_source,
            "total_output_rows": total_output,
            "checked_rows": checked,
            "missing_rows": missing_rows,
            "empty_story_rows": empty_story,
            "short_story_rows": short_story,
            "fallback_to_original_rows": fallback_to_original,
            "unchanged_feature_rows": unchanged_features,
            "error_rows": error_rows,
            "quarantine_rows": len(quarantine_rows),
        },
        "rates": {
            "empty_story_rate": _rate(empty_story, denom),
            "short_story_rate": _rate(short_story, denom),
            "fallback_to_original_rate": _rate(fallback_to_original, denom),
            "unchanged_feature_rate": _rate(unchanged_features, denom),
            "error_rate": _rate(error_rows, max(1, total_source)),
            "quarantine_rate": _rate(len(quarantine_rows), max(1, total_source)),
        },
    }
    violations = []
    rates = report["rates"]
    if rates["empty_story_rate"] > MAX_EMPTY_STORY_RATE:
        violations.append("empty_story_rate_above_threshold")
    if rates["short_story_rate"] > MAX_SHORT_STORY_RATE:
        violations.append("short_story_rate_above_threshold")
    if rates["fallback_to_original_rate"] > MAX_FALLBACK_TO_ORIGINAL_RATE:
        violations.append("fallback_to_original_rate_above_threshold")
    if rates["unchanged_feature_rate"] > MAX_UNCHANGED_FEATURE_RATE:
        violations.append("unchanged_feature_rate_above_threshold")
    if rates["error_rate"] > MAX_ERROR_RATE:
        violations.append("error_rate_above_threshold")
    if rates["quarantine_rate"] > MAX_QUARANTINE_RATE:
        violations.append("quarantine_rate_above_threshold")
    if missing_rows > 0:
        violations.append("missing_rows_detected")
    report["violations"] = violations
    report["passed"] = len(violations) == 0
    return report


def _write_quality_report(path, report):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def _write_manifest(mode, resume, source_rows, output_rows, rows_generated_this_run):
    repo_root = os.path.dirname(os.path.abspath(__file__))
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quality_profile": QUALITY_PROFILE,
        "git": _git_info(repo_root),
        "mode": mode,
        "resume": bool(resume),
        "model": {
            "ollama_url": OLLAMA_URL,
            "ollama_model": OLLAMA_MODEL,
        },
        "runtime_config": {
            "target_years": TARGET_YEARS,
            "min_story_chars": MIN_STORY_CHARS,
            "max_repair_retries": MAX_REPAIR_RETRIES,
            "max_empty_story_rate": MAX_EMPTY_STORY_RATE,
            "max_short_story_rate": MAX_SHORT_STORY_RATE,
            "max_fallback_to_original_rate": MAX_FALLBACK_TO_ORIGINAL_RATE,
            "max_unchanged_feature_rate": MAX_UNCHANGED_FEATURE_RATE,
            "max_error_rate": MAX_ERROR_RATE,
            "max_quarantine_rate": MAX_QUARANTINE_RATE,
        },
        "rows": {
            "source_rows": int(source_rows),
            "output_rows": int(output_rows),
            "rows_generated_this_run": int(rows_generated_this_run),
        },
        "inputs": {
            "stories_jsonl": {"path": IN_JSONL, "sha256": _sha256_of_file(IN_JSONL)},
            "selected_features": {"path": FEATURES_JSON, "sha256": _sha256_of_file(FEATURES_JSON)},
        },
        "outputs": {
            "future_stories_jsonl": {"path": OUT_JSONL, "sha256": _sha256_of_file(OUT_JSONL)},
            "quarantine_jsonl": {"path": OUT_QUARANTINE, "sha256": _sha256_of_file(OUT_QUARANTINE)},
            "quality_report": {"path": OUT_QUALITY, "sha256": _sha256_of_file(OUT_QUALITY)},
            "research_report": {"path": OUT_RESEARCH, "sha256": _sha256_of_file(OUT_RESEARCH)},
            "readme": {"path": OUT_README, "sha256": _sha256_of_file(OUT_README)},
        },
    }
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _compute_efficiency_stats(generation_events, elapsed_seconds):
    gen = _summarize_generation_events(generation_events)
    prompt_chars = int(gen.get("total_prompt_chars", 0))
    raw_chars = int(gen.get("total_raw_chars", 0))
    estimated_prompt_tokens = prompt_chars / 4.0
    estimated_response_tokens = raw_chars / 4.0
    estimated_total_tokens = estimated_prompt_tokens + estimated_response_tokens
    rows = int(gen.get("events", 0))
    elapsed = float(elapsed_seconds) if elapsed_seconds and elapsed_seconds > 0 else 0.0
    return {
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60.0 if elapsed > 0 else 0.0,
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "estimated_response_tokens": estimated_response_tokens,
        "estimated_total_tokens": estimated_total_tokens,
        "estimated_tokens_per_second": (estimated_total_tokens / elapsed) if elapsed > 0 else 0.0,
        "estimated_tokens_per_minute": (estimated_total_tokens * 60.0 / elapsed) if elapsed > 0 else 0.0,
        "rows_generated": rows,
        "rows_per_second": (rows / elapsed) if elapsed > 0 else 0.0,
        "rows_per_minute": (rows * 60.0 / elapsed) if elapsed > 0 else 0.0,
        "token_estimation_method": "chars_div_4_approximation",
    }


def _finalize_and_gate(
    source_recs,
    out_recs,
    quarantine_rows,
    generation_events,
    idx_to_name,
    name_to_idx,
    mode,
    elapsed_seconds,
):
    _write_jsonl(OUT_JSONL, out_recs)
    _write_quarantine(OUT_QUARANTINE, quarantine_rows)
    report = _compute_quality_report(
        source_recs=source_recs,
        out_recs=out_recs,
        quarantine_rows=quarantine_rows,
        idx_to_name=idx_to_name,
        name_to_idx=name_to_idx,
    )
    _write_quality_report(OUT_QUALITY, report)
    research = _compute_research_report(
        mode=mode,
        source_recs=source_recs,
        out_recs=out_recs,
        quarantine_rows=quarantine_rows,
        generation_events=generation_events,
        idx_to_name=idx_to_name,
        name_to_idx=name_to_idx,
    )
    research["efficiency"] = _compute_efficiency_stats(generation_events, elapsed_seconds)
    _write_research_report(OUT_RESEARCH, research)
    if not report["passed"]:
        raise RuntimeError(
            f"Quality gate failed. See {OUT_QUALITY}. Violations: {', '.join(report['violations'])}"
        )


def run_default(idx_to_name, name_to_idx, resume=False):
    run_started = time.time()
    all_recs = list(read_jsonl(IN_JSONL))
    total = len(all_recs)
    quarantine_rows = []
    out_recs = []
    generation_events = []
    start_idx = 0

    if resume:
        if os.path.isfile(OUT_JSONL):
            out_recs = list(read_jsonl(OUT_JSONL))
            if len(out_recs) > total:
                raise RuntimeError("Existing output has more rows than input; refusing resume.")
            start_idx = len(out_recs)
            print(f"Resume enabled. Existing rows: {start_idx}/{total}")
        else:
            print("Resume enabled but no existing output found. Starting fresh.")
    else:
        for p in [OUT_JSONL, OUT_QUARANTINE, OUT_QUALITY, OUT_RESEARCH, OUT_README, OUT_MANIFEST]:
            if os.path.isfile(p):
                os.remove(p)

    t0 = time.time()
    total_chars = 0
    for n, source_rec in enumerate(all_recs[start_idx:], start=start_idx + 1):
        candidate, attempts, chars_used, meta = _generate_with_retries(
            source_rec, idx_to_name, name_to_idx, max_retries=MAX_REPAIR_RETRIES
        )
        generation_events.append(
            {
                "year": source_rec.get("year"),
                "label": source_rec.get("label"),
                "row_index": source_rec.get("row_index"),
                **meta,
            }
        )
        total_chars += chars_used
        if needs_repair(source_rec, candidate, idx_to_name, name_to_idx):
            quarantine_rows.append(
                _make_quarantine_row(
                    source_rec,
                    candidate,
                    reason="default_generation_failed_quality",
                    attempts=attempts,
                )
            )
        out_recs.append(candidate)

        if n % 5 == 0 or n == total:
            elapsed = time.time() - t0
            avg_tps = (total_chars / 4) / elapsed if elapsed > 0 else None
            _progress(n, total, elapsed, avg_tps)
        if SLEEP_BETWEEN_REQUESTS:
            time.sleep(SLEEP_BETWEEN_REQUESTS)
    if total > 0:
        sys.stdout.write("\n")

    elapsed_seconds = time.time() - run_started
    efficiency = _compute_efficiency_stats(generation_events, elapsed_seconds)
    _finalize_and_gate(
        all_recs,
        out_recs,
        quarantine_rows,
        generation_events,
        idx_to_name,
        name_to_idx,
        mode="default",
        elapsed_seconds=elapsed_seconds,
    )
    _write_readme(IN_JSONL, OUT_JSONL, mode="default", efficiency=efficiency)
    _write_manifest(
        mode="default",
        resume=resume,
        source_rows=len(all_recs),
        output_rows=len(out_recs),
        rows_generated_this_run=max(0, len(all_recs) - start_idx),
    )
    print(f"Wrote: {OUT_JSONL}")
    print(f"Wrote: {OUT_QUARANTINE}")
    print(f"Wrote: {OUT_QUALITY}")
    print(f"Wrote: {OUT_RESEARCH}")
    print(f"Wrote: {OUT_MANIFEST}")
    print(f"Wrote: {OUT_README}")


def run_repair_only(idx_to_name, name_to_idx, dry_run=False):
    run_started = time.time()
    if not os.path.isfile(OUT_JSONL):
        raise FileNotFoundError(f"Missing output to repair: {OUT_JSONL}")

    source_recs = list(read_jsonl(IN_JSONL))
    out_recs = list(read_jsonl(OUT_JSONL))

    if len(out_recs) > len(source_recs):
        raise RuntimeError("Output has more rows than input; refusing repair.")
    if len(out_recs) < len(source_recs):
        out_recs.extend({} for _ in range(len(source_recs) - len(out_recs)))

    repair_idx = []
    for i in range(len(source_recs)):
        current = out_recs[i]
        if needs_repair(source_recs[i], current, idx_to_name, name_to_idx):
            repair_idx.append(i)

    print(f"Repair candidates: {len(repair_idx)} / {len(source_recs)}")
    if dry_run:
        return
    if not repair_idx:
        _write_readme(IN_JSONL, OUT_JSONL, mode="repair-only", repaired_count=0)
        print("Nothing to repair.")
        return

    t0 = time.time()
    total_chars = 0
    repaired = 0
    quarantine_rows = []
    generation_events = []

    for n, i in enumerate(repair_idx, start=1):
        source_rec = source_recs[i]
        existing = out_recs[i] if isinstance(out_recs[i], dict) else {}

        raw_existing = str(existing.get("raw_model_output", "") or "")
        parsed_existing = parse_json_relaxed(raw_existing)
        candidate = build_output_record(
            source_rec,
            raw_existing,
            parsed_existing,
            existing.get("error"),
            idx_to_name,
            name_to_idx,
        )

        # If still bad after salvage, regenerate this row.
        attempts = 0
        if needs_repair(source_rec, candidate, idx_to_name, name_to_idx):
            candidate, attempts, chars_used, meta = _generate_with_retries(
                source_rec, idx_to_name, name_to_idx, max_retries=MAX_REPAIR_RETRIES
            )
            generation_events.append(
                {
                    "year": source_rec.get("year"),
                    "label": source_rec.get("label"),
                    "row_index": source_rec.get("row_index"),
                    **meta,
                }
            )
            total_chars += chars_used
            if needs_repair(source_rec, candidate, idx_to_name, name_to_idx):
                quarantine_rows.append(
                    _make_quarantine_row(
                        source_rec,
                        candidate,
                        reason="repair_generation_failed_quality",
                        attempts=attempts,
                    )
                )

        out_recs[i] = candidate
        repaired += 1

        if n % 5 == 0 or n == len(repair_idx):
            elapsed = time.time() - t0
            avg_tps = (total_chars / 4) / elapsed if elapsed > 0 and total_chars > 0 else None
            _progress(n, len(repair_idx), elapsed, avg_tps)
        if SLEEP_BETWEEN_REQUESTS:
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    if repair_idx:
        sys.stdout.write("\n")

    elapsed_seconds = time.time() - run_started
    efficiency = _compute_efficiency_stats(generation_events, elapsed_seconds)
    _finalize_and_gate(
        source_recs,
        out_recs,
        quarantine_rows,
        generation_events,
        idx_to_name,
        name_to_idx,
        mode="repair-only",
        elapsed_seconds=elapsed_seconds,
    )
    _write_readme(
        IN_JSONL,
        OUT_JSONL,
        mode="repair-only",
        repaired_count=repaired,
        efficiency=efficiency,
    )
    _write_manifest(
        mode="repair-only",
        resume=True,
        source_rows=len(source_recs),
        output_rows=len(out_recs),
        rows_generated_this_run=len(repair_idx),
    )
    print(f"Wrote repaired output: {OUT_JSONL}")
    print(f"Wrote: {OUT_QUARANTINE}")
    print(f"Wrote: {OUT_QUALITY}")
    print(f"Wrote: {OUT_RESEARCH}")
    print(f"Wrote: {OUT_MANIFEST}")
    print(f"Wrote: {OUT_README}")


def main():
    args = parse_args()

    if not os.path.isfile(IN_JSONL):
        raise FileNotFoundError(f"Missing input: {IN_JSONL}")

    idx_to_name, name_to_idx = load_feature_maps(FEATURES_JSON)
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.repair_only:
        if args.resume:
            print("--resume is ignored with --repair-only.")
        run_repair_only(idx_to_name, name_to_idx, dry_run=args.dry_run)
    else:
        if args.dry_run:
            print("--dry-run is only used with --repair-only.")
            return
        run_default(idx_to_name, name_to_idx, resume=args.resume)


if __name__ == "__main__":
    main()
