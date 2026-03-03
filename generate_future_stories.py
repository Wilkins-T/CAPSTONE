"""
Generate "future" stories from existing stories, with updated feature lists.

Modes:
- Default: append/resume generation into OUT_JSONL.
- Repair-only: rewrite only bad rows in an existing OUT_JSONL.

Inputs:
  - drive-download-20260219T230708Z-1-001/data_stories/stories.jsonl
  - selected_features.json

Outputs:
  - drive-download-20260219T230708Z-1-001/data_stories_future/future_stories.jsonl
  - drive-download-20260219T230708Z-1-001/data_stories_future/README_future.md
"""

import argparse
import ast
import json
import os
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

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

TARGET_YEARS = [2016, 2018]
SLEEP_BETWEEN_REQUESTS = 0.0


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
1) Write a revised story describing how this sample would likely appear in {years_str}.
2) Provide the updated list of active features that best match the revised story.

Rules:
- Use neutral, forensic language with uncertainty when appropriate.
- Avoid step-by-step instructions, code, or operational guidance.
- The revised story should be whatever length is needed to explain the behavior.
- Feature list must use exact feature names from the provided list.
- If you remove or add features, keep it realistic and minimal.
- Do not invent features outside the provided list.
- Output ONLY valid JSON, no extra text.

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


def needs_repair(rec):
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
    if not str(rec.get("future_story", "")).strip():
        return True
    return False


def _write_jsonl(path, records):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
    os.replace(tmp, path)


def _write_readme(input_path, output_path, mode, repaired_count=None):
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


def run_default(idx_to_name, name_to_idx):
    start_at = count_existing_lines(OUT_JSONL)
    all_recs = list(read_jsonl(IN_JSONL))
    total = len(all_recs)
    if start_at >= total:
        print(f"Nothing to do. {OUT_JSONL} already has {start_at} lines.")
        return

    t0 = time.time()
    total_chars = 0
    with open(OUT_JSONL, "a", encoding="utf-8") as f:
        for n, source_rec in enumerate(all_recs[start_at:], start=start_at + 1):
            prompt = build_prompt(source_rec, TARGET_YEARS)
            raw = ""
            error = None
            parsed = None
            try:
                raw = ollama_generate(prompt)
                parsed = parse_json_relaxed(raw)
            except Exception as exc:
                error = str(exc)
            total_chars += len(prompt) + len(raw)

            out = build_output_record(source_rec, raw, parsed, error, idx_to_name, name_to_idx)
            if out.get("error") == "no_valid_features_after_validation":
                raise RuntimeError(f"No valid features for row_index={source_rec.get('row_index')}")

            f.write(json.dumps(out, ensure_ascii=True) + "\n")

            if n % 5 == 0 or n == total:
                elapsed = time.time() - t0
                avg_tps = (total_chars / 4) / elapsed if elapsed > 0 else None
                _progress(n, total, elapsed, avg_tps)
            if SLEEP_BETWEEN_REQUESTS:
                time.sleep(SLEEP_BETWEEN_REQUESTS)
    if total > 0:
        sys.stdout.write("\n")

    _write_readme(IN_JSONL, OUT_JSONL, mode="default")
    print(f"Wrote: {OUT_JSONL}")
    print(f"Wrote: {OUT_README}")


def run_repair_only(idx_to_name, name_to_idx, dry_run=False):
    if not os.path.isfile(OUT_JSONL):
        raise FileNotFoundError(f"Missing output to repair: {OUT_JSONL}")

    source_recs = list(read_jsonl(IN_JSONL))
    out_recs = list(read_jsonl(OUT_JSONL))

    if len(out_recs) > len(source_recs):
        raise RuntimeError("Output has more rows than input; refusing repair.")

    repair_idx = []
    for i in range(len(source_recs)):
        current = out_recs[i] if i < len(out_recs) else None
        if needs_repair(current):
            repair_idx.append(i)

    print(f"Repair candidates: {len(repair_idx)} / {len(source_recs)}")
    if dry_run:
        return
    if not repair_idx:
        _write_readme(IN_JSONL, OUT_JSONL, mode="repair-only", repaired_count=0)
        print("Nothing to repair.")
        return

    if len(out_recs) < len(source_recs):
        out_recs.extend({} for _ in range(len(source_recs) - len(out_recs)))

    t0 = time.time()
    total_chars = 0
    repaired = 0

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
        if needs_repair(candidate):
            prompt = build_prompt(source_rec, TARGET_YEARS)
            raw = ""
            parsed = None
            error = None
            try:
                raw = ollama_generate(prompt)
                parsed = parse_json_relaxed(raw)
            except Exception as exc:
                error = str(exc)
            total_chars += len(prompt) + len(raw)
            candidate = build_output_record(source_rec, raw, parsed, error, idx_to_name, name_to_idx)

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

    _write_jsonl(OUT_JSONL, out_recs)
    _write_readme(IN_JSONL, OUT_JSONL, mode="repair-only", repaired_count=repaired)
    print(f"Wrote repaired output: {OUT_JSONL}")
    print(f"Wrote: {OUT_README}")


def main():
    args = parse_args()

    if not os.path.isfile(IN_JSONL):
        raise FileNotFoundError(f"Missing input: {IN_JSONL}")

    idx_to_name, name_to_idx = load_feature_maps(FEATURES_JSON)
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.repair_only:
        run_repair_only(idx_to_name, name_to_idx, dry_run=args.dry_run)
    else:
        if args.dry_run:
            print("--dry-run is only used with --repair-only.")
            return
        run_default(idx_to_name, name_to_idx)


if __name__ == "__main__":
    main()
