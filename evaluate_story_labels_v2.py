"""
Direct LLM label evaluation on year-story text for 2016 and 2018.

Inputs:
  drive-download-20260219T230708Z-1-001/data_stories_v2/stories_2016.jsonl
  drive-download-20260219T230708Z-1-001/data_stories_v2/stories_2018.jsonl

Outputs:
  drive-download-20260219T230708Z-1-001/data_story_label_eval_v2/story_label_predictions_2016_2018.jsonl
  drive-download-20260219T230708Z-1-001/data_story_label_eval_v2/story_label_eval_report.json
"""

import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.request

from sklearn.metrics import f1_score


BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drive-download-20260219T230708Z-1-001")
STORIES_DIR = os.path.join(BASE_DIR, "data_stories_v2")
OUT_DIR = os.path.join(BASE_DIR, "data_story_label_eval_v2")
IN_PATHS = [
    os.path.join(STORIES_DIR, "stories_2016.jsonl"),
    os.path.join(STORIES_DIR, "stories_2018.jsonl"),
]
OUT_JSONL = os.path.join(OUT_DIR, "story_label_predictions_2016_2018.jsonl")
OUT_REPORT = os.path.join(OUT_DIR, "story_label_eval_report.json")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

FEW_SHOT_EXAMPLES = [
    {
        "label": 0,
        "story": """ASSESSMENT:

The app exhibits routine observable behavior associated with common Android functionality.

BEHAVIOR:

* The app requests internet and network-state access.
* It uses standard Android services and package metadata APIs.
* It exposes normal launcher activity behavior and basic user-facing components.

RATIONALE:

* Internet and network-state permissions are common in ordinary connected apps.
* Use of package-manager and context-service APIs is consistent with regular app operation.
* The listed activities and intents do not by themselves indicate harmful behavior.""",
        "output": {
            "predicted_label": 0,
            "predicted_label_name": "benign",
            "confidence": 0.67,
            "evidence_for_0": "The story is consistent with ordinary Android app behavior and lacks a strong concentration of clearly harmful actions.",
            "evidence_for_1": "There are permissions and API calls present, but they are not enough on their own to justify label 1.",
            "rationale": "The observable behavior is more consistent with label 0 than label 1."
        },
    },
    {
        "label": 0,
        "story": """ASSESSMENT:

The app shows ordinary multimedia and interface-related behavior.

BEHAVIOR:

* The app uses media playback and video-view related APIs.
* It requests internet connectivity and interacts with standard activities.
* It uses normal UI and lifecycle components.

RATIONALE:

* Media and playback APIs are common in legitimate consumer apps.
* Internet access and ordinary activity execution are not sufficient evidence for label 1.
* The story does not present a concentrated pattern of clearly harmful actions.""",
        "output": {
            "predicted_label": 0,
            "predicted_label_name": "benign",
            "confidence": 0.65,
            "evidence_for_0": "The behavior looks compatible with ordinary app features such as playback, connectivity, and UI interaction.",
            "evidence_for_1": "Some APIs could appear sensitive in isolation, but the story does not show a strong harmful pattern.",
            "rationale": "Overall the evidence favors label 0."
        },
    },
    {
        "label": 1,
        "story": """ASSESSMENT:

The app exhibits a concentrated set of sensitive behaviors affecting device data and communication.

BEHAVIOR:

* The app accesses device identifiers and telephony state.
* It sends network requests while interacting with SMS-related capability.
* It executes system-level commands and combines multiple sensitive permissions.

RATIONALE:

* Device-ID and telephony access combined with outbound communication increases concern.
* SMS capability together with command execution is stronger evidence for label 1.
* The concentration of sensitive behaviors is not well explained by ordinary app functionality.""",
        "output": {
            "predicted_label": 1,
            "predicted_label_name": "malicious",
            "confidence": 0.82,
            "evidence_for_0": "Some individual permissions can appear in normal apps, but the combination here is unusually concentrated.",
            "evidence_for_1": "The story combines identifier access, messaging-related capability, command execution, and network activity.",
            "rationale": "The evidence is more consistent with label 1."
        },
    },
    {
        "label": 1,
        "story": """ASSESSMENT:

The app presents multiple sensitive behaviors involving data collection and privileged control.

BEHAVIOR:

* The app accesses sensitive device state and account-related information.
* It uses restricted APIs and persistent communication-related functionality.
* It combines several elevated permissions with background-oriented behavior.

RATIONALE:

* The accumulation of sensitive access patterns is stronger than expected for ordinary functionality.
* Restricted API use plus broad permission scope increases support for label 1.
* The overall pattern is more concentrated and risky than a typical benign app profile.""",
        "output": {
            "predicted_label": 1,
            "predicted_label_name": "malicious",
            "confidence": 0.79,
            "evidence_for_0": "There may be some overlap with normal apps, but not enough to outweigh the broader concentration of sensitive behaviors.",
            "evidence_for_1": "The story shows a dense grouping of sensitive access, restricted APIs, and background-capable behavior.",
            "rationale": "The total evidence favors label 1."
        },
    },
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--num-ctx", type=int, default=4096)
    p.add_argument("--timeout-seconds", type=int, default=120)
    p.add_argument("--max-retries", type=int, default=1)
    p.add_argument("--retry-backoff-base", type=float, default=0.7)
    p.add_argument("--retry-backoff-factor", type=float, default=1.7)
    p.add_argument("--concurrency", type=int, default=4)
    return p.parse_args()


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_json_relaxed(text: str):
    if not text:
        return None
    raw = text.strip()
    candidates = [raw]
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL):
        stripped = m.group(1).strip()
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


def build_prompt(story: str):
    examples = []
    for ex in FEW_SHOT_EXAMPLES:
        examples.append(
            "Example story:\n"
            f"{ex['story']}\n\n"
            "Example output:\n"
            f"{json.dumps(ex['output'], ensure_ascii=True, indent=2)}"
        )
    example_block = "\n\n".join(examples)
    return f"""
You are classifying an Android app telemetry story.

Label definitions:
- Label 0: benign
- Label 1: malicious

Important calibration rules:
- The dataset is balanced; both labels are common.
- Do not default to label 1 just because the story mentions permissions, network access, suspicious APIs, or device information.
- Many benign apps also request sensitive permissions and call low-level Android APIs.
- Choose label 1 only when the total pattern is more concentrated, harmful, or difficult to explain as ordinary app behavior.
- Consider evidence for both labels before choosing.
- If the evidence is mixed, prefer the better-supported label rather than the more alarming label.

Few-shot examples:
{example_block}

Story:
{story}

Task:
Predict whether the story is more consistent with label 0 or label 1.

Rules:
- Use only the story text.
- Do not use markdown.
- Output only valid JSON.
- Keep the rationale brief and evidence-focused.
- Include short evidence for both labels before the final choice.

JSON schema:
{{
  "predicted_label": 0,
  "predicted_label_name": "benign",
  "confidence": 0.0,
  "evidence_for_0": "string",
  "evidence_for_1": "string",
  "rationale": "string"
}}
""".strip()


def normalize_predicted_label(parsed):
    if not isinstance(parsed, dict):
        return None
    label = parsed.get("predicted_label")
    if isinstance(label, str):
        s = label.strip().lower()
        if s in {"0", "benign"}:
            return 0
        if s in {"1", "malicious"}:
            return 1
    if isinstance(label, (int, float)):
        i = int(label)
        if i in (0, 1):
            return i
    name = str(parsed.get("predicted_label_name", "")).strip().lower()
    if name == "benign":
        return 0
    if name == "malicious":
        return 1
    return None


def generate_job(job):
    rec = job["rec"]
    args = job["args"]
    prompt = build_prompt(str(rec.get("story", "")))
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    raw = ""
    error = None
    parsed = None
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
            error = None
        except Exception as exc:
            error = str(exc)
        rchars += len(raw)
        if normalize_predicted_label(parsed) is not None:
            break
        if a < args.max_retries:
            delay = args.retry_backoff_base * (args.retry_backoff_factor ** (a - 1))
            time.sleep(delay)
            backoff_s += delay

    pred = normalize_predicted_label(parsed)
    pred_name = "malicious" if pred == 1 else "benign" if pred == 0 else None
    confidence = None
    rationale = ""
    if isinstance(parsed, dict):
        try:
            confidence = float(parsed.get("confidence"))
        except Exception:
            confidence = None
        rationale = str(parsed.get("rationale", "")).strip()

    out = {
        "year": rec.get("year"),
        "row_index": rec.get("row_index"),
        "label": rec.get("label"),
        "label_name": rec.get("label_name"),
        "story": rec.get("story", ""),
        "predicted_label": pred,
        "predicted_label_name": pred_name,
        "confidence": confidence,
        "rationale": rationale,
        "raw_model_output": raw,
        "error": error,
        "attempts": attempts,
        "correct": bool(pred == rec.get("label")) if pred is not None else False,
        "generation_meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt_sha256": prompt_hash,
            "prompt_chars": pchars,
            "response_chars": rchars,
            "retry_backoff_seconds": round(backoff_s, 4),
            "model": OLLAMA_MODEL,
        },
    }
    return out


def _progress(n, total, elapsed):
    width = 30
    frac = 1.0 if total == 0 else n / total
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    eta = (total - n) / (n / elapsed) if n > 0 and elapsed > 0 else 0
    sys.stdout.write(f"\r[{bar}] {n}/{total} ETA {int(eta//60):02d}:{int(eta%60):02d}")
    sys.stdout.flush()


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    source = []
    for p in IN_PATHS:
        source.extend(read_jsonl(p))

    start = 0
    if args.resume and os.path.isfile(OUT_JSONL):
        start = sum(1 for _ in read_jsonl(OUT_JSONL))
    mode = "a" if args.resume else "w"
    pending = [{"rec": rec, "args": args} for rec in source[start:]]
    events = []

    with open(OUT_JSONL, mode, encoding="utf-8") as f:
        workers = max(1, args.concurrency)
        t0 = time.time()
        if workers == 1:
            iterator = map(generate_job, pending)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                iterator = ex.map(generate_job, pending)
                for i, out in enumerate(iterator, start=start):
                    f.write(json.dumps(out, ensure_ascii=True) + "\n")
                    events.append(out)
                    n = i + 1
                    if n % 5 == 0 or n == len(source):
                        _progress(n, len(source), time.time() - t0)
            iterator = None

        if iterator is not None:
            for i, out in enumerate(iterator, start=start):
                f.write(json.dumps(out, ensure_ascii=True) + "\n")
                events.append(out)
                n = i + 1
                if n % 5 == 0 or n == len(source):
                    _progress(n, len(source), time.time() - t0)
        if source:
            sys.stdout.write("\n")

    rows = list(read_jsonl(OUT_JSONL))
    valid = [r for r in rows if r.get("predicted_label") in (0, 1)]
    acc = (sum(1 for r in valid if r.get("correct")) / len(valid)) if valid else 0.0
    macro_f1 = (
        f1_score(
            [int(r["label"]) for r in valid],
            [int(r["predicted_label"]) for r in valid],
            average="macro",
            zero_division=0,
        )
        if valid
        else 0.0
    )

    by_year = {}
    for year in [2016, 2018]:
        yr = [r for r in rows if int(r.get("year")) == year]
        yr_valid = [r for r in yr if r.get("predicted_label") in (0, 1)]
        yr_f1 = (
            f1_score(
                [int(r["label"]) for r in yr_valid],
                [int(r["predicted_label"]) for r in yr_valid],
                average="macro",
                zero_division=0,
            )
            if yr_valid
            else None
        )
        by_year[str(year)] = {
            "rows": len(yr),
            "valid_predictions": len(yr_valid),
            "accuracy": round(sum(1 for r in yr_valid if r.get("correct")) / len(yr_valid), 4) if yr_valid else None,
            "macro_f1": round(float(yr_f1), 4) if yr_f1 is not None else None,
        }

    attempts = [int(r.get("attempts", 0)) for r in rows]
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": OLLAMA_MODEL,
        "source_years": [2016, 2018],
        "rows": len(rows),
        "valid_predictions": len(valid),
        "accuracy": round(acc, 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_year": by_year,
        "generation_stats": {
            "avg_attempts": round(float(sum(attempts) / len(attempts)), 4) if attempts else 0.0,
            "median_attempts": float(statistics.median(attempts)) if attempts else 0.0,
            "max_attempts": int(max(attempts)) if attempts else 0,
        },
    }
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {OUT_JSONL}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
