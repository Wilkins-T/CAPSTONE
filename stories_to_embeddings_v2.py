"""
V2 Stage 3: Convert story datasets to dense embeddings using SentenceTransformer.

Inputs:
  - data_stories_v2/stories_<year>.jsonl
  - data_stories_future_v2/future_stories.jsonl

Outputs:
  - data_story_embeddings_v2/data_<year>/X.npy, y.npy
  - data_story_embeddings_v2/data_future/X.npy, y.npy
  - data_story_embeddings_v2/embeddings_quarantine.jsonl
  - data_story_embeddings_v2/embeddings_quality_report.json
  - data_story_embeddings_v2/embeddings_research_report.json
  - data_story_embeddings_v2/embeddings_run_manifest.json
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np


BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drive-download-20260219T230708Z-1-001")
STORIES_DIR = os.path.join(BASE_DIR, "data_stories_v2")
FUTURE_STORIES = os.path.join(BASE_DIR, "data_stories_future_v2", "future_stories.jsonl")
OUT_DIR = os.path.join(BASE_DIR, "data_story_embeddings_v2")
YEARS = [2012, 2014, 2016, 2018]

OUT_QUARANTINE = os.path.join(OUT_DIR, "embeddings_quarantine.jsonl")
OUT_QUALITY = os.path.join(OUT_DIR, "embeddings_quality_report.json")
OUT_RESEARCH = os.path.join(OUT_DIR, "embeddings_research_report.json")
OUT_MANIFEST = os.path.join(OUT_DIR, "embeddings_run_manifest.json")
OUT_README = os.path.join(OUT_DIR, "README_embeddings_v2.md")
QUALITY_PROFILE = "research_v1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-empty-text-rate", type=float, default=0.01)
    p.add_argument("--max-invalid-embedding-rate", type=float, default=0.0)
    p.add_argument("--max-quarantine-rate", type=float, default=0.10)
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


def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def _progress(n, total, elapsed, prefix=""):
    width = 30
    frac = 1.0 if total == 0 else n / total
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    eta = (total - n) / (n / elapsed) if n > 0 and elapsed > 0 else 0
    lead = f"{prefix} " if prefix else ""
    sys.stdout.write(f"\r{lead}[{bar}] {n}/{total} ETA {int(eta//60):02d}:{int(eta%60):02d}")
    sys.stdout.flush()


def _encode_records(records, text_key, out_subdir, model, batch_size, normalize, resume):
    out_dir = os.path.join(OUT_DIR, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "checkpoint.json")

    n = len(records)
    y = np.array([int(r.get("label", 0)) for r in records], dtype=np.int64)
    texts = [str(r.get(text_key, "")).strip() for r in records]

    if resume and os.path.isfile(ckpt_path) and os.path.isfile(os.path.join(out_dir, "X.npy")) and os.path.isfile(os.path.join(out_dir, "y.npy")):
        ckpt = json.load(open(ckpt_path, "r", encoding="utf-8"))
        start = int(ckpt.get("next_index", 0))
        dim = int(ckpt.get("embedding_dim", 384))
        X = np.lib.format.open_memmap(os.path.join(out_dir, "X.npy"), mode="r+", dtype=np.float32, shape=(n, dim))
        y_mm = np.lib.format.open_memmap(os.path.join(out_dir, "y.npy"), mode="r+", dtype=np.int64, shape=(n,))
    else:
        start = 0
        # one probe to discover dim
        probe = model.encode(["probe"], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=normalize)
        dim = int(probe.shape[1])
        X = np.lib.format.open_memmap(os.path.join(out_dir, "X.npy"), mode="w+", dtype=np.float32, shape=(n, dim))
        y_mm = np.lib.format.open_memmap(os.path.join(out_dir, "y.npy"), mode="w+", dtype=np.int64, shape=(n,))

    y_mm[:] = y

    events = []
    quarantine = []
    t0 = time.time()
    for s in range(start, n, batch_size):
        e = min(n, s + batch_size)
        batch = texts[s:e]
        emb = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        ).astype(np.float32)

        X[s:e, :] = emb
        events.append({"start": s, "end": e, "size": e - s})

        for i in range(s, e):
            txt = texts[i]
            row = emb[i - s]
            reason = None
            if not txt:
                reason = "empty_text"
            elif not np.isfinite(row).all():
                reason = "invalid_embedding"
            elif np.linalg.norm(row) == 0.0:
                reason = "zero_norm_embedding"
            if reason:
                quarantine.append(
                    {
                        "dataset": out_subdir,
                        "row_index": i,
                        "label": int(y[i]),
                        "reason": reason,
                    }
                )

        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({"next_index": e, "embedding_dim": dim, "n_records": n}, f)
        _progress(e, n, time.time() - t0, prefix=out_subdir)

    elapsed = time.time() - t0
    if n > 0:
        sys.stdout.write("\n")
    return {
        "out_dir": out_dir,
        "shape": (n, dim),
        "events": events,
        "quarantine": quarantine,
        "elapsed_seconds": elapsed,
        "rows_processed_this_run": max(0, n - start),
    }


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'sentence-transformers'. Install with: pip install sentence-transformers"
        ) from exc

    if not args.resume:
        for p in [OUT_QUARANTINE, OUT_QUALITY, OUT_RESEARCH, OUT_MANIFEST, OUT_README]:
            if os.path.isfile(p):
                os.remove(p)

    t0 = time.time()
    model = SentenceTransformer(args.model_name, device=args.device)

    outputs = []
    all_events = []
    all_quarantine = []
    rows_total = 0
    rows_processed_this_run = 0

    for year in YEARS:
        p = os.path.join(STORIES_DIR, f"stories_{year}.jsonl")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing stories file: {p}")
        recs = read_jsonl(p)
        rows_total += len(recs)
        r = _encode_records(recs, "story", f"data_{year}", model, args.batch_size, args.normalize, args.resume)
        outputs.append((f"data_{year}", r["out_dir"], r["shape"]))
        all_events.extend(r["events"])
        all_quarantine.extend(r["quarantine"])
        rows_processed_this_run += r["rows_processed_this_run"]

    if not os.path.isfile(FUTURE_STORIES):
        raise FileNotFoundError(f"Missing future stories file: {FUTURE_STORIES}")
    future_recs = read_jsonl(FUTURE_STORIES)
    rows_total += len(future_recs)
    r = _encode_records(future_recs, "future_story", "data_future", model, args.batch_size, args.normalize, args.resume)
    outputs.append(("data_future", r["out_dir"], r["shape"]))
    all_events.extend(r["events"])
    all_quarantine.extend(r["quarantine"])
    rows_processed_this_run += r["rows_processed_this_run"]

    _write_jsonl(OUT_QUARANTINE, all_quarantine)

    empty_rate = sum(1 for q in all_quarantine if q.get("reason") == "empty_text") / max(1, rows_total)
    invalid_rate = sum(1 for q in all_quarantine if q.get("reason") in {"invalid_embedding", "zero_norm_embedding"}) / max(1, rows_total)
    quarantine_rate = len(all_quarantine) / max(1, rows_total)

    quality = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quality_profile": QUALITY_PROFILE,
        "thresholds": {
            "max_empty_text_rate": args.max_empty_text_rate,
            "max_invalid_embedding_rate": args.max_invalid_embedding_rate,
            "max_quarantine_rate": args.max_quarantine_rate,
        },
        "counts": {
            "rows_total": rows_total,
            "quarantine_rows": len(all_quarantine),
            "empty_text_rows": sum(1 for q in all_quarantine if q.get("reason") == "empty_text"),
            "invalid_embedding_rows": sum(1 for q in all_quarantine if q.get("reason") in {"invalid_embedding", "zero_norm_embedding"}),
        },
        "rates": {
            "empty_text_rate": empty_rate,
            "invalid_embedding_rate": invalid_rate,
            "quarantine_rate": quarantine_rate,
        },
    }
    violations = []
    if quality["rates"]["empty_text_rate"] > args.max_empty_text_rate:
        violations.append("empty_text_rate_above_threshold")
    if quality["rates"]["invalid_embedding_rate"] > args.max_invalid_embedding_rate:
        violations.append("invalid_embedding_rate_above_threshold")
    if quality["rates"]["quarantine_rate"] > args.max_quarantine_rate:
        violations.append("quarantine_rate_above_threshold")
    quality["violations"] = violations
    quality["passed"] = len(violations) == 0
    with open(OUT_QUALITY, "w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)

    elapsed = time.time() - t0
    est_prompt_tokens = rows_processed_this_run * 16.0
    est_response_tokens = sum(s[2][1] for s in outputs) * rows_processed_this_run if outputs else 0.0
    research = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quality_profile": QUALITY_PROFILE,
        "model": args.model_name,
        "rows_total_output": rows_total,
        "rows_processed_this_run": rows_processed_this_run,
        "embedding_stats": {
            "datasets": [{"name": n, "path": p, "shape": sh} for n, p, sh in outputs],
            "batches_processed": len(all_events),
            "avg_batch_size": float(sum(e["size"] for e in all_events) / len(all_events)) if all_events else 0.0,
        },
        "efficiency": {
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60.0 if elapsed > 0 else 0.0,
            "estimated_prompt_tokens": est_prompt_tokens,
            "estimated_response_tokens": est_response_tokens,
            "estimated_total_tokens": est_prompt_tokens + est_response_tokens,
            "estimated_tokens_per_second": (est_prompt_tokens + est_response_tokens) / elapsed if elapsed > 0 else 0.0,
            "rows_per_second": rows_processed_this_run / elapsed if elapsed > 0 else 0.0,
            "token_estimation_method": "coarse_approximation",
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
            "encoder": args.model_name,
            "batch_size": args.batch_size,
            "normalize": args.normalize,
            "device": args.device,
        },
        "quality_config": {
            "strict_gate": args.strict_gate,
            "max_empty_text_rate": args.max_empty_text_rate,
            "max_invalid_embedding_rate": args.max_invalid_embedding_rate,
            "max_quarantine_rate": args.max_quarantine_rate,
        },
        "inputs": {
            **{f"stories_{y}": {"path": os.path.join(STORIES_DIR, f"stories_{y}.jsonl"), "sha256": _sha256_of_file(os.path.join(STORIES_DIR, f"stories_{y}.jsonl"))} for y in YEARS},
            "future_stories": {"path": FUTURE_STORIES, "sha256": _sha256_of_file(FUTURE_STORIES)},
        },
        "outputs": {
            **{n: {"path": os.path.join(p, "X.npy"), "sha256": _sha256_of_file(os.path.join(p, "X.npy"))} for n, p, _ in outputs},
            "quarantine": {"path": OUT_QUARANTINE, "sha256": _sha256_of_file(OUT_QUARANTINE)},
            "quality": {"path": OUT_QUALITY, "sha256": _sha256_of_file(OUT_QUALITY)},
            "research": {"path": OUT_RESEARCH, "sha256": _sha256_of_file(OUT_RESEARCH)},
            "readme": {"path": OUT_README, "sha256": _sha256_of_file(OUT_README)},
        },
    }
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(OUT_README, "w", encoding="utf-8") as f:
        f.write("# Story Embeddings V2\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Quality profile: {QUALITY_PROFILE}\n")
        f.write(f"- Model: {args.model_name}\n")
        f.write(f"- Normalize embeddings: {args.normalize}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Device: {args.device}\n")
        f.write(f"- Mode: {'resume' if args.resume else 'fresh-run'}\n")
        f.write(f"- Quarantine: {OUT_QUARANTINE}\n")
        f.write(f"- Quality report: {OUT_QUALITY}\n")
        f.write(f"- Research report: {OUT_RESEARCH}\n")
        f.write(f"- Run manifest: {OUT_MANIFEST}\n")
        f.write(f"- Elapsed seconds: {elapsed:.2f}\n")
        for name, path, shape in outputs:
            f.write(f"- {name}: {path}, shape={shape}\n")

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
