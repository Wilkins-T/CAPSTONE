"""
Generate a custom synthetic dataset (X.npy, y.npy) using a local Ollama LLM.

Inputs:
  - data_2012/X.npy, y.npy
  - data_2014/X.npy, y.npy
  - selected_features.json (optional, for human-readable names)

Outputs (default):
  - drive-download-20260219T230708Z-1-001/data_custom/X.npy
  - drive-download-20260219T230708Z-1-001/data_custom/y.npy
  - drive-download-20260219T230708Z-1-001/data_custom/README_synthesis.md
"""

import json
import os
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
OUT_DIR = os.path.join(BASE_DIR, "data_custom")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

N_SYNTH = 6000
N_BENIGN = 3000
N_MALICIOUS = 3000

TOP_K_FEATURES = 40
RNG_SEED = 42
PROB_SMOOTHING = 0.5  # Laplace smoothing to avoid 0/1 probabilities
BETA_KAPPA = 15.0     # per-sample probability jitter (lower => more diversity)
ENSURE_UNIQUENESS = True
DRIFT_HORIZON_MIN = 1.0  # 1.0 ~= one step after 2014 (2016-like)
DRIFT_HORIZON_MAX = 2.0  # 2.0 ~= two steps after 2014 (2018-like)
DRIFT_STRENGTH = 0.85    # damp linear extrapolation to stay plausible
BASE_MUTATION_RATE = 0.01
DRIFT_MUTATION_SCALE = 2.0
MAX_MUTATION_RATE = 0.30
MIN_PROB = 0.001
MAX_PROB = 0.999


def load_features(features_json_path: str):
    if not os.path.isfile(features_json_path):
        return {}
    with open(features_json_path, "r") as f:
        name_to_idx = json.load(f)
    return {int(idx): name for name, idx in name_to_idx.items()}


def load_xy(data_dir: str):
    X = np.load(os.path.join(data_dir, "X.npy"), allow_pickle=True)
    y = np.load(os.path.join(data_dir, "y.npy"), allow_pickle=True)
    if y.ndim > 1:
        y = y.ravel()
    return X.astype(np.int64), y.astype(np.float64)


def summarize_stats(X, y):
    n_features = X.shape[1]
    benign = X[y == 0]
    malicious = X[y == 1]
    n_b = benign.shape[0]
    n_m = malicious.shape[0]
    count_b = benign.sum(axis=0)
    count_m = malicious.sum(axis=0)
    # Laplace smoothing to prevent 0/1 probabilities.
    p_b = (count_b + PROB_SMOOTHING) / (n_b + 2 * PROB_SMOOTHING)
    p_m = (count_m + PROB_SMOOTHING) / (n_m + 2 * PROB_SMOOTHING)
    delta = p_m - p_b
    return {
        "n_features": n_features,
        "n_benign": int(n_b),
        "n_malicious": int(n_m),
        "benign_mean_ones": float(benign.mean()),
        "malicious_mean_ones": float(malicious.mean()),
        "count_b": count_b,
        "count_m": count_m,
        "p_b": p_b,
        "p_m": p_m,
        "delta": delta,
    }


def class_probs(X, y):
    """Return smoothed class-conditional probabilities and class counts."""
    benign = X[y == 0]
    malicious = X[y == 1]
    n_b = benign.shape[0]
    n_m = malicious.shape[0]
    count_b = benign.sum(axis=0)
    count_m = malicious.sum(axis=0)
    p_b = (count_b + PROB_SMOOTHING) / (n_b + 2 * PROB_SMOOTHING)
    p_m = (count_m + PROB_SMOOTHING) / (n_m + 2 * PROB_SMOOTHING)
    return p_b, p_m, n_b, n_m


def build_prompt(stats, idx_to_name):
    delta = stats["delta"]
    p_b = stats["p_b"]
    p_m = stats["p_m"]

    top_pos = np.argsort(-delta)[: TOP_K_FEATURES // 2]
    top_neg = np.argsort(delta)[: TOP_K_FEATURES // 2]

    def _line(i):
        name = idx_to_name.get(int(i), f"feature_{int(i)}")
        return f"{int(i)} | {name} | benign={p_b[i]:.3f} malicious={p_m[i]:.3f} delta={delta[i]:+.3f}"

    lines = ["Top malicious-leaning features:"]
    lines += [_line(i) for i in top_pos]
    lines += ["", "Top benign-leaning features:"]
    lines += [_line(i) for i in top_neg]

    prompt = f"""
    You are designing a controlled synthetic data generation policy for a binary malware classifier.

    Context:
    - We have {stats['n_features']} binary (0/1) features.
    - Data combines 2012 and 2014 distributions.
    - Goal: produce a sampling policy that improves robustness to temporal drift (2016+2018),
    not merely maximize training accuracy.
    - Synthetic data may be moderately extreme but must remain statistically plausible.

    Base dataset statistics:
    - benign samples: {stats['n_benign']}
    - malicious samples: {stats['n_malicious']}
    - benign mean feature density: {stats['benign_mean_ones']:.4f}
    - malicious mean feature density: {stats['malicious_mean_ones']:.4f}

    Feature importance & prevalence summary:
    {os.linesep.join(lines)}

    Interpretation rules:
    - delta modifies Bernoulli sampling probability for a feature in that class.
    Example: delta = +0.10 increases probability of feature=1 by 10%.
    - global_flip_prob randomly flips bits across all features.
    - novelty_rate introduces rare co-occurring feature patterns.

    Design objectives:
    1. Reduce overfitting to highly year-specific features.
    2. Increase robustness to features likely to drift over time.
    3. Maintain approximate class separability.
    4. Avoid collapsing global feature density.

    Constraints:
    - At most 20 entries per adjustment list.
    - delta in [-0.25, 0.25].
    - global_flip_prob in [0.0, 0.02].
    - novelty_rate in [0.0, 0.02].
    - Prefer features from the provided summary.
    - Do NOT invent feature indices.
    - Output must strictly follow schema.

    Return ONLY this JSON object:

    {{
    "benign_adjustments": [{{"feature_index": int, "delta": float}}],
    "malicious_adjustments": [{{"feature_index": int, "delta": float}}],
    "global_flip_prob": float,
    "novelty_rate": float,
    "notes": "brief explanation of drift strategy"
    }}
    """
    return prompt.strip()


def ollama_generate(prompt: str):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 4096},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "")


def _extract_json_object(text: str):
    """Extract first top-level JSON object using brace matching."""
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


def parse_policy(text: str):
    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting first JSON object via brace matching
    candidate = _extract_json_object(text)
    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return None


def repair_policy_with_llm(raw_text: str):
    """Ask the LLM to convert raw output into valid JSON with the required schema."""
    fix_prompt = (
        "Convert the following text into a valid JSON object ONLY. "
        "No code fences, no extra text. "
        "Schema: {"
        "\"benign_adjustments\":[{\"feature_index\":int,\"delta\":float}],"
        "\"malicious_adjustments\":[{\"feature_index\":int,\"delta\":float}],"
        "\"global_flip_prob\":float,"
        "\"novelty_rate\":float,"
        "\"notes\":\"string\""
        "}.\n\n"
        "Text:\n"
        f"{raw_text}\n"
    )
    llm_fixed = ollama_generate(fix_prompt)
    return parse_policy(llm_fixed), llm_fixed


def default_policy():
    """Fallback policy if LLM is unavailable."""
    return {
        "benign_adjustments": [],
        "malicious_adjustments": [],
        "global_flip_prob": 0.0,
        "novelty_rate": 0.0,
        "notes": "fallback policy: no LLM adjustments",
    }


def apply_policy(base_probs, adjustments):
    probs = base_probs.copy()
    for adj in adjustments:
        idx = adj.get("feature_index", adj.get("index"))
        if idx is None:
            continue
        try:
            idx = int(idx)
        except ValueError:
            continue
        delta = float(adj.get("delta", 0.0))
        if idx < 0 or idx >= probs.shape[0]:
            continue
        probs[idx] = np.clip(probs[idx] + delta, 0.01, 0.99)
    return probs


def synthesize_class_from_drift(
    seed_rows,
    p_prev,
    p_recent,
    p_recent_adj,
    n_samples,
    rng,
    flip_prob=0.0,
    novelty_rate=0.0,
):
    """
    Build samples by mutating real 2014 rows toward extrapolated future drift.
    Drift direction comes from (2014 - 2012), then extrapolates across 2016..2018 horizon.
    """
    if seed_rows.shape[0] == 0:
        raise ValueError("seed_rows is empty for class synthesis.")

    pick = rng.integers(0, seed_rows.shape[0], size=n_samples)
    X_seed = seed_rows[pick].astype(np.int64, copy=True)

    horizons = rng.uniform(DRIFT_HORIZON_MIN, DRIFT_HORIZON_MAX, size=(n_samples, 1))
    trend = (p_recent - p_prev)[None, :]
    p_row = p_recent_adj[None, :] + (DRIFT_STRENGTH * horizons * trend)
    p_row = np.clip(p_row, MIN_PROB, MAX_PROB)

    if BETA_KAPPA and BETA_KAPPA > 0:
        alpha = np.clip(p_row * BETA_KAPPA, 1e-3, None)
        beta = np.clip((1.0 - p_row) * BETA_KAPPA, 1e-3, None)
        p_sample = rng.beta(alpha, beta)
    else:
        p_sample = p_row
    X_target = rng.binomial(1, p_sample).astype(np.int64)

    drift_mag = np.abs(DRIFT_STRENGTH * horizons * trend)
    mutate_prob = np.clip(
        BASE_MUTATION_RATE + DRIFT_MUTATION_SCALE * drift_mag,
        0.0,
        MAX_MUTATION_RATE,
    )
    mutate_mask = rng.random(size=X_seed.shape) < mutate_prob
    X = np.where(mutate_mask, X_target, X_seed).astype(np.int64)

    if flip_prob > 0.0:
        mask = rng.random(size=X.shape) < flip_prob
        X = np.logical_xor(X, mask).astype(np.int64)
    if novelty_rate > 0.0:
        rare = np.where((p_prev < 0.02) & (p_recent < 0.02))[0]
        if rare.size > 0:
            n_flip = int(n_samples * novelty_rate)
            rows = rng.integers(0, n_samples, size=n_flip)
            cols = rng.choice(rare, size=n_flip, replace=True)
            X[rows, cols] = 1
    return X


def dedupe_rows(X, rng):
    if not ENSURE_UNIQUENESS:
        return X
    # Deduplicate rows; if duplicates exist, resample them.
    X_unique, idx = np.unique(X, axis=0, return_index=True)
    n_dupe = X.shape[0] - X_unique.shape[0]
    if n_dupe == 0:
        return X
    # Resample duplicates by slight random flips
    X_out = X.copy()
    dup_mask = np.ones(X.shape[0], dtype=bool)
    dup_mask[idx] = False
    dup_idx = np.where(dup_mask)[0]
    for i in dup_idx:
        row = X_out[i].copy()
        # force at least one flip
        flip_count = max(1, int(0.01 * row.size))
        cols = rng.integers(0, row.size, size=flip_count)
        row[cols] = 1 - row[cols]
        X_out[i] = row
    return X_out


def main():
    X12, y12 = load_xy(DATA_2012)
    X14, y14 = load_xy(DATA_2014)
    X_all = np.vstack([X12, X14])
    y_all = np.concatenate([y12, y14])

    idx_to_name = load_features(FEATURES_JSON)
    stats = summarize_stats(X_all, y_all)
    p12_b, p12_m, n12_b, n12_m = class_probs(X12, y12)
    p14_b, p14_m, n14_b, n14_m = class_probs(X14, y14)

    prompt = build_prompt(stats, idx_to_name)
    llm_raw = ""
    llm_fixed = None
    policy = None
    llm_error = None
    try:
        llm_raw = ollama_generate(prompt)
        policy = parse_policy(llm_raw)
        if policy is None:
            policy, llm_fixed = repair_policy_with_llm(llm_raw)
    except Exception as exc:
        llm_error = str(exc)
    if policy is None:
        policy = default_policy()

    rng = np.random.default_rng(RNG_SEED)
    p_b = stats["p_b"]
    p_m = stats["p_m"]

    benign_adj = policy.get("benign_adjustments", [])
    mal_adj = policy.get("malicious_adjustments", [])
    flip_prob = float(policy.get("global_flip_prob", 0.0))
    novelty_rate = float(policy.get("novelty_rate", 0.0))

    # Apply policy to the recent anchor distribution (2014), then drift forward.
    p_b_adj = apply_policy(p14_b, benign_adj)
    p_m_adj = apply_policy(p14_m, mal_adj)

    X14_b = X14[y14 == 0]
    X14_m = X14[y14 == 1]

    X_b = synthesize_class_from_drift(
        X14_b, p12_b, p14_b, p_b_adj, N_BENIGN, rng, flip_prob, novelty_rate
    )
    X_m = synthesize_class_from_drift(
        X14_m, p12_m, p14_m, p_m_adj, N_MALICIOUS, rng, flip_prob, novelty_rate
    )
    X_b = dedupe_rows(X_b, rng)
    X_m = dedupe_rows(X_m, rng)

    X_out = np.vstack([X_b, X_m]).astype(np.int64)
    y_out = np.concatenate([np.zeros(N_BENIGN), np.ones(N_MALICIOUS)]).astype(np.float64)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "X.npy"), X_out)
    np.save(os.path.join(OUT_DIR, "y.npy"), y_out)

    # Human-readable report
    report_path = os.path.join(OUT_DIR, "README_synthesis.md")
    with open(report_path, "w") as f:
        f.write("# Synthetic Data Report\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Source Data\n")
        f.write("- Years: 2012 + 2014\n")
        f.write(f"- Combined samples: {X_all.shape[0]} (benign={stats['n_benign']} malicious={stats['n_malicious']})\n")
        f.write(f"- Features: {stats['n_features']}\n\n")
        f.write("## Generation Targets\n")
        f.write(f"- Synthetic samples: {N_SYNTH} (benign={N_BENIGN} malicious={N_MALICIOUS})\n")
        f.write(f"- Output dir: {OUT_DIR}\n\n")
        f.write("## LLM Policy (raw)\n")
        f.write("```\n")
        f.write(llm_raw.strip() + "\n")
        f.write("```\n\n")
        if llm_error:
            f.write("## LLM Error\n")
            f.write("```\n")
            f.write(llm_error + "\n")
            f.write("```\n\n")
        if llm_fixed:
            f.write("## LLM Fix Attempt Output\n")
            f.write("```\n")
            f.write(llm_fixed.strip() + "\n")
            f.write("```\n\n")
        f.write("## Parsed Policy\n")
        f.write("```\n")
        f.write(json.dumps(policy, indent=2) + "\n")
        f.write("```\n\n")
        f.write("## Summary Stats\n")
        f.write(f"- benign mean ones: {stats['benign_mean_ones']:.4f}\n")
        f.write(f"- malicious mean ones: {stats['malicious_mean_ones']:.4f}\n")
        f.write(f"- 2012 class counts: benign={n12_b} malicious={n12_m}\n")
        f.write(f"- 2014 class counts: benign={n14_b} malicious={n14_m}\n")
        f.write(f"- flip_prob: {flip_prob}\n")
        f.write(f"- novelty_rate: {novelty_rate}\n")
        f.write(f"- prob_smoothing: {PROB_SMOOTHING}\n")
        f.write(f"- beta_kappa: {BETA_KAPPA}\n")
        f.write(f"- drift_horizon_min: {DRIFT_HORIZON_MIN}\n")
        f.write(f"- drift_horizon_max: {DRIFT_HORIZON_MAX}\n")
        f.write(f"- drift_strength: {DRIFT_STRENGTH}\n")
        f.write(f"- base_mutation_rate: {BASE_MUTATION_RATE}\n")
        f.write(f"- drift_mutation_scale: {DRIFT_MUTATION_SCALE}\n")
        f.write(f"- max_mutation_rate: {MAX_MUTATION_RATE}\n")
        f.write(f"- ensure_uniqueness: {ENSURE_UNIQUENESS}\n")

    print(f"Wrote: {os.path.join(OUT_DIR, 'X.npy')}")
    print(f"Wrote: {os.path.join(OUT_DIR, 'y.npy')}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
