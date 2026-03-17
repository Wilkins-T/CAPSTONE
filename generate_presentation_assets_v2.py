"""
Generate presentation assets for V2 pipeline outputs.
"""

import argparse
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


BASE_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(BASE_CODE_DIR, "drive-download-20260219T230708Z-1-001")
EMB_DIR = os.path.join(BASE_DATA_DIR, "data_story_embeddings_v2")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=os.path.join(BASE_CODE_DIR, "presentation_artifacts_v2"))
    return p.parse_args()


def maybe_load_xy(path):
    x = os.path.join(path, "X.npy")
    y = os.path.join(path, "y.npy")
    if not (os.path.isfile(x) and os.path.isfile(y)):
        return None, None
    X = np.load(x, allow_pickle=True)
    yv = np.load(y, allow_pickle=True)
    if yv.ndim > 1:
        yv = yv.ravel()
    return X, yv


def _progress(n, total, elapsed, prefix=""):
    width = 30
    frac = 1.0 if total == 0 else n / total
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    eta = (total - n) / (n / elapsed) if n > 0 and elapsed > 0 else 0
    lead = f"{prefix} " if prefix else ""
    sys.stdout.write(f"\r{lead}[{bar}] {n}/{total} ETA {int(eta//60):02d}:{int(eta%60):02d}")
    sys.stdout.flush()


def main():
    args = parse_args()
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    assets = []
    total_steps = 4
    done = 0
    t0 = time.time()
    _progress(done, total_steps, 0.0, prefix="assets")

    rows = []
    for year in [2012, 2014, 2016, 2018]:
        X, y = maybe_load_xy(os.path.join(EMB_DIR, f"data_{year}"))
        if X is None:
            continue
        rows.append(
            {
                "year": year,
                "n_samples": int(X.shape[0]),
                "embedding_dim": int(X.shape[1]),
                "n_benign": int((y == 0).sum()),
                "n_malicious": int((y == 1).sum()),
                "avg_l2_norm": float(np.linalg.norm(X, axis=1).mean()) if X.size else 0.0,
            }
        )
    done += 1
    _progress(done, total_steps, time.time() - t0, prefix="assets")

    if rows:
        years = [r["year"] for r in rows]
        benign = [r["n_benign"] for r in rows]
        malicious = [r["n_malicious"] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(years, benign, label="Benign", color="#4C78A8")
        ax.bar(years, malicious, bottom=benign, label="Malicious", color="#E45756")
        ax.set_title("V2 Embedding Dataset Class Balance")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        p = os.path.join(out, "v2_class_balance.png")
        fig.savefig(p, dpi=180)
        plt.close(fig)
        assets.append(p)

        norms = [r["avg_l2_norm"] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(years, norms, marker="o")
        ax.set_title("V2 Embedding Norm Trend")
        ax.set_xlabel("Year")
        ax.set_ylabel("Average L2 norm")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        p = os.path.join(out, "v2_embedding_norm_trend.png")
        fig.savefig(p, dpi=180)
        plt.close(fig)
        assets.append(p)
    done += 1
    _progress(done, total_steps, time.time() - t0, prefix="assets")

    rs = os.path.join(BASE_CODE_DIR, "results_summary_v2.json")
    if os.path.isfile(rs):
        data = json.load(open(rs, "r", encoding="utf-8"))
        years = [2012, 2014, 2016, 2018]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, key, title in [
            (axes[0], "cross_year_2012_model", "2012-trained model"),
            (axes[1], "cross_year_custom_model", "Custom-trained model"),
        ]:
            block = data.get(key, {})
            rf = [block.get(str(y), {}).get("rf", {}).get("macro_f1", np.nan) for y in years]
            mlp = [block.get(str(y), {}).get("mlp", {}).get("macro_f1", np.nan) for y in years]
            ax.plot(years, rf, marker="o", label="RF")
            ax.plot(years, mlp, marker="o", label="MLP")
            ax.set_title(title)
            ax.grid(alpha=0.25)
            ax.set_xlabel("Test year")
        axes[0].set_ylabel("Macro F1")
        axes[1].legend()
        fig.tight_layout()
        p = os.path.join(out, "v2_cross_year_performance.png")
        fig.savefig(p, dpi=180)
        plt.close(fig)
        assets.append(p)
    done += 1
    _progress(done, total_steps, time.time() - t0, prefix="assets")

    rep = os.path.join(out, "presentation_report_v2.md")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("# Presentation Report V2\n\n")
        f.write(f"Generated assets: {len(assets)}\n\n")
        for a in assets:
            f.write(f"- {a}\n")
        if rows:
            f.write("\n## Embedding Dataset Overview\n")
            for r in rows:
                f.write(
                    f"- {r['year']}: samples={r['n_samples']}, benign={r['n_benign']}, malicious={r['n_malicious']}, "
                    f"dim={r['embedding_dim']}, avg_l2_norm={r['avg_l2_norm']:.4f}\n"
                )
    assets.append(rep)
    done += 1
    _progress(done, total_steps, time.time() - t0, prefix="assets")
    sys.stdout.write("\n")

    print(f"Generated {len(assets)} assets in {out}")
    for a in assets:
        print(a)


if __name__ == "__main__":
    main()
