"""
Generate presentation-ready artifacts from malware dataset + pipeline outputs.

Outputs include:
- Dataset overview tables and plots (class balance, density, drift)
- Model performance plots (if results_summary.json is available)
- Story pipeline quality plots/stats (if JSONL files are available)
- A markdown report summarizing key numbers and file locations

Usage:
  python generate_presentation_assets.py
  python generate_presentation_assets.py --out-dir /path/to/presentation_artifacts
"""

import argparse
import csv
import glob
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


BASE_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(BASE_CODE_DIR, "drive-download-20260219T230708Z-1-001")
SELECTED_FEATURES = os.path.join(BASE_DATA_DIR, "selected_features.json")

YEARS = [2012, 2014, 2016, 2018]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default=os.path.join(BASE_CODE_DIR, "presentation_artifacts"),
        help="Output directory for generated presentation assets.",
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_feature_names(path, n_features):
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


def maybe_load_xy(year):
    data_dir = os.path.join(BASE_DATA_DIR, f"data_{year}")
    x_path = os.path.join(data_dir, "X.npy")
    y_path = os.path.join(data_dir, "y.npy")
    if not (os.path.isfile(x_path) and os.path.isfile(y_path)):
        return None, None
    x = np.load(x_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)
    if y.ndim > 1:
        y = y.ravel()
    return x.astype(np.int64), y.astype(np.int64)


def dataset_stats(year_to_data):
    rows = []
    for year, (x, y) in year_to_data.items():
        benign = x[y == 0]
        malicious = x[y == 1]
        rows.append(
            {
                "year": year,
                "n_samples": int(x.shape[0]),
                "n_features": int(x.shape[1]),
                "n_benign": int((y == 0).sum()),
                "n_malicious": int((y == 1).sum()),
                "density_all": float(x.mean()),
                "density_benign": float(benign.mean()) if benign.size else 0.0,
                "density_malicious": float(malicious.mean()) if malicious.size else 0.0,
            }
        )
    rows.sort(key=lambda r: r["year"])
    return rows


def write_dataset_csv(rows, out_csv):
    fields = [
        "year",
        "n_samples",
        "n_features",
        "n_benign",
        "n_malicious",
        "density_all",
        "density_benign",
        "density_malicious",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def plot_class_balance(rows, out_png):
    years = [r["year"] for r in rows]
    benign = [r["n_benign"] for r in rows]
    malicious = [r["n_malicious"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(years, benign, label="Benign", color="#4C78A8")
    ax.bar(years, malicious, bottom=benign, label="Malicious", color="#E45756")
    ax.set_title("Class Balance by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Samples")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_density(rows, out_png):
    years = [r["year"] for r in rows]
    all_d = [r["density_all"] for r in rows]
    b_d = [r["density_benign"] for r in rows]
    m_d = [r["density_malicious"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(years, all_d, marker="o", label="All", color="#2E86AB")
    ax.plot(years, b_d, marker="o", label="Benign", color="#1B9E77")
    ax.plot(years, m_d, marker="o", label="Malicious", color="#D95F02")
    ax.set_title("Feature Density Trend by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean activation probability")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_similarity_heatmap(year_to_data, out_png):
    years = sorted(year_to_data)
    prevalences = []
    for year in years:
        x, _ = year_to_data[year]
        prevalences.append(x.mean(axis=0))
    mat = np.vstack(prevalences)

    sims = np.zeros((len(years), len(years)), dtype=np.float64)
    for i in range(len(years)):
        for j in range(len(years)):
            a = mat[i]
            b = mat[j]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            sims[i, j] = float(np.dot(a, b) / denom) if denom > 0 else 0.0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sims, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(years)))
    ax.set_yticks(range(len(years)))
    ax.set_xticklabels(years)
    ax.set_yticklabels(years)
    ax.set_title("Year-to-Year Similarity (Feature Prevalence Cosine)")
    for i in range(len(years)):
        for j in range(len(years)):
            ax.text(j, i, f"{sims[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def top_drift_rows(year_to_data, feature_names, src=2012, dst=2018, top_k=25):
    if src not in year_to_data or dst not in year_to_data:
        return []
    x_src, _ = year_to_data[src]
    x_dst, _ = year_to_data[dst]
    p_src = x_src.mean(axis=0)
    p_dst = x_dst.mean(axis=0)
    delta = p_dst - p_src
    idx = np.argsort(-np.abs(delta))[:top_k]
    rows = []
    for i in idx:
        rows.append(
            {
                "feature_index": int(i),
                "feature_name": feature_names[int(i)],
                "p_src": float(p_src[i]),
                "p_dst": float(p_dst[i]),
                "delta": float(delta[i]),
            }
        )
    return rows


def write_top_drift_csv(rows, out_csv):
    fields = ["feature_index", "feature_name", "p_src", "p_dst", "delta"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def plot_top_drift(rows, out_png, src=2012, dst=2018):
    if not rows:
        return
    names = [r["feature_name"][:40] for r in rows][::-1]
    vals = [r["delta"] for r in rows][::-1]
    colors = ["#D95F02" if v > 0 else "#1B9E77" for v in vals]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(vals)), vals, color=colors)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title(f"Top Feature Drift ({src} -> {dst})")
    ax.set_xlabel("Change in activation probability")
    ax.axvline(0, color="black", linewidth=1)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def find_results_summary():
    preferred = os.path.join(BASE_CODE_DIR, "results_summary.json")
    if os.path.isfile(preferred):
        return preferred
    candidates = glob.glob(os.path.join(BASE_CODE_DIR, "saved_old_tests", "**", "results_summary.json"), recursive=True)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def plot_results_summary(summary_path, out_png):
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    years = [2012, 2014, 2016, 2018]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    plots = [
        ("cross_year_2012_model", "2012-trained model applied across years"),
        ("cross_year_custom_model", "Custom-trained model applied across years"),
    ]

    for ax, (block, title) in zip(axes, plots):
        block_data = data.get(block, {})
        rf = [block_data.get(str(y), {}).get("rf", {}).get("macro_f1", np.nan) for y in years]
        mlp = [block_data.get(str(y), {}).get("mlp", {}).get("macro_f1", np.nan) for y in years]
        ax.plot(years, rf, marker="o", label="RF")
        ax.plot(years, mlp, marker="o", label="MLP")
        ax.set_title(title)
        ax.set_xlabel("Test year")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Macro F1")
    axes[1].legend()
    fig.suptitle("Cross-year model robustness")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def stream_story_stats(jsonl_path, feature_key, story_key, label_key="label"):
    stats = {
        "records": 0,
        "empty_story": 0,
        "story_chars_total": 0,
        "feature_count_total": 0,
        "label_counts": Counter(),
        "validation_note_counts": Counter(),
        "error_count": 0,
    }
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            stats["records"] += 1

            story = (rec.get(story_key) or "").strip()
            if not story:
                stats["empty_story"] += 1
            stats["story_chars_total"] += len(story)

            feats = rec.get(feature_key, []) or []
            stats["feature_count_total"] += len(feats)

            label = rec.get(label_key)
            if label in (0, 1, "0", "1"):
                stats["label_counts"][int(label)] += 1

            note = rec.get("validation_note")
            if note:
                stats["validation_note_counts"][str(note)] += 1

            if rec.get("error"):
                stats["error_count"] += 1

    if stats["records"] > 0:
        stats["avg_story_chars"] = stats["story_chars_total"] / stats["records"]
        stats["avg_feature_count"] = stats["feature_count_total"] / stats["records"]
    else:
        stats["avg_story_chars"] = 0.0
        stats["avg_feature_count"] = 0.0

    return stats


def plot_story_quality(stats_a, stats_b, out_png):
    labels = ["Stage1 stories", "Stage2 future"]
    avg_chars = [stats_a["avg_story_chars"], stats_b["avg_story_chars"]]
    empty_pct = [
        100.0 * stats_a["empty_story"] / max(1, stats_a["records"]),
        100.0 * stats_b["empty_story"] / max(1, stats_b["records"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(labels, avg_chars, color=["#4C78A8", "#F58518"])
    axes[0].set_title("Average story length")
    axes[0].set_ylabel("Characters")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(labels, empty_pct, color=["#54A24B", "#E45756"])
    axes[1].set_title("Empty story rate")
    axes[1].set_ylabel("Percent")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_future_array_density(x_path, out_png):
    x = np.load(x_path, allow_pickle=True)
    nonzero = x.sum(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(nonzero, bins=30, color="#4C78A8", edgecolor="black", alpha=0.8)
    ax.set_title("Future Array Sparsity Distribution")
    ax.set_xlabel("Active features per sample")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    return {
        "rows": int(x.shape[0]),
        "features": int(x.shape[1]),
        "avg_active_features": float(nonzero.mean()) if nonzero.size else 0.0,
        "min_active_features": int(nonzero.min()) if nonzero.size else 0,
        "max_active_features": int(nonzero.max()) if nonzero.size else 0,
    }


def write_summary_report(out_md, items):
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Presentation Artifact Report\n\n")

        f.write("## Generated Assets\n")
        for item in items.get("assets", []):
            f.write(f"- {item}\n")
        f.write("\n")

        if "dataset_rows" in items:
            f.write("## Dataset Overview\n")
            for row in items["dataset_rows"]:
                f.write(
                    f"- {row['year']}: samples={row['n_samples']}, benign={row['n_benign']}, "
                    f"malicious={row['n_malicious']}, density={row['density_all']:.4f}\n"
                )
            f.write("\n")

        if "story_stats" in items:
            s = items["story_stats"]
            f.write("## Stage 1 Story Stats\n")
            f.write(
                f"- records={s['records']}, empty_story={s['empty_story']}, "
                f"avg_story_chars={s['avg_story_chars']:.1f}, avg_active_features={s['avg_feature_count']:.2f}, "
                f"errors={s['error_count']}\n\n"
            )

        if "future_story_stats" in items:
            s = items["future_story_stats"]
            f.write("## Stage 2 Future Story Stats\n")
            f.write(
                f"- records={s['records']}, empty_story={s['empty_story']}, "
                f"avg_story_chars={s['avg_story_chars']:.1f}, avg_active_features={s['avg_feature_count']:.2f}, "
                f"errors={s['error_count']}\n"
            )
            if s["validation_note_counts"]:
                top_notes = s["validation_note_counts"].most_common(5)
                f.write("- top validation notes: " + ", ".join(f"{k}={v}" for k, v in top_notes) + "\n")
            f.write("\n")

        if "future_array_stats" in items:
            s = items["future_array_stats"]
            f.write("## Stage 3 Array Stats\n")
            f.write(
                f"- rows={s['rows']}, features={s['features']}, avg_active_features={s['avg_active_features']:.2f}, "
                f"min_active_features={s['min_active_features']}, max_active_features={s['max_active_features']}\n"
            )


def main():
    args = parse_args()
    out_dir = args.out_dir
    ensure_dir(out_dir)

    assets = []
    summary = {"assets": assets}

    year_to_data = {}
    for year in YEARS:
        x, y = maybe_load_xy(year)
        if x is not None:
            year_to_data[year] = (x, y)

    if year_to_data:
        rows = dataset_stats(year_to_data)
        summary["dataset_rows"] = rows

        csv_path = os.path.join(out_dir, "dataset_overview.csv")
        write_dataset_csv(rows, csv_path)
        assets.append(csv_path)

        p1 = os.path.join(out_dir, "class_balance_by_year.png")
        plot_class_balance(rows, p1)
        assets.append(p1)

        p2 = os.path.join(out_dir, "feature_density_trend.png")
        plot_density(rows, p2)
        assets.append(p2)

        p3 = os.path.join(out_dir, "year_similarity_heatmap.png")
        plot_similarity_heatmap(year_to_data, p3)
        assets.append(p3)

        n_features = rows[0]["n_features"]
        feature_names = load_feature_names(SELECTED_FEATURES, n_features)
        drift_rows = top_drift_rows(year_to_data, feature_names, src=2012, dst=2018, top_k=25)
        if drift_rows:
            drift_csv = os.path.join(out_dir, "top_feature_drift_2012_to_2018.csv")
            write_top_drift_csv(drift_rows, drift_csv)
            assets.append(drift_csv)

            drift_png = os.path.join(out_dir, "top_feature_drift_2012_to_2018.png")
            plot_top_drift(drift_rows, drift_png, src=2012, dst=2018)
            assets.append(drift_png)

    results_summary_path = find_results_summary()
    if results_summary_path:
        perf_png = os.path.join(out_dir, "cross_year_performance.png")
        plot_results_summary(results_summary_path, perf_png)
        assets.append(perf_png)
        summary["results_summary_path"] = results_summary_path

    stories_jsonl = os.path.join(BASE_DATA_DIR, "data_stories", "stories.jsonl")
    future_stories_jsonl = os.path.join(BASE_DATA_DIR, "data_stories_future", "future_stories.jsonl")
    if os.path.isfile(stories_jsonl):
        s1 = stream_story_stats(
            stories_jsonl,
            feature_key="active_features",
            story_key="story",
        )
        summary["story_stats"] = s1
    if os.path.isfile(future_stories_jsonl):
        s2 = stream_story_stats(
            future_stories_jsonl,
            feature_key="future_active_features",
            story_key="future_story",
        )
        summary["future_story_stats"] = s2

    if "story_stats" in summary and "future_story_stats" in summary:
        q_png = os.path.join(out_dir, "story_pipeline_quality.png")
        plot_story_quality(summary["story_stats"], summary["future_story_stats"], q_png)
        assets.append(q_png)

    future_x = os.path.join(BASE_DATA_DIR, "data_future_arrays", "X.npy")
    if os.path.isfile(future_x):
        a_png = os.path.join(out_dir, "future_array_sparsity.png")
        arr_stats = plot_future_array_density(future_x, a_png)
        summary["future_array_stats"] = arr_stats
        assets.append(a_png)

    report_md = os.path.join(out_dir, "presentation_report.md")
    write_summary_report(report_md, summary)
    assets.append(report_md)

    print(f"Generated {len(assets)} assets in: {out_dir}")
    for p in assets:
        print(p)


if __name__ == "__main__":
    main()
