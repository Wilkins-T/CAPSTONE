"""
V2 training/evaluation on story embeddings.

Outputs:
  - results_summary_v2.json
  - results_comparison_v2.png
  - feature_importance_v2.csv
  - feature_importance_v2.json
  - feature_importance_v2.png
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMB_DIR = os.path.join(BASE_DIR, "drive-download-20260219T230708Z-1-001", "data_story_embeddings_v2")
YEARS = [2012, 2014, 2016, 2018]
CUSTOM_DATA_DIR = os.path.join(EMB_DIR, "data_future")

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

RF_N_ESTIMATORS = 1000
RF_MAX_LEAF_NODES = 32

MLP_HIDDEN_1 = 256
MLP_HIDDEN_2 = 128
MLP_BATCH_SIZE = 32
MLP_EPOCHS = 80
MLP_LEARNING_RATE = 1e-4
MLP_WEIGHT_DECAY = 5e-4
MLP_PATIENCE = 15
MLP_USE_SCALER = True
TOP_N = 40


def get_data_dir(year: int) -> str:
    return os.path.join(EMB_DIR, f"data_{year}")


def load_xy(data_dir: str):
    x_path = os.path.join(data_dir, "X.npy")
    y_path = os.path.join(data_dir, "y.npy")
    if not (os.path.isfile(x_path) and os.path.isfile(y_path)):
        raise FileNotFoundError(f"Missing {x_path} or {y_path}")
    X = np.load(x_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)
    if y.ndim > 1:
        y = y.ravel()
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    return X, y


def evaluate_rf_on_data(model, X_test, y_test):
    pred = model.predict(X_test)
    return accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro", zero_division=0)


def run_rf(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_leaf_nodes=RF_MAX_LEAF_NODES,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    acc, f1 = evaluate_rf_on_data(model, X_test, y_test)
    return model, acc, f1


class MLP(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(n_features, MLP_HIDDEN_1)
        self.fc2 = nn.Linear(MLP_HIDDEN_1, MLP_HIDDEN_2)
        self.fc3 = nn.Linear(MLP_HIDDEN_2, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def run_mlp(X_train, y_train, X_test, y_test, n_classes, progress_label="MLP"):
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_STATE)

    scaler = None
    if MLP_USE_SCALER:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train
    )

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)), batch_size=MLP_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), batch_size=MLP_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), batch_size=MLP_BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(X_train.shape[1], n_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=MLP_LEARNING_RATE, weight_decay=MLP_WEIGHT_DECAY)

    best_val = float("inf")
    best_state = None
    patience = 0

    t0 = time.time()
    for epoch in range(MLP_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += loss_fn(out, yb).item() * yb.size(0)
                n_val += yb.size(0)
        val_loss = val_loss / max(1, n_val)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= MLP_PATIENCE:
                _progress(epoch + 1, MLP_EPOCHS, time.time() - t0, prefix=progress_label)
                sys.stdout.write("\n")
                break
        _progress(epoch + 1, MLP_EPOCHS, time.time() - t0, prefix=progress_label)
    else:
        sys.stdout.write("\n")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(pred.tolist())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return model, scaler, acc, f1


def eval_mlp(model, scaler, X, y):
    if scaler is not None:
        X = scaler.transform(X)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    loader = DataLoader(TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)), batch_size=256, shuffle=False)
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(pred.tolist())
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro", zero_division=0)


def _f(v):
    return round(float(v), 4) if v is not None else None


def _model_sort_key(label: str):
    if label.isdigit():
        return (0, int(label))
    return (1, label)


def _progress(n, total, elapsed, prefix=""):
    width = 30
    frac = 1.0 if total == 0 else n / total
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    eta = (total - n) / (n / elapsed) if n > 0 and elapsed > 0 else 0
    lead = f"{prefix} " if prefix else ""
    sys.stdout.write(f"\r{lead}[{bar}] {n}/{total} ETA {int(eta//60):02d}:{int(eta%60):02d}")
    sys.stdout.flush()


def _save_feature_importance(importance_by_model):
    if not importance_by_model:
        return

    labels = sorted(importance_by_model.keys(), key=_model_sort_key)
    csv_path = os.path.join(BASE_DIR, "feature_importance_v2.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("rank,model,embedding_dim,importance\n")
        for m in labels:
            lst = importance_by_model[m]
            for r, (dim, imp) in enumerate(lst, 1):
                f.write(f"{r},{m},{dim},{imp:.6f}\n")

    json_path = os.path.join(BASE_DIR, "feature_importance_v2.json")
    j = {
        m: [{"rank": r, "embedding_dim": dim, "importance": round(imp, 6)} for r, (dim, imp) in enumerate(importance_by_model[m], 1)]
        for m in labels
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(j, f, indent=2)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed; skipping feature importance plot")
        return

    n_models = len(labels)
    n_cols = min(2, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, max(5, 4 * n_rows)))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, m in enumerate(labels):
        ax = axes[i]
        top = importance_by_model[m][:TOP_N]
        dims = [str(d) for d, _ in top]
        vals = [v for _, v in top]
        ypos = np.arange(len(dims))[::-1]
        ax.barh(ypos, vals, color="steelblue")
        ax.set_yticks(ypos)
        ax.set_yticklabels(dims, fontsize=7)
        ax.set_title(f"Top {len(top)} dims — {m}")
        ax.set_xlabel("Importance")
        ax.grid(axis="x", alpha=0.3)

    for j in range(n_models, len(axes)):
        axes[j].set_visible(False)

    plot_path = os.path.join(BASE_DIR, "feature_importance_v2.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_results(results):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed; skipping results plot")
        return

    years = [str(y) for y in YEARS if str(y) in results.get("same_year", {})]
    if not years:
        return

    has_custom = any(str(y) in results.get("cross_year_custom_model", {}) for y in years)
    w = 0.13 if has_custom else 0.2
    x = np.arange(len(years))
    fig, axes = plt.subplots(1, 2, figsize=(14 if has_custom else 12, 5))

    same_rf = [results["same_year"][y]["rf"]["accuracy"] for y in years]
    same_mlp = [results["same_year"][y]["mlp"]["accuracy"] for y in years]
    c2012 = results.get("cross_year_2012_model", {})
    c_rf = [c2012.get(y, {}).get("rf", {}).get("accuracy", 0.0) for y in years]
    c_mlp = [c2012.get(y, {}).get("mlp", {}).get("accuracy", 0.0) for y in years]

    custom = results.get("cross_year_custom_model", {})
    custom_label = custom.get("_label", "Custom")
    cu_rf = [custom.get(y, {}).get("rf", {}).get("accuracy", 0.0) for y in years]
    cu_mlp = [custom.get(y, {}).get("mlp", {}).get("accuracy", 0.0) for y in years]

    ax = axes[0]
    ax.bar(x - 2.5 * w, same_rf, w, label="Same-year RF", color="steelblue")
    ax.bar(x - 1.5 * w, same_mlp, w, label="Same-year MLP", color="darkorange")
    ax.bar(x - 0.5 * w, c_rf, w, label="2012 model (RF)", color="lightblue")
    ax.bar(x + 0.5 * w, c_mlp, w, label="2012 model (MLP)", color="navajowhite")
    if has_custom:
        ax.bar(x + 1.5 * w, cu_rf, w, label=f"{custom_label} (RF)", color="plum")
        ax.bar(x + 2.5 * w, cu_mlp, w, label=f"{custom_label} (MLP)", color="lightgreen")
    ax.set_title("Accuracy by test year (V2)")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    same_rf_f1 = [results["same_year"][y]["rf"]["macro_f1"] for y in years]
    same_mlp_f1 = [results["same_year"][y]["mlp"]["macro_f1"] for y in years]
    c_rf_f1 = [c2012.get(y, {}).get("rf", {}).get("macro_f1", 0.0) for y in years]
    c_mlp_f1 = [c2012.get(y, {}).get("mlp", {}).get("macro_f1", 0.0) for y in years]
    cu_rf_f1 = [custom.get(y, {}).get("rf", {}).get("macro_f1", 0.0) for y in years]
    cu_mlp_f1 = [custom.get(y, {}).get("mlp", {}).get("macro_f1", 0.0) for y in years]

    ax = axes[1]
    ax.bar(x - 2.5 * w, same_rf_f1, w, label="Same-year RF", color="steelblue")
    ax.bar(x - 1.5 * w, same_mlp_f1, w, label="Same-year MLP", color="darkorange")
    ax.bar(x - 0.5 * w, c_rf_f1, w, label="2012 model (RF)", color="lightblue")
    ax.bar(x + 0.5 * w, c_mlp_f1, w, label="2012 model (MLP)", color="navajowhite")
    if has_custom:
        ax.bar(x + 1.5 * w, cu_rf_f1, w, label=f"{custom_label} (RF)", color="plum")
        ax.bar(x + 2.5 * w, cu_mlp_f1, w, label=f"{custom_label} (MLP)", color="lightgreen")
    ax.set_title("Macro F1 by test year (V2)")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    out = os.path.join(BASE_DIR, "results_comparison_v2.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    year_data = {}
    importance_by_model = {}
    t_same = time.time()
    for i, year in enumerate(YEARS, start=1):
        d = get_data_dir(year)
        if not os.path.isdir(d):
            warnings.warn(f"Missing {d}, skipping")
            _progress(i, len(YEARS), time.time() - t_same, prefix="same-year")
            continue
        X, y = load_xy(d)
        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        rf, rf_acc, rf_f1 = run_rf(Xtr, ytr, Xte, yte)
        mlp, scaler, mlp_acc, mlp_f1 = run_mlp(
            Xtr, ytr, Xte, yte, n_classes=len(np.unique(y)), progress_label=f"MLP {year}"
        )
        imp = [(int(i), float(v)) for i, v in enumerate(rf.feature_importances_)]
        imp.sort(key=lambda t: t[1], reverse=True)
        importance_by_model[str(year)] = imp

        year_data[year] = {
            "X_test": Xte,
            "y_test": yte,
            "rf": rf,
            "mlp": mlp,
            "scaler": scaler,
            "same_rf_acc": rf_acc,
            "same_rf_f1": rf_f1,
            "same_mlp_acc": mlp_acc,
            "same_mlp_f1": mlp_f1,
        }
        _progress(i, len(YEARS), time.time() - t_same, prefix="same-year")
    if YEARS:
        sys.stdout.write("\n")

    if not year_data:
        print("No year embedding data found.")
        return

    results = {
        "same_year": {
            str(y): {
                "rf": {"accuracy": _f(year_data[y]["same_rf_acc"]), "macro_f1": _f(year_data[y]["same_rf_f1"])} ,
                "mlp": {"accuracy": _f(year_data[y]["same_mlp_acc"]), "macro_f1": _f(year_data[y]["same_mlp_f1"])} ,
            }
            for y in year_data
        },
        "cross_year_2012_model": {},
        "cross_year_custom_model": {},
    }

    if 2012 in year_data:
        rf = year_data[2012]["rf"]
        mlp = year_data[2012]["mlp"]
        scaler = year_data[2012]["scaler"]
        t_cross_2012 = time.time()
        n_eval = len(year_data)
        for i, y in enumerate(year_data, start=1):
            Xte, yte = year_data[y]["X_test"], year_data[y]["y_test"]
            ra, rf1 = evaluate_rf_on_data(rf, Xte, yte)
            ma, mf1 = eval_mlp(mlp, scaler, Xte, yte)
            results["cross_year_2012_model"][str(y)] = {
                "rf": {"accuracy": _f(ra), "macro_f1": _f(rf1)},
                "mlp": {"accuracy": _f(ma), "macro_f1": _f(mf1)},
            }
            _progress(i, n_eval, time.time() - t_cross_2012, prefix="cross-2012")
        if n_eval:
            sys.stdout.write("\n")

    if os.path.isdir(CUSTOM_DATA_DIR):
        Xc, yc = load_xy(CUSTOM_DATA_DIR)
        tr, te = train_test_split(np.arange(len(yc)), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=yc)
        rf_c, _, _ = run_rf(Xc[tr], yc[tr], Xc[te], yc[te])
        mlp_c, scaler_c, _, _ = run_mlp(
            Xc[tr], yc[tr], Xc[te], yc[te], n_classes=len(np.unique(yc)), progress_label="MLP custom"
        )
        imp = [(int(i), float(v)) for i, v in enumerate(rf_c.feature_importances_)]
        imp.sort(key=lambda t: t[1], reverse=True)
        importance_by_model["custom (data_future)"] = imp

        results["cross_year_custom_model"] = {"_label": "custom (data_future)"}
        t_cross_custom = time.time()
        n_eval = len(year_data)
        for i, y in enumerate(year_data, start=1):
            Xte, yte = year_data[y]["X_test"], year_data[y]["y_test"]
            ra, rf1 = evaluate_rf_on_data(rf_c, Xte, yte)
            ma, mf1 = eval_mlp(mlp_c, scaler_c, Xte, yte)
            results["cross_year_custom_model"][str(y)] = {
                "rf": {"accuracy": _f(ra), "macro_f1": _f(rf1)},
                "mlp": {"accuracy": _f(ma), "macro_f1": _f(mf1)},
            }
            _progress(i, n_eval, time.time() - t_cross_custom, prefix="cross-custom")
        if n_eval:
            sys.stdout.write("\n")

    out = os.path.join(BASE_DIR, "results_summary_v2.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _plot_results(results)
    _save_feature_importance(importance_by_model)
    print(f"Wrote {out}")
    print(f"Wrote {os.path.join(BASE_DIR, 'results_comparison_v2.png')}")
    print(f"Wrote {os.path.join(BASE_DIR, 'feature_importance_v2.csv')}")
    print(f"Wrote {os.path.join(BASE_DIR, 'feature_importance_v2.json')}")
    print(f"Wrote {os.path.join(BASE_DIR, 'feature_importance_v2.png')}")


if __name__ == "__main__":
    main()
