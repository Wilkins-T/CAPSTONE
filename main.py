"""
Train Random Forest and MLP per year (2012, 2014, 2016, 2018).
Compare same-year performance vs applying the 2012 model to other years.
Outputs: results_summary.json and results_comparison.png.
"""

import json
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "drive-download-20260219T230708Z-1-001",
)
YEARS = [2012, 2014, 2016, 2018]

# Custom training set: trained once, then tested on each year's test set (like 2012 model).
# Option A: list of years to combine (e.g. [2012, 2014] = train on 2012+2014 train data).
# Option B: path to a folder with X.npy and y.npy (overrides CUSTOM_TRAIN_YEARS if set).
CUSTOM_TRAIN_YEARS = None  # e.g. [2012, 2014, 2016] or None to disable
CUSTOM_DATA_DIR = None     # e.g. os.path.join(BASE_DATA_DIR, "data_custom") or None

FEATURES_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "drive-download-20260219T230708Z-1-001",
    "selected_features.json",
)
RANDOM_STATE = 42 # seed for reproducibility;
TEST_SIZE = 0.2 # fraction of data used for testing
VAL_SIZE = 0.1  # fraction of train used for validation (early stopping)

# Random Forest (from Random Forest.ipynb)
RF_N_ESTIMATORS = 5000 # number of trees in the forest
RF_MAX_LEAF_NODES = 16 # maximum number of leaves in each tree

# MLP (from Multi-Layer Perceptron.ipynb style)
MLP_HIDDEN_1 = 300 # number of neurons in the first hidden layer
MLP_HIDDEN_2 = 100 # number of neurons in the second hidden layer
MLP_BATCH_SIZE = 16 # batch size for training
MLP_EPOCHS = 100  # max epochs; early stopping may stop sooner
MLP_LEARNING_RATE = 0.00001 # learning rate for the optimizer
MLP_WEIGHT_DECAY = 5e-4 # L2 regularization strength
MLP_PATIENCE = 20  # stop if val loss doesn't improve for this many epochs
MLP_USE_SCALER = True  # scale/normalize features for MLP
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_IMPORTANCE_TOP_N = 40  # number of top features to show in the graph


def load_feature_names(features_json_path: str):
    """
    Load selected_features.json (feature_name -> index) and return index -> feature_name.
    Returns dict[int, str] and list of (index, name) in index order for alignment with RF.
    """
    if not os.path.isfile(features_json_path):
        return {}
    with open(features_json_path) as f:
        name_to_idx = json.load(f)
    return {int(idx): name for name, idx in name_to_idx.items()}


def compute_feature_importance(rf_model, index_to_name: dict, n_features: int):
    """
    Get RF feature_importances_ and pair with feature names from selected_features.json.
    Returns list of (feature_name, importance) sorted by importance descending.
    """
    imp = rf_model.feature_importances_
    out = []
    for i in range(min(len(imp), n_features)):
        name = index_to_name.get(i, f"feature_{i}")
        out.append((name, float(imp[i])))
    # sort by importance descending
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def _model_sort_key(label: str):
    """Sort key: numeric model labels (2012, 2014, ...) first, then 'custom (...)' last."""
    if label.isdigit():
        return (0, int(label))
    return (1, label)


def save_and_plot_feature_importance_per_model(
    importance_by_model: dict, out_dir: str
):
    """
    importance_by_model: dict[model_label, list of (feature_name, importance)].
    Keys are model identifiers (e.g. "2012", "2014", "custom (2012+2014)").
    Save one CSV and one JSON; plot one figure with a subplot per trained model.
    """
    if not importance_by_model:
        return
    out_dir = out_dir or RESULTS_DIR
    labels_sorted = sorted(importance_by_model.keys(), key=_model_sort_key)

    # CSV: rank, model, feature_name, importance (one row per feature per model)
    csv_path = os.path.join(out_dir, "feature_importance.csv")
    with open(csv_path, "w") as f:
        f.write("rank,model,feature_name,importance\n")
        for model_label in labels_sorted:
            lst = importance_by_model[model_label]
            for r, (name, imp) in enumerate(lst, 1):
                name_esc = name.replace('"', '""') if '"' in name else name
                if "," in name or '"' in name_esc:
                    name_esc = f'"{name_esc}"'
                model_esc = model_label.replace('"', '""') if '"' in model_label else model_label
                if "," in model_label or '"' in model_esc:
                    model_esc = f'"{model_esc}"'
                f.write(f"{r},{model_esc},{name_esc},{imp:.6f}\n")
    print(f"Feature importance table (per model) saved to {csv_path}")

    # JSON: { "2012": [ {rank, feature_name, importance}, ... ], "custom (...)": [...], ... }
    json_path = os.path.join(out_dir, "feature_importance.json")
    json_data = {}
    for model_label in labels_sorted:
        lst = importance_by_model[model_label]
        json_data[model_label] = [
            {"rank": r, "feature_name": name, "importance": round(imp, 6)}
            for r, (name, imp) in enumerate(lst, 1)
        ]
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Feature importance JSON (per model) saved to {json_path}")

    # Plot: one subplot per trained model, top N features each
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed; skipping feature importance plot.")
        return

    n_models = len(labels_sorted)
    n_cols = min(2, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(7 * n_cols, max(5, 4 * n_rows))
    )
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, model_label in enumerate(labels_sorted):
        ax = axes[idx]
        lst = importance_by_model[model_label]
        top = lst[: FEATURE_IMPORTANCE_TOP_N]
        names = [n[:50] + ("..." if len(n) > 50 else "") for n, _ in top]
        values = [v for _, v in top]
        y_pos = np.arange(len(names))[::-1]
        ax.barh(y_pos, values, color="steelblue", edgecolor="navy", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {len(top)} features — {model_label}")
        ax.grid(axis="x", alpha=0.3)

    for j in range(n_models, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Feature importance plot (per model) saved to {plot_path}")


def get_data_dir(year: int) -> str:
    """Return path to data_YYYY folder."""
    return os.path.join(BASE_DATA_DIR, f"data_{year}")


def load_data(data_dir: str):
    """Load X.npy and y.npy from data_dir. Returns (X, y_encoded, label_encoder, y_raw)."""
    X_path = os.path.join(data_dir, "X.npy")
    y_path = os.path.join(data_dir, "y.npy")
    if not os.path.isfile(X_path) or not os.path.isfile(y_path):
        raise FileNotFoundError(
            f"Data not found. Expected {X_path} and {y_path}."
        )
    X = np.load(X_path, allow_pickle=True)
    y_raw = np.load(y_path, allow_pickle=True)
    if y_raw.ndim > 1:
        y_raw = y_raw.ravel()
    if not np.isfinite(X).all():
        n_bad = np.logical_not(np.isfinite(X)).sum()
        warnings.warn(f"Replacing {n_bad} non-finite values in X with 0.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return X, y, le, y_raw


def evaluate_rf_on_data(rf_model, X_test, y_test):
    """Evaluate a trained RF on (X_test, y_test). Returns (accuracy, macro_f1)."""
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    return acc, f1


def evaluate_mlp_on_data(model, X_test, y_test, scaler, device, batch_size=256):
    """Evaluate a trained MLP on (X_test, y_test), optionally scaling X_test with scaler."""
    if scaler is not None:
        X_test = scaler.transform(X_test)
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=batch_size,
        shuffle=False,
    )
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(device)
            out = model(X_b)
            _, pred = torch.max(out, 1)
            y_true.extend(y_b.tolist())
            y_pred.extend(pred.cpu().tolist())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, f1


def run_random_forest(X_train, y_train, X_test, y_test, verbose=True):
    """Train and evaluate Random Forest. Returns (model, accuracy, macro_f1)."""
    if verbose:
        print("\n  [RF] Training...")
    rnd_clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_leaf_nodes=RF_MAX_LEAF_NODES,
        random_state=RANDOM_STATE,
    )
    rnd_clf.fit(X_train, y_train)
    acc, f1 = evaluate_rf_on_data(rnd_clf, X_test, y_test)
    if verbose:
        print(f"  [RF] Test accuracy: {acc:.4f}, macro F1: {f1:.4f}")
    return rnd_clf, acc, f1


class MLP(nn.Module):
    """
    Multi-layer perceptron for tabular data (reference: Multi-Layer Perceptron.ipynb).
    Input size and number of classes are set from data.
    """

    def __init__(self, n_features: int, n_classes: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_features, MLP_HIDDEN_1)
        self.fc2 = nn.Linear(MLP_HIDDEN_1, MLP_HIDDEN_2)
        self.fc3 = nn.Linear(MLP_HIDDEN_2, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_epoch(model, data_loader, optimizer, loss_fn, device):
    """Run one training epoch; return mean loss and accuracy."""
    model.train()
    total_loss = 0.0
    total, correct = 0, 0

    for X_batch, y_batch in data_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = loss_fn(predictions, y_batch)
        loss.backward()
        optimizer.step()

        _, pred = torch.max(predictions.data, 1)
        total_loss += loss.item()
        total += y_batch.size(0)
        correct += (pred == y_batch).sum().item()

    return total_loss / total, correct / total


def compute_loss_accuracy(model, data_loader, loss_fn, device):
    """Compute mean loss and accuracy without training (e.g. for validation)."""
    model.eval()
    total_loss = 0.0
    total, correct = 0, 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            _, pred = torch.max(predictions.data, 1)
            total_loss += loss.item() * y_batch.size(0)
            total += y_batch.size(0)
            correct += (pred == y_batch).sum().item()

    return total_loss / total, correct / total


def evaluate_mlp(model, data_loader, device):
    """Return predictions and (optionally) labels; compute loss if loss_fn given."""
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch)
            _, pred = torch.max(predictions.data, 1)
            y_true.extend(y_batch.tolist())
            y_pred.extend(pred.cpu().tolist())

    return y_true, y_pred


def run_mlp(X_train, y_train, X_test, y_test, n_classes: int, verbose=True):
    """Train and evaluate MLP. Returns (model, scaler, accuracy, macro_f1). scaler is None if not using scaling."""
    if verbose:
        print("\n  [MLP] Training...")

    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X_train.shape[1]
    scaler = None

    if MLP_USE_SCALER:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train
    )

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
        batch_size=MLP_BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=MLP_BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=MLP_BATCH_SIZE,
        shuffle=False,
    )

    model = MLP(n_features=n_features, n_classes=n_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=MLP_LEARNING_RATE,
        weight_decay=MLP_WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, MLP_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = compute_loss_accuracy(model, val_loader, loss_fn, device)

        if verbose and epoch % 10 == 0:
            print(
                f"    epoch {epoch}: train acc={train_acc:.4f} loss={train_loss:.4f} | "
                f"val acc={val_acc:.4f} val loss={val_loss:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= MLP_PATIENCE:
                if verbose:
                    print(f"    Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    y_true, y_pred = evaluate_mlp(model, test_loader, device)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    if verbose:
        print(f"  [MLP] Test accuracy: {acc:.4f}, macro F1: {f1:.4f}")

    return model, scaler, acc, f1


def _align_labels_to_encoder(y_raw, encoder):
    """Filter samples to those with labels in encoder.classes_ and return encoded y."""
    mask = np.isin(y_raw, encoder.classes_)
    if not np.all(mask):
        n_drop = np.sum(~mask)
        warnings.warn(f"Dropping {n_drop} test samples with labels not in encoder (cross-year).")
    y_encoded = encoder.transform(y_raw[mask])
    return mask, y_encoded


def plot_results(results: dict, out_path: str):
    """Generate comparison graphs and save to out_path."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed; skipping plots.")
        return

    years = [str(y) for y in YEARS if str(y) in results.get("same_year", {})]
    if not years:
        return

    custom_data = results.get("cross_year_custom_model") or {}
    has_custom = any(str(y) in custom_data for y in years)
    w = 0.13 if has_custom else 0.2
    fig, axes = plt.subplots(1, 2, figsize=(14 if has_custom else 12, 5))
    n = len(years)
    x = np.arange(n)
    same_rf = [results["same_year"][y]["rf"]["accuracy"] for y in years]
    same_mlp = [results["same_year"][y]["mlp"]["accuracy"] for y in years]
    cross_2012 = results.get("cross_year_2012_model", {})
    cross_rf = []
    cross_mlp = []
    for i, y in enumerate(years):
        r = cross_2012.get(y, {}).get("rf", {})
        m = cross_2012.get(y, {}).get("mlp", {})
        cross_rf.append(r.get("accuracy") if r.get("accuracy") is not None else (same_rf[i] if y == "2012" else 0.0))
        cross_mlp.append(m.get("accuracy") if m.get("accuracy") is not None else (same_mlp[i] if y == "2012" else 0.0))

    custom_rf, custom_mlp = [], []
    if has_custom:
        for y in years:
            r = custom_data.get(y, {}).get("rf", {})
            m = custom_data.get(y, {}).get("mlp", {})
            custom_rf.append(r.get("accuracy") if r.get("accuracy") is not None else 0.0)
            custom_mlp.append(m.get("accuracy") if m.get("accuracy") is not None else 0.0)
    custom_label = custom_data.get("_label", "Custom")

    # --- Accuracy ---
    ax = axes[0]
    ax.bar(x - 2.5 * w, same_rf, w, label="Same-year RF", color="steelblue")
    ax.bar(x - 1.5 * w, same_mlp, w, label="Same-year MLP", color="darkorange")
    ax.bar(x - 0.5 * w, cross_rf, w, label="2012 model (RF)", color="lightblue", edgecolor="gray")
    ax.bar(x + 0.5 * w, cross_mlp, w, label="2012 model (MLP)", color="navajowhite", edgecolor="gray")
    if has_custom:
        ax.bar(x + 1.5 * w, custom_rf, w, label=f"{custom_label} (RF)", color="plum", edgecolor="gray")
        ax.bar(x + 2.5 * w, custom_mlp, w, label=f"{custom_label} (MLP)", color="lightgreen", edgecolor="gray")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by test year")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # --- Macro F1 ---
    ax = axes[1]
    same_rf_f1 = [results["same_year"][y]["rf"]["macro_f1"] for y in years]
    same_mlp_f1 = [results["same_year"][y]["mlp"]["macro_f1"] for y in years]
    cross_rf_f1 = []
    cross_mlp_f1 = []
    for i, y in enumerate(years):
        r = cross_2012.get(y, {}).get("rf", {})
        m = cross_2012.get(y, {}).get("mlp", {})
        cross_rf_f1.append(r.get("macro_f1") if r.get("macro_f1") is not None else (same_rf_f1[i] if y == "2012" else 0.0))
        cross_mlp_f1.append(m.get("macro_f1") if m.get("macro_f1") is not None else (same_mlp_f1[i] if y == "2012" else 0.0))
    custom_rf_f1, custom_mlp_f1 = [], []
    if has_custom:
        for y in years:
            r = custom_data.get(y, {}).get("rf", {})
            m = custom_data.get(y, {}).get("mlp", {})
            custom_rf_f1.append(r.get("macro_f1") if r.get("macro_f1") is not None else 0.0)
            custom_mlp_f1.append(m.get("macro_f1") if m.get("macro_f1") is not None else 0.0)
    ax.bar(x - 2.5 * w, same_rf_f1, w, label="Same-year RF", color="steelblue")
    ax.bar(x - 1.5 * w, same_mlp_f1, w, label="Same-year MLP", color="darkorange")
    ax.bar(x - 0.5 * w, cross_rf_f1, w, label="2012 model (RF)", color="lightblue", edgecolor="gray")
    ax.bar(x + 0.5 * w, cross_mlp_f1, w, label="2012 model (MLP)", color="navajowhite", edgecolor="gray")
    if has_custom:
        ax.bar(x + 1.5 * w, custom_rf_f1, w, label=f"{custom_label} (RF)", color="plum", edgecolor="gray")
        ax.bar(x + 2.5 * w, custom_mlp_f1, w, label=f"{custom_label} (MLP)", color="lightgreen", edgecolor="gray")
    ax.set_ylabel("Macro F1")
    ax.set_title("Macro F1 by test year")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all years and train per-year models
    year_data = {}
    for year in YEARS:
        data_dir = get_data_dir(year)
        if not os.path.isdir(data_dir):
            warnings.warn(f"Skipping {year}: directory not found: {data_dir}")
            continue
        try:
            X, y, encoder, y_raw = load_data(data_dir)
        except FileNotFoundError as e:
            warnings.warn(f"Skipping {year}: {e}")
            continue

        n_classes = len(np.unique(y))
        # Stratified split; then get raw labels for test set (for cross-year eval)
        idx = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_train_raw, y_test_raw = y_raw[train_idx], y_raw[test_idx]

        print(f"\n{'='*60}")
        print(f"YEAR {year}  (train: {len(X_train)}, test: {len(X_test)}, classes: {n_classes})")
        print("=" * 60)

        rf_model, rf_acc, rf_f1 = run_random_forest(
            X_train, y_train, X_test, y_test, verbose=True
        )
        mlp_model, mlp_scaler, mlp_acc, mlp_f1 = run_mlp(
            X_train, y_train, X_test, y_test, n_classes=n_classes, verbose=True
        )

        year_data[year] = {
            "rf_model": rf_model,
            "mlp_model": mlp_model,
            "mlp_scaler": mlp_scaler,
            "encoder": encoder,
            "X_train": X_train,
            "y_train": y_train,
            "y_train_raw": y_train_raw,
            "X_test": X_test,
            "y_test": y_test,
            "y_test_raw": y_test_raw,
            "same_rf_acc": rf_acc,
            "same_rf_f1": rf_f1,
            "same_mlp_acc": mlp_acc,
            "same_mlp_f1": mlp_f1,
        }

    if not year_data:
        print("No year data loaded. Exiting.")
        return

    def _f(v):
        return round(float(v), 4) if v is not None else None

    # Build same-year results
    results = {
        "same_year": {
            str(y): {
                "rf": {"accuracy": _f(year_data[y]["same_rf_acc"]), "macro_f1": _f(year_data[y]["same_rf_f1"])},
                "mlp": {"accuracy": _f(year_data[y]["same_mlp_acc"]), "macro_f1": _f(year_data[y]["same_mlp_f1"])},
            }
            for y in year_data
        },
        "cross_year_2012_model": {},
        "cross_year_custom_model": {},
    }

    # Cross-year: apply 2012 model to other years' test sets
    if 2012 in year_data:
        rf_2012 = year_data[2012]["rf_model"]
        mlp_2012 = year_data[2012]["mlp_model"]
        mlp_scaler_2012 = year_data[2012]["mlp_scaler"]
        encoder_2012 = year_data[2012]["encoder"]

        for test_year in year_data:
            if test_year == 2012:
                results["cross_year_2012_model"]["2012"] = {
                    "rf": {"accuracy": _f(year_data[2012]["same_rf_acc"]), "macro_f1": _f(year_data[2012]["same_rf_f1"])},
                    "mlp": {"accuracy": _f(year_data[2012]["same_mlp_acc"]), "macro_f1": _f(year_data[2012]["same_mlp_f1"])},
                }
                continue
            X_other = year_data[test_year]["X_test"]
            y_raw_other = year_data[test_year]["y_test_raw"]
            mask, y_other_encoded = _align_labels_to_encoder(y_raw_other, encoder_2012)
            X_other_filt = X_other[mask]
            if len(X_other_filt) == 0:
                results["cross_year_2012_model"][str(test_year)] = {
                    "rf": {"accuracy": None, "macro_f1": None, "note": "no test samples after label alignment"},
                    "mlp": {"accuracy": None, "macro_f1": None, "note": "no test samples after label alignment"},
                }
                continue
            rf_acc_c, rf_f1_c = evaluate_rf_on_data(rf_2012, X_other_filt, y_other_encoded)
            mlp_acc_c, mlp_f1_c = evaluate_mlp_on_data(
                mlp_2012, X_other_filt, y_other_encoded, mlp_scaler_2012, device
            )
            results["cross_year_2012_model"][str(test_year)] = {
                "rf": {"accuracy": _f(rf_acc_c), "macro_f1": _f(rf_f1_c)},
                "mlp": {"accuracy": _f(mlp_acc_c), "macro_f1": _f(mlp_f1_c)},
            }
            print(f"\n2012 model on {test_year} test: RF acc={rf_acc_c:.4f} F1={rf_f1_c:.4f} | MLP acc={mlp_acc_c:.4f} F1={mlp_f1_c:.4f}")

    # Custom training set: train on user-defined data, test on each year
    custom_label = None
    if CUSTOM_DATA_DIR and os.path.isdir(CUSTOM_DATA_DIR):
        try:
            X_c, y_c, encoder_c, y_raw_c = load_data(CUSTOM_DATA_DIR)
            idx = np.arange(len(X_c))
            train_idx, test_idx = train_test_split(
                idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_c
            )
            X_c_train, X_c_test = X_c[train_idx], X_c[test_idx]
            y_c_train, y_c_test = y_c[train_idx], y_c[test_idx]
            n_classes_c = len(np.unique(y_c))
            custom_label = f"custom ({os.path.basename(CUSTOM_DATA_DIR)})"
            print(f"\n{'='*60}")
            print(f"CUSTOM TRAINING: {custom_label}  (train: {len(X_c_train)}, test: {len(X_c_test)})")
            print("=" * 60)
            rf_custom, _, _ = run_random_forest(X_c_train, y_c_train, X_c_test, y_c_test, verbose=True)
            mlp_custom, mlp_scaler_custom, _, _ = run_mlp(
                X_c_train, y_c_train, X_c_test, y_c_test, n_classes=n_classes_c, verbose=True
            )
            encoder_custom = encoder_c
        except FileNotFoundError as e:
            warnings.warn(f"Custom data dir skipped: {e}")
            rf_custom = mlp_custom = mlp_scaler_custom = encoder_custom = None
    elif CUSTOM_TRAIN_YEARS:
        years_to_combine = [y for y in CUSTOM_TRAIN_YEARS if y in year_data]
        if not years_to_combine:
            warnings.warn("CUSTOM_TRAIN_YEARS set but no matching year data; skipping custom model.")
            rf_custom = mlp_custom = mlp_scaler_custom = encoder_custom = None
        else:
            X_list = [year_data[y]["X_train"] for y in years_to_combine]
            y_raw_list = [year_data[y]["y_train_raw"] for y in years_to_combine]
            X_c_train = np.vstack(X_list)
            y_raw_c_train = np.concatenate(y_raw_list)
            encoder_custom = LabelEncoder()
            y_c_train = encoder_custom.fit_transform(y_raw_c_train)
            n_classes_c = len(encoder_custom.classes_)
            custom_label = "custom (" + "+".join(map(str, years_to_combine)) + ")"
            print(f"\n{'='*60}")
            print(f"CUSTOM TRAINING: {custom_label}  (train: {len(X_c_train)} samples)")
            print("=" * 60)
            rf_custom, _, _ = run_random_forest(
                X_c_train, y_c_train, X_c_train, y_c_train, verbose=True
            )
            # For MLP we need a val split from the combined data
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_c_train, y_c_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_c_train
            )
            mlp_custom, mlp_scaler_custom, _, _ = run_mlp(
                X_tr, y_tr, X_val, y_val, n_classes=n_classes_c, verbose=True
            )
    else:
        rf_custom = mlp_custom = mlp_scaler_custom = encoder_custom = None

    if rf_custom is not None and custom_label:
        results["cross_year_custom_model"] = {"_label": custom_label}
        for test_year in year_data:
            X_other = year_data[test_year]["X_test"]
            y_raw_other = year_data[test_year]["y_test_raw"]
            mask, y_other_encoded = _align_labels_to_encoder(y_raw_other, encoder_custom)
            X_other_filt = X_other[mask]
            if len(X_other_filt) == 0:
                results["cross_year_custom_model"][str(test_year)] = {
                    "rf": {"accuracy": None, "macro_f1": None},
                    "mlp": {"accuracy": None, "macro_f1": None},
                }
                continue
            rf_acc_c, rf_f1_c = evaluate_rf_on_data(rf_custom, X_other_filt, y_other_encoded)
            mlp_acc_c, mlp_f1_c = evaluate_mlp_on_data(
                mlp_custom, X_other_filt, y_other_encoded, mlp_scaler_custom, device
            )
            results["cross_year_custom_model"][str(test_year)] = {
                "rf": {"accuracy": _f(rf_acc_c), "macro_f1": _f(rf_f1_c)},
                "mlp": {"accuracy": _f(mlp_acc_c), "macro_f1": _f(mlp_f1_c)},
            }
            print(f"\nCustom model on {test_year} test: RF acc={rf_acc_c:.4f} F1={rf_f1_c:.4f} | MLP acc={mlp_acc_c:.4f} F1={mlp_f1_c:.4f}")

    # Feature importance per trained model (each year's RF + custom RF if present)
    index_to_name = load_feature_names(FEATURES_JSON)
    importance_by_model = {}
    for year in year_data:
        rf_model = year_data[year]["rf_model"]
        n_features = year_data[year]["X_train"].shape[1]
        importance_by_model[str(year)] = compute_feature_importance(
            rf_model, index_to_name, n_features
        )
    if rf_custom is not None and custom_label:
        n_features_custom = getattr(rf_custom, "n_features_in_", None)
        if n_features_custom:
            importance_by_model[custom_label] = compute_feature_importance(
                rf_custom, index_to_name, n_features_custom
            )
    save_and_plot_feature_importance_per_model(
        importance_by_model,
        out_dir=RESULTS_DIR,
    )

    # Save JSON (convert non-JSON-serializable if any)
    results_path = os.path.join(RESULTS_DIR, "results_summary.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults summary saved to {results_path}")

    # Plots
    plot_path = os.path.join(RESULTS_DIR, "results_comparison.png")
    plot_results(results, plot_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
