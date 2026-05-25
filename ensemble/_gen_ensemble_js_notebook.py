#!/usr/bin/env python3
"""Generate ensemble_js_margin_variance.ipynb (nbformat 4, clean outputs)."""
import json
from pathlib import Path

NB = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": [],
}


def md(s: str):
    NB["cells"].append(
        {
            "cell_type": "markdown",
            "id": f"md{len(NB['cells'])}",
            "metadata": {},
            "source": s.splitlines(keepends=True),
        }
    )


def code(s: str):
    NB["cells"].append(
        {
            "cell_type": "code",
            "id": f"code{len(NB['cells'])}",
            "metadata": {},
            "source": s.splitlines(keepends=True),
            "outputs": [],
            "execution_count": None,
        }
    )


md(
    """# Training-Time Ensemble — JS Divergence + Margin + Variance (+ Mean)

This notebook loads **pre-computed `.npy` probability files** from a cache folder.
**No base models** are loaded or run (Colab / T4 friendly).

## Models (3)

Only **BERT**, **mDeBERTa**, and **Qwen** — Gemma is removed everywhere.

## Feature design (~18–21 dimensions)

Instead of Shannon entropy alone, we stack interpretable **disagreement** and **confidence** signals:

| Block | Size | Description |
|-------|------|-------------|
| Raw probabilities | 9 | 3 models × 3 NLI classes (order: BERT, mDeBERTa, Qwen) |
| Margin | 3 | Per model: largest prob − second-largest (prediction confidence) |
| Jensen–Shannon | 3 | Symmetric divergence between each **pair** of model distributions (BERT↔mDeBERTa, BERT↔Qwen, mDeBERTa↔Qwen) |
| Variance (per class) | 3 | For each class, variance of that class’s probability across the 3 models |
| Mean (per class) | 3 | *(optional)* Mean probability per class across models |

With all toggles on: **9 + 3 + 3 + 3 + 3 = 21** features.  
Turning off the optional mean block yields **18** features (still includes JS + margin + variance).

## Cache layout

Point `CACHE_DIR` at a folder that contains **only** these files per config/split:

```
{config}__{split}__bert_probs.npy
{config}__{split}__mdeberta_probs.npy
{config}__{split}__qwen_probs.npy
```

## Ablation strategy

After the full run, we remove **one feature group at a time** and retrain meta-learners, reporting deltas vs the full model on **`trglue_mnli::test_matched`** (instructor focus).
"""
)

code(
    """import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from datasets import load_dataset
from IPython.display import display

from collections import Counter

from scipy.stats import entropy as scipy_entropy

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
"""
)

code(
    """SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ── Three base models only (no Gemma) ─────────────────────────────────────
MODEL_ORDER = ["bert", "mdeberta", "qwen"]

DATASET_NAME = "yilmazzey/sdp2-nli"
CONFIGS      = ["snli_tr_1_1", "multinli_tr_1_1", "trglue_mnli"]
LABEL_MAP    = {0: "entailment", 1: "neutral", 2: "contradiction"}
LABEL_NAMES  = [LABEL_MAP[i] for i in range(3)]
NUM_LABELS   = 3

EVAL_SPLITS = {
    "snli_tr_1_1":     ["test"],
    "multinli_tr_1_1": ["validation_matched", "validation_mismatched"],
    "trglue_mnli":     ["test_matched", "test_mismatched"],
}

# Static ensemble weights (Gemma mass redistributed to remaining models; renormalized)
STATIC_WEIGHTED_WEIGHTS = {"bert": 0.25 / 0.74, "mdeberta": 0.12 / 0.74, "qwen": 0.37 / 0.74}
ROUTER_W = {"bert": 0.0, "mdeberta": 0.25, "qwen": 0.75}
CLASS_WEIGHTS_V1 = {
    0: {"bert": 0.0, "mdeberta": 0.10, "qwen": 0.90},
    1: {"bert": 0.0, "mdeberta": 0.15, "qwen": 0.85},
    2: {"bert": 0.0, "mdeberta": 0.10, "qwen": 0.90},
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ── Cache produced by your feature notebook (3 × .npy per split) ───────────
CACHE_DIR = Path("stacking_cache_train_split_3model_probs")
ARTIFACT_DIR = Path("stacking_artifacts_js21")
ARTIFACT_DIR.mkdir(exist_ok=True)

assert CACHE_DIR.exists(), (
    f"Cache folder not found: {CACHE_DIR.resolve()}\\n"
    "Set CACHE_DIR to the folder with bert / mdeberta / qwen *_probs.npy files."
)
print("Cache folder:", CACHE_DIR.resolve())

META_EPOCHS             = 40
EARLY_STOPPING_PATIENCE = 8
META_BATCH_SIZE         = 128
LEARNING_RATE           = 1e-3
WEIGHT_DECAY            = 1e-4
"""
)

md("## Load labels from HuggingFace (no model inference)")

code(
    """# Read HF token from environment — never hard-code tokens.
hf_token = os.environ.get("HF_TOKEN", None)
if hf_token:
    from huggingface_hub import login
    login(token=hf_token)
    print("Logged in to HuggingFace.")
else:
    print("HF_TOKEN not set — assuming public dataset or already logged in.")

datasets_hf = {}
for cfg in CONFIGS:
    print(f"Loading labels: {DATASET_NAME} :: {cfg}")
    datasets_hf[cfg] = load_dataset(DATASET_NAME, cfg)
    print("  splits:", list(datasets_hf[cfg].keys()))
print("Label loading complete.")
"""
)

md("## Cache loading utilities")

code(
    """def cache_name(config: str, split: str, model_key: str) -> Path:
    safe_cfg = config.replace("/", "_")
    safe_sp = split.replace("/", "_")
    return CACHE_DIR / f"{safe_cfg}__{safe_sp}__{model_key}_probs.npy"


def load_probs_from_cache(config: str, split: str) -> dict:
    \"\"\"Load three model probability arrays from .npy. No inference.\"\"\"
    probs = {}
    for model_key in MODEL_ORDER:
        p = cache_name(config, split, model_key)
        if not p.exists():
            raise FileNotFoundError(f"Missing cache file: {p}")
        probs[model_key] = np.asarray(np.load(p), dtype=np.float64)
    shapes = {k: v.shape for k, v in probs.items()}
    print(f"  Loaded {config}/{split} — shapes: {shapes}")
    return probs
"""
)

md(
    """## Feature engineering (`build_X`)

**Jensen–Shannon divergence** between two discrete distributions \\(P\\) and \\(Q\\):

\\[
\\mathrm{JS}(P,Q)=\\frac{1}{2}D_{\\mathrm{KL}}(P\\|M)+\\frac{1}{2}D_{\\mathrm{KL}}(Q\\|M),\\quad M=\\frac{P+Q}{2}
\\]

We implement this in log-space (natural log) for numerical stability; `scipy.stats.entropy` is used for the KL terms row-wise.

**Margin** (per model): if sorted class probabilities are \\(p_{(1)}\\ge p_{(2)}\\ge p_{(3)}\\), margin \\(=p_{(1)}-p_{(2)}\\) (how decisive the argmax is).

**Variance / mean** (per NLI class): across the three models, for each class index \\(c\\), summarize the three values \\(p^{(\\mathrm{bert})}_c, p^{(\\mathrm{mdeb})}_c, p^{(\\mathrm{qwen})}_c\\).
"""
)

code(
    """@dataclass
class FeatureFlags:
    \"\"\"Toggle feature groups for ablation studies. Raw probs (9) are always on.\"\"\"
    use_margin: bool = True      # +3  confidence per model
    use_js: bool = True          # +3  pairwise disagreement
    use_var: bool = True         # +3  spread per class across models
    use_mean_prob: bool = True   # +3  consensus level per class (optional block)


def js_divergence_rows(p, q):
    \"\"\"Jensen–Shannon divergence per row; uses scipy.stats.entropy for each KL term (axis=1).\"\"\"
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-10, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-10, 1.0)
    m = 0.5 * (p + q)
    m = np.clip(m, 1e-10, 1.0)
    kl_pm = scipy_entropy(p, m, axis=1)
    kl_qm = scipy_entropy(q, m, axis=1)
    return 0.5 * (kl_pm + kl_qm)


def margin_per_row(probs):
    \"\"\"max - second_max for each row, shape (N,).\"\"\"
    s = np.sort(probs, axis=1)
    return (s[:, -1] - s[:, -2]).astype(np.float64)


def feature_layout(flags: FeatureFlags):
    \"\"\"Return ordered list of (block_name, n_cols) for documentation / naming.\"\"\"
    blocks = [("raw_probs", 9)]
    if flags.use_margin:
        blocks.append(("margin", 3))
    if flags.use_js:
        blocks.append(("js_div", 3))
    if flags.use_var:
        blocks.append(("var_per_class", 3))
    if flags.use_mean_prob:
        blocks.append(("mean_per_class", 3))
    return blocks


def build_feature_names(flags: FeatureFlags):
    names = []
    for m in MODEL_ORDER:
        for c in range(3):
            names.append(f"{m}_p{c}")
    if flags.use_margin:
        for m in MODEL_ORDER:
            names.append(f"margin_{m}")
    if flags.use_js:
        names.extend(["js_bert_mdeberta", "js_bert_qwen", "js_mdeberta_qwen"])
    if flags.use_var:
        names.extend(["var_class_entailment", "var_class_neutral", "var_class_contradiction"])
    if flags.use_mean_prob:
        names.extend(["mean_class_entailment", "mean_class_neutral", "mean_class_contradiction"])
    return names


def build_X(probs_dict: dict, flags: Optional[FeatureFlags] = None) -> np.ndarray:
    \"\"\"
    Stack meta-features for 3 models. Column order:
      [9 raw probs][optional 3 margins][optional 3 JS][optional 3 var][optional 3 mean]

    Memory: single hstack of float32 at the end; intermediates are vectorized numpy.
    \"\"\"
    if flags is None:
        flags = FeatureFlags()

    bert = np.asarray(probs_dict["bert"], dtype=np.float64)
    mdeb = np.asarray(probs_dict["mdeberta"], dtype=np.float64)
    qwen = np.asarray(probs_dict["qwen"], dtype=np.float64)
    assert bert.shape[1] == 3 and mdeb.shape[1] == 3 and qwen.shape[1] == 3

    stack = np.stack([bert, mdeb, qwen], axis=1)  # (N, 3, 3)

    parts = [stack.reshape(bert.shape[0], 9)]

    if flags.use_margin:
        margins = np.stack([margin_per_row(bert), margin_per_row(mdeb), margin_per_row(qwen)], axis=1)
        parts.append(margins)

    if flags.use_js:
        js_bm = js_divergence_rows(bert, mdeb).reshape(-1, 1)
        js_bq = js_divergence_rows(bert, qwen).reshape(-1, 1)
        js_mq = js_divergence_rows(mdeb, qwen).reshape(-1, 1)
        parts.append(np.hstack([js_bm, js_bq, js_mq]))

    if flags.use_var or flags.use_mean_prob:
        # variance / mean across models for each class
        vcls = np.var(stack, axis=1)
        mcls = np.mean(stack, axis=1)
        if flags.use_var:
            parts.append(vcls)
        if flags.use_mean_prob:
            parts.append(mcls)

    X = np.hstack(parts).astype(np.float32)
    return X


# Default full feature flags used for main training
DEFAULT_FLAGS = FeatureFlags()
FULL_FEATURE_NAMES = build_feature_names(DEFAULT_FLAGS)
print("Full feature dim:", len(FULL_FEATURE_NAMES), "| names:", FULL_FEATURE_NAMES)
"""
)

md("## Static ensemble helpers (3 models)")

code(
    """def onehot_from_probs(p: np.ndarray) -> np.ndarray:
    out = np.zeros_like(p)
    out[np.arange(len(p)), p.argmax(axis=1)] = 1.0
    return out


def weighted_static_pred(bert_p, mdeb_p, qwen_p):
    w = np.array([STATIC_WEIGHTED_WEIGHTS[m] for m in MODEL_ORDER], dtype=np.float32)
    probs = np.stack([bert_p, mdeb_p, qwen_p], axis=0)
    return np.einsum("m,mnc->nc", w, probs).argmax(axis=1)


def majority_vote_pred(bert_p, mdeb_p, qwen_p):
    votes = np.stack([bert_p.argmax(1), mdeb_p.argmax(1), qwen_p.argmax(1)], axis=1)
    return np.array(
        [int(Counter(row.tolist()).most_common(1)[0][0]) for row in votes],
        dtype=np.int64,
    )


def class_conditional_routing_pred(bert_p, mdeb_p, qwen_p):
    oh = np.stack(
        [onehot_from_probs(bert_p), onehot_from_probs(mdeb_p), onehot_from_probs(qwen_p)],
        axis=0,
    )
    router_w = np.array([ROUTER_W[m] for m in MODEL_ORDER], dtype=np.float32)
    routed_class = np.einsum("m,mnc->nc", router_w, oh).argmax(axis=1)
    final = np.empty(len(routed_class), dtype=np.int64)
    for c in range(NUM_LABELS):
        mask = routed_class == c
        if not mask.any():
            continue
        w_c = np.array([CLASS_WEIGHTS_V1[c][m] for m in MODEL_ORDER], dtype=np.float32)
        final[mask] = np.einsum("m,mnc->nc", w_c, oh[:, mask, :]).argmax(axis=1)
    return final
"""
)

md("## Meta-learner architectures")

code(
    """class StackingMLP(nn.Module):
    \"\"\"Two-hidden-layer MLP; `in_dim` matches current feature vector length.\"\"\"
    def __init__(self, in_dim: int, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class StackingBiLSTM(nn.Module):
    \"\"\"
    Grouped BiLSTM over **3 model timesteps**.
    Each timestep = that model's 3 class probabilities plus optional scalar margin
    → 3 or 4 input features per step. Global tail (JS, var, mean blocks) is concatenated
    to the final BiLSTM hidden state before the classifier.
    \"\"\"
    def __init__(self, per_model_dim: int, global_dim: int, hidden_size: int = 64, n_classes: int = 3):
        super().__init__()
        assert per_model_dim in (3, 4)
        self.per_model_dim = per_model_dim
        self.global_dim = global_dim
        self.lstm = nn.LSTM(
            input_size=per_model_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size * 2 + global_dim, n_classes)

    def forward(self, x):
        b = x.size(0)
        seq = x[:, : 3 * self.per_model_dim].contiguous().view(b, 3, self.per_model_dim)
        tail = x[:, 3 * self.per_model_dim :]
        out, _ = self.lstm(seq)
        last = out[:, -1, :]
        if self.global_dim > 0:
            last = torch.cat([last, tail], dim=-1)
        return self.fc(self.drop(last))


class StackingLSTM_flat(nn.Module):
    \"\"\"Treat every scalar feature as its own timestep (input_size=1).\"\"\"
    def __init__(self, in_dim: int, hidden_size: int = 64, n_classes: int = 3):
        super().__init__()
        self.in_dim = in_dim
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):
        seq = x.unsqueeze(-1)
        out, _ = self.lstm(seq)
        return self.fc(self.drop(out[:, -1, :]))
"""
)

md("## Training loop")

code(
    """def _train_torch_model(model, X_train, y_train, model_name="model"):
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    n = len(X_t)
    n_val = max(1, int(0.10 * n))
    train_ds, val_ds = random_split(
        TensorDataset(X_t, y_t),
        [n - n_val, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=META_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=META_BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=META_EPOCHS, eta_min=1e-5)

    best_state, best_val, bad_epochs = None, float("inf"), 0
    model = model.to(DEVICE)

    for epoch in range(META_EPOCHS):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_losses.append(criterion(model(xb), yb).item())

        mean_tr = float(np.mean(tr_losses)) if tr_losses else float("nan")
        mean_val = float(np.mean(val_losses)) if val_losses else float("nan")
        print(f"{model_name} epoch {epoch+1:02d}/{META_EPOCHS} | train_loss={mean_tr:.4f}  val_loss={mean_val:.4f}")

        if mean_val < best_val - 1e-5:
            best_val = mean_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping {model_name} at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _get_accuracy_batched(model, X_data, y_true, batch_size):
    model.eval()
    preds, y_list = [], []
    ds = TensorDataset(torch.tensor(X_data, dtype=torch.float32), torch.tensor(y_true, dtype=torch.long))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            preds.append(model(xb).argmax(-1).cpu().numpy())
            y_list.append(yb.numpy())
    return accuracy_score(np.concatenate(y_list), np.concatenate(preds))


def train_meta_learners(X_train: np.ndarray, y_train: np.ndarray, flags: FeatureFlags):
    in_dim = X_train.shape[1]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train).astype(np.float32)

    mlp = StackingMLP(in_dim=in_dim, n_classes=NUM_LABELS)
    mlp = _train_torch_model(mlp, Xs, y_train, model_name="MLP")

    svm = LinearSVC(C=1.0, class_weight="balanced", random_state=SEED, max_iter=20000)
    svm.fit(Xs, y_train)
    print("LinearSVC fitted.")

    per_model_dim = 4 if flags.use_margin else 3
    global_dim = in_dim - 3 * per_model_dim
    bilstm = StackingBiLSTM(per_model_dim=per_model_dim, global_dim=global_dim, hidden_size=64, n_classes=NUM_LABELS)
    bilstm = _train_torch_model(bilstm, Xs, y_train, model_name="BiLSTM")

    flat_lstm = StackingLSTM_flat(in_dim=in_dim, hidden_size=64, n_classes=NUM_LABELS)
    flat_lstm = _train_torch_model(flat_lstm, Xs, y_train, model_name="FlatLSTM")

    mlp_acc = _get_accuracy_batched(mlp, Xs, y_train, META_BATCH_SIZE)
    bi_acc = _get_accuracy_batched(bilstm, Xs, y_train, META_BATCH_SIZE)
    fl_acc = _get_accuracy_batched(flat_lstm, Xs, y_train, META_BATCH_SIZE)
    print(
        f"Train acc | MLP: {mlp_acc:.4f} | SVC: {(svm.predict(Xs)==y_train).mean():.4f} "
        f"| BiLSTM: {bi_acc:.4f} | FlatLSTM: {fl_acc:.4f}"
    )

    return {
        "scaler": scaler,
        "mlp": mlp,
        "svm": svm,
        "bilstm": bilstm,
        "flat_lstm": flat_lstm,
        "feature_flags": flags,
        "in_dim": in_dim,
        "per_model_dim": per_model_dim,
        "global_dim": global_dim,
    }
"""
)

md("## Metrics and display helpers")

code(
    """def compute_metrics_dict(y_true, y_pred):
    acc = float(accuracy_score(y_true, y_pred))
    f1m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_each = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_per = {LABEL_NAMES[i]: float(f1_each[i]) for i in range(NUM_LABELS)}
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return {"accuracy": acc, "f1_macro": f1m, "f1_per_class": f1_per, "cm": cm}


def plot_confusion(cm, title, save_path=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print("Saved figure:", save_path)
    plt.show()


def append_row(rows, split_key, method, y_true, y_pred, row_kind="computed"):
    m = compute_metrics_dict(y_true, y_pred)
    rows.append(
        {
            "split": split_key,
            "method": method,
            "row_kind": row_kind,
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
            "f1_entailment": m["f1_per_class"]["entailment"],
            "f1_neutral": m["f1_per_class"]["neutral"],
            "f1_contradiction": m["f1_per_class"]["contradiction"],
        }
    )


def append_reference_row(rows, split_key):
    ref_acc = 0.8408 if split_key == "trglue_mnli::test_matched" else float("nan")
    rows.append(
        {
            "split": split_key,
            "method": "REF_published_static_weighted_84.08pct",
            "row_kind": "reference_published",
            "accuracy": ref_acc,
            "f1_macro": float("nan"),
            "f1_entailment": float("nan"),
            "f1_neutral": float("nan"),
            "f1_contradiction": float("nan"),
        }
    )
"""
)

md("## Main training + evaluation loop")

code(
    """METHOD_TAG = "Meta_train_only_js_margin_var"

rows = []
trained_meta = {}
train_feature_bank = {}
eval_feature_bank = {}

for cfg in CONFIGS:
    print("\\n" + "=" * 100)
    print(f"Config: {cfg}")

    y_train = np.array(datasets_hf[cfg]["train"]["label"], dtype=np.int64)
    train_probs = load_probs_from_cache(cfg, "train")
    X_train = build_X(train_probs, DEFAULT_FLAGS)
    print(f"  Train features shape: {X_train.shape}")

    meta = train_meta_learners(X_train, y_train, DEFAULT_FLAGS)
    trained_meta[cfg] = meta
    train_feature_bank[cfg] = {"X_train": X_train, "y_train": y_train, "probs": train_probs}
    eval_feature_bank[cfg] = {}

    for sp in EVAL_SPLITS[cfg]:
        split_key = f"{cfg}::{sp}"
        y_true = np.array(datasets_hf[cfg][sp]["label"], dtype=np.int64)
        eval_probs = load_probs_from_cache(cfg, sp)
        X_eval = build_X(eval_probs, DEFAULT_FLAGS)
        X_eval_s = meta["scaler"].transform(X_eval).astype(np.float32)

        eval_feature_bank[cfg][sp] = {"X_eval_s": X_eval_s, "y_true": y_true, "probs": eval_probs}

        Xt = torch.tensor(X_eval_s, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            pred_mlp = meta["mlp"](Xt).argmax(-1).cpu().numpy()
            pred_bilstm = meta["bilstm"](Xt).argmax(-1).cpu().numpy()
            pred_flat = meta["flat_lstm"](Xt).argmax(-1).cpu().numpy()
        pred_svm = meta["svm"].predict(X_eval_s)

        pred_bert = eval_probs["bert"].argmax(axis=1)
        pred_mdeb = eval_probs["mdeberta"].argmax(axis=1)
        pred_qwen = eval_probs["qwen"].argmax(axis=1)
        pred_majority = majority_vote_pred(eval_probs["bert"], eval_probs["mdeberta"], eval_probs["qwen"])
        pred_weighted = weighted_static_pred(eval_probs["bert"], eval_probs["mdeberta"], eval_probs["qwen"])
        pred_route = class_conditional_routing_pred(eval_probs["bert"], eval_probs["mdeberta"], eval_probs["qwen"])

        append_row(rows, split_key, f"Meta_MLP_{METHOD_TAG}", y_true, pred_mlp)
        append_row(rows, split_key, f"Meta_LinearSVC_{METHOD_TAG}", y_true, pred_svm)
        append_row(rows, split_key, f"Meta_BiLSTM_{METHOD_TAG}", y_true, pred_bilstm)
        append_row(rows, split_key, f"Meta_FlatLSTM_{METHOD_TAG}", y_true, pred_flat)
        append_row(rows, split_key, "BERT", y_true, pred_bert)
        append_row(rows, split_key, "mDeBERTa", y_true, pred_mdeb)
        append_row(rows, split_key, "Qwen", y_true, pred_qwen)
        append_row(rows, split_key, "Majority_vote", y_true, pred_majority)
        append_row(rows, split_key, "Weighted_static_hand_weights", y_true, pred_weighted)
        append_row(rows, split_key, "Class_conditional_routing_computed", y_true, pred_route)
        append_reference_row(rows, split_key)

        plot_dir = ARTIFACT_DIR / "figures"
        plot_dir.mkdir(exist_ok=True)
        sk = split_key.replace("::", "_")
        plot_confusion(compute_metrics_dict(y_true, pred_mlp)["cm"], f"MLP CM — {split_key}", plot_dir / f"{sk}_mlp.png")
        plot_confusion(compute_metrics_dict(y_true, pred_bilstm)["cm"], f"BiLSTM CM — {split_key}", plot_dir / f"{sk}_bilstm.png")
        plot_confusion(compute_metrics_dict(y_true, pred_flat)["cm"], f"FlatLSTM CM — {split_key}", plot_dir / f"{sk}_flatlstm.png")

results_df = pd.DataFrame(rows).sort_values(
    ["split", "row_kind", "accuracy"], ascending=[True, True, False], na_position="last"
)
out_csv = ARTIFACT_DIR / "stacking_results_js_margin_var.csv"
results_df.to_csv(out_csv, index=False)
print("Saved:", out_csv)
display(results_df)
"""
)

md("## Styled results table")

code(
    """metric_cols = ["accuracy", "f1_macro", "f1_entailment", "f1_neutral", "f1_contradiction"]
pivot_df = results_df.pivot_table(index=["split", "method", "row_kind"], values=metric_cols, aggfunc="first").reset_index()
styled = (
    pivot_df.style.format({m: "{:.4f}" for m in metric_cols})
    .background_gradient(subset=["accuracy", "f1_macro"], cmap="YlGn")
    .set_properties(**{"text-align": "left"})
    .set_caption("Train-split-only stacking — JS divergence + margin + variance (+ mean)")
)
display(styled)
"""
)

md(
    """## Feature importance analysis

- **MLP first-layer weight norms** (structural proxy).
- **Block permutation importance** on **held-out eval** data (not train): shuffle columns within a semantic block and measure accuracy drop.
- **LinearSVC** coefficient norms.

Block definitions follow the column order produced by `build_X` with `DEFAULT_FLAGS`.
"""
)

code(
    """def _normalize_importance(vals, names):
    vals = np.array(vals, dtype=np.float64)
    s = vals.sum()
    if s <= 0:
        s = 1.0
    return pd.DataFrame({"feature": names, "importance": vals / s}).sort_values("importance", ascending=False)


def _permutation_importance_custom_blocks(Xs, y, predict_fn, block_defs, n_repeats=5, seed=SEED):
    rng = np.random.default_rng(seed)
    base_acc = accuracy_score(y, predict_fn(Xs))
    names, imps = [], []
    for name, cols in block_defs:
        names.append(name)
        drops = []
        for _ in range(n_repeats):
            Xp = Xs.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, cols] = Xp[perm][:, cols]
            drops.append(base_acc - accuracy_score(y, predict_fn(Xp)))
        imps.append(float(np.mean(drops)))
    return pd.DataFrame({"feature": names, "importance": imps}).sort_values("importance", ascending=False)


def importance_blocks_for_flags(flags: FeatureFlags):
    \"\"\"Return (block_name, column_indices) aligned with `build_feature_names(flags)`.\"\"\"
    names = build_feature_names(flags)
    blocks = [("raw_probs", list(range(9)))]
    off = 9
    if flags.use_margin:
        blocks.append(("margin", list(range(off, off + 3))))
        off += 3
    if flags.use_js:
        blocks.append(("js_divergence", list(range(off, off + 3))))
        off += 3
    if flags.use_var:
        blocks.append(("var_per_class", list(range(off, off + 3))))
        off += 3
    if flags.use_mean_prob:
        blocks.append(("mean_per_class", list(range(off, off + 3))))
        off += 3
    return blocks, names


importance_rows = []

for cfg in CONFIGS:
    eval_sp = "test_matched" if cfg == "trglue_mnli" else EVAL_SPLITS[cfg][0]
    eb = eval_feature_bank[cfg][eval_sp]
    X_eval_s = eb["X_eval_s"]
    y_eval = eb["y_true"]

    print("\\n" + "#" * 90)
    print(f"Feature importance: {cfg}  (eval split: {eval_sp}, n={len(y_eval)})")

    meta = trained_meta[cfg]
    flags = meta["feature_flags"]
    block_defs, F_NAMES = importance_blocks_for_flags(flags)

    def predict_mlp(X_in):
        with torch.no_grad():
            return meta["mlp"](torch.tensor(X_in, dtype=torch.float32).to(DEVICE)).argmax(-1).cpu().numpy()

    def predict_bilstm(X_in):
        with torch.no_grad():
            return meta["bilstm"](torch.tensor(X_in, dtype=torch.float32).to(DEVICE)).argmax(-1).cpu().numpy()

    mlp_w = meta["mlp"].net[0].weight.detach().cpu().numpy()
    mlp_scores = np.linalg.norm(mlp_w, axis=0)
    print("\\nMLP first-layer feature norms (normalized):")
    mlp_norm_df = _normalize_importance(mlp_scores, F_NAMES)
    display(mlp_norm_df)

    print("MLP permutation importance (eval data):")
    mlp_perm_df = _permutation_importance_custom_blocks(X_eval_s, y_eval, predict_mlp, block_defs)
    display(mlp_perm_df)

    print("BiLSTM permutation importance (eval data):")
    bilstm_perm_df = _permutation_importance_custom_blocks(X_eval_s, y_eval, predict_bilstm, block_defs)
    display(bilstm_perm_df)

    svm_scores = np.linalg.norm(meta["svm"].coef_, axis=0)
    print("LinearSVC coefficient norms (normalized):")
    svm_norm_df = _normalize_importance(svm_scores, F_NAMES)
    display(svm_norm_df)

    for df, method_name in [
        (mlp_norm_df, "MLP_FirstLayerNorm"),
        (mlp_perm_df, "MLP_PermutationImportance_eval"),
        (bilstm_perm_df, "BiLSTM_PermutationImportance_eval"),
        (svm_norm_df, "LinearSVC_CoefNorm"),
    ]:
        d = df.copy()
        d["config"] = cfg
        d["method"] = method_name
        importance_rows.append(d)

all_importances_df = pd.concat(importance_rows, ignore_index=True)
imp_csv = ARTIFACT_DIR / "feature_importances_js_margin_var.csv"
all_importances_df.to_csv(imp_csv, index=False)
print("\\nSaved:", imp_csv)
"""
)

md(
    """## Ablation studies (one feature group removed at a time)

Primary comparison: **`trglue_mnli::test_matched`** — same split as instructor feedback.

Each row retrains **MLP, BiLSTM, and Flat LSTM** from scratch on the training split with `StandardScaler`, then evaluates on the ablation eval set.
"""
)

code(
    """CFG_ABLATION = "trglue_mnli"
SP_ABLATION = "test_matched"

meta_abl = trained_meta[CFG_ABLATION]
X_train_full = train_feature_bank[CFG_ABLATION]["X_train"]
y_train_abl = train_feature_bank[CFG_ABLATION]["y_train"]
train_probs_abl = train_feature_bank[CFG_ABLATION]["probs"]
eval_probs_abl = eval_feature_bank[CFG_ABLATION][SP_ABLATION]["probs"]
y_true_abl = eval_feature_bank[CFG_ABLATION][SP_ABLATION]["y_true"]


def _baseline_acc(learner_key: str):
    tag = {"MLP": f"Meta_MLP_{METHOD_TAG}", "BiLSTM": f"Meta_BiLSTM_{METHOD_TAG}", "FlatLSTM": f"Meta_FlatLSTM_{METHOD_TAG}"}[learner_key]
    return results_df.loc[(results_df["split"] == f"{CFG_ABLATION}::{SP_ABLATION}") & (results_df["method"] == tag), "accuracy"].values[0]


baseline_mlp = _baseline_acc("MLP")
baseline_bilstm = _baseline_acc("BiLSTM")
baseline_flat_lstm = _baseline_acc("FlatLSTM")
print(f"Baselines ({CFG_ABLATION}::{SP_ABLATION}) — MLP: {baseline_mlp:.4f}  BiLSTM: {baseline_bilstm:.4f}  FlatLSTM: {baseline_flat_lstm:.4f}")

ablation_summary = []


def run_group_ablation(name: str, flags: FeatureFlags):
    \"\"\"Train all three torch meta-learners + report vs full-feature baseline.\"\"\"
    print(f"\\n{'='*70}\\nAblation: {name}  |  flags={flags}")

    X_tr = build_X(train_probs_abl, flags)
    X_ev = build_X(eval_probs_abl, flags)
    in_dim = X_tr.shape[1]
    per_model_dim = 4 if flags.use_margin else 3
    global_dim = in_dim - 3 * per_model_dim

    sc = StandardScaler().fit(X_tr)
    Xs_tr = sc.transform(X_tr).astype(np.float32)
    Xs_ev = sc.transform(X_ev).astype(np.float32)

    mlp = StackingMLP(in_dim=in_dim, n_classes=NUM_LABELS)
    mlp = _train_torch_model(mlp, Xs_tr, y_train_abl, model_name=f"MLP_{name}")
    with torch.no_grad():
        pred_m = mlp(torch.tensor(Xs_ev, dtype=torch.float32).to(DEVICE)).argmax(-1).cpu().numpy()
    acc_m = accuracy_score(y_true_abl, pred_m)
    f1_m = f1_score(y_true_abl, pred_m, average="macro", zero_division=0)
    print(f"  MLP      Acc={acc_m:.4f}  F1={f1_m:.4f}  Δ={acc_m-baseline_mlp:+.4f}")
    ablation_summary.append({"learner": "MLP", "config": name, "accuracy": acc_m, "f1_macro": f1_m, "delta_vs_baseline": acc_m - baseline_mlp})

    bi = StackingBiLSTM(per_model_dim=per_model_dim, global_dim=global_dim)
    bi = _train_torch_model(bi, Xs_tr, y_train_abl, model_name=f"BiLSTM_{name}")
    with torch.no_grad():
        pred_b = bi(torch.tensor(Xs_ev, dtype=torch.float32).to(DEVICE)).argmax(-1).cpu().numpy()
    acc_b = accuracy_score(y_true_abl, pred_b)
    f1_b = f1_score(y_true_abl, pred_b, average="macro", zero_division=0)
    print(f"  BiLSTM   Acc={acc_b:.4f}  F1={f1_b:.4f}  Δ={acc_b-baseline_bilstm:+.4f}")
    ablation_summary.append({"learner": "BiLSTM", "config": name, "accuracy": acc_b, "f1_macro": f1_b, "delta_vs_baseline": acc_b - baseline_bilstm})

    fl = StackingLSTM_flat(in_dim=in_dim)
    fl = _train_torch_model(fl, Xs_tr, y_train_abl, model_name=f"FlatLSTM_{name}")
    with torch.no_grad():
        pred_f = fl(torch.tensor(Xs_ev, dtype=torch.float32).to(DEVICE)).argmax(-1).cpu().numpy()
    acc_f = accuracy_score(y_true_abl, pred_f)
    f1_f = f1_score(y_true_abl, pred_f, average="macro", zero_division=0)
    print(f"  FlatLSTM Acc={acc_f:.4f}  F1={f1_f:.4f}  Δ={acc_f-baseline_flat_lstm:+.4f}")
    ablation_summary.append({"learner": "FlatLSTM", "config": name, "accuracy": acc_f, "f1_macro": f1_f, "delta_vs_baseline": acc_f - baseline_flat_lstm})


# Full model already in results_df; optional explicit re-run for sanity:
# run_group_ablation("Full_repeat", FeatureFlags())

run_group_ablation("No_JS", FeatureFlags(use_js=False))
run_group_ablation("No_Margin", FeatureFlags(use_margin=False))
run_group_ablation("No_Variance", FeatureFlags(use_var=False))
run_group_ablation("No_MeanProb", FeatureFlags(use_mean_prob=False))
run_group_ablation("No_JS_No_Margin", FeatureFlags(use_js=False, use_margin=False))
run_group_ablation("RawProbs_only", FeatureFlags(use_margin=False, use_js=False, use_var=False, use_mean_prob=False))

print("\\n" + "=" * 70)
print("Per-model normalization ablation (full DEFAULT_FLAGS layout)")


def per_model_norm_groups(flags: FeatureFlags):
    \"\"\"Return list of (group_name, column_indices).\"\"\"
    names = build_feature_names(flags)
    groups = []
    i = 0
    for m in MODEL_ORDER:
        groups.append((f"{m}_probs", [i, i + 1, i + 2]))
        i += 3
    if flags.use_margin:
        for mi, m in enumerate(MODEL_ORDER):
            groups.append((f"{m}_margin", [i + mi]))
        i += 3
    # tail: global stack
    if i < len(names):
        groups.append(("global_tail", list(range(i, len(names)))))
    return groups


def fit_apply_per_model_norm(X_train, X_eval, flags):
    groups = per_model_norm_groups(flags)
    scalers = {}
    X_tr = X_train.copy().astype(np.float32)
    X_ev = X_eval.copy().astype(np.float32)
    for gname, cols in groups:
        sc = StandardScaler().fit(X_tr[:, cols])
        X_tr[:, cols] = sc.transform(X_train[:, cols])
        X_ev[:, cols] = sc.transform(X_eval[:, cols])
        scalers[gname] = sc
    sc_global = StandardScaler().fit(X_tr)
    return sc_global.transform(X_tr), sc_global.transform(X_ev), scalers, sc_global


flags_full = DEFAULT_FLAGS
X_tr_raw = build_X(train_probs_abl, flags_full)
X_ev_raw = build_X(eval_probs_abl, flags_full)
Xs_tr_pm, Xs_ev_pm, _, sc_g = fit_apply_per_model_norm(X_tr_raw.copy(), X_ev_raw.copy(), flags_full)

in_dim = Xs_tr_pm.shape[1]
mlp_pm = StackingMLP(in_dim=in_dim)
mlp_pm = _train_torch_model(mlp_pm, Xs_tr_pm.astype(np.float32), y_train_abl, model_name="MLP_per_model_norm")
with torch.no_grad():
    pred_pm = mlp_pm(torch.tensor(Xs_ev_pm, dtype=torch.float32).to(DEVICE)).argmax(-1).cpu().numpy()
print(f"MLP per-model norm — Acc={accuracy_score(y_true_abl, pred_pm):.4f}")

per_model_dim = 4
global_dim = in_dim - 3 * per_model_dim
bi_pm = StackingBiLSTM(per_model_dim=per_model_dim, global_dim=global_dim)
bi_pm = _train_torch_model(bi_pm, Xs_tr_pm.astype(np.float32), y_train_abl, model_name="BiLSTM_per_model_norm")
with torch.no_grad():
    pred_bpm = bi_pm(torch.tensor(Xs_ev_pm, dtype=torch.float32).to(DEVICE)).argmax(-1).cpu().numpy()
print(f"BiLSTM per-model norm — Acc={accuracy_score(y_true_abl, pred_bpm):.4f}")

ablation_summary.append(
    {
        "learner": "MLP",
        "config": "Per_model_norm",
        "accuracy": accuracy_score(y_true_abl, pred_pm),
        "f1_macro": f1_score(y_true_abl, pred_pm, average="macro", zero_division=0),
        "delta_vs_baseline": accuracy_score(y_true_abl, pred_pm) - baseline_mlp,
    }
)
ablation_summary.append(
    {
        "learner": "BiLSTM",
        "config": "Per_model_norm",
        "accuracy": accuracy_score(y_true_abl, pred_bpm),
        "f1_macro": f1_score(y_true_abl, pred_bpm, average="macro", zero_division=0),
        "delta_vs_baseline": accuracy_score(y_true_abl, pred_bpm) - baseline_bilstm,
    }
)

abl_df = pd.DataFrame(ablation_summary)
abl_df["baseline_acc"] = abl_df["learner"].map(
    {"MLP": baseline_mlp, "BiLSTM": baseline_bilstm, "FlatLSTM": baseline_flat_lstm}
)
abl_df = abl_df.sort_values(["learner", "accuracy"], ascending=[True, False])
display(
    abl_df.style.format(
        {"accuracy": "{:.4f}", "f1_macro": "{:.4f}", "delta_vs_baseline": "{:+.4f}", "baseline_acc": "{:.4f}"}
    )
    .background_gradient(subset=["accuracy"], cmap="YlGn")
    .set_caption(f"Feature-group ablations (+ per-model norm) on {CFG_ABLATION}::{SP_ABLATION}")
)
abl_csv = ARTIFACT_DIR / "ablation_feature_groups_js_margin_var.csv"
abl_df.to_csv(abl_csv, index=False)
print("Saved:", abl_csv)
"""
)

md("## Save trained meta-learners")

code(
    """import shutil

models_dir = ARTIFACT_DIR / "trained_models"
models_dir.mkdir(exist_ok=True, parents=True)

for cfg, meta in trained_meta.items():
    joblib.dump(meta["scaler"], models_dir / f"{cfg}_scaler.joblib")
    joblib.dump(meta["feature_flags"], models_dir / f"{cfg}_feature_flags.joblib")
    torch.save(meta["mlp"].state_dict(), models_dir / f"{cfg}_mlp.pt")
    joblib.dump(meta["svm"], models_dir / f"{cfg}_svm.joblib")
    torch.save(meta["bilstm"].state_dict(), models_dir / f"{cfg}_bilstm.pt")
    torch.save(meta["flat_lstm"].state_dict(), models_dir / f"{cfg}_flat_lstm.pt")
    print(f"Saved meta-learners for {cfg}")

shutil.make_archive(str(ARTIFACT_DIR) + "_zipbundle", "zip", ARTIFACT_DIR)
print("Zipped:", str(ARTIFACT_DIR) + "_zipbundle.zip")
"""
)

md(
    """## Design note: grouped BiLSTM vs flat LSTM

- **Grouped BiLSTM** reads **3 timesteps** (one per base model). Each timestep uses that model’s **3 softmax probabilities** and optionally its **margin**, so the recurrent state can relate evidence and confidence **within** the same model in one step.

- **Flat LSTM** walks the **flattened feature vector** one scalar at a time. It can still learn, but long-range interactions (e.g. a JS term far from the relevant probability block) require more steps; it is kept here as an explicit **sanity / comparison** baseline, as in the original notebook.
"""
)

out_path = Path("/Users/zeynep_yilmaz/Desktop/Turkish_NLI/src/ensemble_js_margin_variance.ipynb")
out_path.write_text(json.dumps(NB, indent=1))
print("Wrote", out_path)
