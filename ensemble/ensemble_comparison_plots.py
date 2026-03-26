from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2}
LABEL_NAMES = ["Entailment", "Neutral", "Contradiction"]
MODEL_ORDER = ["bert", "mdeberta", "gemma", "qwen"]
MODEL_PRETTY = {
    "bert": "BERT",
    "mdeberta": "mDeBERTa",
    "gemma": "Gemma",
    "qwen": "Qwen",
}
PRED_KEY_FULL = {
    "bert": "bert_allnli_tr",
    "mdeberta": "mdeberta",
    "gemma": "gemma",
    "qwen": "qwen",
}
PRED_COL_HARD = {
    "bert": "bert_pred",
    "mdeberta": "mdeberta_pred",
    "gemma": "gemma_pred",
    "qwen": "qwen_pred",
}


OPTIMISED_W = {"bert": 0.25, "mdeberta": 0.12, "gemma": 0.26, "qwen": 0.37}

PROFILES = {
    "V1 (user spec)": {
        "router": {"bert": 0.0, "mdeberta": 0.0, "gemma": 0.50, "qwen": 0.50},
        "classes": {
            0: {"bert": 0.0, "mdeberta": 0.05, "gemma": 0.25, "qwen": 0.70},
            1: {"bert": 0.0, "mdeberta": 0.10, "gemma": 0.70, "qwen": 0.20},
            2: {"bert": 0.0, "mdeberta": 0.05, "gemma": 0.25, "qwen": 0.70},
        },
    },
    "V2 (Gemma trust)": {
        "router": {"bert": 0.0, "mdeberta": 0.0, "gemma": 0.60, "qwen": 0.40},
        "classes": {
            0: {"bert": 0.0, "mdeberta": 0.0, "gemma": 0.40, "qwen": 0.60},
            1: {"bert": 0.0, "mdeberta": 0.0, "gemma": 0.80, "qwen": 0.20},
            2: {"bert": 0.0, "mdeberta": 0.0, "gemma": 0.40, "qwen": 0.60},
        },
    },
    "V3 (mDeBERTa router)": {
        "router": {"bert": 0.0, "mdeberta": 0.15, "gemma": 0.45, "qwen": 0.40},
        "classes": {
            0: {"bert": 0.0, "mdeberta": 0.10, "gemma": 0.30, "qwen": 0.60},
            1: {"bert": 0.0, "mdeberta": 0.10, "gemma": 0.65, "qwen": 0.25},
            2: {"bert": 0.0, "mdeberta": 0.10, "gemma": 0.30, "qwen": 0.60},
        },
    },
    "V4 (aggressive Gemma neutral)": {
        "router": {"bert": 0.0, "mdeberta": 0.0, "gemma": 0.50, "qwen": 0.50},
        "classes": {
            0: {"bert": 0.0, "mdeberta": 0.0, "gemma": 0.35, "qwen": 0.65},
            1: {"bert": 0.0, "mdeberta": 0.0, "gemma": 0.90, "qwen": 0.10},
            2: {"bert": 0.0, "mdeberta": 0.0, "gemma": 0.35, "qwen": 0.65},
        },
    },
    "V5 (Gemma only on neutral, Qwen only on rest)": {
        "router": {"bert": 0.0, "mdeberta": 0.0, "gemma": 0.50, "qwen": 0.50},
        "classes": {
            0: {"bert": 0.0, "mdeberta": 0.0, "gemma": 0.0, "qwen": 1.0},
            1: {"bert": 0.0, "mdeberta": 0.0, "gemma": 1.0, "qwen": 0.0},
            2: {"bert": 0.0, "mdeberta": 0.0, "gemma": 0.0, "qwen": 1.0},
        },
    },
    "V6 (3-way with mDeBERTa boost on con)": {
        "router": {"bert": 0.0, "mdeberta": 0.10, "gemma": 0.45, "qwen": 0.45},
        "classes": {
            0: {"bert": 0.0, "mdeberta": 0.10, "gemma": 0.30, "qwen": 0.60},
            1: {"bert": 0.0, "mdeberta": 0.05, "gemma": 0.75, "qwen": 0.20},
            2: {"bert": 0.0, "mdeberta": 0.25, "gemma": 0.35, "qwen": 0.40},
        },
    },
}


def build_one_hot(preds_array: np.ndarray, n_classes: int = 3) -> np.ndarray:
    oh = np.zeros((len(preds_array), n_classes), dtype=float)
    for c in range(n_classes):
        oh[preds_array == c, c] = 1.0
    return oh


def fast_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    f1s = np.empty(n_classes, dtype=float)
    for c in range(n_classes):
        pred_c = y_pred == c
        true_c = y_true == c
        tp = int((pred_c & true_c).sum())
        denom = 2 * tp + int((pred_c & ~true_c).sum()) + int((~pred_c & true_c).sum())
        f1s[c] = (2 * tp / denom) if denom > 0 else 0.0
    return f1s


def eval_weights(weights: dict[str, float], oh: np.ndarray, gold: np.ndarray) -> dict[str, float]:
    w = np.array([weights[m] for m in MODEL_ORDER], dtype=float)
    scores = np.einsum("m,mnc->nc", w, oh)
    preds = np.argmax(scores, axis=1)
    f1s = fast_f1(gold, preds)
    return {
        "accuracy": float((preds == gold).mean()),
        "f1_macro": float(f1s.mean()),
        "f1_entailment": float(f1s[0]),
        "f1_neutral": float(f1s[1]),
        "f1_contradiction": float(f1s[2]),
    }


def class_conditional_ensemble(
    oh: np.ndarray,
    gold: np.ndarray,
    router_w: dict[str, float],
    class_weights: dict[int, dict[str, float]],
) -> dict[str, float]:
    n = oh.shape[1]
    w_router = np.array([router_w[m] for m in MODEL_ORDER], dtype=float)
    scores_router = np.einsum("m,mnc->nc", w_router, oh)
    prelim_class = np.argmax(scores_router, axis=1)

    final_preds = np.empty(n, dtype=int)
    for c in range(3):
        mask = prelim_class == c
        if not mask.any():
            continue
        w_c = np.array([class_weights[c][m] for m in MODEL_ORDER], dtype=float)
        scores_c = np.einsum("m,mnc->nc", w_c, oh[:, mask, :])
        final_preds[mask] = np.argmax(scores_c, axis=1)

    f1s = fast_f1(gold, final_preds)
    return {
        "accuracy": float((final_preds == gold).mean()),
        "f1_macro": float(f1s.mean()),
        "f1_entailment": float(f1s[0]),
        "f1_neutral": float(f1s[1]),
        "f1_contradiction": float(f1s[2]),
    }


def to_label_id(v) -> int:
    if isinstance(v, (int, np.integer)):
        return int(v)
    return LABEL_MAP[str(v).strip().lower()]


def load_full_arrays(full_json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(full_json_path.read_text())
    examples = data["per_example_results"]
    gold = np.array([int(ex["gold_label"]) for ex in examples], dtype=int)
    oh = np.stack(
        [
            build_one_hot(
                np.array([int(ex["predictions"][PRED_KEY_FULL[m]]) for ex in examples], dtype=int)
            )
            for m in MODEL_ORDER
        ],
        axis=0,
    )
    return oh, gold


def load_hard_arrays(hard_csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(hard_csv_path)
    gold = df["true_label"].map(to_label_id).to_numpy(dtype=int)
    oh = np.stack(
        [
            build_one_hot(df[PRED_COL_HARD[m]].map(to_label_id).to_numpy(dtype=int))
            for m in MODEL_ORDER
        ],
        axis=0,
    )
    return oh, gold


def find_best_grid_weights(metrics_ensemble_csv: Path) -> dict[str, float]:
    df = pd.read_csv(metrics_ensemble_csv)
    best_row = df[df["tag"].fillna("").str.contains("global-best-accuracy", regex=False)]
    row = best_row.iloc[0] if not best_row.empty else df.iloc[0]
    return {
        "bert": float(row["w_bert"]),
        "mdeberta": float(row["w_mdeberta"]),
        "gemma": float(row["w_gemma"]),
        "qwen": float(row["w_qwen"]),
    }


def choose_best_class_profile(oh_hard: np.ndarray, gold_hard: np.ndarray) -> tuple[str, dict]:
    best_name = None
    best_metrics = None
    best_acc = -1.0
    for name, profile in PROFILES.items():
        metrics = class_conditional_ensemble(
            oh_hard, gold_hard, profile["router"], profile["classes"]
        )
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_name = name
            best_metrics = metrics
    assert best_name is not None and best_metrics is not None
    return best_name, best_metrics


def make_plots(
    agg_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    model_order = [
        "BERT",
        "mDeBERTa",
        "Gemma",
        "Qwen",
        "Best Grid Ensemble",
        "Best Class-Routed Ensemble",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)
    for ax, subset_name in zip(axes, ["9008", "BERT hard"]):
        sub = agg_df[agg_df["Subset"] == subset_name].copy()
        sub_long = sub.melt(
            id_vars=["Model", "Subset"],
            value_vars=["Accuracy", "F1 Macro"],
            var_name="Metric",
            value_name="Score",
        )
        sns.barplot(
            data=sub_long,
            x="Model",
            y="Score",
            hue="Metric",
            order=model_order,
            palette={"Accuracy": "#2f4b7c", "F1 Macro": "#665191"},
            ax=ax,
        )
        ax.set_title(f"{subset_name}: Accuracy vs F1 Macro", fontweight="bold")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=28)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=10, padding=2)
        ax.legend(loc="upper left")

    plt.suptitle("Single Models vs Best Ensembles", fontsize=20, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(output_dir / "ensemble_overall_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(22, 8), sharey=True)
    class_palette = {"Entailment": "#2ca02c", "Neutral": "#ff7f0e", "Contradiction": "#d62728"}
    for ax, subset_name in zip(axes, ["9008", "BERT hard"]):
        sub = per_class_df[per_class_df["Subset"] == subset_name].copy()
        sns.barplot(
            data=sub,
            x="Model",
            y="F1",
            hue="Class",
            order=model_order,
            hue_order=["Entailment", "Neutral", "Contradiction"],
            palette=class_palette,
            ax=ax,
        )
        ax.set_title(f"{subset_name}: Per-class F1", fontweight="bold")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("")
        ax.set_ylabel("F1")
        ax.tick_params(axis="x", rotation=28)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=9, padding=1)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
            title="Class",
        )

    plt.suptitle(
        "Per-class Comparison (Entailment / Neutral / Contradiction)",
        fontsize=20,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "ensemble_per_class_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "src" / "ensemble" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_candidates = [
        root / "data",
        root / "dashboard" / "data",
    ]

    def pick_file(name: str) -> Path:
        for d in data_candidates:
            p = d / name
            if p.exists():
                return p
        raise FileNotFoundError(f"Could not find {name} in: {data_candidates}")

    full_json = pick_file("trglue_test_matched_four_models_results.json")
    hard_csv = pick_file("error_analysis.csv")
    ens_csv = pick_file("metrics_ensemble.csv")

    oh_full, gold_full = load_full_arrays(full_json)
    oh_hard, gold_hard = load_hard_arrays(hard_csv)

    best_grid_w = find_best_grid_weights(ens_csv)
    best_class_name, _ = choose_best_class_profile(oh_hard, gold_hard)
    best_class_profile = PROFILES[best_class_name]

    configs = {
        "BERT": {"type": "flat", "weights": {"bert": 1.0, "mdeberta": 0.0, "gemma": 0.0, "qwen": 0.0}},
        "mDeBERTa": {"type": "flat", "weights": {"bert": 0.0, "mdeberta": 1.0, "gemma": 0.0, "qwen": 0.0}},
        "Gemma": {"type": "flat", "weights": {"bert": 0.0, "mdeberta": 0.0, "gemma": 1.0, "qwen": 0.0}},
        "Qwen": {"type": "flat", "weights": {"bert": 0.0, "mdeberta": 0.0, "gemma": 0.0, "qwen": 1.0}},
        "Best Grid Ensemble": {"type": "flat", "weights": best_grid_w},
        "Best Class-Routed Ensemble": {
            "type": "routed",
            "router": best_class_profile["router"],
            "classes": best_class_profile["classes"],
        },
    }

    agg_rows = []
    class_rows = []
    for name, cfg in configs.items():
        if cfg["type"] == "flat":
            m_full = eval_weights(cfg["weights"], oh_full, gold_full)
            m_hard = eval_weights(cfg["weights"], oh_hard, gold_hard)
        else:
            m_full = class_conditional_ensemble(oh_full, gold_full, cfg["router"], cfg["classes"])
            m_hard = class_conditional_ensemble(oh_hard, gold_hard, cfg["router"], cfg["classes"])

        agg_rows.append({"Model": name, "Subset": "9008", "Accuracy": m_full["accuracy"], "F1 Macro": m_full["f1_macro"]})
        agg_rows.append({"Model": name, "Subset": "BERT hard", "Accuracy": m_hard["accuracy"], "F1 Macro": m_hard["f1_macro"]})

        for label in LABEL_NAMES:
            key = f"f1_{label.lower()}"
            class_rows.append({"Model": name, "Subset": "9008", "Class": label, "F1": m_full[key]})
            class_rows.append({"Model": name, "Subset": "BERT hard", "Class": label, "F1": m_hard[key]})

    agg_df = pd.DataFrame(agg_rows)
    per_class_df = pd.DataFrame(class_rows)

    agg_df.to_csv(out_dir / "ensemble_overall_comparison_scores.csv", index=False)
    per_class_df.to_csv(out_dir / "ensemble_per_class_scores.csv", index=False)
    make_plots(agg_df, per_class_df, out_dir)

    print(f"Best class-routed profile on hard set: {best_class_name}")
    print(f"Saved outputs to: {out_dir}")
    print("- ensemble_overall_comparison.png")
    print("- ensemble_per_class_comparison.png")
    print("- ensemble_overall_comparison_scores.csv")
    print("- ensemble_per_class_scores.csv")


if __name__ == "__main__":
    main()
