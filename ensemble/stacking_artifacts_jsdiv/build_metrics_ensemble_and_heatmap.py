#!/usr/bin/env python3
"""
Build dashboard-compatible metrics_ensemble.csv from stacking_results_jsdiv.csv
and save accuracy / F1 heatmaps (separate figures).

When merging into the project-wide `dashboard/data/metrics_ensemble.csv`, append
only **Meta_***, **Majority_vote**, and **Weighted_static** rows (tags
`stacking-meta-jsdiv` / `preset-ensemble-jsdiv`) so base-model rows are not
duplicated.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "stacking_results_jsdiv.csv"
OUT_CSV = HERE / "metrics_ensemble.csv"
OUT_HEAT_ACC = HERE / "heatmap_jsdiv_accuracy.png"
OUT_HEAT_F1 = HERE / "heatmap_jsdiv_f1_macro.png"

# Align with dashboard/data/metrics_ensemble.csv schema
WEIGHT_COLS = ["w_bert", "w_mdeberta", "w_gemma", "w_qwen"]

MODEL_MAP = {
    "BERT": ("bert-base-turkish-cased-allnli_tr", (1.0, 0.0, 0.0, 0.0)),
    "mDeBERTa": ("mDeBERTa-v3-base-mnli-xnli", (0.0, 1.0, 0.0, 0.0)),
    "Qwen": ("Qwen2-7B-Instruct", (0.0, 0.0, 0.0, 1.0)),
}


def method_to_tag(method: str) -> str:
    if method.startswith("Meta_"):
        return "stacking-meta-jsdiv"
    if method in ("Majority_vote", "Weighted_static"):
        return "preset-ensemble-jsdiv"
    if method in MODEL_MAP:
        return "individual-model"
    return "other-jsdiv"


def method_to_model_id(method: str, split: str) -> str:
    if method in MODEL_MAP:
        return MODEL_MAP[method][0]
    suffix = split.replace("::", "__")
    return f"{method}_{suffix}"


def method_to_weights(method: str) -> tuple[float, float, float, float]:
    if method in MODEL_MAP:
        return MODEL_MAP[method][1]
    return (float("nan"),) * 4


def main() -> None:
    df = pd.read_csv(RESULTS)
    df = df[df["row_kind"].eq("computed")].copy()

    rows = []
    for _, r in df.iterrows():
        split = r["split"]
        method = r["method"]
        wb, wm, wg, wq = method_to_weights(method)
        rows.append(
            {
                "model_id": method_to_model_id(method, split),
                "tag": method_to_tag(method),
                "w_bert": wb,
                "w_mdeberta": wm,
                "w_gemma": wg,
                "w_qwen": wq,
                "accuracy": r["accuracy"],
                "f1_macro": r["f1_macro"],
                "f1_entailment": r["f1_entailment"],
                "f1_neutral": r["f1_neutral"],
                "f1_contradiction": r["f1_contradiction"],
                "data_source": split,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print("Wrote", OUT_CSV.resolve(), "rows:", len(out))

    # --- Heatmaps (split × method) -----------------------------------------
    pivot_acc = df.pivot(index="split", columns="method", values="accuracy")
    pivot_f1 = df.pivot(index="split", columns="method", values="f1_macro")

    split_order = [
        "multinli_tr_1_1::validation_matched",
        "multinli_tr_1_1::validation_mismatched",
        "snli_tr_1_1::test",
        "trglue_mnli::test_matched",
        "trglue_mnli::test_mismatched",
    ]
    pivot_acc = pivot_acc.reindex([s for s in split_order if s in pivot_acc.index])
    pivot_f1 = pivot_f1.reindex([s for s in split_order if s in pivot_f1.index])

    def short_split(s: str) -> str:
        return (
            s.replace("multinli_tr_1_1::", "MNLI ")
            .replace("snli_tr_1_1::", "SNLI ")
            .replace("trglue_mnli::", "TrGLUE ")
            .replace("_", " ")
        )

    pivot_acc.index = [short_split(i) for i in pivot_acc.index]
    pivot_f1.index = [short_split(i) for i in pivot_f1.index]

    plt.rcParams.update({"figure.figsize": (14, 5), "font.size": 9})
    for pivot, title, path, vmin in [
        (pivot_acc, "JS-div ensemble run — accuracy", OUT_HEAT_ACC, 0.6),
        (pivot_f1, "JS-div ensemble run — F1 macro", OUT_HEAT_F1, 0.6),
    ]:
        fig, ax = plt.subplots(figsize=(max(12, pivot.shape[1] * 0.9), 4.2))
        sns.heatmap(
            pivot.astype(float),
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            vmin=vmin,
            vmax=1.0,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": title.split("—")[-1].strip()},
        )
        ax.set_title(title)
        ax.set_xlabel("Method")
        ax.set_ylabel("Eval split")
        plt.xticks(rotation=35, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print("Wrote", path.resolve())


if __name__ == "__main__":
    main()
