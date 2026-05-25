# Explainable AI Report (16-Feature Stacking Meta-Learners)

Folder analyzed: `xai_plots_archive_16feat`

## Executive Summary

- The strongest contradiction signal in MLP SHAP is `qwen_p2` (mean |SHAP| = 0.0859).
- On average across classes, the most influential model block is `qwen` (relative contribution = 0.4203).
- Global class-wise SHAP summaries (MLP/BiLSTM/LinearSVC) and local force/waterfall/LIME examples are available and suitable for appendix figures.

## Contradiction-Class Feature Importance (MLP SHAP)

```
     feature  mean_abs_shap
     qwen_p2       0.085895
     qwen_p1       0.067889
 mdeberta_p2       0.067181
     bert_p1       0.056732
     bert_p2       0.038202
 mdeberta_p1       0.018894
 mdeberta_p0       0.018848
entropy_bert       0.012374
```

## Model-Block Importance by Class (MLP SHAP)

```
        class    block  mean_abs_shap  relative
contradiction     qwen       0.170353  0.409264
   entailment mdeberta       0.153252  0.402549
      neutral     qwen       0.209456  0.455710
```

## Interpretation Notes

- If contradiction top features are mostly `*_p2` probabilities from specific base models, the meta-learner is relying on direct contradiction confidence rather than uncertainty cues.
- If entropy features (`entropy_*`) appear in top ranks, uncertainty is contributing meaningfully to final decisions.
- Dominance of a single block can improve peak performance but may reduce robustness if that base model fails on domain shift.
- Compare local explanations (`force_*`, `waterfall_*`, `lime_*`) between correct and error contradiction cases to identify systematic failure patterns.

## Figure Inventory (selected)

### Global SHAP figures (examples)

- `mlp_bar_contradiction.png`
- `mlp_bar_entailment.png`
- `mlp_bar_neutral.png`
- `mlp_block_importance_contradiction.png`
- `mlp_block_importance_entailment.png`
- `mlp_block_importance_neutral.png`
- `mlp_summary_contradiction.png`
- `mlp_summary_entailment.png`
- `mlp_summary_neutral.png`

### Local explanation figures (examples)

- `force_example_contradiction_correct.png`
- `force_example_contradiction_error.png`
- `force_example_entailment_correct.png`
- `force_example_generic_error.png`
- `lime_mlp_contradiction_correct.png`
- `lime_mlp_contradiction_error.png`
- `lime_mlp_entailment_correct.png`
- `lime_mlp_generic_error.png`
- `waterfall_example_contradiction_correct.png`
- `waterfall_example_contradiction_error.png`
- `waterfall_example_entailment_correct.png`
- `waterfall_example_generic_error.png`