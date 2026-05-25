# Comparison Against Base 4 Models (from data/four_models_aggregates.json)

## Base models (TrGLUE test_matched)

```
           source                       name  accuracy  f1_macro  f1_entailment  f1_neutral  f1_contradiction
base_4models_data         Qwen 2 7B Instruct    0.8186  0.818052       0.825973    0.801596          0.826587
base_4models_data             Gemma 3 27B IT    0.8134  0.812177       0.816695    0.763894          0.855941
base_4models_data mDeBERTa-v3-base-mnli-xnli    0.7965  0.793514       0.825740    0.751181          0.803622
base_4models_data           BERT (allnli_tr)    0.7501  0.743400       0.830000    0.650000          0.740000
```

Best base model: **Qwen 2 7B Instruct** | acc=0.8186, macro-F1=0.8181

## Best ensemble result in current artifacts

**Meta_BiLSTM** | 16f acc=0.8511, macro-F1=0.8511
- Accuracy gain vs best base: +0.0325
- Macro-F1 gain vs best base: +0.0330
- Entailment F1 gain: +0.0265
- Neutral F1 gain: +0.0437
- Contradiction F1 gain: +0.0289

## Top ensemble methods vs best base (16-feature)

```
                              name  accuracy_16feat  f1_macro_16feat  accuracy_gain_vs_best_base_16feat  f1_macro_gain_vs_best_base_16feat
                       Meta_BiLSTM         0.851132         0.851101                           0.032532                           0.033049
                          Meta_MLP         0.850244         0.850178                           0.031644                           0.032126
                     Meta_FlatLSTM         0.850022         0.849987                           0.031422                           0.031935
                    Meta_LinearSVC         0.839032         0.838813                           0.020432                           0.020761
                              Qwen         0.815608         0.814991                          -0.002992                          -0.003060
      Weighted_static_hand_weights         0.813832         0.812314                          -0.004768                          -0.005738
                          mDeBERTa         0.796514         0.793475                          -0.022086                          -0.024577
Class_conditional_routing_computed         0.773424         0.770488                          -0.045176                          -0.047564
```