# Weekend Performance Report: 12-Feature vs 16-Feature Ensembles

## Executive Summary

- Best overall method on **16 features (mean accuracy)**: `Meta_BiLSTM` (0.8408).
- Best overall method on **16 features (mean macro-F1)**: `Meta_BiLSTM` (0.8403).
- Best overall method on **12 features (mean accuracy)**: `Meta_FlatLSTM` (0.8394).
- Best overall method on **12 features (mean macro-F1)**: `Meta_FlatLSTM` (0.8389).
- Largest average accuracy gain from 16 features: `Meta_BiLSTM` (+0.0021).
- Largest average accuracy drop from 16 features: `Meta_LinearSVC` (-0.0002).

## Most Successful Meta-Learners (averaged across splits)

```
   method_norm  accuracy_12feat  accuracy_16feat  f1_macro_12feat  f1_macro_16feat  accuracy_delta_16_minus_12  f1_macro_delta_16_minus_12
   Meta_BiLSTM         0.838737         0.840800         0.838230         0.840302                    0.002064                    0.002072
      Meta_MLP         0.838223         0.840088         0.837732         0.839583                    0.001865                    0.001851
 Meta_FlatLSTM         0.839447         0.839473         0.838914         0.838990                    0.000027                    0.000076
Meta_LinearSVC         0.834532         0.834301         0.833690         0.833535                   -0.000231                   -0.000155
```

## Best Meta-Learner per Split (16-feature setting)

```
                                 split method_norm  accuracy_16feat  f1_macro_16feat
   multinli_tr_1_1::validation_matched    Meta_MLP         0.801305         0.800693
multinli_tr_1_1::validation_mismatched Meta_BiLSTM         0.813028         0.812074
                     snli_tr_1_1::test Meta_BiLSTM         0.835607         0.835251
             trglue_mnli::test_matched Meta_BiLSTM         0.851132         0.851101
          trglue_mnli::test_mismatched Meta_BiLSTM         0.903439         0.902888
```

## Interpretation

- 16-feature setup tends to help methods that can exploit entropy signals and richer interactions (especially sequence/meta neural learners).
- Some methods show small negative deltas, suggesting they do not consistently benefit from extra entropy features or may need hyperparameter retuning.
- For presentation: emphasize Meta_BiLSTM vs Meta_MLP trends and include split-level stability from matched/mismatched sets.
