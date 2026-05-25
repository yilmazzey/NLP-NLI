# Comparison vs Base 4 Models (TrGLUE test_matched)

Best base model (among BERT/mDeBERTa/Gemma/Qwen): **Qwen** with accuracy 0.8156, macro-F1 0.8150.
Best ensemble method: **Meta_BiLSTM** with accuracy 0.8511, macro-F1 0.8511.
Accuracy gain vs best base: +0.0355
Macro-F1 gain vs best base: +0.0361

Label-wise F1 gain (best ensemble vs best base):
- entailment: +0.0294
- neutral: +0.0466
- contradiction: +0.0323

Top ensemble methods (16-feature) on this split:
```
                       method_norm  accuracy_16feat  f1_macro_16feat  f1_entailment_16feat  f1_neutral_16feat  f1_contradiction_16feat
                       Meta_BiLSTM         0.851132         0.851101              0.852515           0.845293                 0.855495
                          Meta_MLP         0.850244         0.850178              0.852211           0.842724                 0.855600
                     Meta_FlatLSTM         0.850022         0.849987              0.851412           0.842650                 0.855898
                    Meta_LinearSVC         0.839032         0.838813              0.845290           0.825922                 0.845226
      Weighted_static_hand_weights         0.813832         0.812314              0.822387           0.781622                 0.832933
Class_conditional_routing_computed         0.773424         0.770488              0.823082           0.726908                 0.761474
```

Base 4 comparison (12f/16f + data reference):
```
method_norm              model_id_data  accuracy_12feat  accuracy_16feat  accuracy_data  f1_macro_12feat  f1_macro_16feat  f1_macro_data  f1_entailment_12feat  f1_entailment_16feat  f1_entailment_data  f1_neutral_12feat  f1_neutral_16feat  f1_neutral_data  f1_contradiction_12feat  f1_contradiction_16feat  f1_contradiction_data
       Qwen          Qwen2-7B-Instruct         0.815608         0.815608       0.818717         0.814991         0.814991       0.818052              0.823082              0.823082            0.825973           0.798702           0.798702         0.801596                 0.823190                 0.823190               0.826587
   mDeBERTa mDeBERTa-v3-base-mnli-xnli         0.796514         0.796514       0.796514         0.793475         0.793475       0.793514              0.826087              0.826087            0.825740           0.750851           0.750851         0.751181                 0.803488                 0.803488               0.803622
       BERT    bert-base-turkish-cased         0.659192         0.659192       0.332260         0.639190         0.639190       0.314136              0.721935              0.721935            0.179872           0.491156           0.491156         0.360549                 0.704479                 0.704479               0.401985
      Gemma              gemma-2-2b-it         0.399312         0.399312       0.791963         0.357893         0.357893       0.791804              0.496634              0.496634            0.787986           0.238667           0.238667         0.777860                 0.338378                 0.338378               0.809564
```