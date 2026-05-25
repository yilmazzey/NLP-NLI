import json
from pathlib import Path

# Load the individual results
with open("results/metrics.json", "r") as f:
    individual = json.load(f)

# Dataset sizes (approximate)
sizes = {
    "snli_tr_1_1": {"test": 10000},
    "multinli_tr_1_1": {"validation_matched": 10000, "validation_mismatched": 10000},
    "trglue_mnli": {"test_matched": 3000, "test_mismatched": 3000}
}

# Calculate weighted averages for each combination
combinations = {
    "snli+mnli": [
        ("snli_tr_1_1", "test"),
        ("multinli_tr_1_1", "validation_matched"),
        ("multinli_tr_1_1", "validation_mismatched")
    ],
    "snli+trglue": [
        ("snli_tr_1_1", "test"),
        ("trglue_mnli", "test_matched"),
        ("trglue_mnli", "test_mismatched")
    ],
    "mnli+trglue": [
        ("multinli_tr_1_1", "validation_matched"),
        ("multinli_tr_1_1", "validation_mismatched"),
        ("trglue_mnli", "test_matched"),
        ("trglue_mnli", "test_mismatched")
    ],
    "snli+mnli+trglue": [
        ("snli_tr_1_1", "test"),
        ("multinli_tr_1_1", "validation_matched"),
        ("multinli_tr_1_1", "validation_mismatched"),
        ("trglue_mnli", "test_matched"),
        ("trglue_mnli", "test_mismatched")
    ]
}

weighted_averages = {}

for combo_name, splits in combinations.items():
    total_size = sum(sizes[cfg][split] for cfg, split in splits)
    
    acc_sum = 0
    f1_macro_sum = 0
    f1_ent_sum = 0
    f1_neu_sum = 0
    f1_con_sum = 0
    
    for cfg, split in splits:
        weight = sizes[cfg][split] / total_size
        metrics = individual[cfg][split]
        
        acc_sum += metrics["accuracy"] * weight
        f1_macro_sum += metrics["f1_macro"] * weight
        f1_ent_sum += metrics["f1_per_class"]["entailment"] * weight
        f1_neu_sum += metrics["f1_per_class"]["neutral"] * weight
        f1_con_sum += metrics["f1_per_class"]["contradiction"] * weight
    
    weighted_averages[combo_name] = {
        "accuracy": acc_sum,
        "f1_macro": f1_macro_sum,
        "f1_per_class": {
            "entailment": f1_ent_sum,
            "neutral": f1_neu_sum,
            "contradiction": f1_con_sum
        }
    }

# Save weighted averages
with open("results/metrics_weighted_average.json", "w") as f:
    json.dump(weighted_averages, f, indent=2)

# Print comparison
print("=" * 80)
print("COMPARISON: Weighted Average (Individual) vs Combined Dataset Results")
print("=" * 80)

with open("results_combined/metrics.json", "r") as f:
    combined = json.load(f)

for combo_name in weighted_averages.keys():
    print(f"\n{combo_name.upper()}")
    print("-" * 80)
    
    wa = weighted_averages[combo_name]
    cb = combined[combo_name]
    
    print(f"  Accuracy:")
    print(f"    Weighted Avg: {wa['accuracy']:.4f}")
    print(f"    Combined:     {cb['accuracy']:.4f}")
    print(f"    Difference:   {(cb['accuracy'] - wa['accuracy']):.4f}")
    
    print(f"\n  F1 Macro:")
    print(f"    Weighted Avg: {wa['f1_macro']:.4f}")
    print(f"    Combined:     {cb['f1_macro']:.4f}")
    print(f"    Difference:   {(cb['f1_macro'] - wa['f1_macro']):.4f}")
    
    print(f"\n  F1 per class:")
    for label in ["entailment", "neutral", "contradiction"]:
        diff = cb['f1_per_class'][label] - wa['f1_per_class'][label]
        print(f"    {label:13s}: WA={wa['f1_per_class'][label]:.4f}, Combined={cb['f1_per_class'][label]:.4f}, Diff={diff:+.4f}")

print("\n" + "=" * 80)
print("Saved weighted averages to: results/metrics_weighted_average.json")
print("=" * 80)
