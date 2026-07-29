#!/usr/bin/env python3
"""
scripts/prune_features.py
--------------------------
Feature Importance Analyzer & Noise Pruning Engine for Onyx ML Gatekeeper.

1. Inspects data/onyx_ml_gatekeeper.json across all decision trees.
2. Calculates relative feature importance (split frequency & node depth weight).
3. Ranks features from highest to lowest impact.
4. Drops noise features (< 1.0% contribution) and triggers model re-training for max precision.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_JSON = os.path.join(DATA_DIR, "onyx_ml_gatekeeper.json")
FEATURES_OUTPUT = os.path.join(DATA_DIR, "onyx_ml_features.json")

def traverse_tree(node, feature_counts, depth=0):
    """Recursively traverses a decision tree node to accumulate feature split frequency."""
    if node.get('is_leaf', False):
        return
        
    feat_idx = node['feat_idx']
    # Weight splits closer to the root slightly higher (depth decay)
    weight = 1.0 / (1.0 + 0.2 * depth)
    feature_counts[feat_idx] = feature_counts.get(feat_idx, 0.0) + weight
    
    if 'left' in node:
        traverse_tree(node['left'], feature_counts, depth + 1)
    if 'right' in node:
        traverse_tree(node['right'], feature_counts, depth + 1)

def main():
    print("=" * 75)
    print("  ONYX QUANTITATIVE DESK — FEATURE IMPORTANCE PRUNING ENGINE")
    print("=" * 75)
    
    if not os.path.exists(MODEL_JSON):
        print(f"[ERROR] Trained model JSON not found at {MODEL_JSON}. Run scripts/train_ml_model.py first.")
        sys.exit(1)
        
    with open(MODEL_JSON, "r") as f:
        model_data = json.load(f)
        
    feature_cols = model_data['feature_cols']
    trees = model_data['trees']
    
    feature_counts = {}
    for tree in trees:
        traverse_tree(tree, feature_counts)
        
    total_score = sum(feature_counts.values()) + 1e-8
    
    # Calculate percentage importances
    rankings = []
    for idx, col in enumerate(feature_cols):
        score = feature_counts.get(idx, 0.0)
        pct = (score / total_score) * 100.0
        rankings.append((col, pct, score))
        
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    print(f"[1/3] Analyzed {len(trees)} Decision Trees across {len(feature_cols)} Predictor Indicators.\n")
    print(f"{'Rank':<5} | {'Feature Indicator Name':<30} | {'Importance %':<15} | {'Status'}")
    print("-" * 75)
    
    kept_features = []
    pruned_features = []
    
    for i, (col, pct, score) in enumerate(rankings, 1):
        if pct >= 1.0:
            status = "✅ KEEP (High Impact)"
            kept_features.append(col)
        else:
            status = "❌ PRUNE (Noise < 1.0%)"
            pruned_features.append(col)
            
        print(f"#{i:<4} | {col:<30} | {pct:>6.2f}%         | {status}")
        
    print("-" * 75)
    print(f"\n[SUMMARY] Total Features: {len(feature_cols)} | Kept: {len(kept_features)} | Pruned Noise: {len(pruned_features)}")
    
    if pruned_features:
        print("\n[NOISE DISCARDED]: " + ", ".join(pruned_features))
    else:
        print("\nAll features are contributing > 1.0% to predictions!")

    # Retrain model with pruned features
    print("\n[2/3] Retraining ML Gatekeeper using only the top potent features...")
    
    # Save pruned features list
    with open(FEATURES_OUTPUT, "w") as f:
        json.dump({
            "features": kept_features,
            "pruned_features": pruned_features,
            "prune_threshold_pct": 1.0,
            "updated_at": pd.Timestamp.now().isoformat()
        }, f, indent=2)
        
    print(f"  --> Saved pruned feature schema to {FEATURES_OUTPUT}")
    
    # Trigger model retrain script with pruned features
    print("[3/3] Launching streamlined model training...")
    os.system(f"python3 {os.path.join(PROJECT_ROOT, 'scripts', 'train_ml_model.py')}")

if __name__ == "__main__":
    main()
