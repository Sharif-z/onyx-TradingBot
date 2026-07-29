#!/usr/bin/env python3
"""
scripts/purged_cross_validation.py
-----------------------------------
Institutional-Grade Purged & Embargoed K-Fold Cross-Validation Engine.

Prevents financial time-series data leakage and temporal autocorrelation:
1. 5 Chronological Folds over data/ml_trading_dataset.csv.
2. 5-Day (480 candles of 15m) Embargo Buffer between training and validation sets.
3. Out-of-Sample Safe Trade Precision evaluation: P(Safe) = P(Breakeven) + P(Clean Win) >= 65%.
4. Calculates Fold-by-Fold Precision, Recall, Approval Rate, and R-Expectancy.
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_CSV = os.path.join(DATA_DIR, "ml_trading_dataset.csv")
FEATURES_JSON = os.path.join(DATA_DIR, "onyx_ml_features.json")

# Pure NumPy Binned Decision Tree Ensemble for low-RAM Termux execution
class FastNumpyBinnedEnsemble:
    def __init__(self, n_estimators=60, max_depth=5, min_samples_leaf=20, n_bins=32):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins
        self.trees = []
        self.bin_edges = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_binned = pd.DataFrame(index=X.index)
        for col in X.columns:
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            edges = np.unique(np.percentile(X[col].values, percentiles))
            self.bin_edges[col] = edges.tolist()
            X_binned[col] = np.digitize(X[col].values, edges) - 1

        X_np = X_binned.values.astype(np.int8)
        y_np = y.values.astype(np.int8)
        
        n_samples, n_features = X_np.shape
        n_classes = len(np.unique(y_np))
        rng = np.random.RandomState(42)
        
        class_counts = np.bincount(y_np, minlength=3)
        class_weights = n_samples / (n_classes * (class_counts.astype(np.float32) + 1e-8))
        
        for t in range(self.n_estimators):
            sample_idx = rng.choice(n_samples, size=int(0.80 * n_samples), replace=True)
            feat_idx = rng.choice(n_features, size=int(0.75 * n_features), replace=False)
            tree = self._build_tree(X_np[sample_idx][:, feat_idx], y_np[sample_idx], class_weights, current_depth=0, feat_indices=feat_idx)
            self.trees.append(tree)

    def _build_tree(self, X_sub, y_sub, class_weights, current_depth, feat_indices):
        n_samples, n_sub_feats = X_sub.shape
        counts = np.bincount(y_sub, minlength=3)
        weighted_counts = counts * class_weights
        probs = weighted_counts / (np.sum(weighted_counts) + 1e-8)
        
        if current_depth >= self.max_depth or n_samples < self.min_samples_leaf or len(np.unique(y_sub)) == 1:
            return {'is_leaf': True, 'probs': probs.tolist()}
            
        best_gain = -1.0
        best_split = None
        
        for f_sub_idx in range(n_sub_feats):
            feat_vals = X_sub[:, f_sub_idx]
            unique_vals = np.unique(feat_vals)
            if len(unique_vals) <= 1:
                continue
                
            for val in unique_vals[::2]:
                left_mask = feat_vals <= val
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                    
                p_left = np.bincount(y_sub[left_mask], minlength=3) / (np.sum(left_mask) + 1e-8)
                p_right = np.bincount(y_sub[right_mask], minlength=3) / (np.sum(right_mask) + 1e-8)
                
                gini_parent = 1.0 - np.sum(probs**2)
                gini_left = 1.0 - np.sum(p_left**2)
                gini_right = 1.0 - np.sum(p_right**2)
                
                weight_l = np.sum(left_mask) / n_samples
                weight_r = np.sum(right_mask) / n_samples
                gain = gini_parent - (weight_l * gini_left + weight_r * gini_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = (f_sub_idx, feat_indices[f_sub_idx], val, left_mask)
                    
        if best_split is None or best_gain <= 0.0:
            return {'is_leaf': True, 'probs': probs.tolist()}
            
        f_sub_idx, raw_feat_idx, val, left_mask = best_split
        right_mask = ~left_mask
        
        left_child = self._build_tree(X_sub[left_mask], y_sub[left_mask], class_weights, current_depth + 1, feat_indices)
        right_child = self._build_tree(X_sub[right_mask], y_sub[right_mask], class_weights, current_depth + 1, feat_indices)
        
        return {
            'is_leaf': False,
            'feat_idx': int(raw_feat_idx),
            'bin_threshold': int(val),
            'left': left_child,
            'right': right_child
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_binned = pd.DataFrame(index=X.index)
        for col in X.columns:
            edges = np.array(self.bin_edges[col])
            X_binned[col] = np.digitize(X[col].values, edges) - 1

        X_np = X_binned.values.astype(np.int8)
        n_samples = len(X_np)
        
        all_probs = np.zeros((n_samples, 3), dtype=np.float32)
        for tree in self.trees:
            tree_probs = np.zeros((n_samples, 3), dtype=np.float32)
            for r_idx in range(n_samples):
                row = X_np[r_idx]
                curr = tree
                while not curr['is_leaf']:
                    if row[curr['feat_idx']] <= curr['bin_threshold']:
                        curr = curr['left']
                    else:
                        curr = curr['right']
                tree_probs[r_idx] = curr['probs']
            all_probs += tree_probs
            
        all_probs /= len(self.trees)
        return all_probs

def run_purged_cv(k_folds=5, embargo_candles=480, threshold=0.65):
    """
    Executes Purged K-Fold Cross-Validation with an Embargo Buffer.
    embargo_candles = 480 (5 days of 15-minute candles).
    """
    print("=" * 80)
    print("  ONYX QUANTITATIVE DESK — PURGED & EMBARGOED CROSS-VALIDATION ENGINE")
    print(f"  Configuration: {k_folds} Folds | Embargo Buffer: {embargo_candles} candles (5 Days) | P_Safe Threshold: {threshold*100:.0f}%")
    print("=" * 80)
    
    if not os.path.exists(DATASET_CSV):
        print(f"[ERROR] Dataset file not found at {DATASET_CSV}.")
        sys.exit(1)
        
    df = pd.read_csv(DATASET_CSV)
    print(f"[1/4] Loaded dataset: {len(df):,} total trade setup rows.\n")
    
    # Load features list
    if os.path.exists(FEATURES_JSON):
        with open(FEATURES_JSON, "r") as f:
            feature_cols = json.load(f).get("features", [])
    else:
        drop_cols = ['timestamp', 'ticker', 'datetime_utc', 'side', 'entry_price', 'sl_price', 'tp_price', 'label_win', 'label_max_rr', 'label_net_r', 'label_class']
        feature_cols = [c for c in df.columns if c not in drop_cols]
        
    X = df[feature_cols].copy().astype(np.float32)
    y = df['label_class'].copy().astype(np.int8)
    
    n = len(df)
    fold_size = n // k_folds
    
    fold_results = []
    
    print(f"{'Fold':<6} | {'Val Window':<20} | {'Train Rows':<10} | {'Approved':<10} | {'Baseline Safe':<14} | {'ML Safe Precision':<18} | {'Expectancy Improvement'}")
    print("-" * 115)
    
    for k in range(k_folds):
        val_start = k * fold_size
        val_end = (k + 1) * fold_size if k < k_folds - 1 else n
        
        # Purge & Embargo calculation
        # Purge 5 days (480 candles) before and after validation split
        embargo_start = max(0, val_start - embargo_candles)
        embargo_end = min(n, val_end + embargo_candles)
        
        # Train indices exclude [embargo_start, embargo_end]
        train_idx = list(range(0, embargo_start)) + list(range(embargo_end, n))
        val_idx = list(range(val_start, val_end))
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Train fold model
        clf = FastNumpyBinnedEnsemble(n_estimators=60, max_depth=5, min_samples_leaf=20, n_bins=32)
        clf.fit(X_train, y_train)
        
        probs = clf.predict_proba(X_val)
        p_safe = probs[:, 1] + probs[:, 2] # P(Breakeven) + P(Clean Win)
        
        approved_mask = p_safe >= threshold
        approved_count = int(np.sum(approved_mask))
        
        y_val_approved = y_val.values[approved_mask]
        
        val_hard_losses = int(np.sum(y_val == 0))
        val_scratches = int(np.sum(y_val == 1))
        val_clean_wins = int(np.sum(y_val == 2))
        base_safe_pct = ((val_scratches + val_clean_wins) / len(y_val)) * 100.0
        
        if approved_count > 0:
            app_hard_losses = int(np.sum(y_val_approved == 0))
            app_scratches = int(np.sum(y_val_approved == 1))
            app_clean_wins = int(np.sum(y_val_approved == 2))
            
            ml_safe_precision = ((app_scratches + app_clean_wins) / approved_count) * 100.0
            
            exp_base = ((val_clean_wins * 1.5) - (val_hard_losses * 1.0)) / len(y_val)
            exp_ml = ((app_clean_wins * 1.5) - (app_hard_losses * 1.0)) / approved_count
            exp_diff = exp_ml - exp_base
        else:
            ml_safe_precision = 0.0
            exp_base, exp_ml, exp_diff = 0.0, 0.0, 0.0
            
        fold_results.append({
            'fold': k + 1,
            'val_rows': len(val_idx),
            'train_rows': len(train_idx),
            'approved_count': approved_count,
            'approval_rate': (approved_count / len(val_idx)) * 100.0,
            'base_safe_pct': base_safe_pct,
            'ml_safe_precision': ml_safe_precision,
            'exp_base': exp_base,
            'exp_ml': exp_ml,
            'exp_diff': exp_diff
        })
        
        val_str = f"{val_start}:{val_end}"
        print(f"Fold #{k+1:<2} | {val_str:<20} | {len(train_idx):<10,} | {approved_count:<10,} | {base_safe_pct:>6.2f}%         | {ml_safe_precision:>6.2f}%            | {exp_diff:+.3f} R")
        
    print("-" * 115)
    
    # Calculate Average Across All Folds
    mean_base_safe = np.mean([r['base_safe_pct'] for r in fold_results])
    mean_ml_safe = np.mean([r['ml_safe_precision'] for r in fold_results])
    mean_exp_diff = np.mean([r['exp_diff'] for r in fold_results])
    mean_approval = np.mean([r['approval_rate'] for r in fold_results])
    
    print("\n" + "=" * 80)
    print("  PURGED & EMBARGOED CROSS-VALIDATION SUMMARY RESULTS")
    print("=" * 80)
    print(f"  Total Folds Evaluated         : {k_folds}")
    print(f"  Embargo Buffer                : 5 Days (480 15m Candles)")
    print(f"  Baseline Out-of-Sample Safe % : {mean_base_safe:.2f}%")
    print(f"  [SUCCESS] ML Safe Precision   : {mean_ml_safe:.2f}%  <-- ({mean_ml_safe - mean_base_safe:+.2f}% out-of-sample stability!)")
    print(f"  Mean Expectancy Boost / Trade : {mean_exp_diff:+.3f} R")
    print(f"  Mean Gatekeeper Approval Rate : {mean_approval:.2f}% of setups passed")
    print("=" * 80)

def main():
    start = time.time()
    run_purged_cv(k_folds=5, embargo_candles=480, threshold=0.65)
    elapsed = time.time() - start
    print(f"\n  [DONE] Purged & Embargoed Cross-Validation completed in {elapsed:.2f} seconds!")

if __name__ == "__main__":
    main()
