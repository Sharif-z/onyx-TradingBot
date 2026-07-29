#!/usr/bin/env python3
"""
scripts/train_ml_model.py
----------------------------
Institutional-Grade Memory-Efficient ML Model Trainer for Onyx Gatekeeper.

Designed for low RAM consumption on mobile tablets / Termux.
Supports:
1. Scikit-Learn HistGradientBoostingClassifier (if sklearn installed).
2. Pure NumPy/SciPy Low-Memory Vectorized Binned Ensemble Engine (Fallback).
3. 80/20 Chronological Split (No lookahead bias).
4. Safe Trade Probability: P(Safe) = P(Breakeven) + P(Clean Win) >= 65%.
5. Model export to data/onyx_ml_gatekeeper.json or data/onyx_ml_gatekeeper.joblib.
"""

import os
import sys
import time
import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_CSV = os.path.join(DATA_DIR, "ml_trading_dataset.csv")
MODEL_OUTPUT_JOBLIB = os.path.join(DATA_DIR, "onyx_ml_gatekeeper.joblib")
MODEL_OUTPUT_JSON = os.path.join(DATA_DIR, "onyx_ml_gatekeeper.json")
FEATURES_OUTPUT = os.path.join(DATA_DIR, "onyx_ml_features.json")

# Pure NumPy Binned Decision Tree Ensemble for low-RAM Termux execution
class FastNumpyBinnedEnsemble:
    def __init__(self, n_estimators=100, max_depth=6, min_samples_leaf=20, n_bins=32):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins
        self.trees = []
        self.bin_edges = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        print(f"  [NUMPY ENGINE] Quantizing {X.shape[1]} features into {self.n_bins} histogram bins...")
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
        
        print(f"  [NUMPY ENGINE] Building {self.n_estimators} binned decision trees in memory-efficient chunks...")
        rng = np.random.RandomState(42)
        
        # Calculate class weights for imbalance handling
        class_counts = np.bincount(y_np)
        class_weights = n_samples / (n_classes * class_counts.astype(np.float32))
        
        for t in range(self.n_estimators):
            # 80% bootstrap sample
            sample_idx = rng.choice(n_samples, size=int(0.80 * n_samples), replace=True)
            feat_idx = rng.choice(n_features, size=int(0.75 * n_features), replace=False)
            
            tree = self._build_tree(X_np[sample_idx][:, feat_idx], y_np[sample_idx], class_weights, current_depth=0, feat_indices=feat_idx)
            self.trees.append(tree)
            if (t + 1) % 25 == 0:
                print(f"    --> Trained {t+1}/{self.n_estimators} decision trees...")

    def _build_tree(self, X_sub, y_sub, class_weights, current_depth, feat_indices):
        n_samples, n_sub_feats = X_sub.shape
        counts = np.bincount(y_sub, minlength=3)
        weighted_counts = counts * class_weights
        probs = weighted_counts / (np.sum(weighted_counts) + 1e-8)
        
        if current_depth >= self.max_depth or n_samples < self.min_samples_leaf or len(np.unique(y_sub)) == 1:
            return {'is_leaf': True, 'probs': probs.tolist()}
            
        best_gain = -1.0
        best_split = None
        
        # Random subset of candidate splits for memory speed
        for f_sub_idx in range(n_sub_feats):
            feat_vals = X_sub[:, f_sub_idx]
            unique_vals = np.unique(feat_vals)
            if len(unique_vals) <= 1:
                continue
                
            for val in unique_vals[::2]: # test every 2nd bin
                left_mask = feat_vals <= val
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                    
                # Gini impurity gain
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
        # Bin input data using saved bin edges
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

    def save_json(self, filepath: str, feature_cols: list):
        data = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'n_bins': self.n_bins,
            'bin_edges': self.bin_edges,
            'feature_cols': feature_cols,
            'trees': self.trees
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
        print(f"  --> Saved Pure NumPy Model: {filepath}")

def main():
    print("=" * 75)
    print("  ONYX QUANTITATIVE DESK — ML GATEKEEPER MODEL TRAINER")
    print("  Mobile & Termux Low-Memory Engine")
    print("=" * 75)
    
    if not os.path.exists(DATASET_CSV):
        print(f"[ERROR] Dataset file not found at {DATASET_CSV}. Run scripts/harvest_ml_dataset.py first.")
        sys.exit(1)
        
    print(f"[1/5] Loading dataset from {DATASET_CSV}...")
    df = pd.read_csv(DATASET_CSV)
    print(f"  --> Loaded {len(df):,} setup rows.")
    
    target_col = 'label_class'
    drop_cols = [
        'timestamp', 'ticker', 'datetime_utc', 'side', 
        'entry_price', 'sl_price', 'tp_price', 
        'label_win', 'label_max_rr', 'label_net_r', 'label_class'
    ]
    
    if os.path.exists(FEATURES_OUTPUT):
        try:
            with open(FEATURES_OUTPUT, "r") as f:
                meta = json.load(f)
                if "features" in meta and len(meta["features"]) > 0:
                    feature_cols = [c for c in meta["features"] if c in df.columns]
                    print(f"[2/5] Using {len(feature_cols)} PRUNED predictor features from {FEATURES_OUTPUT}.")
                else:
                    feature_cols = [c for c in df.columns if c not in drop_cols]
                    print(f"[2/5] Extracted {len(feature_cols)} predictor features.")
        except Exception:
            feature_cols = [c for c in df.columns if c not in drop_cols]
            print(f"[2/5] Extracted {len(feature_cols)} predictor features.")
    else:
        feature_cols = [c for c in df.columns if c not in drop_cols]
        print(f"[2/5] Extracted {len(feature_cols)} predictor features.")
    
    X = df[feature_cols].copy().astype(np.float32)
    y = df[target_col].copy().astype(np.int8)
    
    # 80/20 Chronological Split (No shuffle)
    train_size = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    print(f"  --> Chronological Split: Train Set = {len(X_train):,} rows | Test Set = {len(X_test):,} rows")
    
    # Check if Scikit-Learn is installed
    use_sklearn = False
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        import joblib
        use_sklearn = True
        print("[3/5] Using Scikit-Learn HistGradientBoostingClassifier...")
    except ImportError:
        print("[3/5] Scikit-Learn not found -> Using Pure NumPy Vectorized Binned Engine (0% RAM Bottleneck!)...")
        
    start_time = time.time()
    
    if use_sklearn:
        clf = HistGradientBoostingClassifier(
            class_weight='balanced',
            max_iter=150,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=25,
            l2_regularization=1.5,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=15
        )
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        joblib.dump(clf, MODEL_OUTPUT_JOBLIB, compress=3)
        print(f"  --> Scikit-Learn Model Saved: {MODEL_OUTPUT_JOBLIB}")
    else:
        clf = FastNumpyBinnedEnsemble(n_estimators=80, max_depth=5, min_samples_leaf=20, n_bins=32)
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        clf.save_json(MODEL_OUTPUT_JSON, feature_cols)
        
    elapsed = time.time() - start_time
    print(f"  --> Training & Forward Prediction complete in {elapsed:.2f} seconds.")
    
    # 4. Evaluate Gatekeeper Threshold Logic
    print("\n[4/5] Evaluating Gatekeeper Safe Trade Threshold Logic (Unseen Test Data)...")
    
    p_loss = probs[:, 0]
    p_breakeven = probs[:, 1]
    p_win = probs[:, 2]
    
    # Safe Trade Probability = P(Breakeven) + P(Clean Win) = 1.0 - P(Hard Loss)
    p_safe = p_breakeven + p_win
    
    threshold = 0.65
    approved_mask = p_safe >= threshold
    
    approved_count = int(np.sum(approved_mask))
    total_test = len(y_test)
    approval_rate = (approved_count / total_test) * 100.0
    
    y_test_approved = y_test.values[approved_mask]
    
    approved_hard_losses = int(np.sum(y_test_approved == 0))
    approved_scratches = int(np.sum(y_test_approved == 1))
    approved_clean_wins = int(np.sum(y_test_approved == 2))
    
    safe_precision = ((approved_scratches + approved_clean_wins) / max(approved_count, 1)) * 100.0
    win_precision = (approved_clean_wins / max(approved_count, 1)) * 100.0
    
    base_hard_losses = int(np.sum(y_test == 0))
    base_scratches = int(np.sum(y_test == 1))
    base_clean_wins = int(np.sum(y_test == 2))
    base_safe_precision = ((base_scratches + base_clean_wins) / total_test) * 100.0
    
    print("\n" + "=" * 75)
    print("  FORWARD TEST SET PERFORMANCE EVALUATION (UNSEEN FUTURE DATA)")
    print("=" * 75)
    print(f"  Total Test Trades Evaluated   : {total_test:,}")
    print(f"  Baseline Safe Trade Rate       : {base_safe_precision:.2f}% (Hard Losses = {base_hard_losses/total_test*100:.2f}%)")
    print("-" * 75)
    print(f"  ML GATEKEEPER APPROVED TRADES (P_Safe >= {threshold*100:.0f}%):")
    print(f"  Approved Trades Count         : {approved_count:,} ({approval_rate:.2f}% of all setups)")
    print(f"  Clean Wins (+1.5R Hit)        : {approved_clean_wins:,} ({win_precision:.2f}% of approved)")
    print(f"  Breakeven Scratches (0.0R)    : {approved_scratches:,} ({approved_scratches/max(approved_count, 1)*100:.2f}% of approved)")
    print(f"  Hard Losses (-1.0R SL)        : {approved_hard_losses:,} ({approved_hard_losses/max(approved_count, 1)*100:.2f}% of approved)")
    print(f"  [SUCCESS] Safe Trade Precision: {safe_precision:.2f}% (Hard Loss Reduction: {100 - (approved_hard_losses/max(approved_count, 1)*100):.2f}% Safe)")
    print("=" * 75)
    
    # Calculate Expected Net R per Approved Trade
    net_r_approved = (approved_clean_wins * 1.5) + (approved_scratches * 0.0) - (approved_hard_losses * 1.0)
    avg_r_per_trade = net_r_approved / max(approved_count, 1)
    
    net_r_baseline = (base_clean_wins * 1.5) + (base_scratches * 0.0) - (base_hard_losses * 1.0)
    avg_r_baseline = net_r_baseline / total_test
    
    print(f"\n  💰 EXPECTED RETURN METRICS (R-MULTIPLE EXPECTANCY):")
    print(f"  Baseline Expectancy / Trade   : {avg_r_baseline:+.3f} R")
    print(f"  ML Approved Expectancy / Trade: {avg_r_per_trade:+.3f} R  <-- ({avg_r_per_trade - avg_r_baseline:+.3f} R improvement!)")
    print(f"  Total Net R Return (Test Set) : {net_r_approved:+.1f} R across {approved_count} trades")
    print("=" * 75)

    # 5. Export Features JSON Config
    print("\n[5/5] Saving features config metadata...")
    with open(FEATURES_OUTPUT, "w") as f:
        json.dump({
            "features": feature_cols,
            "threshold": threshold,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "train_size": train_size,
            "test_size": total_test,
            "safe_precision": safe_precision,
            "avg_r_per_trade": avg_r_per_trade,
            "engine": "sklearn" if use_sklearn else "numpy_binned"
        }, f, indent=2)
        
    print(f"  --> Saved Features Metadata: {FEATURES_OUTPUT}")
    print("\n  [DONE] Onyx ML Gatekeeper model training complete!")

if __name__ == "__main__":
    main()
