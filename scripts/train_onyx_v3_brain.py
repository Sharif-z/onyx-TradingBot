#!/usr/bin/env python3
"""
scripts/train_onyx_v3_brain.py
--------------------------------
Onyx V3 Low-RAM Universal Brain Trainer (Chunked Ingestion Engine).

Features:
- Streamed Data Ingestion: Reads dataset in 50,000-row chunks to keep peak RAM < 120MB.
- Filters strictly for 5 target core coins: ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'LINK/USDT'].
- Memory Optimization: Casts float columns to float32 and integer targets to int8.
- 100-Tree Fast Vectorized Binary Ensemble.
- Saves trained model to data/onyx_ml_gatekeeper.json and data/onyx_ml_features.json.
"""

import os
import sys
import time
import json
import gc
from datetime import datetime, timezone
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_CSV = os.path.join(DATA_DIR, "ml_trading_dataset_50coins.csv")
MODEL_OUTPUT_JSON = os.path.join(DATA_DIR, "onyx_ml_gatekeeper.json")
FEATURES_OUTPUT = os.path.join(DATA_DIR, "onyx_ml_features.json")

TARGET_TICKERS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "LINK/USDT"]

FEATURE_COLUMNS = [
    'risk_dist_pct', 'ema9_ema21_diff_pct', 'close_ema9_diff_pct', 
    'hma200_dist_pct', 'hma800_dist_pct', 'hma200_slope_pct', 
    'rsi', 'rsi_slope3', 'adx', 'adx_slope3', 'atr_pct', 'atr_squeeze_ratio', 
    'vol_ratio', 'vol_surge_ratio', 'body_to_range_ratio', 
    'upper_wick_ratio', 'lower_wick_ratio', 'ha_body_to_range_ratio', 
    'ha_upper_wick_ratio', 'ha_lower_wick_ratio', 
    'hour_of_day', 'day_of_week', 'hour_sin', 'hour_cos'
]

class FastVectorizedBinaryEnsemble:
    """Low-RAM Fast Vectorized 100-Tree Binary Binned Classifier."""
    def __init__(self, n_estimators=100, max_depth=6, min_samples_leaf=50, n_bins=32):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins
        self.trees = []
        self.bin_edges = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        print(f"  [QUANTIZATION] Quantizing {X.shape[1]} features into {self.n_bins} percentiles...")
        X_binned = pd.DataFrame(index=X.index)
        for col in X.columns:
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            edges = np.unique(np.percentile(X[col].values, percentiles))
            self.bin_edges[col] = edges.tolist()
            X_binned[col] = np.digitize(X[col].values, edges) - 1

        X_np = X_binned.values.astype(np.int8)
        y_np = y.values.astype(np.int8)
        
        n_samples, n_features = X_np.shape
        n_classes = 2
        
        print(f"  [TRAINING] Fitting {self.n_estimators} binary decision trees across {n_samples:,} setup vectors...")
        rng = np.random.RandomState(42)
        
        class_counts = np.bincount(y_np, minlength=2)
        class_weights = n_samples / (n_classes * np.maximum(class_counts, 1).astype(np.float32))
        
        for t in range(self.n_estimators):
            sample_idx = rng.choice(n_samples, size=int(0.85 * n_samples), replace=True)
            feat_idx = rng.choice(n_features, size=int(0.80 * n_features), replace=False)
            
            tree = self._build_tree(X_np[sample_idx][:, feat_idx], y_np[sample_idx], class_weights, current_depth=0, feat_indices=feat_idx)
            self.trees.append(tree)
            if (t + 1) % 25 == 0 or (t + 1) == self.n_estimators:
                print(f"    --> Built {t+1}/{self.n_estimators} Decision Trees...")

    def _build_tree(self, X_sub, y_sub, class_weights, current_depth, feat_indices):
        n_samples = len(y_sub)
        counts = np.bincount(y_sub, minlength=2)[:2]
        weighted_counts = counts * class_weights[:2]
        total_w = np.sum(weighted_counts) + 1e-8
        probs = weighted_counts / total_w
        
        if current_depth >= self.max_depth or n_samples < self.min_samples_leaf or len(np.unique(y_sub)) == 1:
            return {'is_leaf': True, 'probs': [round(float(p), 4) for p in probs]}
            
        n_features = X_sub.shape[1]
        best_gain = -1.0
        best_feat = None
        best_thresh = None
        
        parent_gini = 1.0 - np.sum(probs**2)
        
        for f_local in range(n_features):
            feat_vals = X_sub[:, f_local]
            
            max_b = self.n_bins + 1
            hist = np.bincount(feat_vals.astype(np.int32) * 2 + y_sub, minlength=max_b * 2)[:max_b * 2].reshape(max_b, 2)
            left_counts = np.cumsum(hist, axis=0)
            right_counts = counts - left_counts
            
            left_weighted = left_counts * class_weights[:2]
            right_weighted = right_counts * class_weights[:2]
            
            left_total = np.sum(left_weighted, axis=1, keepdims=True) + 1e-8
            right_total = np.sum(right_weighted, axis=1, keepdims=True) + 1e-8
            
            left_probs = left_weighted / left_total
            right_probs = right_weighted / right_total
            
            left_gini = 1.0 - np.sum(left_probs**2, axis=1)
            right_gini = 1.0 - np.sum(right_probs**2, axis=1)
            
            left_n = np.sum(left_counts, axis=1)
            p_left = left_n / float(n_samples)
            p_right = 1.0 - p_left
            
            gain = parent_gini - (p_left * left_gini + p_right * right_gini)
            
            valid_mask = (left_n >= 20) & ((n_samples - left_n) >= 20)
            if not np.any(valid_mask):
                continue
                
            gain[~valid_mask] = -1.0
            best_idx = np.argmax(gain)
            max_g = gain[best_idx]
            
            if max_g > best_gain:
                best_gain = max_g
                best_feat = f_local
                best_thresh = int(best_idx)
                
        if best_gain <= 0.0001 or best_feat is None:
            return {'is_leaf': True, 'probs': [round(float(p), 4) for p in probs]}
            
        left_mask = X_sub[:, best_feat] <= best_thresh
        right_mask = ~left_mask
        
        global_feat_idx = int(feat_indices[best_feat])
        
        left_node = self._build_tree(X_sub[left_mask], y_sub[left_mask], class_weights, current_depth + 1, feat_indices)
        right_node = self._build_tree(X_sub[right_mask], y_sub[right_mask], class_weights, current_depth + 1, feat_indices)
        
        return {
            'is_leaf': False,
            'feat_idx': global_feat_idx,
            'bin_threshold': best_thresh,
            'left': left_node,
            'right': right_node
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_binned = pd.DataFrame(index=X.index)
        for col in X.columns:
            edges = np.array(self.bin_edges[col])
            X_binned[col] = np.digitize(X[col].values, edges) - 1
            
        X_np = X_binned.values.astype(np.int8)
        n_samples = len(X_np)
        probs_all = np.zeros((n_samples, 2), dtype=np.float32)
        
        for tree in self.trees:
            probs_tree = np.zeros((n_samples, 2), dtype=np.float32)
            
            def traverse(node, mask):
                if not np.any(mask):
                    return
                if node.get('is_leaf', False):
                    probs_tree[mask] = np.array(node['probs'], dtype=np.float32)
                    return
                f_idx = node['feat_idx']
                thresh = node['bin_threshold']
                left_mask = mask & (X_np[:, f_idx] <= thresh)
                right_mask = mask & (~left_mask)
                traverse(node['left'], left_mask)
                traverse(node['right'], right_mask)

            traverse(tree, np.ones(n_samples, dtype=bool))
            probs_all += probs_tree
                
        probs_all /= len(self.trees)
        return probs_all

def main():
    print("=" * 80)
    print("  ONYX V3 LOW-RAM UNIVERSAL BRAIN — STREAMED CHUNKED TRAINER")
    print("=" * 80)
    
    if not os.path.exists(DATASET_CSV):
        print(f"[ERROR] Dataset file not found at {DATASET_CSV}!")
        sys.exit(1)
        
    print(f"[*] Streamed Ingestion: Reading {DATASET_CSV} in 50,000-row chunks...")
    print(f"[*] Target Filter Assets: {', '.join(TARGET_TICKERS)}")
    
    filtered_chunks = []
    total_scanned_rows = 0
    chunk_count = 0
    
    for chunk in pd.read_csv(DATASET_CSV, chunksize=50000):
        chunk_count += 1
        total_scanned_rows += len(chunk)
        
        # Lowercase column names immediately
        chunk.columns = [str(c).strip().lower() for c in chunk.columns]
        
        # Filter for target core coins
        f_chunk = chunk[chunk['ticker'].isin(TARGET_TICKERS)].copy()
        if not f_chunk.empty:
            # Memory optimization: Downcast dtypes to float32 & int8
            for col in FEATURE_COLUMNS:
                if col in f_chunk.columns:
                    f_chunk[col] = f_chunk[col].astype(np.float32)
                    
            if 'label_win' in f_chunk.columns:
                f_chunk['label_win'] = f_chunk['label_win'].astype(np.int8)
            elif 'label_class' in f_chunk.columns:
                f_chunk['label_win'] = (f_chunk['label_class'] == 2).astype(np.int8)
                
            filtered_chunks.append(f_chunk)
            
        if chunk_count % 10 == 0:
            print(f"  -> Processed {total_scanned_rows:,} rows across {chunk_count} chunks... (Filtered rows: {sum(len(c) for c in filtered_chunks):,})")

    if not filtered_chunks:
        print("[ERROR] No setup rows found for target tickers!")
        sys.exit(1)
        
    df = pd.concat(filtered_chunks, ignore_index=True)
    del filtered_chunks
    gc.collect()
    
    print("\n" + "=" * 80)
    print(f"[*] Streamed Ingestion Complete: {total_scanned_rows:,} rows scanned.")
    print(f"[*] Filtered Dataset: {len(df):,} setup rows for core coins ({', '.join(TARGET_TICKERS)})")
    print(f"[*] RAM Memory Usage: ~{df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB (Optimized for Termux)")
    print("=" * 80)
    
    X = df[FEATURE_COLUMNS].copy()
    y = df['label_win'].astype(int)
    
    print(f"[*] Target Distribution (Binary): {np.bincount(y)}")
    win_pct = (np.sum(y == 1) / len(y)) * 100.0
    print(f"[*] Raw Base Win Rate: {win_pct:.2f}% ({np.sum(y==1):,} wins / {np.sum(y==0):,} losses)")
    
    split_idx = int(0.80 * len(df))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"[*] Train set: {len(X_train):,} rows | Holdout Test set: {len(X_test):,} rows")
    
    model = FastVectorizedBinaryEnsemble(n_estimators=100, max_depth=6, min_samples_leaf=50, n_bins=32)
    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start_time
    print(f"[*] Model training completed in {elapsed:.2f} seconds!")
    
    print("[*] Evaluating Holdout Test Set Predictions...")
    test_probs = model.predict_proba(X_test)
    p_win_test = test_probs[:, 1]
    
    p75_thresh = float(np.percentile(p_win_test, 75))
    p90_thresh = float(np.percentile(p_win_test, 90))
    
    print("\n" + "=" * 80)
    print("  MODEL HOLDOUT EVALUATION & PERCENTILE MATRIX")
    print("=" * 80)
    print(f"  P75 Threshold (Tier 1 Standard):  {p75_thresh*100:.2f}%")
    print(f"  P90 Threshold (Tier 2 Sniper):    {p90_thresh*100:.2f}%")
    
    t1_mask = p_win_test >= p75_thresh
    if np.sum(t1_mask) > 0:
        t1_winrate = (np.sum(y_test.values[t1_mask] == 1) / np.sum(t1_mask)) * 100.0
        print(f"  🟢 Tier 1 Standard Win Rate (Top 25% Setups): {t1_winrate:.2f}% ({np.sum(t1_mask):,} trades)")
        
    t2_mask = p_win_test >= p90_thresh
    if np.sum(t2_mask) > 0:
        t2_winrate = (np.sum(y_test.values[t2_mask] == 1) / np.sum(t2_mask)) * 100.0
        print(f"  🔥 Tier 2 Sniper Win Rate (Top 10% Setups):   {t2_winrate:.2f}% ({np.sum(t2_mask):,} trades)")
        
    export_payload = {
        'model_type': 'FastVectorizedBinaryEnsemble',
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth,
        'features': FEATURE_COLUMNS,
        'bin_edges': model.bin_edges,
        'trees': model.trees,
        'percentiles': {
            'p75_threshold': round(p75_thresh, 4),
            'p90_threshold': round(p90_thresh, 4)
        },
        'metadata': {
            'trained_at': datetime.now(timezone.utc).isoformat(),
            'total_samples': len(df),
            'win_samples': int(np.sum(y == 1)),
            'loss_samples': int(np.sum(y == 0))
        }
    }
    
    with open(MODEL_OUTPUT_JSON, 'w') as f:
        json.dump(export_payload, f, indent=2)
    print(f"\n[EXPORT SUCCESS] Onyx V3 Model exported to {MODEL_OUTPUT_JSON}")
    
    features_payload = {
        'feature_columns': FEATURE_COLUMNS,
        'tier1_threshold': 0.51,
        'tier2_threshold': 0.53,
        'n_features': len(FEATURE_COLUMNS),
        'metadata': {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'baseline_winrate': round(win_pct, 2)
        }
    }
    with open(FEATURES_OUTPUT, 'w') as f:
        json.dump(features_payload, f, indent=2)
    print(f"[EXPORT SUCCESS] Onyx V3 Feature Spec exported to {FEATURES_OUTPUT}")
    
    print("=" * 80)
    print("  ONYX V3 BRAIN TRAINING COMPLETE 🚀")
    print("=" * 80)

if __name__ == "__main__":
    main()
