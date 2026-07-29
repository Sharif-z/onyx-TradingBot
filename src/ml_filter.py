#!/usr/bin/env python3
"""
src/ml_filter.py
------------------
Live Machine Learning Gatekeeper Filter for Onyx Trading Bot.

Loads the trained pure NumPy binned ensemble model (data/onyx_ml_gatekeeper.json)
and feature schema (data/onyx_ml_features.json) to evaluate live trade triggers
in real-time during bot execution.
"""

import os
import sys
import json
import math
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_JSON = os.path.join(DATA_DIR, "onyx_ml_gatekeeper.json")
FEATURES_JSON = os.path.join(DATA_DIR, "onyx_ml_features.json")

class MLFilter:
    def __init__(self, model_path: str = MODEL_JSON, features_path: str = FEATURES_JSON):
        self.model_path = model_path
        self.features_path = features_path
        self.is_loaded = False
        self.trees = []
        self.bin_edges = {}
        self.feature_cols = []
        self.threshold = 0.65
        
        self.load_model()
        
    def load_model(self):
        """Loads trained model weights and feature quantization bin edges from disk."""
        if not os.path.exists(self.model_path):
            print(f"[ML_FILTER WARN] Model binary missing at {self.model_path}. ML Gatekeeper disabled.")
            self.is_loaded = False
            return
            
        try:
            with open(self.model_path, "r") as f:
                data = json.load(f)
                self.trees = data.get("trees", [])
                self.bin_edges = data.get("bin_edges", {})
                self.feature_cols = data.get("feature_cols", [])
                
            if os.path.exists(self.features_path):
                with open(self.features_path, "r") as f:
                    f_meta = json.load(f)
                    self.threshold = f_meta.get("threshold", 0.65)
                    
            self.is_loaded = len(self.trees) > 0 and len(self.feature_cols) > 0
            if self.is_loaded:
                print(f"[ML_FILTER SUCCESS] Loaded Onyx ML Gatekeeper ({len(self.trees)} trees, {len(self.feature_cols)} features, Threshold: {self.threshold*100:.0f}%).")
        except Exception as e:
            print(f"[ML_FILTER ERROR] Failed to load ML model: {e}")
            self.is_loaded = False

    def extract_live_features(self, active_row: pd.Series, prev_row: pd.Series, side: str) -> dict:
        """Extracts 24 normalized technical & market state features from live tick row."""
        try:
            entry_price = float(active_row['close'])
            atr = float(active_row['atr']) if 'atr' in active_row and not pd.isna(active_row['atr']) else entry_price * 0.015
            
            candle_range = (float(active_row['high']) - float(active_row['low'])) + 1e-8
            body_size = abs(float(active_row['close']) - float(active_row['open']))
            
            ha_open = float(active_row['ha_open']) if 'ha_open' in active_row else float(active_row['open'])
            ha_high = float(active_row['ha_high']) if 'ha_high' in active_row else float(active_row['high'])
            ha_low = float(active_row['ha_low']) if 'ha_low' in active_row else float(active_row['low'])
            ha_close = float(active_row['ha_close']) if 'ha_close' in active_row else float(active_row['close'])
            
            ha_range = (ha_high - ha_low) + 1e-8
            ha_body_size = abs(ha_close - ha_open)
            
            dt = pd.Timestamp.now(tz='UTC')
            if 'timestamp' in active_row and not pd.isna(active_row['timestamp']):
                dt = pd.to_datetime(active_row['timestamp'], unit='ms', utc=True)
                
            prev_trend = float(prev_row['trend_baseline']) if prev_row is not None and 'trend_baseline' in prev_row and not pd.isna(prev_row['trend_baseline']) else float(active_row['trend_baseline'])
            
            vol_sma20 = float(active_row['vol_sma20']) if 'vol_sma20' in active_row and not pd.isna(active_row['vol_sma20']) else float(active_row['volume'])
            vol_max20 = float(active_row['vol_max20']) if 'vol_max20' in active_row and not pd.isna(active_row['vol_max20']) else vol_sma20 * 1.5
            
            atr_sma100 = float(active_row['atr_sma100']) if 'atr_sma100' in active_row and not pd.isna(active_row['atr_sma100']) else atr
            
            rsi = float(active_row['rsi']) if 'rsi' in active_row and not pd.isna(active_row['rsi']) else 50.0
            adx = float(active_row['adx']) if 'adx' in active_row and not pd.isna(active_row['adx']) else 20.0
            
            rsi_slope3 = float(active_row['rsi_slope3']) if 'rsi_slope3' in active_row and not pd.isna(active_row['rsi_slope3']) else 0.0
            adx_slope3 = float(active_row['adx_slope3']) if 'adx_slope3' in active_row and not pd.isna(active_row['adx_slope3']) else 0.0
            
            features = {
                'risk_dist_pct': (1.5 * atr / entry_price) * 100.0,
                'ema9_ema21_diff_pct': ((float(active_row['ema9']) - float(active_row['ema21'])) / float(active_row['ema21'])) * 100.0,
                'close_ema9_diff_pct': ((entry_price - float(active_row['ema9'])) / float(active_row['ema9'])) * 100.0,
                'hma200_dist_pct': ((entry_price - float(active_row['trend_baseline'])) / float(active_row['trend_baseline'])) * 100.0,
                'hma800_dist_pct': ((entry_price - float(active_row['macro_baseline'])) / float(active_row['macro_baseline'])) * 100.0,
                'hma200_slope_pct': ((float(active_row['trend_baseline']) - prev_trend) / (prev_trend + 1e-8)) * 100.0,
                'rsi': rsi,
                'rsi_slope3': rsi_slope3,
                'adx': adx,
                'adx_slope3': adx_slope3,
                'atr_pct': (atr / entry_price) * 100.0,
                'atr_squeeze_ratio': atr / (atr_sma100 + 1e-8),
                'vol_ratio': float(active_row['volume']) / (vol_sma20 + 1e-8),
                'vol_surge_ratio': float(active_row['volume']) / (vol_max20 + 1e-8),
                'body_to_range_ratio': body_size / candle_range,
                'upper_wick_ratio': (float(active_row['high']) - max(float(active_row['open']), entry_price)) / candle_range,
                'lower_wick_ratio': (min(float(active_row['open']), entry_price) - float(active_row['low'])) / candle_range,
                'ha_body_to_range_ratio': ha_body_size / ha_range,
                'ha_upper_wick_ratio': (ha_high - max(ha_open, ha_close)) / ha_range,
                'ha_lower_wick_ratio': (min(ha_open, ha_close) - ha_low) / ha_range,
                'hour_of_day': dt.hour,
                'day_of_week': dt.weekday(),
                'hour_sin': math.sin(2 * math.pi * dt.hour / 24.0),
                'hour_cos': math.cos(2 * math.pi * dt.hour / 24.0),
            }
            return features
        except Exception as e:
            print(f"[ML_FILTER ERROR] Feature extraction exception: {e}")
            return {}

    def predict_safe_probability(self, features: dict) -> tuple[float, float, float]:
        """Quantizes live feature dict into bins and computes P(Loss), P(Breakeven), P(Win)."""
        if not self.is_loaded:
            return 0.0, 0.5, 0.5
            
        try:
            # Quantize features into bin indices
            binned_vector = []
            for col in self.feature_cols:
                val = float(features.get(col, 0.0))
                edges = np.array(self.bin_edges.get(col, [0.0, 1.0]))
                bin_idx = int(np.digitize([val], edges)[0] - 1)
                binned_vector.append(bin_idx)
                
            n_samples = 1
            all_probs = np.zeros(3, dtype=np.float32)
            
            for tree in self.trees:
                curr = tree
                while not curr.get('is_leaf', False):
                    feat_idx = curr['feat_idx']
                    thresh = curr['bin_threshold']
                    if binned_vector[feat_idx] <= thresh:
                        curr = curr['left']
                    else:
                        curr = curr['right']
                all_probs += np.array(curr['probs'], dtype=np.float32)
                
            all_probs /= len(self.trees)
            p_loss, p_be, p_win = float(all_probs[0]), float(all_probs[1]), float(all_probs[2])
            return p_loss, p_be, p_win
        except Exception as e:
            print(f"[ML_FILTER ERROR] Inference prediction failed: {e}")
            return 0.0, 0.5, 0.5

    def evaluate(self, active_row: pd.Series, prev_row: pd.Series, side: str, dry_run: bool = True, custom_threshold: float = None) -> dict:
        """
        Evaluates live tick trigger. Returns evaluation dictionary:
        { 'approved': bool, 'p_safe': float, 'p_win': float, 'p_loss': float, 'reason': str }
        """
        threshold = custom_threshold if custom_threshold is not None else self.threshold
        
        if not self.is_loaded:
            return {
                'approved': True,
                'p_safe': 1.0,
                'p_win': 0.5,
                'p_loss': 0.0,
                'reason': '[ML_GATE] Model disabled or missing — Allowing trade by default.'
            }
            
        feats = self.extract_live_features(active_row, prev_row, side)
        if not feats:
            return {
                'approved': True,
                'p_safe': 1.0,
                'p_win': 0.5,
                'p_loss': 0.0,
                'reason': '[ML_GATE] Feature extraction failed — Allowing trade.'
            }
            
        p_loss, p_be, p_win = self.predict_safe_probability(feats)
        p_safe = p_be + p_win # 1.0 - P(Hard Loss)
        
        is_approved = p_safe >= threshold
        
        if is_approved:
            reason = f"[ML_APPROVED] Safe Prob {p_safe*100:.1f}% >= {threshold*100:.0f}% threshold (Win: {p_win*100:.1f}%, BE: {p_be*100:.1f}%)."
        else:
            if dry_run:
                reason = f"[ML_DRY_RUN_VETO] Low Safe Prob {p_safe*100:.1f}% < {threshold*100:.0f}% threshold. (Dry-Run: Trade Allowed)."
            else:
                reason = f"[ML_VETO] High Loss Risk! Safe Prob {p_safe*100:.1f}% < {threshold*100:.0f}% threshold (Loss Risk: {p_loss*100:.1f}%)."
                
        return {
            'approved': is_approved if not dry_run else True,
            'p_safe': round(p_safe, 4),
            'p_win': round(p_win, 4),
            'p_loss': round(p_loss, 4),
            'dry_run': dry_run,
            'reason': reason
        }
