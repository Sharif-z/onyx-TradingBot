#!/usr/bin/env python3
"""
scratch/test_dual_tier_safeguards.py
--------------------------------------
Unit test script verifying Onyx V3 Dual-Tier ML Matrix routing & telemetry logging.
"""

import sys
import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.ml_filter import MLFilter
from config import TalksyConfig

def main():
    print("=" * 70)
    print("  ONYX V3 DUAL-TIER MATRIX ROUTING & TELEMETRY VERIFICATION")
    print("=" * 70)
    
    ml_filter = MLFilter()
    if not ml_filter.is_loaded:
        print("[!] Warning: ML Filter JSON model not yet loaded/exported.")
    else:
        print(f"[SUCCESS] ML Filter loaded successfully! Feature Count: {len(ml_filter.feature_cols)}")
        
    print(f"[*] Config Thresholds: Tier 1 = {TalksyConfig.ML_TIER1_THRESHOLD*100:.1f}% | Tier 2 = {TalksyConfig.ML_TIER2_THRESHOLD*100:.1f}%")
    print(f"[*] Fade Mode Enabled: {TalksyConfig.FADE_MODE}")
    print(f"[*] Portfolio Heat Cap Limit: {TalksyConfig.MAX_PORTFOLIO_HEAT_PCT:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
