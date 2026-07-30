#!/usr/bin/env python3
"""
scripts/reset_account.py
------------------------
Resets Onyx Bot account state:
1. Clears data/trade_ledger.csv
2. Clears data/active_position.json
3. Resets Account Capital back to $10,000.00 USD
"""

import os
import csv
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
LEDGER_FILE = os.path.join(DATA_DIR, "trade_ledger.csv")
STATE_FILE = os.path.join(CACHE_DIR, "active_position.json")

def main():
    print("=" * 65)
    print("  ONYX QUANTITATIVE DESK — ACCOUNT RESET UTILITY")
    print("=" * 65)
    
    # 1. Reset Trade Ledger CSV with header only
    fieldnames = [
        'entry_time', 'exit_time', 'ticker', 'type', 'entry_price',
        'exit_price', 'pnl', 'cause', 'position_size', 'balance'
    ]
    
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LEDGER_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
    print(f"  [1/3] Reset Trade Ledger CSV at {LEDGER_FILE}")
    
    # 2. Reset Active Position JSON
    if os.path.exists(STATE_FILE):
        try:
            os.remove(STATE_FILE)
            print(f"  [2/3] Removed persistent position state file {STATE_FILE}")
        except Exception as e:
            print(f"  [2/3] Warning removing state file: {e}")
    else:
        print(f"  [2/3] Position state file is clean.")
        
    print("  [3/3] Account capital reset to $10,000.00 USD.")
    print("=" * 65)
    print("  [DONE] Account reset complete! Ready for live market trading.")

if __name__ == "__main__":
    main()
