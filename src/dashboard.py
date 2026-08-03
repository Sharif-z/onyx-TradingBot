# src/dashboard.py
import os
import time
from datetime import datetime
from colorama import init, Fore, Back, Style
from tabulate import tabulate

# Initialize colorama for cross-platform support
init(autoreset=True)

class Dashboard:
    @staticmethod
    def clear_screen():
        """Clear console screen using ANSI codes to reduce flicker."""
        print("\033[H\033[2J", end="")

    @staticmethod
    def render(
        payloads: dict,
        focused_ticker: str,
        trade_ledger: list,
        ticking_interval: int = 20
    ):
        """Render the beautiful dark-themed ASCII Terminal Dashboard representing the multi-symbol trading desk."""
        Dashboard.clear_screen()
        
        # High-Contrast Pitch Black Box Header
        print(f"{Fore.CYAN}{Style.BRIGHT}┌" + "─"*68 + "┐")
        print(f"{Fore.CYAN}{Style.BRIGHT}│ {Fore.WHITE}{Style.BRIGHT}            ONYX  –  QUANTITATIVE DESK TERMINAL               {Fore.CYAN}│")
        print(f"{Fore.CYAN}{Style.BRIGHT}├" + "─"*68 + "┤")
        
        # Part 1: Portfolio Desk Overview Grid
        now_str = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
        
        sess_status = "ACTIVE"
        sess_countdown = ""
        for t in payloads.keys():
            p = payloads[t]
            if 'session_status' in p:
                sess_status = p['session_status']
                sess_countdown = p.get('session_countdown', '')
                break
                
        if sess_status == "ACTIVE":
            session_indicator = f"{Fore.GREEN}● ACTIVE"
        else:
            session_indicator = f"{Fore.RED}○ INACTIVE ({sess_countdown})"
            
        print(f" {Fore.LIGHTBLACK_EX}SYNC TIME: {Fore.WHITE}{now_str} │ {Fore.LIGHTBLACK_EX}SESSION: {session_indicator}")
        print(f" {Fore.CYAN}{Style.BRIGHT}PORTFOLIO DESK OVERVIEW:")
        
        overview_rows = []
        for t in sorted(payloads.keys()):
            p = payloads.get(t)
            if p:
                t_state = p.get('state', 'IDLE')
                t_price = p.get('current_price', 0.0)
                t_entry = p.get('entry_price')
                t_contracts = p.get('contracts', 0.0)
                
                # Calculate individual position unclosed PnL
                if t_state == "LONG" and t_entry and t_contracts:
                    t_pnl = t_contracts * (t_price - t_entry)
                    t_pnl_pct = (t_pnl / (t_entry * t_contracts) * 100.0) if (t_entry and t_contracts and (t_entry * t_contracts != 0)) else 0.0
                elif t_state == "SHORT" and t_entry and t_contracts:
                    t_pnl = t_contracts * (t_entry - t_price)
                    t_pnl_pct = (t_pnl / (t_entry * t_contracts) * 100.0) if (t_entry and t_contracts and (t_entry * t_contracts != 0)) else 0.0
                else:
                    t_pnl = 0.0
                    t_pnl_pct = 0.0
                
                # Sentiment indicators
                t_bull = p.get('bull_pct', 50)
                t_bear = p.get('bear_pct', 50)
                
                pnl_color = Fore.GREEN if t_pnl >= 0 else Fore.RED
                pnl_sign = "+" if t_pnl >= 0 else ""
                
                state_color = Fore.WHITE + Style.DIM if t_state == "IDLE" else (Fore.GREEN + Style.BRIGHT if t_state == "LONG" else Fore.RED + Style.BRIGHT)
                
                pnl_str = f"{pnl_color}{pnl_sign}${t_pnl:.2f} ({pnl_sign}{t_pnl_pct:.2f}%)" if t_state != "IDLE" else f"{Fore.LIGHTBLACK_EX}$0.00 (0.00%)"
                
                # Highlight active focused symbol with a arrow indicator
                ticker_label = f"{Fore.YELLOW}{Style.BRIGHT}→ {t:<9}" if t == focused_ticker else f"  {t:<9}"
                
                # Price dynamic precision formatting
                is_major = t.startswith("BTC/") or t.startswith("ETH/")
                prec = 2 if is_major else (3 if t_price < 200 else 2)
                price_str = f"{Fore.CYAN}${t_price:,.{prec}f}"
                
                overview_rows.append([
                    ticker_label,
                    f"{state_color}{t_state:<6}",
                    price_str,
                    pnl_str,
                    f"{Fore.GREEN}{t_bull}%{Fore.WHITE}/{Fore.RED}{t_bear}%",
                    f"{Fore.MAGENTA}{p.get('candle_close_count_wrong_side', 0)}"
                ])
                
        print(tabulate(overview_rows, headers=["Symbol", "State", "Price", "Open PnL", "Bull/Bear", "Panic"], tablefmt="plain"))
        print(f"{Fore.CYAN}{Style.BRIGHT}├" + "─"*68 + "┤")
        
        # Part 2: Detailed HUD for Focused Ticker
        p = payloads.get(focused_ticker)
        if not p:
            print(f"  {Fore.RED}No detailed telemetry payload for focused symbol {focused_ticker}.")
            print(f"{Fore.CYAN}{Style.BRIGHT}└" + "─"*68 + "┘")
            return
            
        t_state = p.get('state', 'IDLE')
        t_price = p.get('current_price', 0.0)
        t_balance = p.get('balance', 10000.0)
        t_portfolio = p.get('portfolio_val', t_balance)
        t_pnl = t_portfolio - t_balance
        
        state_color = Fore.WHITE + Style.DIM if t_state == "IDLE" else (Fore.GREEN + Style.BRIGHT if t_state == "LONG" else Fore.RED + Style.BRIGHT)
        print(f" {Fore.YELLOW}{Style.BRIGHT}FOCUSED TICKER DETAIL: {Fore.WHITE}{focused_ticker} ({t_state})")
        
        # Row 2: Live Price & Targets
        price_color = Fore.CYAN
        if t_state == "LONG" and p.get('entry_price') is not None:
            price_color = Fore.GREEN if t_price >= p['entry_price'] else Fore.RED
        elif t_state == "SHORT" and p.get('entry_price') is not None:
            price_color = Fore.GREEN if t_price <= p['entry_price'] else Fore.RED
            
        is_major = focused_ticker.startswith("BTC/") or focused_ticker.startswith("ETH/")
        prec = 2 if is_major else (3 if t_price < 200 else 2)
        
        print(f" {Fore.LIGHTBLACK_EX}LIVE PRICE: {price_color}{t_price:,.{prec}f} USD")
        
        if t_state != "IDLE" and p.get('entry_price') is not None:
            ep = p['entry_price']
            sl_val = p.get('exit_sl')
            tp_val = p.get('exit_tp')
            sl_str = f"{sl_val:.{prec}f}" if sl_val is not None else '---'
            tp_str = f"{tp_val:.{prec}f}" if tp_val is not None else '---'
            print(f" {Fore.LIGHTBLACK_EX}TARGETS:    {Fore.LIGHTBLACK_EX}ENTRY: {Fore.YELLOW}{ep:.{prec}f} │ SL: {Fore.RED}{sl_str} │ TP: {Fore.GREEN}{tp_str}")
            pnl_color = Fore.GREEN if t_pnl >= 0 else Fore.RED
            pnl_sign = "+" if t_pnl >= 0 else ""
            print(f" {Fore.LIGHTBLACK_EX}OPEN PNL:   {pnl_color}{pnl_sign}${t_pnl:.2f} USD")
        else:
            print(f" {Fore.LIGHTBLACK_EX}POSITIONS:  {Fore.LIGHTBLACK_EX}No active positions. Scanning orderbooks...")
            
        print(f"{Fore.LIGHTBLACK_EX}─"*70)
        
        # Sentiment bar
        bull_pct = p.get('bull_pct', 50)
        bear_pct = p.get('bear_pct', 50)
        bar_len = 30
        bull_chars = int((bull_pct / 100.0) * bar_len)
        bear_chars = bar_len - bull_chars
        meter_bar = f"{Fore.GREEN}{'█'*bull_chars}{Fore.RED}{'░'*bear_chars}"
        print(f" {Fore.WHITE}{Style.BRIGHT}PREDICTA V4 SENTIMENT: [{meter_bar}{Fore.WHITE}] {Fore.GREEN}{bull_pct}% BULL {Fore.WHITE}/ {Fore.RED}{bear_pct}% BEAR")
        print(f"{Fore.LIGHTBLACK_EX}─"*70)
        
        # Order Book
        order_book = p.get('order_book')
        if order_book and 'bids' in order_book and 'asks' in order_book:
            bids = order_book['bids']
            asks = order_book['asks']
            if bids and asks:
                total_bids = sum(b[1] for b in bids[:3])
                total_asks = sum(a[1] for a in asks[:3])
                sum_vol = total_bids + total_asks
                imbalance = ((total_bids - total_asks) / sum_vol * 100.0) if sum_vol > 0 else 0.0
                imb_color = Fore.GREEN if imbalance >= 0 else Fore.RED
                imb_sign = "+" if imbalance >= 0 else ""
                spread = asks[0][0] - bids[0][0]
                
                print(f" {Fore.LIGHTBLACK_EX}ORDER BOOK: Imbalance: {imb_color}{imb_sign}{imbalance:.1f}%{Fore.LIGHTBLACK_EX} │ Spread: {Fore.CYAN}${spread:.2f}")
                bid_str = f"{Fore.GREEN}${bids[0][0]:,.{prec}f} ({bids[0][1]:.3f} units)"
                ask_str = f"{Fore.RED}${asks[0][0]:,.{prec}f} ({asks[0][1]:.3f} units)"
                print(f"            Top Bid: {bid_str:<40} Top Ask: {ask_str}")
                print(f"{Fore.LIGHTBLACK_EX}─"*70)
                
        # Technical Indicator Matrix
        indicators = p.get('indicators', {})
        mom_status = "STRONG" if indicators.get('delta_mom') else "WEAK"
        mom_color = Fore.GREEN if mom_status == "STRONG" else Fore.RED
        
        ha_close_val = indicators.get('ha_close', 0.0)
        trend_val = indicators.get('trend_val')
        ema9_val = indicators.get('ema9', 0.0)
        ema21_val = indicators.get('ema21', 0.0)
        atr_val = indicators.get('atr', 0.0)
        
        trend_str = f"{trend_val:.{prec}f}" if trend_val is not None else "N/A"
        
        ind_table = [
            ["HA Close", f"{ha_close_val:.{prec}f}", "RSI (14)", f"{indicators.get('rsi', 0.0):.1f}"],
            ["200 HMA Trend", trend_str, "Stoch K/D", f"{indicators.get('stoch_k', 0.0):.1f}/{indicators.get('stoch_d', 0.0):.1f}"],
            ["EMA 9 / 21", f"{ema9_val:.{prec}f}/{ema21_val:.{prec}f}", "ADX / DI Diff", f"{indicators.get('adx', 0.0):.1f} | {indicators.get('di_diff', 0.0):+.1f}"],
            ["ATR (14)", f"{atr_val:.{prec}f}", "Vol Momentum", f"{mom_color}{mom_status}"]
        ]
        print(f"{Fore.CYAN}{Style.BRIGHT}  TECHNICAL INDICATORS MATRIX")
        print(tabulate(ind_table, headers=["Indicator", "Value", "Indicator", "Value"], tablefmt="plain"))
        print(f"{Fore.LIGHTBLACK_EX}─"*70)
        
        # Win rate / ledger
        wins = 0
        win_rate = 0.0
        total_trades = len(trade_ledger)
        if total_trades > 0:
            wins = sum(1 for t in trade_ledger if t.get('pnl', 0.0) > 0)
            win_rate = (wins / total_trades) * 100.0
            
        print(f" {Fore.LIGHTBLACK_EX}ACCOUNT:    {Fore.WHITE}Cash: {Fore.GREEN}${t_balance:,.2f} {Fore.WHITE}│ Total Valuation: {Fore.GREEN}${t_portfolio:,.2f}")
        print(f" {Fore.LIGHTBLACK_EX}WIN RATE:   {Fore.CYAN}{win_rate:.1f}% ({wins} wins / {total_trades} total)")
        print(f"{Fore.LIGHTBLACK_EX}─"*70)
        
        # Recent executed transactions
        print(f"{Fore.MAGENTA}{Style.BRIGHT}  RECENT EXECUTED TRANSACTIONS")
        if trade_ledger:
            headers = ["Exit Time", "Type", "Symbol", "Size", "Entry", "Exit", "PnL ($)"]
            ledger_rows = []
            for t in trade_ledger[-3:]:
                pnl = t.get('pnl', 0.0)
                pnl_str = f"{Fore.GREEN}{pnl:+.2f}" if pnl > 0 else f"{Fore.RED}{pnl:.2f}"
                row_type = f"{Fore.GREEN}LONG" if t.get('type') == "LONG" else f"{Fore.RED}SHORT"
                coin = t.get('ticker', 'BTC/USDT').split('/')[0]
                size_str = f"{t.get('position_size', 0.0)} {coin}"
                t_ticker = t.get('ticker', 'BTC/USDT')
                t_is_major = t_ticker.startswith("BTC/") or t_ticker.startswith("ETH/")
                t_prec = 2 if t_is_major else (3 if t.get('entry_price', 0.0) < 200 else 2)
                
                ledger_rows.append([
                    t.get('exit_time', '').split(' ')[-1],
                    row_type,
                    t_ticker,
                    size_str,
                    f"{t.get('entry_price', 0.0):.{t_prec}f}",
                    f"{t.get('exit_price', 0.0):.{t_prec}f}",
                    pnl_str
                ])
            print(tabulate(ledger_rows, headers=headers, tablefmt="plain"))
        else:
            print(f"  {Fore.LIGHTBLACK_EX}No completed trades in this session yet.")
            
        print(f"{Fore.CYAN}{Style.BRIGHT}└" + "─"*68 + "┘")
        print(f" {Fore.LIGHTBLACK_EX}Console ticking every {ticking_interval} seconds. Press Ctrl+C to terminate.")
