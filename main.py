# main.py
import argparse
import asyncio
import sys
import warnings
# Suppress dependency and version warning logs (e.g. yfinance curl_cffi alerts)
warnings.filterwarnings("ignore")

import uvicorn
from config import TalksyConfig
from src.data_source import DataSourceManager
from src.bot import TalksyBot
from src.dashboard import Dashboard
from src.web_server import app, broadcast_telemetry

async def main_loop(sim_mode: bool):
    # Load configuration
    config = TalksyConfig()
    
    # Override simulation mode from CLI args if specified
    if sim_mode:
        config.SIMULATION_MODE = True
        interval = config.SIMULATION_TICK_SECS
    else:
        interval = config.TICKING_INTERVAL_SECS
        
    print(f"[SYSTEM] Starting Onyx Bot...")
    print(f"[SYSTEM] Mode: {'SIMULATION (Accelerated)' if config.SIMULATION_MODE else 'LIVE (Standard 3m ticks)'}")
    print(f"[SYSTEM] Primary Ticker: {config.TICKER_CCXT} | Timeframe: {config.TIMEFRAME_PRIMARY}")
    print(f"[SYSTEM] Ticking Interval: {interval} seconds")
    print(f"[SYSTEM] Local Web Dashboard: http://{config.WEB_HOST}:{config.WEB_PORT}")
    await asyncio.sleep(1.5)
    
    # Initialize components
    data_manager = DataSourceManager(
        simulation_mode=config.SIMULATION_MODE,
        exchange_id=config.EXCHANGE_ID,
        tickers=config.TICKERS,
        timeframe=config.TIMEFRAME_PRIMARY,
        lookback_days=config.LOOKBACK_DAYS
    )
    
    bot = TalksyBot(
        config=config,
        simulation_mode=config.SIMULATION_MODE
    )
    bot.data_manager = data_manager
    
    # Inject bot instance into FastAPI app state
    app.state.bot = bot
    
    # Configure and start programmatic Uvicorn web server
    uvicorn_config = uvicorn.Config(
        app=app,
        host=config.WEB_HOST,
        port=config.WEB_PORT,
        log_level="warning",
        ws_ping_interval=20,
        ws_ping_timeout=20
    )
    server = uvicorn.Server(uvicorn_config)
    server_task = asyncio.create_task(server.serve())
    
    try:
        while True:
            # 1. Fetch historical and active candle data for all tickers
            try:
                dfs = await data_manager.fetch_candles()
                books = await data_manager.fetch_order_books(limit=5)
            except Exception as e:
                print(f"\n[ERROR] Ingestion loop connection failure: {e}. Retrying next tick...")
                await asyncio.sleep(interval)
                continue
                
            # 2. Feed data into state machine
            payloads = await bot.process_tick(dfs, books)
            
            # 3. Broadcast telemetry update to Web UI WebSockets
            try:
                await broadcast_telemetry(payloads)
            except Exception as e:
                print(f"[WARN] Failed to broadcast telemetry: {e}")
            
            # 4. Render local dashboard console overview as well
            Dashboard.render(
                payloads=payloads,
                focused_ticker=config.TICKER,
                trade_ledger=bot.trade_ledger,
                ticking_interval=interval
            )
            
            # 5. Wait for the next tick
            await asyncio.sleep(interval)
            
    except asyncio.CancelledError:
        print("\n[SYSTEM] Loop execution cancelled.")
    finally:
        # Stop Uvicorn server task
        server.should_exit = True
        await asyncio.sleep(0.5)
        if not server_task.done():
            server_task.cancel()
            
        # Clean data manager connection
        await data_manager.close()
        
        # Summary report
        print("\n" + "="*50)
        print("  ONYX - BOT TERMINATION REPORT")
        print("="*50)
        print(f" Final Account Balance: ${bot.balance:,.2f} USD")
        pnl_session = bot.balance - config.INITIAL_BALANCE
        pnl_color = "+" if pnl_session >= 0 else ""
        print(f" Session Net Return:    {pnl_color}${pnl_session:,.2f} USD ({pnl_color}{(pnl_session/config.INITIAL_BALANCE)*100.0:.2f}%)")
        print(f" Total Closed Trades:   {len(bot.trade_ledger)}")
        print(f" Saved Trade Ledger to: {config.LEDGER_FILE}")
        print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Onyx - Web Trading Desk Console")
    parser.add_argument(
        "--sim", 
        action="store_true", 
        help="Run in accelerated simulation mode (updates every few seconds with synthetic candles)"
    )
    args = parser.parse_args()
    
    try:
        asyncio.run(main_loop(sim_mode=args.sim))
    except KeyboardInterrupt:
        print("\n[SYSTEM] Keyboard Interrupt received. Shutting down gracefully...")
        sys.exit(0)
