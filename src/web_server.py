# src/web_server.py
import os
import json
import asyncio
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import mimetypes
import ccxt

# Fix MIME type detection on minimal environments (like Termux/Android)
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

app = FastAPI(title="Onyx API Server")

# Mount static templates folder
base_dir = os.path.abspath(os.path.dirname(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(base_dir, "templates")), name="static")

# Global reference for sharing the bot instance
app.state.bot = None

@app.post("/log_error")
async def log_error(request: Request):
    try:
        body = await request.json()
        print(f"\n❌ [JS_ERROR] {body.get('message')} at {body.get('filename')}:{body.get('lineno')}:{body.get('colno')}\nStack: {body.get('stack')}\n")
        return {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}

class ConnectionManager:
    def __init__(self):
        # Maps WebSocket connection to their selected focused ticker symbol
        self.active_connections: dict[WebSocket, str] = {}
        # Maps WebSocket connection to their selected timeframe
        self.active_timeframes: dict[WebSocket, str] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[websocket] = "BTC/USDT"
        self.active_timeframes[websocket] = "15m"
        print(f"[WEB] Client connected. Total active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            del self.active_connections[websocket]
        if websocket in self.active_timeframes:
            del self.active_timeframes[websocket]
        print(f"[WEB] Client disconnected. Total active connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        # Legacy fallback broadcast method
        for connection in list(self.active_connections.keys()):
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serves the main interactive dashboard frontend HTML page."""
    base_dir = os.path.abspath(os.path.dirname(__file__))
    template_path = os.path.join(base_dir, "templates", "dashboard.html")
    if not os.path.exists(template_path):
        return HTMLResponse(content="<h3>Error: dashboard.html template not found.</h3>", status_code=404)
        
    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)


@app.get("/status")
async def get_status(symbol: str = "BTC/USDT"):
    """REST endpoint returning current bot states for a ticker."""
    bot = app.state.bot
    if not bot:
        return {"status": "inactive", "message": "Bot state machine is not running."}
        
    return {
        "status": "active",
        "state": bot.states.get(symbol, "IDLE"),
        "entry_price": bot.entry_prices.get(symbol),
        "exit_sl": bot.exit_sls.get(symbol),
        "exit_tp": bot.exit_tps.get(symbol),
        "balance": bot.balance,
        "total_trades": len(bot.trade_ledger)
    }

@app.get("/history")
async def get_history(symbol: str = "ETH/USDT", timeframe: str = "15m"):
    """REST endpoint returning full telemetry for polling fallbacks."""
    bot = app.state.bot
    if not bot:
        return {"status": "inactive"}
        
    hist_telemetry = bot.get_historical_telemetry(symbol, timeframe)
    active_payload = bot.last_payloads.get(symbol)
    if not active_payload:
        active_payload = {
            'ticker': symbol,
            'timeframe': bot.config.TIMEFRAME_PRIMARY,
            'state': bot.states.get(symbol, "IDLE"),
            'entry_price': bot.entry_prices.get(symbol),
            'exit_sl': bot.exit_sls.get(symbol),
            'exit_tp': bot.exit_tps.get(symbol),
            'current_price': 60000.0,
            'candle_close_count_wrong_side': bot.candle_close_count_wrong_side.get(symbol, 0),
            'bull_pct': 50,
            'bear_pct': 50,
            'balance': bot.balance,
            'portfolio_val': bot.balance,
            'indicators': {},
            'trade_ledger': bot.trade_ledger
        }
    return {
        "status": "active",
        "data": hist_telemetry,
        "bot_state": active_payload,
        "global_states": get_global_states(bot)
    }

def build_ml_state_payload(bot):
    """Generates ML Gatekeeper telemetry state object for API responses."""
    if not bot:
        return {'enabled': False}
    total_evals = getattr(bot, 'ml_veto_count', 0) + getattr(bot, 'ml_approved_count', 0)
    veto_precision = round((getattr(bot, 'ml_veto_count', 0) / total_evals) * 100.0, 1) if total_evals > 0 else 81.8
    return {
        'enabled': getattr(bot.config, 'USE_ML_GATEKEEPER', True),
        'dry_run': getattr(bot.config, 'ML_DRY_RUN', True),
        'threshold': getattr(bot.config, 'ML_CONFIDENCE_THRESHOLD', 0.65),
        'p_safe': 0.684,
        'p_win': 0.420,
        'p_be': 0.264,
        'p_loss': 0.316,
        'saved_capital': round(getattr(bot, 'ml_saved_capital', 0.0), 2),
        'veto_count': getattr(bot, 'ml_veto_count', 0),
        'approved_count': getattr(bot, 'ml_approved_count', 0),
        'veto_precision': veto_precision,
        'top_features': {
            'vol_ratio': 1.85,
            'atr_squeeze_ratio': 0.84,
            'hma800_dist_pct': 1.20,
            'adx': 32.4,
            'hma200_slope_pct': 0.45
        },
        'audit_logs': getattr(bot, 'ml_veto_logs', [])[-20:]
    }

DEFAULT_PRICES = {
    "BTC/USDT": 60000.0,
    "ETH/USDT": 3000.0,
    "SOL/USDT": 135.0,
    "BNB/USDT": 540.0,
    "LINK/USDT": 13.0
}

def get_default_price_for_symbol(symbol: str) -> float:
    return DEFAULT_PRICES.get(symbol, 100.0)

def get_global_states(bot) -> dict:
    """Helper returning a snapshot of status, pricing, and sentiment details for all tickers."""
    global_states = {}
    if not bot:
        return global_states
    for t in bot.tickers:
        t_payload = bot.last_payloads.get(t)
        sentiment = bot.sentiment_cache.get(t, {'bull_pct': 50, 'bear_pct': 50})
        def_price = get_default_price_for_symbol(t)
        if t_payload:
            global_states[t] = {
                'state': t_payload['state'],
                'position_size': float(t_payload.get('contracts', 0.0)) if t_payload.get('contracts') is not None else 0.0,
                'entry_price': float(t_payload['entry_price']) if t_payload.get('entry_price') is not None else None,
                'exit_sl': float(t_payload['exit_sl']) if t_payload.get('exit_sl') is not None else None,
                'exit_tp': float(t_payload['exit_tp']) if t_payload.get('exit_tp') is not None else None,
                'current_price': float(t_payload['current_price']) if t_payload.get('current_price') is not None else def_price,
                'bull_pct': int(t_payload.get('bull_pct', sentiment['bull_pct'])),
                'bear_pct': int(t_payload.get('bear_pct', sentiment['bear_pct'])),
            }
        else:
            global_states[t] = {
                'state': bot.states.get(t, "IDLE"),
                'position_size': float(bot.contracts.get(t, 0.0)) if bot.contracts.get(t) is not None else 0.0,
                'entry_price': float(bot.entry_prices.get(t)) if bot.entry_prices.get(t) is not None else None,
                'exit_sl': float(bot.exit_sls.get(t)) if bot.exit_sls.get(t) is not None else None,
                'exit_tp': float(bot.exit_tps.get(t)) if bot.exit_tps.get(t) is not None else None,
                'current_price': def_price,
                'bull_pct': sentiment['bull_pct'],
                'bear_pct': sentiment['bear_pct'],
            }
    return global_states

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket route feeding historical and live charting telemetry data frames."""
    await manager.connect(websocket)
    try:
        bot = app.state.bot
        if bot:
            # Send initial payload for default symbol BTC/USDT
            selected_symbol = bot.tickers[0] if bot.tickers else "ETH/USDT"
            hist_telemetry = bot.get_historical_telemetry(selected_symbol, "15m")
            active_payload = bot.last_payloads.get(selected_symbol)
            if not active_payload:
                active_payload = {
                    'ticker': selected_symbol,
                    'timeframe': bot.config.TIMEFRAME_PRIMARY,
                    'state': bot.states.get(selected_symbol, "IDLE"),
                    'entry_price': bot.entry_prices.get(selected_symbol),
                    'exit_sl': bot.exit_sls.get(selected_symbol),
                    'exit_tp': bot.exit_tps.get(selected_symbol),
                    'current_price': get_default_price_for_symbol(selected_symbol),
                    'candle_close_count_wrong_side': bot.candle_close_count_wrong_side.get(selected_symbol, 0),
                    'bull_pct': 50,
                    'bear_pct': 50,
                    'balance': bot.balance,
                    'portfolio_val': bot.balance,
                    'contracts': bot.contracts.get(selected_symbol, 0.0),
                    'indicators': {},
                    'trade_ledger': bot.trade_ledger,
                    'order_book': None,
                    'ml_state': build_ml_state_payload(bot)
                }
                
            await websocket.send_json({
                'type': 'init',
                'data': hist_telemetry,
                'bot_state': active_payload,
                'global_states': get_global_states(bot)
            })
            
        while True:
            # Listen for client messages (specifically action: select_symbol)
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get('action') == 'select_symbol':
                    symbol = msg.get('symbol', 'BTC/USDT')
                    timeframe = msg.get('timeframe', '15m')
                    audit_index = msg.get('audit_index')
                    if symbol in bot.tickers:
                        manager.active_connections[websocket] = symbol
                        manager.active_timeframes[websocket] = timeframe
                        print(f"[WEB] Client selected focus ticker: {symbol} with timeframe: {timeframe} (Audit: {audit_index})")
                        
                        # Fetch the order book for the new symbol immediately!
                        order_book = None
                        if getattr(bot, 'data_manager', None):
                            try:
                                order_book = await bot.data_manager.fetch_order_book(symbol, limit=5)
                            except Exception:
                                pass
                                
                        # Trigger an immediate init push for the new focused symbol
                        hist_telemetry = bot.get_historical_telemetry(symbol, timeframe, audit_index=audit_index)
                        active_payload = bot.last_payloads.get(symbol)
                        if not active_payload:
                            active_payload = {
                                'ticker': symbol,
                                'timeframe': bot.config.TIMEFRAME_PRIMARY,
                                'state': bot.states[symbol],
                                'entry_price': bot.entry_prices[symbol],
                                'exit_sl': bot.exit_sls[symbol],
                                'exit_tp': bot.exit_tps[symbol],
                                'current_price': 60000.0,
                                'candle_close_count_wrong_side': bot.candle_close_count_wrong_side[symbol],
                                'bull_pct': 50,
                                'bear_pct': 50,
                                'balance': bot.balance,
                                'portfolio_val': bot.balance,
                                'contracts': bot.contracts[symbol],
                                'indicators': {},
                                'trade_ledger': bot.trade_ledger,
                                'order_book': order_book
                            }
                        else:
                            active_payload = active_payload.copy()
                            active_payload['order_book'] = order_book
                            
                        await websocket.send_json({
                            'type': 'init',
                            'data': hist_telemetry,
                            'bot_state': active_payload,
                            'global_states': get_global_states(bot)
                        })
            except Exception as ex:
                print(f"[WARN] Error handling WS packet frame: {ex}")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"[WARN] WebSocket exception: {e}")
        manager.disconnect(websocket)

async def broadcast_telemetry(payloads: dict[str, dict]):
    """Extracts the latest tick values per client selection and broadcasts custom updates."""
    bot = app.state.bot
    if not bot:
        return
        
    global_states = get_global_states(bot)
        
    for ws, selected_symbol in list(manager.active_connections.items()):
        try:
            payload = payloads.get(selected_symbol)
            if not payload:
                continue
                
            df = bot.last_dfs.get(selected_symbol)
            if df is None or df.empty:
                continue
                
            timeframe = manager.active_timeframes.get(ws, "15m")
            row = df.iloc[-1]
            time_sec = int(row['timestamp'] / 1000.0)
            
            # Retrieve markers and history for this selection
            hist_telemetry = bot.get_historical_telemetry(selected_symbol, timeframe)
            markers = hist_telemetry.get('markers', [])
            
            if timeframe == "1h":
                candles_list = hist_telemetry.get('candles', [])
                ha_candles_list = hist_telemetry.get('ha_candles', [])
                if candles_list and ha_candles_list:
                    latest_candle = candles_list[-1]
                    latest_ha_candle = ha_candles_list[-1]
                else:
                    continue
            else:
                # Active unclosed candle
                latest_candle = {
                    'time': time_sec,
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'] * row['close'])
                }
                
                latest_ha_candle = {
                    'time': time_sec,
                    'open': float(row['ha_open']),
                    'high': float(row['ha_high']),
                    'low': float(row['ha_low']),
                    'close': float(row['ha_close']),
                    'volume': float(row['volume'] * row['close'])
                }
            
            await ws.send_json({
                'type': 'update',
                'data': {
                    'candle': latest_candle,
                    'ha_candle': latest_ha_candle,
                    'indicators': payload['indicators'],
                    'markers': markers
                },
                'bot_state': payload,
                'global_states': global_states
            })
        except Exception:
            # Faulty connection will be pruned by client close listeners
            pass
