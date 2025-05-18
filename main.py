from fastapi import FastAPI
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator

app = FastAPI()

class StockRequest(BaseModel):
    tickers: list

@app.post("/analyze")
def analyze_stocks(request: StockRequest):
    results = []
    for symbol in request.tickers:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="10mo", interval="1d")
            if hist.empty:
                continue

            rsi = RSIIndicator(close=hist["Close"], window=14)
            hist["RSI"] = rsi.rsi()
            latest_rsi = hist["RSI"].iloc[-1]
            info = stock.info

            if not info or "currentPrice" not in info:
                continue

            results.append({
                "name": info.get("shortName", symbol),
                "symbol": symbol,
                "price": info.get("currentPrice"),
                "pe_ratio": info.get("trailingPE"),
                "rsi": round(latest_rsi, 2),
                "signal": (
                    "BUY" if latest_rsi < 30 else
                    "SELL" if latest_rsi > 70 else
                    "HOLD"
                )
            })
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue
    return {"analysis": results}