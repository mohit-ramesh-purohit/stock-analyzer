from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands

app = FastAPI()

class StockRequest(BaseModel):
    tickers: List[str]

@app.post("/analyze")
def analyze_stocks(request: StockRequest):
    results = []

    for symbol in request.tickers:
        try:
            print(f"Analyzing {symbol}")
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1mo", interval="1d")

            # Skip if there's not enough data for technical indicators
            if hist.empty or len(hist) < 14:
                print(f"Insufficient data for {symbol}")
                continue

            # RSI Calculation
            rsi_indicator = RSIIndicator(close=hist["Close"], window=14)
            hist["RSI"] = rsi_indicator.rsi()
            latest_rsi = hist["RSI"].dropna().iloc[-1]

            # MACD Calculation
            macd_indicator = MACD(close=hist["Close"])
            hist["MACD"] = macd_indicator.macd()
            hist["MACD_SIGNAL"] = macd_indicator.macd_signal()
            macd_cross = hist["MACD"].iloc[-1] > hist["MACD_SIGNAL"].iloc[-1]

            # ADX Calculation
            adx = ADXIndicator(high=hist["High"], low=hist["Low"], close=hist["Close"])
            latest_adx = adx.adx().iloc[-1]

            # Bollinger Bands
            bb = BollingerBands(close=hist["Close"])
            bb_upper = bb.bollinger_hband().iloc[-1]
            bb_lower = bb.bollinger_lband().iloc[-1]
            bb_mid = bb.bollinger_mavg().iloc[-1]
            price = hist["Close"].iloc[-1]

            # Signal logic
            signal = "HOLD"
            if latest_rsi < 30 and macd_cross and price < bb_mid:
                signal = "BUY"
            elif latest_rsi > 70 and not macd_cross and price > bb_mid:
                signal = "SELL"

            info = stock.info
            results.append({
                "name": info.get("shortName", symbol),
                "symbol": symbol,
                "price": round(info.get("currentPrice", price), 2),
                "pe_ratio": round(info.get("trailingPE", 0), 2),
                "rsi": round(latest_rsi, 2),
                "adx": round(latest_adx, 2),
                "macd_cross": macd_cross,
                "bollinger_upper": round(bb_upper, 2),
                "bollinger_lower": round(bb_lower, 2),
                "signal": signal
            })

        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            continue

    return {"analysis": results}