from fastapi import FastAPI
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

app = FastAPI()

class StockRequest(BaseModel):
    tickers: list

@app.post("/analyze")
def analyze_stocks(request: StockRequest):
    results = []

    for symbol in request.tickers:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="6mo", interval="1d")

            if hist.empty or len(hist) < 200:
                continue

            hist["SMA_50"] = SMAIndicator(close=hist["Close"], window=50).sma_indicator()
            hist["SMA_200"] = SMAIndicator(close=hist["Close"], window=200).sma_indicator()
            macd_indicator = MACD(close=hist["Close"])
            hist["MACD"] = macd_indicator.macd_diff()
            hist["MACD_line"] = macd_indicator.macd()
            hist["MACD_signal"] = macd_indicator.macd_signal()
            hist["ADX"] = ADXIndicator(high=hist["High"], low=hist["Low"], close=hist["Close"]).adx()
            hist["RSI"] = RSIIndicator(close=hist["Close"], window=14).rsi()
            bb = BollingerBands(close=hist["Close"], window=20, window_dev=2)
            hist["BB_upper"] = bb.bollinger_hband()
            hist["BB_lower"] = bb.bollinger_lband()
            hist["BB_width"] = hist["BB_upper"] - hist["BB_lower"]

            latest = hist.iloc[-1]
            info = stock.info

            if not info or "currentPrice" not in info:
                continue

            # Strategy scoring
            strategy_score = 0
            rationale = []

            if latest["SMA_50"] > latest["SMA_200"]:
                strategy_score += 15
                rationale.append("Bullish SMA crossover")
            if latest["MACD"] > 0:
                strategy_score += 10
                rationale.append("MACD above signal")
            if latest["ADX"] > 25:
                strategy_score += 10
                rationale.append("Strong trend (ADX > 25)")
            if latest["RSI"] < 30:
                strategy_score += 15
                rationale.append("RSI in oversold zone")
            elif latest["RSI"] > 70:
                strategy_score -= 10
                rationale.append("RSI in overbought zone")
            if latest["Close"] < latest["BB_lower"]:
                strategy_score += 10
                rationale.append("Price near lower Bollinger Band")
            if hist["Volume"].iloc[-1] > hist["Volume"].rolling(window=20).mean().iloc[-1]:
                strategy_score += 10
                rationale.append("Volume spike detected")
            if latest["BB_width"] < 0.05 * latest["Close"]:
                strategy_score += 10
                rationale.append("Bollinger Band squeeze")

            signal = "HOLD"
            if strategy_score >= 60:
                signal = "BUY"
            elif strategy_score <= 20:
                signal = "SELL"

            results.append({
                "name": info.get("shortName", symbol),
                "symbol": symbol,
                "price": info.get("currentPrice"),
                "pe_ratio": info.get("trailingPE"),
                "sector": info.get("sector"),
                "sma_50": round(latest["SMA_50"], 2),
                "sma_200": round(latest["SMA_200"], 2),
                "macd": round(latest["MACD"], 2),
                "macd_line": round(latest["MACD_line"], 2),
                "macd_signal": round(latest["MACD_signal"], 2),
                "adx": round(latest["ADX"], 2),
                "rsi": round(latest["RSI"], 2),
                "bb_upper": round(latest["BB_upper"], 2),
                "bb_lower": round(latest["BB_lower"], 2),
                "bb_width": round(latest["BB_width"], 2),
                "volume": int(hist["Volume"].iloc[-1]),
                "signal": signal,
                "strategy_score": strategy_score,
                "rationale": rationale
            })

        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue

    return {"analysis": results}
