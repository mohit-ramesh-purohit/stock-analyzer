from fastapi import FastAPI
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, MACD, ADXIndicator
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
            if hist.empty or len(hist) < 50:
                continue

            close = hist["Close"]
            volume = hist["Volume"]

            # Indicators
            rsi = RSIIndicator(close).rsi().iloc[-1]
            sma50 = SMAIndicator(close, window=50).sma_indicator().iloc[-1]
            sma200 = SMAIndicator(close, window=200).sma_indicator().iloc[-1]
            macd_line = MACD(close).macd().iloc[-1]
            macd_signal = MACD(close).macd_signal().iloc[-1]
            adx = ADXIndicator(high=hist["High"], low=hist["Low"], close=close).adx().iloc[-1]
            bb = BollingerBands(close)
            bb_width = (bb.bollinger_hband() - bb.bollinger_lband()).iloc[-1]
            volume_surge = volume.iloc[-1] > 1.5 * volume.rolling(window=20).mean().iloc[-1]

            # Scoring logic
            score = 0
            if rsi < 35: score += 20
            if sma50 > sma200: score += 20
            if macd_line > macd_signal: score += 20
            if adx > 25: score += 15
            if bb_width < 0.05 * close.iloc[-1]: score += 10
            if volume_surge: score += 15

            signal = "BUY" if score >= 75 else "HOLD" if score >= 50 else "AVOID"

            results.append({
                "symbol": symbol,
                "price": round(close.iloc[-1], 2),
                "rsi": round(rsi, 2),
                "sma50": round(sma50, 2),
                "sma200": round(sma200, 2),
                "macd": round(macd_line, 2),
                "macd_signal": round(macd_signal, 2),
                "adx": round(adx, 2),
                "bb_width": round(bb_width, 2),
                "volume_surge": volume_surge,
                "strategy_score": score,
                "signal": signal
            })
        except Exception as e:
            results.append({
                "symbol": symbol,
                "error": str(e)
            })

    return {"analysis": results}