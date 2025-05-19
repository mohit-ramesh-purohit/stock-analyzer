from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands

app = FastAPI()

class StockRequest(BaseModel):
    tickers: List[str]

@app.post("/analyze")
def analyze_stocks(request: StockRequest):
    results = []

    for symbol in request.tickers:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1mo", interval="1d")

            if hist.empty or len(hist) < 20:
                continue

            hist["RSI"] = RSIIndicator(close=hist["Close"], window=14).rsi()
            hist["SMA_50"] = SMAIndicator(close=hist["Close"], window=50).sma_indicator()
            hist["SMA_200"] = SMAIndicator(close=hist["Close"], window=200).sma_indicator()
            macd = MACD(close=hist["Close"])
            hist["MACD"] = macd.macd()
            hist["MACD_signal"] = macd.macd_signal()
            hist["ADX"] = ADXIndicator(high=hist["High"], low=hist["Low"], close=hist["Close"]).adx()
            bollinger = BollingerBands(close=hist["Close"])
            hist["bb_bbm"] = bollinger.bollinger_mavg()
            hist["bb_bbh"] = bollinger.bollinger_hband()
            hist["bb_bbl"] = bollinger.bollinger_lband()
            hist["Volume"] = hist["Volume"]

            latest = hist.iloc[-1]
            price = stock.info.get("currentPrice", None)
            pe_ratio = stock.info.get("trailingPE", None)
            name = stock.info.get("shortName", symbol)

            rsi_score = 1 if latest["RSI"] < 30 else -1 if latest["RSI"] > 70 else 0
            macd_score = 1 if latest["MACD"] > latest["MACD_signal"] else -1
            adx_score = 1 if latest["ADX"] > 25 else 0
            bb_score = 1 if latest["Close"] < latest["bb_bbl"] else -1 if latest["Close"] > latest["bb_bbh"] else 0
            sma_score = 1 if latest["SMA_50"] > latest["SMA_200"] else -1
            total_score = rsi_score + macd_score + adx_score + bb_score + sma_score

            if total_score >= 3:
                signal = "STRONG BUY"
            elif total_score >= 1:
                signal = "BUY"
            elif total_score <= -2:
                signal = "SELL"
            else:
                signal = "HOLD"

            results.append({
                "name": name,
                "symbol": symbol,
                "price": price,
                "pe_ratio": pe_ratio,
                "rsi": round(latest["RSI"], 2),
                "macd": round(latest["MACD"], 2),
                "macd_signal": round(latest["MACD_signal"], 2),
                "adx": round(latest["ADX"], 2),
                "bb_upper": round(latest["bb_bbh"], 2),
                "bb_lower": round(latest["bb_bbl"], 2),
                "sma_50": round(latest["SMA_50"], 2),
                "sma_200": round(latest["SMA_200"], 2),
                "volume": int(latest["Volume"]),
                "strategy_score": total_score,
                "signal": signal
            })

        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            continue

    return {"analysis": results}