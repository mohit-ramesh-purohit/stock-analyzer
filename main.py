from fastapi import FastAPI
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands
import logging

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockRequest(BaseModel):
    tickers: list

@app.post("/analyze")
def analyze_stocks(request: StockRequest):
    results = []
    for symbol in request.tickers:
        try:
            logger.info(f"Analyzing {symbol}")
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1mo", interval="1d")
            
            if hist is None or hist.empty or len(hist) < 14:
                logger.warning(f"Insufficient data for {symbol}")
                continue

            # Indicators
            hist["RSI"] = RSIIndicator(close=hist["Close"], window=14).rsi()
            hist["SMA_50"] = SMAIndicator(close=hist["Close"], window=50).sma_indicator()
            hist["SMA_200"] = SMAIndicator(close=hist["Close"], window=200).sma_indicator()
            hist["MACD"] = MACD(close=hist["Close"]).macd()
            hist["ADX"] = ADXIndicator(high=hist["High"], low=hist["Low"], close=hist["Close"]).adx()
            bb = BollingerBands(close=hist["Close"])
            hist["BB_upper"] = bb.bollinger_hband()
            hist["BB_lower"] = bb.bollinger_lband()

            latest = hist.iloc[-1]
            info = stock.info

            signal = "HOLD"
            if latest["RSI"] < 30 and latest["MACD"] > 0 and latest["ADX"] > 20:
                signal = "BUY"
            elif latest["RSI"] > 70 and latest["MACD"] < 0:
                signal = "SELL"

            results.append({
                "name": info.get("shortName", symbol),
                "symbol": symbol,
                "price": info.get("currentPrice"),
                "pe_ratio": info.get("trailingPE"),
                "rsi": round(latest["RSI"], 2),
                "sma_50": round(latest["SMA_50"], 2) if not pd.isna(latest["SMA_50"]) else None,
                "sma_200": round(latest["SMA_200"], 2) if not pd.isna(latest["SMA_200"]) else None,
                "macd": round(latest["MACD"], 2) if not pd.isna(latest["MACD"]) else None,
                "adx": round(latest["ADX"], 2) if not pd.isna(latest["ADX"]) else None,
                "bb_upper": round(latest["BB_upper"], 2) if not pd.isna(latest["BB_upper"]) else None,
                "bb_lower": round(latest["BB_lower"], 2) if not pd.isna(latest["BB_lower"]) else None,
                "signal": signal
            })
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            continue
    return {"analysis": results}