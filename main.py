from fastapi import FastAPI
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

app = FastAPI()

class StockRequest(BaseModel):
    tickers: list

@app.post("/analyze")
def analyze_stocks(request: StockRequest):
    results = []
    
    for symbol in request.tickers:
        try:
            print(f"Analyzing {symbol}")
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1mo", interval="1d")
            if hist.empty:
                print(f"No data for {symbol}, skipping.")
                continue

            hist.dropna(inplace=True)

            # Add indicators
            hist["RSI"] = RSIIndicator(close=hist["Close"]).rsi()
            hist["SMA_50"] = SMAIndicator(close=hist["Close"], window=50).sma_indicator()
            hist["SMA_200"] = SMAIndicator(close=hist["Close"], window=200).sma_indicator()
            hist["MACD"] = MACD(close=hist["Close"]).macd_diff()
            hist["BB_upper"] = BollingerBands(close=hist["Close"]).bollinger_hband()
            hist["BB_lower"] = BollingerBands(close=hist["Close"]).bollinger_lband()
            hist["ADX"] = ADXIndicator(high=hist["High"], low=hist["Low"], close=hist["Close"]).adx()
            hist["OBV"] = OnBalanceVolumeIndicator(close=hist["Close"], volume=hist["Volume"]).on_balance_volume()

            latest = hist.iloc[-1]
            info = stock.info

            result = {
                "name": info.get("shortName", symbol),
                "symbol": symbol,
                "price": latest["Close"],
                "pe_ratio": info.get("trailingPE", "N/A"),
                "rsi": round(latest["RSI"], 2),
                "sma_50": round(latest["SMA_50"], 2) if pd.notna(latest["SMA_50"]) else None,
                "sma_200": round(latest["SMA_200"], 2) if pd.notna(latest["SMA_200"]) else None,
                "macd": round(latest["MACD"], 2) if pd.notna(latest["MACD"]) else None,
                "bollinger_upper": round(latest["BB_upper"], 2) if pd.notna(latest["BB_upper"]) else None,
                "bollinger_lower": round(latest["BB_lower"], 2) if pd.notna(latest["BB_lower"]) else None,
                "adx": round(latest["ADX"], 2) if pd.notna(latest["ADX"]) else None,
                "volume": int(latest["Volume"]),
                "signal": (
                    "BUY" if latest["RSI"] < 30 else
                    "SELL" if latest["RSI"] > 70 else
                    "HOLD"
                )
            }

            results.append(result)
        
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            continue

    return {"analysis": results}