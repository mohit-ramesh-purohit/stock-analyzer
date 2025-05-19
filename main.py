from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import yfinance as yf
import pandas as pd
import logging
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands
from ta.trend import ADXIndicator

app = FastAPI()

logging.basicConfig(level=logging.INFO)

class StockRequest(BaseModel):
    tickers: List[str]

@app.post("/analyze")
def analyze_stocks(request: StockRequest):
    results = []

    for symbol in request.tickers:
        try:
            logging.info(f"Analyzing {symbol}")
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1mo", interval="1d")

            if hist.empty or len(hist) < 20:
                logging.warning(f"Not enough data to analyze {symbol}")
                continue

            hist.dropna(inplace=True)

            # Indicators
            rsi = RSIIndicator(close=hist["Close"], window=14).rsi()
            macd_line = MACD(close=hist["Close"]).macd()
            sma_50 = SMAIndicator(close=hist["Close"], window=50).sma_indicator()
            sma_200 = SMAIndicator(close=hist["Close"], window=200).sma_indicator()
            adx = ADXIndicator(high=hist["High"], low=hist["Low"], close=hist["Close"]).adx()
            bb = BollingerBands(close=hist["Close"], window=20)
            upper_band = bb.bollinger_hband()
            lower_band = bb.bollinger_lband()
            volume = hist["Volume"]

            # Current values
            latest_rsi = round(rsi.iloc[-1], 2)
            latest_macd = round(macd_line.iloc[-1], 2)
            latest_adx = round(adx.iloc[-1], 2)
            current_price = round(hist["Close"].iloc[-1], 2)
            avg_volume = int(volume.mean())

            info = stock.info

            def determine_signal():
                if latest_rsi < 30 and latest_macd > 0 and current_price < lower_band.iloc[-1]:
                    return "BUY"
                elif latest_rsi > 70 or current_price > upper_band.iloc[-1]:
                    return "SELL"
                else:
                    return "HOLD"

            results.append({
                "name": info.get("shortName", symbol),
                "symbol": symbol,
                "price": current_price,
                "pe_ratio": info.get("trailingPE", None),
                "rsi": latest_rsi,
                "macd": latest_macd,
                "adx": latest_adx,
                "volume_avg": avg_volume,
                "signal": determine_signal()
            })

        except Exception as e:
            logging.error(f"Error analyzing {symbol}: {e}")
            continue

    return {"analysis": results}