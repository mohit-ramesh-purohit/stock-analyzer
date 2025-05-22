from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import yfinance as yf
import pandas as pd
import numpy as np
import logging
from scipy.stats import median_abs_deviation

app = FastAPI()
logging.basicConfig(level=logging.INFO)

class StockRequest(BaseModel):
    tickers: List[str]

def calculate_indicators(df):
    if len(df) < 50:
        raise ValueError("Not enough data to calculate indicators")

    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["Bollinger_Upper"] = df["EMA_20"] + 2 * df["Close"].rolling(window=20).std()
    df["Bollinger_Lower"] = df["EMA_20"] - 2 * df["Close"].rolling(window=20).std()
    df["BBW"] = (df["Bollinger_Upper"] - df["Bollinger_Lower"]) / df["EMA_20"]

    df["Volume_SMA_20"] = df["Volume"].rolling(window=20).mean()

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = low.diff()
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    tr_smooth = atr

    plus_di = 100 * (plus_dm.rolling(14).mean() / tr_smooth)
    minus_di = 100 * (minus_dm.rolling(14).mean() / tr_smooth)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df["ADX"] = dx.rolling(14).mean().abs()
    df["ATR_percent"] = (atr / df["Close"]) * 100

    low_14 = df["Low"].rolling(window=14).min()
    high_14 = df["High"].rolling(window=14).max()
    df["%K"] = 100 * ((df["Close"] - low_14) / (high_14 - low_14))
    df["%D"] = df["%K"].rolling(window=3).mean()

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    money_flow = typical_price * df["Volume"]
    pos_flow = money_flow.where(typical_price > typical_price.shift(), 0)
    neg_flow = money_flow.where(typical_price < typical_price.shift(), 0)
    pos_sum = pos_flow.rolling(14).sum()
    neg_sum = neg_flow.rolling(14).sum()
    df["MFI"] = 100 - (100 / (1 + (pos_sum / neg_sum)))

    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv

    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma_tp = tp.rolling(window=20).mean()
    mad_tp = tp.rolling(window=20).apply(lambda x: median_abs_deviation(x, scale='normal'), raw=True)
    df["CCI"] = (tp - sma_tp) / (0.015 * mad_tp)

    df["Williams_%R"] = -100 * ((high_14 - df["Close"]) / (high_14 - low_14))

    factor = 3.0
    atr_10 = tr.rolling(10).mean()
    hl2 = (df["High"] + df["Low"]) / 2
    upper_band = hl2 + (factor * atr_10)
    lower_band = hl2 - (factor * atr_10)
    supertrend = [np.nan] * len(df)
    direction = [True] * len(df)

    for i in range(1, len(df)):
        curr_close = df["Close"].iloc[i]
        if curr_close > upper_band.iloc[i - 1]:
            direction[i] = True
        elif curr_close < lower_band.iloc[i - 1]:
            direction[i] = False
        else:
            direction[i] = direction[i - 1]
        supertrend[i] = lower_band.iloc[i] if direction[i] else upper_band.iloc[i]

    df["SuperTrend"] = supertrend
    df["ST_Direction"] = direction

    df["Donchian_High_20"] = df["High"].rolling(window=20).max()
    df["Donchian_Low_20"] = df["Low"].rolling(window=20).min()
    df["Donchian_Mid"] = (df["Donchian_High_20"] + df["Donchian_Low_20"]) / 2

    open_ = df["Open"]
    num = ((close - open_) + 2*(close.shift(1) - open_.shift(1)) +
           2*(close.shift(2) - open_.shift(2)) + (close.shift(3) - open_.shift(3))) / 6
    den = ((high - low) + 2*(high.shift(1) - low.shift(1)) +
           2*(high.shift(2) - low.shift(2)) + (high.shift(3) - low.shift(3))) / 6
    rvi = num.rolling(window=10).mean() / den.rolling(window=10).mean()
    df["RVI"] = rvi
    df["RVI_Signal"] = rvi.rolling(window=4).mean()

    # --- Ultimate Oscillator (UO) ---
    bp = df["Close"] - pd.concat([df["Low"], df["Close"].shift(1)], axis=1).min(axis=1)
    tr_uo = pd.concat([
        df["High"] - df["Low"],
        abs(df["High"] - df["Close"].shift(1)),
        abs(df["Low"] - df["Close"].shift(1))
    ], axis=1).max(axis=1)

    avg7 = bp.rolling(7).sum() / tr_uo.rolling(7).sum()
    avg14 = bp.rolling(14).sum() / tr_uo.rolling(14).sum()
    avg28 = bp.rolling(28).sum() / tr_uo.rolling(28).sum()
    df["UO"] = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7

    return df

def analyze_stock(ticker):
    try:
        logging.info(f"Analyzing {ticker}")
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")

        if hist.empty or len(hist) < 100:
            raise ValueError("Insufficient historical data")

        df = calculate_indicators(hist)
        df = df.dropna()
        if df.empty:
            raise ValueError("No usable data after dropping NaNs")

        current = df.iloc[-1]
        strategy_score = 0

        if current["SMA_50"] > current["SMA_200"]:
            strategy_score += 1
        if current["Close"] > current["EMA_20"]:
            strategy_score += 1
        if current["RSI"] < 30:
            strategy_score += 1
        elif current["RSI"] > 70:
            strategy_score -= 1
        if current["MACD"] > current["Signal_Line"]:
            strategy_score += 1
        if current["Close"] < current["Bollinger_Lower"]:
            strategy_score += 1
        elif current["Close"] > current["Bollinger_Upper"]:
            strategy_score -= 1
        if current["Volume"] > current["Volume_SMA_20"]:
            strategy_score += 1
        if current["ADX"] > 20:
            strategy_score += 1
        if current["ATR_percent"] > 1.5:
            strategy_score += 1
        if current["MFI"] < 20:
            strategy_score += 1
        elif current["MFI"] > 80:
            strategy_score -= 1
        if current["CCI"] < -100:
            strategy_score += 1
        elif current["CCI"] > 100:
            strategy_score -= 1
        if current["ST_Direction"]:
            strategy_score += 1
        else:
            strategy_score -= 1
        if current["Close"] > current["Donchian_High_20"]:
            strategy_score += 1
        elif current["Close"] < current["Donchian_Low_20"]:
            strategy_score -= 1
        if current["RVI"] > current["RVI_Signal"]:
            strategy_score += 1
        elif current["RVI"] < current["RVI_Signal"]:
            strategy_score -= 1
        if current["UO"] < 30:
            strategy_score += 1
        elif current["UO"] > 70:
            strategy_score -= 1

        signal = (
            "BUY" if strategy_score >= 4 else
            "HOLD" if strategy_score >= 2 else
            "SELL"
        )

        return {
            "name": stock.info.get("shortName", ticker),
            "symbol": ticker,
            "price": round(current["Close"], 2),
            "pe_ratio": round(stock.info.get("trailingPE", 0), 2),
            "rsi": round(current["RSI"], 2),
            "adx": round(current["ADX"], 2),
            "atr_percent": round(current["ATR_percent"], 2),
            "bbw": round(current["BBW"], 3),
            "%K": round(current["%K"], 2),
            "%D": round(current["%D"], 2),
            "mfi": round(current["MFI"], 2),
            "obv": int(current["OBV"]),
            "cci": round(current["CCI"], 2),
            "Williams_%R": round(current["Williams_%R"], 2),
            "supertrend": round(current["SuperTrend"], 2),
            "st_direction": "Bullish" if current["ST_Direction"] else "Bearish",
            "donchian_high": round(current["Donchian_High_20"], 2),
            "donchian_low": round(current["Donchian_Low_20"], 2),
            "donchian_mid": round(current["Donchian_Mid"], 2),
            "rvi": round(current["RVI"], 4),
            "rvi_signal": round(current["RVI_Signal"], 4),
            "uo": round(current["UO"], 2),
            "signal": signal,
            "score": strategy_score
        }

    except Exception as e:
        logging.error(f"Error analyzing {ticker}: {e}")
        return {
            "symbol": ticker,
            "error": str(e),
            "signal": "NO DATA"
        }

@app.post("/analyze")
def analyze_stocks(request: StockRequest):
    results = [analyze_stock(ticker) for ticker in request.tickers]
    return {"analysis": results}