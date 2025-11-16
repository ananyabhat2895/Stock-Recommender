##############################################
# PRECOMPUTE ALL TECHNICAL INDICATORS
# Stores into: stockDB.precomputed_daily
##############################################

from pymongo import MongoClient
import pandas as pd
import numpy as np

# ---------------------------------------------
# 1. CONNECT TO MONGODB
# ---------------------------------------------
client = MongoClient("mongodb://localhost:27017")
db = client["stockDB"]

daily_collection = db["daily_candles"]
precomputed_collection = db["precomputed_daily"]


# ---------------------------------------------
# 2. INDICATOR FUNCTIONS
# ---------------------------------------------
def sma(series, window):
    return series.rolling(window).mean()

def ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def macd(close):
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist

def volatility(close, window=20):
    logret = np.log(close).diff()
    vol = logret.rolling(window).std() * np.sqrt(252)
    return vol.fillna(0)

def volume_zscore(volume, window=20):
    mean_vol = volume.rolling(window).mean()
    std_vol = volume.rolling(window).std()
    return ((volume - mean_vol) / std_vol).fillna(0)

def compute_vwap(df):
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical_price * df["volume"]
    return (pv.cumsum() / df["volume"].cumsum()).fillna(0)


# ---------------------------------------------
# 3. COMPUTE FEATURES FOR ONE SYMBOL
# ---------------------------------------------
def compute_for_symbol(symbol):
    print(f"\nProcessing: {symbol}")

    cursor = daily_collection.find({"symbol": symbol})
    df = pd.DataFrame(list(cursor))

    if df.empty:
        print(f"⚠ No data found for: {symbol}")
        return None

    df["date"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("date").reset_index(drop=True)

    close = df["close"]
    volume = df["volume"]

    # BASIC INDICATORS
    df["RSI"] = rsi(close)
    df["SMA_20"] = sma(close, 20)
    df["SMA_50"] = sma(close, 50)

    df["EMA_12"] = ema(close, 12)
    df["EMA_26"] = ema(close, 26)

    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(close)
    df["Volatility_20"] = volatility(close)

    # RETURNS
    df["Return_1d"] = close.pct_change(1)
    df["Return_5d"] = close.pct_change(5)
    df["Return_10d"] = close.pct_change(10)
    df["Return_20d"] = close.pct_change(20)

    # MOMENTUM
    df["Momentum_20"] = (close - close.shift(20)) / close.shift(20)

    # OPTIONAL (extra useful)
    df["Volume_Z"] = volume_zscore(volume)
    df["VWAP"] = compute_vwap(df)
    df["VWAP_Distance"] = (close - df["VWAP"]) / df["VWAP"]

    # FILL NULLS
    df = df.fillna(0)

    return df


# ---------------------------------------------
# 4. SAVE INTO NEW COLLECTION
# ---------------------------------------------
def save_precomputed(df, symbol):
    precomputed_collection.delete_many({"symbol": symbol})
    precomputed_collection.insert_many(df.to_dict(orient="records"))
    print(f"✅ Saved {len(df)} rows for: {symbol}")


# ---------------------------------------------
# 5. RUN FOR ALL SYMBOLS
# ---------------------------------------------
if __name__ == "__main__":
    print("Fetching symbols...\n")

    symbols = daily_collection.distinct("symbol")

    for symbol in symbols:
        df = compute_for_symbol(symbol)
        if df is not None:
            save_precomputed(df, symbol)

    print("\n🎉 All symbols processed successfully!")
