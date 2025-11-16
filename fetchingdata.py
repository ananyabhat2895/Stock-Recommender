import os
import pandas as pd
import requests
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from pymongo import MongoClient
from pymongo.errors import BulkWriteError
from dotenv import load_dotenv

# ---------------- LOAD ENV ----------------
load_dotenv()

DHAN_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

if not DHAN_TOKEN: raise Exception("❌ Missing DHAN_ACCESS_TOKEN in .env")
if not MONGO_URI: raise Exception("❌ Missing MONGO_URI in .env")
if not MONGO_DB_NAME: raise Exception("❌ Missing MONGO_DB_NAME in .env")

# ---------------- CONNECT TO MONGO ----------------
client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]

daily_collection = db["daily_candles"]
intraday_collection = db["intraday_candles"]

print("✅ Connected to MongoDB")

# ---------------- DHAN HELPERS ----------------
def dhan_headers():
    return {
        "access-token": DHAN_TOKEN,
        "Content-Type": "application/json",
    }

def convert_to_docs(inst, data, timeframe):
    docs = []
    symbol = inst["SYMBOL_NAME"]

    for i in range(len(data["timestamp"])):
        ts = data["timestamp"][i]
        dt = datetime.fromtimestamp(ts)

        docs.append({
            "symbol": symbol,
            "securityId": str(inst["SECURITY_ID"]),
            "exchangeSegment": inst["exchangeSegment"],
            "instrument": inst["INSTRUMENT"],
            "timeframe": timeframe,
            "datetime": dt,
            "timestamp": ts,
            "open": float(data["open"][i]),
            "high": float(data["high"][i]),
            "low": float(data["low"][i]),
            "close": float(data["close"][i]),
            "volume": int(data["volume"][i])
        })
    return docs

# ---------------- FETCH FULL DAILY ----------------
def fetch_full_daily(inst):
    start = (date.today() - relativedelta(years=2)).strftime("%Y-%m-%d")
    end = date.today().strftime("%Y-%m-%d")

    payload = {
        "securityId": str(inst["SECURITY_ID"]),
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY",
        "expiryCode": 0,
        "oi": False,
        "fromDate": start,
        "toDate": end,
    }

    r = requests.post("https://api.dhan.co/v2/charts/historical",
                      json=payload, headers=dhan_headers())
    r.raise_for_status()
    return r.json()

# ---------------- FETCH FULL INTRADAY ----------------
def fetch_full_intraday(inst):
    start = (datetime.now() - relativedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S")
    end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    payload = {
        "securityId": str(inst["SECURITY_ID"]),
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY",
        "interval": "1",
        "oi": False,
        "fromDate": start,
        "toDate": end,
    }

    r = requests.post("https://api.dhan.co/v2/charts/intraday",
                      json=payload, headers=dhan_headers())
    r.raise_for_status()
    return r.json()

# ---------------- MAIN PROCESS ----------------
def main():
    print("📥 Loading NSE_EQ_EQUITY_ONLY.csv…")

    df = pd.read_csv("NSE_EQ_EQUITY_ONLY.csv")
    df["SERIES"] = df["SERIES"].str.strip()
    df["exchangeSegment"] = "NSE_EQ"

    df = df[(df["INSTRUMENT"] == "EQUITY") & (df["SERIES"] == "EQ")]

    print(f"✔ Total companies: {len(df)}")

    total_daily = 0
    total_intraday = 0

    for index, inst in df.iterrows():
        symbol = inst["SYMBOL_NAME"]
        underlying = inst["UNDERLYING_SYMBOL"]

        print("\n==============================")
        print(f"📊 {index+1}. Fetching for: {symbol} - {underlying}")
        print("==============================")

        # DAILY
        try:
            json_daily = fetch_full_daily(inst)
            docs_daily = convert_to_docs(inst, json_daily, "1D")

            if docs_daily:
                try:
                    daily_collection.insert_many(docs_daily, ordered=False)
                    print(f"✔ Inserted {len(docs_daily)} DAILY candles")
                    total_daily += len(docs_daily)
                except BulkWriteError:
                    pass
        except Exception as e:
            print(f"❌ DAILY ERROR for {symbol}: {e}")

        # INTRADAY
        try:
            json_intr = fetch_full_intraday(inst)
            docs_intr = convert_to_docs(inst, json_intr, "1m")

            if docs_intr:
                try:
                    intraday_collection.insert_many(docs_intr, ordered=False)
                    print(f"✔ Inserted {len(docs_intr)} INTRADAY candles")
                    total_intraday += len(docs_intr)
                except BulkWriteError:
                    pass
        except Exception as e:
            print(f"❌ INTRADAY ERROR for {symbol}: {e}")

    print("\n🎉 COMPLETED FULL HISTORIC UPLOAD")
    print(f"📅 TOTAL DAILY INSERTED   : {total_daily}")
    print(f"⏱ TOTAL INTRADAY INSERTED: {total_intraday}")


if __name__ == "__main__":
    main()
