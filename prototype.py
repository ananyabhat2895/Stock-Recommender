import os
import pandas as pd
import requests
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from pymongo import MongoClient
from dotenv import load_dotenv

# ---------------- LOAD ENV ----------------
load_dotenv()

DHAN_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

if not DHAN_TOKEN:
    raise Exception("❌ Missing DHAN_ACCESS_TOKEN in .env")
if not MONGO_URI:
    raise Exception("❌ Missing MONGO_URI in .env")
if not MONGO_DB_NAME:
    raise Exception("❌ Missing MONGO_DB_NAME in .env")

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

def get_daily_dates():
    today = date.today()
    start = today - relativedelta(years=2)
    return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

def get_intraday_dates():
    now = datetime.now()
    start = now - relativedelta(days=90)
    return start.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S")


# ---------------- DHAN API CALLS ----------------
def fetch_daily(inst):
    start, end = get_daily_dates()

    payload = {
        "securityId": str(inst["SECURITY_ID"]),
        "exchangeSegment": inst["exchangeSegment"],
        "instrument": inst["INSTRUMENT"],
        "expiryCode": 0,
        "oi": False,
        "fromDate": start,
        "toDate": end,
    }

    r = requests.post("https://api.dhan.co/v2/charts/historical",
                      json=payload,
                      headers=dhan_headers())
    r.raise_for_status()
    return r.json()


def fetch_intraday(inst):
    start, end = get_intraday_dates()

    payload = {
        "securityId": str(inst["SECURITY_ID"]),
        "exchangeSegment": inst["exchangeSegment"],
        "instrument": inst["INSTRUMENT"],
        "interval": "1",
        "oi": False,
        "fromDate": start,
        "toDate": end,
    }

    r = requests.post("https://api.dhan.co/v2/charts/intraday",
                      json=payload,
                      headers=dhan_headers())
    r.raise_for_status()
    return r.json()


# ---------------- TRANSFORM INTO MONGO DOCS ----------------
def convert_to_mongo_docs(inst, data, timeframe):
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
            "datetime": dt,        # TIME-SERIES FIELD
            "timestamp": ts,
            "open": float(data["open"][i]),
            "high": float(data["high"][i]),
            "low": float(data["low"][i]),
            "close": float(data["close"][i]),
            "volume": int(data["volume"][i])
        })

    return docs


def needs_daily_update(symbol):
    last = daily_collection.find({"symbol": symbol}).sort("datetime", -1).limit(1)
    last_doc = next(last, None)

    if not last_doc:
        return True  # first time fetch

    last_date = last_doc["datetime"].date()
    today = date.today()

    return last_date < today    # True if new daily candle is required


def needs_intraday_update(symbol):
    last = intraday_collection.find({"symbol": symbol}).sort("datetime", -1).limit(1)
    last_doc = next(last, None)

    if not last_doc:
        return True  # first time fetch (90 days)

    last_dt = last_doc["datetime"].replace(second=0, microsecond=0)
    now = datetime.now().replace(second=0, microsecond=0)

    return last_dt < now    # True if new minute exists

def update_stock(stock):
    symbol = stock["SYMBOL_NAME"]
    print(f"\n🔄 Checking updates for {symbol}")

    # ---------------- DAILY UPDATE ----------------
    if needs_daily_update(symbol):
        print("📅 New daily data found → fetching...")
        daily_docs = fetch_incremental_daily(stock)
        if daily_docs:
            try:
                daily_collection.insert_many(daily_docs, ordered=False)
                print(f"✔ Added {len(daily_docs)} daily candles")
            except:
                pass
    else:
        print("📅 Daily candles already up-to-date")

    # ---------------- INTRADAY UPDATE ----------------
    if needs_intraday_update(symbol):
        print("⏱ New intraday data found → fetching...")
        intr_docs = fetch_incremental_intraday(stock)
        if intr_docs:
            try:
                intraday_collection.insert_many(intr_docs, ordered=False)
                print(f"✔ Added {len(intr_docs)} intraday candles")
            except:
                pass
    else:
        print("⏱ Intraday candles already up-to-date")




# ---------------- MAIN FUNCTION ----------------
def main():
    print("📥 Loading NSE_EQ_EQUITY_ONLY.csv…")

    df = pd.read_csv("NSE_EQ_EQUITY_ONLY.csv")

    df["SERIES"] = df["SERIES"].str.strip()
    df["exchangeSegment"] = "NSE_EQ"

    df = df[
        (df["INSTRUMENT"] == "EQUITY") &
        (df["SERIES"] == "EQ")
    ]

    print(f"✔ Valid stocks found: {len(df)}")

    top10 = df.head(10)

    print("\n📌 Selected TOP 10 companies:")
    print(top10["SYMBOL_NAME"].tolist())

    for _, inst in top10.iterrows():

        symbol = inst["SYMBOL_NAME"]
        print(f"\n==============================")
        print(f"Fetching: {symbol}")
        print("==============================")

        # DAILY
        try:
            daily_json = fetch_daily(inst)
            daily_docs = convert_to_mongo_docs(inst, daily_json, "1D")

            if daily_docs:
                daily_collection.insert_many(daily_docs, ordered=False)
                print(f"✔ Inserted {len(daily_docs)} DAILY candles")
        except Exception as e:
            print(f"❌ DAILY ERROR for {symbol}: {e}")

        # INTRADAY
        try:
            intraday_json = fetch_intraday(inst)
            intraday_docs = convert_to_mongo_docs(inst, intraday_json, "1m")

            if intraday_docs:
                intraday_collection.insert_many(intraday_docs, ordered=False)
                print(f"✔ Inserted {len(intraday_docs)} INTRADAY candles")
        except Exception as e:
            print(f"❌ INTRADAY ERROR for {symbol}: {e}")


if __name__ == "__main__":
    main()
