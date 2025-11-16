import pandas as pd
import yfinance as yf
import re

# ================
# CLEAN SYMBOL
# ================
def to_yahoo_symbol(dhan_symbol):
    if not isinstance(dhan_symbol, str):
        return None
    cleaned = re.sub(r"[&\-\.\s]", "", dhan_symbol).upper()
    return cleaned + ".NS"

# ================
# GET MARKET CAP
# ================
def get_market_cap(yahoo_symbol):
    try:
        stock = yf.Ticker(yahoo_symbol)
        mcap = stock.fast_info.get("market_cap")
        return mcap if mcap and mcap > 0 else None
    except:
        return None

# ================
# CLASSIFY MARKET CAP
# ================
def classify(mcap):
    if mcap is None:
        return "UNKNOWN"
    cr = mcap / 1e7
    if cr > 20000:
        return "LARGE_CAP"
    elif cr >= 5000:
        return "MID_CAP"
    else:
        return "SMALL_CAP"

# ================
# LOAD NSE INDEX LISTS
# ================
def load_index_list(file):
    try:
        df = pd.read_csv(file)
        return set(df["symbol"].str.upper())
    except:
        return set()

large_set = load_index_list("nifty50.csv") | load_index_list("niftynext50.csv")
mid_set = load_index_list("midcap150.csv")
small_set = load_index_list("smallcap250.csv")

# ================
# MAIN PROCESS
# ================
df = pd.read_csv("NSE_EQ_EQUITY_ONLY.csv")

df["DhanSymbol"] = df["UNDERLYING_SYMBOL"]
df["YahooSymbol"] = df["DhanSymbol"].apply(to_yahoo_symbol)

cap_category = []
market_cap = []

for i, row in df.iterrows():
    sym = row["DhanSymbol"].upper()

    # 1️⃣ NSE INDEX MATCHING (100% reliable)
    if sym in large_set:
        cap_category.append("LARGE_CAP")
        market_cap.append(None)
        continue
    if sym in mid_set:
        cap_category.append("MID_CAP")
        market_cap.append(None)
        continue
    if sym in small_set:
        cap_category.append("SMALL_CAP")
        market_cap.append(None)
        continue

    # 2️⃣ YAHOO FALLBACK FOR OTHERS
    ysym = row["YahooSymbol"]
    mcap = get_market_cap(ysym)
    c = classify(mcap)

    cap_category.append(c)
    market_cap.append(mcap)

# Save results
df["market_cap"] = market_cap
df["cap_category"] = cap_category

df.to_csv("NSE_with_cap_category.csv", index=False)
print("\n🎉 Done → NSE_with_cap_category.csv")
