"""
XGBoost Stock Return Model + Backtesting (R2-Optimised + Sector/Cap Filters)

Pipeline:
1. Load sector + market-cap info from CSV (NSE.csv)
2. Filter list of symbols by chosen sectors + cap categories
3. Map CSV SYMBOL_NAME -> MongoDB symbol (full company name)
4. Load precomputed indicators for ONLY those companies from MongoDB (stockDB.precomputed_daily)
5. Normalize technical indicators PER SYMBOL (ticker)
6. Create FUTURE 120-day return label per stock (FwdReturn_120d)
7. Build MARKET-NEUTRAL + SMOOTHED target for regression (median smoothing + quantile clipping)
8. Add extra alpha factors (lagged returns, momentum, rolling vol/volume, ranks)
9. Clean data (remove penny stocks, zero-volume, old regime)
10. Train XGBoost regression model on smoothed, market-neutral target
11. Evaluate on held-out time-based test set (MSE / MAE / R2 / Direction Accuracy)
12. Backtest: every 120 days, pick top-K predicted stocks and simulate portfolio
13. Compute top-N stocks for latest date (highest predicted forward return)
14. Plot Actual vs Predicted portfolio value
"""

import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from datetime import datetime
from pymongo import MongoClient

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt


# =====================================================
# 1. MongoDB helpers
# =====================================================

def get_mongo_client(uri: str = "mongodb://localhost:27017") -> MongoClient:
    return MongoClient(uri)


def fetch_precomputed_for_symbol(
    client: MongoClient,
    mongo_symbol_name: str,
    db: str = "stockDB",
    coll: str = "precomputed_daily",
) -> pd.DataFrame:
    """
    Fetch all precomputed rows for a company from Mongo and sort by date.

    NOTE: mongo_symbol_name is the FULL COMPANY NAME
    as it appears in Mongo's 'symbol' field, e.g. 'FACT LTD'.
    """
    c = client[db][coll].find({"symbol": mongo_symbol_name})
    df = pd.DataFrame(list(c))
    if df.empty:
        return df

    # Use "date" if present, else fall back to "datetime"
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        df["date"] = pd.to_datetime(df["datetime"])

    df = df.sort_values("date").reset_index(drop=True)
    return df


# =====================================================
# 2. Dataset preparation helpers
# =====================================================

def prepare_dataset(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """
    Time-based split: first 80% train, last 20% test.
    Drops rows where label or any feature is NaN.
    """
    df = df.copy().reset_index(drop=True)
    df = df.dropna(subset=[label_col] + feature_cols).reset_index(drop=True)

    n = len(df)
    split_at = int(n * (1 - test_size))

    train = df.iloc[:split_at]
    test = df.iloc[split_at:]

    X_train = train[feature_cols].values
    y_train = train[label_col].values
    X_test = test[feature_cols].values
    y_test = test[label_col].values
    dates_test = test["date"]

    return X_train, X_test, y_train, y_test, dates_test


def train_xgb_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray = None,
    y_valid: np.ndarray = None,
    params: Dict = None,
    num_boost_round: int = 1200,
) -> xgb.XGBRegressor:
    """
    Train XGBoost regressor with stronger capacity + regularisation.
    """

    if params is None:
        params = {
            "n_estimators": num_boost_round,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 2,
            "reg_lambda": 5.0,      # L2 regularisation
            "reg_alpha": 1.0,       # L1 regularisation
            "objective": "reg:squarederror",
            "random_state": 42,
        }

    model = xgb.XGBRegressor(**params)

    if X_valid is not None and y_valid is not None:
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    else:
        model.fit(X_train, y_train, verbose=False)

    return model


def predict_returns(
    model: xgb.XGBRegressor,
    scaler: StandardScaler,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Attach predicted forward return to each row."""
    X = scaler.transform(df[feature_cols].values)
    preds = model.predict(X)

    out = df[["date", "symbol", "close"]].copy()
    out["pred_return_fwd"] = preds
    return out


# =====================================================
# 3. BACKTESTING
# =====================================================

def backtest_model(
    full_df: pd.DataFrame,
    model: xgb.XGBRegressor,
    scaler: StandardScaler,
    feature_cols: List[str],
    label_col: str,               # actual future return column, e.g. "FwdReturn_120d"
    top_k: int = 10,
    holding_period: int = 120,
    initial_capital: float = 100000.0,
) -> Tuple[List[float], List[float]]:
    """
    Backtest with rebalancing every `holding_period` days.

    At each rebalance date:
    - Use model to predict next forward return for all stocks on that date
    - Pick top_k predicted stocks
    - Invest equally in them
    - Actual portfolio evolution uses REAL forward return from label_col
      aligned with today's row (no look-ahead).
    """

    df = full_df.copy()

    # Ensure sorted and no NaNs on features/label
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=feature_cols + [label_col]).reset_index(drop=True)

    dates = df["date"].unique()

    portfolio_values_actual = [initial_capital]
    portfolio_values_pred = [initial_capital]

    capital_actual = initial_capital
    capital_pred = initial_capital

    # Step by holding_period
    for i in range(0, len(dates), holding_period):

        current_date = dates[i]

        today_rows = df[df["date"] == current_date].dropna(subset=feature_cols + [label_col])
        if len(today_rows) < top_k:
            continue

        # Predict for all stocks available today
        X_today = scaler.transform(today_rows[feature_cols].values)
        preds = model.predict(X_today)

        today_rows = today_rows.copy()
        today_rows["pred_return"] = preds

        # Pick top-k by predicted return
        selected = today_rows.sort_values("pred_return", ascending=False).head(top_k)

        # Predicted portfolio evolution (using predicted forward return)
        predicted_growth = selected["pred_return"].mean()
        capital_pred = capital_pred * (1 + predicted_growth)

        # Actual portfolio evolution using REAL forward return from label_col
        actual_growth = selected[label_col].mean()
        capital_actual = capital_actual * (1 + actual_growth)

        portfolio_values_actual.append(capital_actual)
        portfolio_values_pred.append(capital_pred)

    return portfolio_values_actual, portfolio_values_pred


def plot_backtest(portfolio_actual: List[float], portfolio_predicted: List[float]) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_actual, label="Actual Portfolio Value")
    plt.plot(portfolio_predicted, label="Predicted Portfolio Value", linestyle="--")
    plt.title("Backtest: Actual vs Predicted Portfolio Value (Horizon steps)")
    plt.xlabel("Backtest Steps (each ≈ holding_period trading days)")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.show()


# =====================================================
# 4. NSE Sector/Cap helpers + Top stocks
# =====================================================

def load_and_filter_nse_list(
    csv_path: str = "NSE.csv",
    allowed_sectors: List[str] = None,
    allowed_caps: List[str] = None,
) -> pd.DataFrame:
    """
    Load NSE symbol list with sector + cap info and filter.

    Returns a FILTERED DATAFRAME with at least:
    - symbol       (ticker)
    - SYMBOL_NAME  (full name matching Mongo 'symbol')
    - sector
    - cap_category
    """
    df = pd.read_csv(csv_path)

    required_cols = ["symbol", "SYMBOL_NAME", "sector", "cap_category"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}' column")

    df["sector"] = df["sector"].fillna("UNKNOWN").astype(str)
    df["cap_category"] = df["cap_category"].fillna("UNKNOWN").astype(str)
    df["symbol"] = df["symbol"].astype(str)
    df["SYMBOL_NAME"] = df["SYMBOL_NAME"].astype(str)

    print("\n🔍 CSV unique sectors:", df["sector"].unique())
    print("🔍 CSV unique cap categories:", df["cap_category"].unique())

    if allowed_sectors:
        df = df[df["sector"].isin(allowed_sectors)]

    if allowed_caps:
        df = df[df["cap_category"].isin(allowed_caps)]

    return df[["symbol", "SYMBOL_NAME", "sector", "cap_category"]]


def get_top_predicted_stocks(
    df_predicted: pd.DataFrame,
    top_n: int = 10,
    latest_only: bool = True,
) -> pd.DataFrame:
    """
    Returns top_n stocks with highest predicted forward return.

    If latest_only=True, only considers the latest date in df_predicted.
    """
    df = df_predicted.copy()

    if latest_only:
        last_date = df["date"].max()
        df = df[df["date"] == last_date]

    df_sorted = df.sort_values("pred_return_fwd", ascending=False)
    return df_sorted.head(top_n)[["date", "symbol", "close", "pred_return_fwd"]]


# =====================================================
# 5. FULL PIPELINE (LOAD + TRAIN + BACKTEST)
# =====================================================

def example_run():

    MONGO_URI = "mongodb://localhost:27017"
    DB = "stockDB"
    COLL = "precomputed_daily"

    # Forward horizon (in trading days)
    HORIZON = 120  # ~6 months

    client = get_mongo_client(MONGO_URI)

    # --- BASE FEATURE COLUMNS (from your precompute.py) ---
    base_feature_cols = [
        "RSI",
        "SMA_20", "SMA_50",
        "EMA_12", "EMA_26",
        "MACD", "MACD_signal", "MACD_hist",
        "Volatility_20",

        "Return_1d", "Return_5d", "Return_10d", "Return_20d",
        "Momentum_20",

        "ATR_14",
        "BB_Width",
        "ROC_10",
        "Stoch_K", "Stoch_D",
        "Williams_R",
        "OBV",
        "Volume_Z",
        "VWAP_Distance",
    ]

    # --------------------------------------------------
    # 5.1 Select companies based on sector + cap filters
    # --------------------------------------------------
    # Adjust these as needed
    allowed_sectors = ["Basic Materials"]
    allowed_caps = ["LARGE_CAP"]

    filtered_df = load_and_filter_nse_list(
        csv_path="NSE.csv",
        allowed_sectors=allowed_sectors,
        allowed_caps=allowed_caps,
    )

    filtered_symbols = filtered_df["symbol"].tolist()
    print("\n================ SYMBOL FILTER ================")
    print("Allowed sectors   :", allowed_sectors if allowed_sectors else "ALL")
    print("Allowed cap groups:", allowed_caps if allowed_caps else "ALL")
    print("Filtered symbols  :", len(filtered_symbols))
    print("Sample symbols    :", filtered_symbols[:10])
    print("================================================\n")

    if filtered_df.empty:
        print("No symbols matched the sector/cap filters. Exiting.")
        return

    # --------------------------------------------------
    # 5.2 Load those companies' data from Mongo
    #     Match by SYMBOL_NAME (full company name)
    # --------------------------------------------------
    dfs = []
    not_found = []

    for _, row in filtered_df.iterrows():
        ticker = row["symbol"]           # e.g. FACT
        full_name = row["SYMBOL_NAME"]   # e.g. FACT LTD

        print(f"🔎 Fetching from Mongo for: {ticker} - {full_name}")
        df_sym = fetch_precomputed_for_symbol(client, full_name, DB, COLL)

        if df_sym.empty:
            print("   ❌ Not found in MongoDB")
            not_found.append(ticker)
            continue

        # Overwrite / attach ticker as the 'symbol' used in the model
        df_sym["symbol"] = ticker
        dfs.append(df_sym)
        print(f"   ✅ Found {len(df_sym)} rows")

    if not dfs:
        print("\n❌ No data found in MongoDB for any of the filtered companies. Exiting.")
        return

    full_df = pd.concat(dfs).reset_index(drop=True)
    full_df = full_df.sort_values(["date", "symbol"]).reset_index(drop=True)

    # Stats
    print(f"\n📊 Total filtered companies       : {len(filtered_df)}")
    print(f"✅ Companies found in MongoDB     : {full_df['symbol'].nunique()}")
    print(f"❌ Companies missing in MongoDB   : {len(not_found)}")
    if not_found:
        print("   Missing tickers (first 20):", not_found[:20])
    print(f"📈 Total historical rows loaded   : {len(full_df):,}")
    print("Full combined dataset (after symbol filter):", full_df.shape)

    # --------------------------------------------------
    # 5.3 Filter out very old data (regime shifts)
    # --------------------------------------------------
    full_df = full_df[full_df["date"] >= pd.Timestamp("2018-01-01")].reset_index(drop=True)

    # --------------------------------------------------
    # 5.4 Remove days with no volume & penny stocks
    # --------------------------------------------------
    if "volume" in full_df.columns:
        full_df = full_df[full_df["volume"] > 0]

    # Remove penny stocks (you can adjust this threshold)
    full_df = full_df[full_df["close"] > 50].reset_index(drop=True)

    # --------------------------------------------------
    # 5.5 Normalize base indicators PER SYMBOL (ticker)
    # --------------------------------------------------
    for col in base_feature_cols:
        if col in full_df.columns:
            full_df[col] = full_df.groupby("symbol")[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-6)
            )

    # --------------------------------------------------
    # 5.6 Add additional predictive alpha factors
    # --------------------------------------------------
    # Raw 1-day returns (for rolling stats)
    full_df["Ret_1d_raw"] = full_df.groupby("symbol")["close"].pct_change()

    # 60-day momentum (from price)
    full_df["Momentum_60"] = full_df.groupby("symbol")["close"].pct_change(60)

    # Volume % change (volume shock)
    if "volume" in full_df.columns:
        full_df["Vol_Change"] = full_df.groupby("symbol")["volume"].pct_change()
        full_df["Volume_RollMean_20"] = (
            full_df.groupby("symbol")["volume"]
            .transform(lambda x: x.rolling(20, min_periods=10).mean())
        )

    # Distance from 52-week high
    full_df["High_252"] = full_df.groupby("symbol")["close"].transform(
        lambda x: x.rolling(252, min_periods=20).max()
    )
    full_df["Pct_From_52W_High"] = full_df["close"] / full_df["High_252"] - 1

    # Rolling volatility of daily returns
    full_df["Volatility_60_extra"] = (
        full_df.groupby("symbol")["Ret_1d_raw"]
        .transform(lambda x: x.rolling(60, min_periods=20).std())
    )

    # --------------------------------------------------
    # 5.7 Create FUTURE HORIZON-day return + market-neutral + smoothing
    # --------------------------------------------------
    fwd_col = f"FwdReturn_{HORIZON}d"

    # Raw forward stock return
    full_df[fwd_col] = (
        full_df.groupby("symbol")["close"].shift(-HORIZON) / full_df["close"] - 1
    )

    # Build a simple "market index" as avg close across all stocks per date
    mkt = full_df.groupby("date")["close"].mean().to_frame("mkt_close").reset_index()
    mkt[f"MktFwdReturn_{HORIZON}d"] = (
        mkt["mkt_close"].shift(-HORIZON) / mkt["mkt_close"] - 1
    )

    # Merge market forward return back into full_df
    full_df = full_df.merge(
        mkt[["date", f"MktFwdReturn_{HORIZON}d"]],
        on="date",
        how="left",
    )

    # Market-neutral forward return: stock - market
    full_df["FwdReturn_mkt_neutral"] = (
        full_df[fwd_col] - full_df[f"MktFwdReturn_{HORIZON}d"]
    )

    # Smooth the target with a rolling window (median, per symbol)
    full_df["FwdReturn_smoothed"] = full_df.groupby("symbol")["FwdReturn_mkt_neutral"].transform(
        lambda x: x.rolling(5, min_periods=1).median()
    )

    # Quantile clipping of the smoothed target (instead of fixed ±0.30)
    valid_target = full_df["FwdReturn_smoothed"].dropna()
    if len(valid_target) > 0:
        lower_q = valid_target.quantile(0.01)
        upper_q = valid_target.quantile(0.99)
    else:
        lower_q, upper_q = -0.30, 0.30

    full_df["Target"] = full_df["FwdReturn_smoothed"].clip(lower_q, upper_q)

    # --------------------------------------------------
    # 5.8 Smooth noisy indicators (optional, helps stability)
    # --------------------------------------------------
    smooth_list = ["RSI", "MACD", "Volatility_20"]
    for col in smooth_list:
        if col in full_df.columns:
            full_df[col + "_smoothed"] = full_df.groupby("symbol")[col].transform(
                lambda x: x.rolling(5, min_periods=1).median()
            )

    # --------------------------------------------------
    # 5.9 Lag features (to avoid look-ahead, add memory)
    # --------------------------------------------------
    lag_defs = {
        "Return_1d": "Return_1d_lag1",
        "Return_5d": "Return_5d_lag1",
        "Momentum_20": "Momentum_20_lag1",
        "RSI": "RSI_lag1",
        "MACD": "MACD_lag1",
    }

    for src_col, lag_col in lag_defs.items():
        if src_col in full_df.columns:
            full_df[lag_col] = full_df.groupby("symbol")[src_col].shift(1)

    # --------------------------------------------------
    # 5.10 Cross-sectional ranks (per date)
    # --------------------------------------------------
    if "RSI" in full_df.columns:
        full_df["RSI_rank"] = full_df.groupby("date")["RSI"].rank(pct=True)
    if "Momentum_60" in full_df.columns:
        full_df["Momentum_60_rank"] = full_df.groupby("date")["Momentum_60"].rank(pct=True)
    if "Volatility_20" in full_df.columns:
        full_df["Volatility_20_rank"] = full_df.groupby("date")["Volatility_20"].rank(pct=True)

    # --------------------------------------------------
    # 5.11 Final feature columns (only those that exist)
    # --------------------------------------------------
    extra_features = [
        "Momentum_60",
        "Vol_Change",
        "Pct_From_52W_High",
        "RSI_smoothed",
        "MACD_smoothed",
        "Volatility_20_smoothed",
        "Volatility_60_extra",
        "Volume_RollMean_20",
        "Return_1d_lag1",
        "Return_5d_lag1",
        "Momentum_20_lag1",
        "RSI_lag1",
        "MACD_lag1",
        "RSI_rank",
        "Momentum_60_rank",
        "Volatility_20_rank",
    ]
    available_cols = set(full_df.columns)
    feature_cols = [c for c in base_feature_cols + extra_features if c in available_cols]

    print("\nUsing feature columns:")
    for c in feature_cols:
        print(" -", c)

    LABEL_COL = "Target"

    # --------------------------------------------------
    # 5.12 Train/test split (time-based)
    # --------------------------------------------------
    X_train, X_test, y_train, y_test, dates_test = prepare_dataset(
        full_df, LABEL_COL, feature_cols
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = train_xgb_regressor(
        X_train_s,
        y_train,
        X_valid=X_test_s,
        y_valid=y_test,
        num_boost_round=1200,
    )

    # --------------------------------------------------
    # 5.13 Evaluation
    # --------------------------------------------------
    y_pred = model.predict(X_test_s)
    print("\nEvaluation on held-out test set (Target = smoothed, market-neutral):")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R2 :", r2_score(y_test, y_pred))

    pred_up = y_pred > 0
    true_up = y_test > 0
    direction_acc = (pred_up == true_up).mean()
    print("Direction Accuracy (%):", direction_acc * 100.0)

    # --------------------------------------------------
    # 5.14 Save model & scaler
    # --------------------------------------------------
    joblib.dump(model, "xgb_fwd_return_model.joblib")
    joblib.dump(scaler, "xgb_fwd_return_scaler.joblib")
    print("\n✅ Model + scaler saved to disk.\n")

    # --------------------------------------------------
    # 5.15 Predict on full_df and show BEST stocks now
    # --------------------------------------------------
    df_pred_all = predict_returns(model, scaler, full_df, feature_cols)

    print("\n📈 TOP 10 PREDICTED STOCKS (latest date):")
    best_now = get_top_predicted_stocks(df_pred_all, top_n=10, latest_only=True)
    print(best_now.to_string(index=False))

    # --------------------------------------------------
    # 5.16 Backtest (portfolio simulation)
    # --------------------------------------------------
    print("\nRunning Backtest...")

    actual_vals, predicted_vals = backtest_model(
        full_df,
        model,
        scaler,
        feature_cols,
        label_col=fwd_col,         # use REAL (raw) forward return for actual money path
        top_k=10,
        holding_period=HORIZON,
        initial_capital=100000,
    )

    print("Backtest Completed. Plotting...")
    plot_backtest(actual_vals, predicted_vals)


if __name__ == "__main__":
    example_run()




