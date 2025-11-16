"""
XGBoost Stock Return Model + Backtesting (R2-Optimised)

Pipeline:
1. Load precomputed indicators from MongoDB (stockDB.precomputed_daily)
2. Normalize technical indicators PER SYMBOL
3. Create FUTURE 120-day return label per stock (FwdReturn_120d)
4. Build MARKET-NEUTRAL + SMOOTHED target for regression
5. Add extra alpha factors (momentum, volume change, 52W-high distance)
6. Clean data (remove penny stocks, zero-volume, old regime)
7. Train XGBoost regression model on smoothed, market-neutral target
8. Evaluate on held-out time-based test set (MSE / MAE / R2 / Direction Accuracy)
9. Backtest: every 120 days, pick top-K predicted stocks and simulate portfolio
10. Plot Actual vs Predicted portfolio value

Author: (your name)
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
    symbol: str,
    db: str = "stockDB",
    coll: str = "precomputed_daily",
) -> pd.DataFrame:
    """Fetch all precomputed rows for a symbol from Mongo and sort by date."""
    c = client[db][coll].find({"symbol": symbol})
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
    num_boost_round: int = 900,
) -> xgb.XGBRegressor:
    """
    Train XGBoost regressor with stronger capacity + regularisation.
    """

    if params is None:
        params = {
            "n_estimators": num_boost_round,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "min_child_weight": 2,
            "reg_lambda": 4.0,      # L2 regularisation
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

        # Actual portfolio evolution using REAL future return from label_col
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
# 4. FULL PIPELINE (LOAD + TRAIN + BACKTEST)
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

    # -----------------------------
    # 4.1 Load all symbols' data
    # -----------------------------
    symbols = client[DB][COLL].distinct("symbol")
    print(f"Found total {len(symbols)} symbols in DB.")

    # Limit for speed (tune as needed)
    symbols = symbols[:500]
    print(f"Using first {len(symbols)} symbols only.")

    dfs = []
    for sym in symbols:
        df = fetch_precomputed_for_symbol(client, sym, DB, COLL)
        if df.empty:
            continue
        df["symbol"] = sym
        dfs.append(df)

    if not dfs:
        print("No data found in DB.")
        return

    full_df = pd.concat(dfs).reset_index(drop=True)
    full_df = full_df.sort_values(["date", "symbol"]).reset_index(drop=True)
    print("Full combined dataset:", full_df.shape)

    # --------------------------------------------------
    # 4.2 Filter out very old data (regime shifts)
    # --------------------------------------------------
    full_df = full_df[full_df["date"] >= pd.Timestamp("2018-01-01")].reset_index(drop=True)

    # --------------------------------------------------
    # 4.3 Remove days with no volume & penny stocks
    # --------------------------------------------------
    if "volume" in full_df.columns:
        full_df = full_df[full_df["volume"] > 0]

    # Remove penny stocks (you can adjust this threshold)
    full_df = full_df[full_df["close"] > 50].reset_index(drop=True)

    # --------------------------------------------------
    # 4.4 Normalize features PER SYMBOL (base indicators)
    # --------------------------------------------------
    for col in base_feature_cols:
        if col in full_df.columns:
            full_df[col] = full_df.groupby("symbol")[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-6)
            )

    # --------------------------------------------------
    # 4.5 Add additional predictive alpha factors
    # --------------------------------------------------

    # 120-day momentum (price change over 120 days)
    full_df["Momentum_60"] = full_df.groupby("symbol")["close"].pct_change(60)

    # Volume % change (volume shock)
    if "volume" in full_df.columns:
        full_df["Vol_Change"] = full_df.groupby("symbol")["volume"].pct_change()

    # Distance from 52-week high
    full_df["High_252"] = full_df.groupby("symbol")["close"].transform(
        lambda x: x.rolling(252, min_periods=20).max()
    )
    full_df["Pct_From_52W_High"] = full_df["close"] / full_df["High_252"] - 1

    # --------------------------------------------------
    # 4.6 Create FUTURE HORIZON-day return + market-neutral + smoothing
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

    # Smooth the target with a rolling window (per symbol)
    full_df["FwdReturn_smoothed"] = full_df.groupby("symbol")["FwdReturn_mkt_neutral"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    # Clip extreme values for training stability
    full_df["Target"] = full_df["FwdReturn_smoothed"].clip(-0.30, 0.30)

    # --------------------------------------------------
    # 4.7 Smooth noisy indicators (optional, helps stability)
    # --------------------------------------------------
    smooth_list = ["RSI", "MACD", "Volatility_20"]
    for col in smooth_list:
        if col in full_df.columns:
            full_df[col + "_smoothed"] = full_df.groupby("symbol")[col].transform(
                lambda x: x.rolling(5, min_periods=1).mean()
            )

    # --------------------------------------------------
    # 4.8 Final feature columns (only those that exist)
    # --------------------------------------------------
    extra_features = [
        "Momentum_60",
        "Vol_Change",
        "Pct_From_52W_High",
        "RSI_smoothed",
        "MACD_smoothed",
        "Volatility_20_smoothed",
    ]
    available_cols = set(full_df.columns)
    feature_cols = [c for c in base_feature_cols + extra_features if c in available_cols]

    print("Using feature columns:")
    for c in feature_cols:
        print(" -", c)

    LABEL_COL = "Target"  # smoothed, market-neutral label for training

    # --------------------------------------------------
    # 4.9 Train/test split (time-based)
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
        num_boost_round=900,
    )

    # --------------------------------------------------
    # 4.10 Evaluation
    # --------------------------------------------------
    y_pred = model.predict(X_test_s)
    print("\nEvaluation on held-out test set (Target = smoothed, market-neutral):")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R2 :", r2_score(y_test, y_pred))

    # Directional accuracy (UP/DOWN) on the target
    pred_up = y_pred > 0
    true_up = y_test > 0
    direction_acc = (pred_up == true_up).mean()
    print("Direction Accuracy (%):", direction_acc * 100.0)

    # --------------------------------------------------
    # 4.11 Save model & scaler
    # --------------------------------------------------
    joblib.dump(model, "xgb_fwd_return_model.joblib")
    joblib.dump(scaler, "xgb_fwd_return_scaler.joblib")
    print("\n✅ Model + scaler saved to disk.\n")

    # --------------------------------------------------
    # 4.12 Backtest (portfolio simulation)
    # --------------------------------------------------
    print("Running Backtest...")

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
