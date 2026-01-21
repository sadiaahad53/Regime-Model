import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds features for the HMM:
      - Returns: log returns
      - Volatility: rolling std of returns
      - MA_9 / MA_21: optional trend filters (kept for signals)
    Requires 'Close' column.
    """
    out = df.copy()

    if "Close" not in out.columns:
        raise ValueError("Expected column 'Close' in price dataframe")

    # log returns
    out["Returns"] = np.log(out["Close"]).diff()

    # rolling volatility (20 trading days)
    out["Volatility"] = out["Returns"].rolling(20).std()

    # optional moving averages (trend confirmation)
    out["MA_9"] = out["Close"].rolling(9).mean()
    out["MA_21"] = out["Close"].rolling(21).mean()

    out = out.dropna()
    return out
