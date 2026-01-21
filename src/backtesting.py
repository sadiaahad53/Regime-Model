import numpy as np
import pandas as pd


def run_backtest(df: pd.DataFrame, annual_trading_days: int = 252) -> dict:
    """
    Simple long/flat backtest using Main_Signal or HMM_Signal if available.

    Required:
      - Close
      - Returns (log returns)

    Uses:
      - Main_Signal if present else HMM_Signal else MA_Signal
      Signal convention: 1 = in market, 0 = out of market

    Outputs:
      - Benchmark_Sharpe
      - Strategy_Sharpe
      - Benchmark_CAGR
      - Strategy_CAGR
      - Benchmark_TotalReturn
      - Strategy_TotalReturn
    """
    out = df.copy()

    if "Returns" not in out.columns:
        raise ValueError("Missing 'Returns' column. Run build_features() first.")

    # pick signal column
    signal_col = None
    for c in ["Main_Signal", "HMM_Signal", "MA_Signal"]:
        if c in out.columns:
            signal_col = c
            break

    if signal_col is None:
        raise ValueError("No signal column found (expected Main_Signal/HMM_Signal/MA_Signal).")

    # ensure binary 0/1
    out[signal_col] = (out[signal_col].fillna(0) > 0).astype(int)

    # Strategy return = signal shifted by 1 day (avoid lookahead)
    out["Strategy_Returns"] = out["Returns"] * out[signal_col].shift(1).fillna(0)

    # Benchmark returns = always in market
    out["Benchmark_Returns"] = out["Returns"]

    def sharpe(x: pd.Series) -> float:
        x = x.dropna()
        if x.std() == 0 or len(x) < 2:
            return float("nan")
        return (x.mean() / x.std()) * np.sqrt(annual_trading_days)

    def total_return(x: pd.Series) -> float:
        return float(np.exp(x.sum()) - 1.0)

    def cagr(x: pd.Series) -> float:
        x = x.dropna()
        if len(x) < 2:
            return float("nan")
        years = len(x) / annual_trading_days
        if years <= 0:
            return float("nan")
        tr = np.exp(x.sum())
        return float(tr ** (1 / years) - 1)

    results = {
        "Benchmark_Sharpe": sharpe(out["Benchmark_Returns"]),
        "Strategy_Sharpe": sharpe(out["Strategy_Returns"]),
        "Benchmark_CAGR": cagr(out["Benchmark_Returns"]),
        "Strategy_CAGR": cagr(out["Strategy_Returns"]),
        "Benchmark_TotalReturn": total_return(out["Benchmark_Returns"]),
        "Strategy_TotalReturn": total_return(out["Strategy_Returns"]),
    }

    return results
