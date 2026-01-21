import pandas as pd


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates simple signals from:
      - Regime (from HMM)
      - Optional moving-average trend confirmation (MA_9 vs MA_21)

    Output columns:
      - HMM_Signal  (1 if in "good" regimes, else 0)
      - MA_Signal   (1 if MA_9 > MA_21, else 0)
      - Main_Signal (HMM_Signal AND MA_Signal)

    Notes:
      - We don't guess regime labels; we keep it simple:
        choose favorable regimes as the top half by average Returns.
    """
    out = df.copy()

    if "Regime" not in out.columns:
        raise ValueError("Missing 'Regime'. Run train_hmm() first.")
    if "Returns" not in out.columns:
        raise ValueError("Missing 'Returns'. Run build_features() first.")

    # Determine "favorable" regimes data-driven:
    # take top half of regimes by mean returns.
    regime_means = out.groupby("Regime")["Returns"].mean().sort_values(ascending=False)
    n_regimes = len(regime_means)
    if n_regimes < 2:
        raise ValueError("Need at least 2 regimes to generate signals.")

    top_k = max(1, n_regimes // 2)  # top half
    favorable_states = set(regime_means.head(top_k).index.tolist())

    out["HMM_Signal"] = out["Regime"].isin(favorable_states).astype(int)

    # Optional MA trend confirmation
    if "MA_9" in out.columns and "MA_21" in out.columns:
        out["MA_Signal"] = (out["MA_9"] > out["MA_21"]).astype(int)
    else:
        out["MA_Signal"] = 1  # if MAs not present, don't block trades

    out["Main_Signal"] = (out["HMM_Signal"] & out["MA_Signal"]).astype(int)

    return out
