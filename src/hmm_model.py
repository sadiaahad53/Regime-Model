import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def train_hmm(
    df: pd.DataFrame,
    n_states: int = 4,
    covariance_type: str = "full",
) -> pd.DataFrame:
    """
    Trains a Gaussian Hidden Markov Model using:
      - Returns
      - Volatility

    Adds columns:
      - Regime
      - Regime_Prob_0 ... Regime_Prob_N
    """

    features = df[["Returns", "Volatility"]].values

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=500,
        random_state=42,
    )

    model.fit(features)

    regimes = model.predict(features)
    probs = model.predict_proba(features)

    df = df.copy()
    df["Regime"] = regimes

    for i in range(n_states):
        df[f"Regime_Prob_{i}"] = probs[:, i]

    return df
