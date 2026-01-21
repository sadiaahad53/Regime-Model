import os
import yaml
from src.data_loader import DataLoader, FMPConfig
from src.feature_engineering import build_features
from src.hmm_model import train_hmm
from src.signal_generation import generate_signals
from src.backtesting import run_backtest


def main():

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise RuntimeError("FMP_API_KEY not set")

    fmp_config = FMPConfig(api_key=api_key)

    data_loader = DataLoader(
        fmp_config=fmp_config,
        symbol=config["data"]["symbol"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
        raw_data_path=config["data"]["raw_data_path"],
    )

    df = data_loader.load_data()

    df = build_features(df)

    df = train_hmm(
        df,
        n_states=config["model"]["n_states"],
        covariance_type=config["model"]["covariance_type"],
    )

    df = generate_signals(df)

    results = run_backtest(
        df,
        annual_trading_days=config["backtesting"]["annual_trading_days"],
    )

    print("\n========== RESULTS ==========")
    for k, v in results.items():
        print(f"{k}: {v:.3f}")


if __name__ == "__main__":
    main()
