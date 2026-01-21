import os
import pandas as pd
from dataclasses import dataclass


@dataclass
class FMPConfig:
    api_key: str


class DataLoader:
    def __init__(
        self,
        fmp_config: FMPConfig,
        symbol: str,
        start_date: str,
        end_date: str,
        raw_data_path: str,
    ):
        self.fmp = fmp_config
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data_path = raw_data_path

    def load_data(self) -> pd.DataFrame:
        if os.path.exists(self.raw_data_path):
            print("Loading cached data")
            df = pd.read_csv(self.raw_data_path, parse_dates=["Date"])
            df.set_index("Date", inplace=True)
            return df.sort_index()

        print(f"Downloading from FMP (Stable): {self.symbol}")
        df = self._fetch_fmp()

        os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
        df.to_csv(self.raw_data_path)

        return df

    def _fetch_fmp(self) -> pd.DataFrame:
        import requests

        url = "https://financialmodelingprep.com/stable/historical-price-eod/full"
        params = {
            "symbol": self.symbol,
            "from": self.start_date,
            "to": self.end_date,
            "apikey": self.fmp.api_key,
        }

        print("HITTING:", url)

        r = requests.get(url, params=params, timeout=30)

        if r.status_code != 200:
            try:
                msg = r.json()
            except Exception:
                msg = r.text
            raise RuntimeError(f"FMP request failed ({r.status_code}): {msg}")

        payload = r.json()

        # ✅ FMP can return:
        # 1) dict: {"symbol": "...", "historical": [...]}
        # 2) dict: {"data": [...]}
        # 3) list: [...]
        if isinstance(payload, dict):
            hist = payload.get("historical") or payload.get("data")
        elif isinstance(payload, list):
            hist = payload
        else:
            hist = None

        if not hist:
            raise RuntimeError(f"Empty/unknown FMP response type: {type(payload)}")

        df = pd.DataFrame(hist)

        # Normalize column names
        df.rename(
            columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "adjClose": "Adj Close",
                "volume": "Volume",
            },
            inplace=True,
        )

        if "Date" not in df.columns:
            raise RuntimeError(f"Missing 'date' field in FMP data. Columns: {list(df.columns)}")

        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        # Make numeric (safe)
        for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df.dropna(subset=["Close"], inplace=True)
        return df
