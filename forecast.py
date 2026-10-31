import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

SYMBOL = "BTCUSDT"
INTERVAL_MS = 5 * 60 * 1000
LOOKBACK = 400
HORIZON = 12
OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)

def fetch_candles():
    endpoints = [
        "https://api.binance.com/api/v3/klines",
        "https://data-api.binance.vision/api/v3/klines",
    ]
    last_error = None
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, params={"symbol": SYMBOL, "interval": "5m", "limit": 500}, timeout=25)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list) or len(data) < LOOKBACK + 1:
                raise ValueError("Invalid or insufficient Binance OHLCV response")
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            complete = [k for k in data if int(k[6]) < now_ms]
            if len(complete) < LOOKBACK:
                raise ValueError("Fewer than 400 complete candles returned")
            rows = complete[-LOOKBACK:]
            df = pd.DataFrame({
                "timestamps": pd.to_datetime([int(k[0]) for k in rows], unit="ms", utc=True).tz_localize(None),
                "open": [float(k[1]) for k in rows],
                "high": [float(k[2]) for k in rows],
                "low": [float(k[3]) for k in rows],
                "close": [float(k[4]) for k in rows],
                "volume": [float(k[5]) for k in rows],
                "amount": [float(k[7]) for k in rows],
            })
            if df[["open", "high", "low", "close"]].isna().any().any():
                raise ValueError("Candles contain NaN")
            if not df.timestamps.is_monotonic_increasing:
                raise ValueError("Candles not in chronological order")
            if (df.timestamps.diff().iloc[1:] != pd.Timedelta(minutes=5)).any():
                raise ValueError("Historical M5 candles contain gaps")
            return df, endpoint
        except Exception as exc:
            last_error = exc
            print(f"Market data endpoint unavailable: {endpoint}: {exc}", file=sys.stderr)
    raise RuntimeError(f"Could not obtain genuine Binance candles: {last_error}")

def main():
    from model import Kronos, KronosTokenizer, KronosPredictor

    history, endpoint = fetch_candles()
    print("Fetched completed Binance candles through", history.timestamps.iloc[-1])
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
    model.eval()
    torch.manual_seed(42)
    np.random.seed(42)
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=400)
    x = history[["open", "high", "low", "close", "volume", "amount"]]
    future_times = pd.Series(pd.date_range(
        history.timestamps.iloc[-1] + pd.Timedelta(minutes=5),
        periods=HORIZON, freq="5min"
    ))
    pred = predictor.predict(
        df=x,
        x_timestamp=history.timestamps.reset_index(drop=True),
        y_timestamp=future_times,
        pred_len=HORIZON,
        T=1.0,
        top_p=0.9,
        sample_count=1
    )
    if len(pred) != HORIZON or not {"open", "high", "low", "close"}.issubset(pred.columns):
        raise ValueError("Kronos did not return the expected OHLC predictions")
    forecast = pred.reset_index(drop=True)[["open", "high", "low", "close"]].copy()
    if not np.isfinite(forecast.to_numpy()).all():
        raise ValueError("Nonfinite Kronos output")
    forecast.insert(0, "time_utc", future_times.dt.strftime("%Y-%m-%dT%H:%M:%SZ").values)
    forecast.to_csv(OUTPUT / "forecast.csv", index=False)
    meta = {
        "model": "NeoQuasar/Kronos-mini",
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-2k",
        "source": "Binance BTCUSDT spot",
        "feed_url": endpoint,
        "timeframe": "5m",
        "last_completed_candle_utc": history.timestamps.iloc[-1].isoformat() + "Z",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "not_exness": True,
        "predictions": forecast.to_dict(orient="records")
    }
    (OUTPUT / "forecast.json").write_text(json.dumps(meta, indent=2))
    print(forecast.to_string(index=False))
    print("Saved genuine model output.")

if __name__ == "__main__":
    main()
