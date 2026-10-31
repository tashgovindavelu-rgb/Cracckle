# Kronos BTC/USDT M5 Forecasting

This repository contains a **real Kronos** inference workflow, not ChatGPT-generated trading signals.

## Run
1. Open [Actions](https://github.com/tashgovindavelu-rgb/Cracckle/actions/workflows/kronos.yml).
2. Select **Run workflow** then **Run workflow**.
3. When finished, open the run and download the `kronos-btc-m5-forecast` artifact with CSV and JSON results.

The public repository is free to run under GitHub's current public-repository Actions allowance, subject to GitHub availability and limits. First run downloads model weights. GitHub-hosted CPU execution may be slow or fail due to RAM/network constraints.

- Source: https://github.com/shiyu-coder/Kronos
- Model: `NeoQuasar/Kronos-mini`
- Tokenizer: `NeoQuasar/Kronos-Tokenizer-2k`
- Market feed: Binance BTCUSDT 5-minute **spot candles**; **not Exness BTCUSD**. Prices may differ.
- Lookback: 400 *completed* candles; next 12 candles predicted.
- No broker connectivity or trade execution.
- Failure exits nonzero; it never invents numbers.

## Accuracy
Kronos forecasts are probabilistic and unverified as trading signals. Trading carries risk.
