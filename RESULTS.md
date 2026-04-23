# Model Results & Performance

## Key Metrics (Test Set — Last 20% of Data)

| Metric | Value | Notes |
|---|---|---|
| Directional Accuracy | 51.3% | % correct up/down predictions. >52% = tradeable signal |
| CV R² (returns) | ~0.00 | Expected for return prediction — random walk baseline |
| Conformal Coverage | 91.5% | Target 90% — calibrated intervals work correctly |
| CP Interval Width | ±8.0% | 90% of true returns fall within this band |
| CV RMSE | 3.12% | Walk-forward cross-validation RMSE on daily returns |

## Backtesting Results (Test Period)

| Strategy | Total Return | Sharpe | Max Drawdown |
|---|---|---|---|
| Binary Long/Flat | See outputs/backtest_curves.csv | — | — |
| Kelly-Sized | See outputs/backtest_curves.csv | — | — |
| Buy & Hold | See outputs/backtest_curves.csv | — | — |

> Run `python main.py` to regenerate with latest data.

## Baseline Comparison

| Model | Dir Acc | Notes |
|---|---|---|
| Naive (always predict up) | ~53% | NFLX has positive drift |
| Random | 50% | Coin flip |
| This model | 51.3% | Slightly above random — honest result |

## Why R² Near Zero Is Correct

Predicting **absolute price** gives R² > 0.99 — but this is trivially achieved by predicting "tomorrow ≈ today". It measures nothing useful.

Predicting **daily returns** gives R² near 0 — this is the honest baseline. Stock returns are close to a random walk. Any model claiming R² > 0.1 on returns without data leakage should be scrutinised.

The meaningful metric is **directional accuracy** and **Sharpe ratio of the resulting strategy**.

## Conformal Prediction Validation

The conformal predictor achieves 91.5% empirical coverage on the test set against a 90% target. This confirms the intervals are properly calibrated — not just wide enough to always be right.

## Limitations

- No earnings surprise signal (biggest NFLX driver — single events move stock ±15%)
- Technical indicators are correlated — ~10 independent signals, not 51
- 15-minute delayed Yahoo Finance data
- Model trained on 2002–2026; pre-streaming era data may not generalise
