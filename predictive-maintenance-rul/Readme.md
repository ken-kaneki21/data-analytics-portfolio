# Predictive Maintenance – Remaining Useful Life (RUL)

## Objective
Predict Remaining Useful Life (RUL) of aircraft engines using NASA CMAPSS FD001 dataset and evaluate fleet-level risk using machine learning and Power BI.

## Dataset
- Source: NASA CMAPSS FD001
- 100 engines
- Multivariate sensor time-series
- RUL capped at 125 cycles for stability

## Feature Engineering
- Lag features (up to 3 cycles)
- Rolling statistics (mean, std over 5, 10, 20 cycles)
- Engine-level validation splits (no data leakage)

## Models
- Mean Predictor (baseline)
- Ridge Regression
- Random Forest Regressor (final)
- Histogram Gradient Boosting (comparison)

## Evaluation
- MAE, RMSE
- Within ±20 cycles accuracy
- Cross-engine backtesting (3 folds)

## Results (RUL Capped at 125)
- MAE ≈ 14
- RMSE ≈ 19
- ±20 cycles accuracy ≈ 74%

## Dashboard
Power BI dashboard includes:
- Fleet risk overview
- Engine-level drilldown
- Actual vs predicted RUL trajectories
- Risk buckets: Critical / Warning / Healthy

## Tools
Python, scikit-learn, Power BI, Pandas, NumPy
