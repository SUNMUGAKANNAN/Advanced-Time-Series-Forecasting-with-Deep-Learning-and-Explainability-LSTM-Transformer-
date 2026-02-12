# Time Series Forecasting using LSTM (PyTorch)

##**Project Overview**
This project implements a deep learning approach for time series forecasting using a Long Short-Term Memory (LSTM) neural network built with PyTorch. The model predicts future values from historical time series data that contains trend, seasonality, and noise patterns.

The project also compares LSTM performance with traditional statistical forecasting methods such as ARIMA and Exponential Smoothing (ETS), and includes model explainability using SHAP.

---

## Objectives
- Generate synthetic time series data with realistic patterns
- Build an LSTM forecasting model using PyTorch
- Compare performance with classical forecasting models
- Evaluate using multiple error metrics
- Provide model explainability using SHAP

---

## Technologies Used
- Python
- PyTorch
- NumPy & Pandas
- Matplotlib
- Scikit-learn
- Statsmodels (ARIMA & ETS)
- SHAP (Explainable AI)

---

## Dataset Description
The dataset is synthetically generated to simulate real-world electricity consumption patterns including:

- Trend component
- Daily seasonality
- Weekly seasonality
- Heteroscedastic noise

---

## Model Architecture

### LSTM Network
- Input Size: 1 feature
- Hidden Layers: 2
- Hidden Units: 64
- Forecast Horizon: 24 time steps

The LSTM captures temporal dependencies and long-term patterns in sequential data.

---

## Evaluation Metrics
The model is evaluated using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

---

## Baseline Models for Comparison
Traditional forecasting methods used:

- ARIMA
- Exponential Smoothing (ETS)

Performance comparison is presented in a results table.

---

## Explainable AI (SHAP)
SHAP is used to interpret model predictions and understand feature importance across time steps.

---

```bash
pip install numpy pandas matplotlib torch scikit-learn statsmodels shap
