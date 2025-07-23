# Conformal Prediction for Systematic Trading Backtesting

This project implements a walk-forward backtesting framework for financial trading strategies in MATLAB. It enhances a baseline LSTM model with a Conformal Prediction (CP) layer to provide statistically robust confidence intervals for return forecasts. The entire system is designed to be managed and optimized using MATLAB's Experiment Manager.

## Project Files

- `runConformalBacktest.m`: The main MATLAB function for the Experiment Manager. It contains the full walk-forward logic with periodic retraining and selectable signal generation rules.
- `test.m`: A standalone script for diagnostic purposes, such as validating the empirical coverage rate of the conformal prediction intervals. **(Recommended to rename to `validate_coverage.m`)**.
- `runLSTMBACKtest.m`: A baseline backtest using only the raw LSTM model, useful for performance comparison.
- `BacktestStrategiesUsingDeepLearningExample.mlx`: The original MathWorks example notebook that this project is based on.
- `energyPrices.mat`: The raw energy futures price data required to run the backtests.
- `lstmBacktestNetwork.mat`: A pre-trained version of the LSTM network for quick tests.

## How to Run

### 1. Run the Full Experiment
1.  Open MATLAB and ensure the project folder is in the MATLAB path.
2.  Open the **Experiment Manager** app from the **APPS** tab.
3.  Create a new "General Purpose" experiment.
4.  Set the **Experiment Function** to `runConformalBacktest`.
5.  In the **Hyperparameters** section, define the parameters you wish to test (e.g., `alpha`, `retrainFreq`, `signalLogic`).
6.  Click **Run** to execute the backtest sweep.

### 2. Run Diagnostics
1.  Open the `test.m` script in MATLAB.
2.  Modify the parameters in the first section of the script as needed.
3.  Click **Run**. The script will output the step-by-step validation of the conformal prediction intervals and a final summary of the coverage rate.

## Dependencies
- MATLAB (R2023b or newer recommended)
- Deep Learning Toolbox
- Financial Toolbox
- Statistics and Machine Learning Toolbox