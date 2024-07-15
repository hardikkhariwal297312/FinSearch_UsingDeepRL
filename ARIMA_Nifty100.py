import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima.arima.utils import ndiffs
from pmdarima import auto_arima
import statsmodels.api as sm

# Load the training and testing datasets
train_df = pd.read_csv("Nifty100_3yrs.csv")
test_df = pd.read_csv("Nifty100.csv")

# Convert the 'Date' column to datetime
train_df['Date'] = pd.to_datetime(train_df['Date'])
test_df['Date'] = pd.to_datetime(test_df['Date'])

# Set the 'Date' column as the index
train_df.set_index('Date', inplace=True)
test_df.set_index('Date', inplace=True)

# Extract the 'Close' column for modeling
train_data = train_df['Close']
test_data = test_df['Close']

# Perform differencing test on the training data
train_data_diff = train_data.diff().dropna()
d_value = ndiffs(train_data_diff, test="adf")

# Train an ARIMA model using the training data
model = sm.tsa.ARIMA(train_data_diff, order=(5, 2, 0))
model_fit = model.fit()

# Forecast using the trained ARIMA model
forecast_diff = model_fit.forecast(steps=len(test_data))

# Inverse the differencing to get forecasted values
forecast = train_data.iloc[-1] + np.cumsum(forecast_diff)

# Calculate forecasted daily returns for the testing dataset
forecasted_returns = np.diff(forecast) / forecast[:-1]

# Calculate cumulative returns for  forecasted returns
cumulative_forecasted_returns = np.cumprod(1 + forecasted_returns) - 1

# Calculate annualized mean return and annualized volatility for forecasted returns


annualized_mean_return_forecasted = np.mean(forecasted_returns) * 252
annualized_volatility_forecasted = np.std(forecasted_returns) * np.sqrt(252)
sharpe_ratio_forecasted = (annualized_mean_return_forecasted - 0.02) / annualized_volatility_forecasted
max_drawdown_forecasted = (np.maximum.accumulate(cumulative_forecasted_returns) - cumulative_forecasted_returns).max()

print("\nForecasted Returns Performance:")
print("Annualized Mean Return:", annualized_mean_return_forecasted)
print("Annualized Volatility:", annualized_volatility_forecasted)
print("Sharpe Ratio:", sharpe_ratio_forecasted)
print("Maximum Drawdown:", max_drawdown_forecasted)



