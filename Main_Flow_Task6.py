import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

df = pd.read_csv(r"A:\PycharmProjects\Main__Flow_Task4\Main_flow_Task4\sales_data.csv", parse_dates=['Date'])

# Set Date as index and explicitly set the frequency to daily (D)
df.set_index('Date', inplace=True)
df = df.asfreq('D')

df['7-Day MA'] = df['Sales'].rolling(window=7).mean()

# Trend Analysis Plot (Historical Sales)
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Sales'], label='Daily Sales', alpha=0.5)
plt.plot(df.index, df['7-Day MA'], label='7-Day Moving Average', color='red', linewidth=2)
plt.title('Sales Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Prepare training and test data
train_data = df['Sales'][:int(0.8 * len(df))]
test_data = df['Sales'][int(0.8 * len(df)):]

# Fit ARIMA model
model = ARIMA(train_data, order=(5, 1, 0))
model_fit = model.fit()

# Forecast future sales
forecast = model_fit.forecast(steps=len(test_data))

# Forecasting Plot (Predicted vs Actual Sales)
plt.figure(figsize=(14, 6))
plt.plot(df.index[:len(train_data)], train_data, label='Training Data')
plt.plot(df.index[len(train_data):], test_data, label='Actual Sales')
plt.plot(df.index[len(train_data):], forecast, label='Forecasted Sales', color='green', linestyle='--')
plt.title('Sales Forecast with ARIMA')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Create a DataFrame for the forecasted sales
forecasted_sales_df = pd.DataFrame({
    'Date': df.index[len(train_data):],
    'Actual Sales': test_data,
    'Forecasted Sales': forecast
})

# Display the forecasted sales (for deliverable)
print(forecasted_sales_df)

# Calculate RMSE and MAPE
rmse = np.sqrt(mean_squared_error(test_data, forecast))
print(f'RMSE: {rmse}')

mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
print(f'MAPE: {mape}%')
