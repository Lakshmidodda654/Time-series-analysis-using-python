import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load the time series data from a CSV file
data = pd.read_csv('time_series_data.csv')

# Convert the date column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Set the date column as the index
data.set_index('Date', inplace=True)

# Visualize the time series data
plt.figure(figsize=(10, 6))
plt.plot(data)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Time Series Data')
plt.show()

# Split the data into training and testing sets
train_data = data.iloc[:-12]  # Use all but the last 12 months for training
test_data = data.iloc[-12:]   # Use the last 12 months for testing

# Fit the ARIMA model
model = ARIMA(train_data, order=(1, 1, 1))  # (p, d, q) values can be adjusted
model_fit = model.fit()

# Make predictions
predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

# Visualize the predicted values against the actual values
plt.figure(figsize=(10, 6))
plt.plot(train_data, label='Training Data')
plt.plot(test_data, label='Actual Data')
plt.plot(test_data.index, predictions, label='Predicted Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Time Series Prediction')
plt.legend()
plt.show()