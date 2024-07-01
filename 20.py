import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Generate synthetic data
date_range = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
np.random.seed(42)
sales_data = np.random.poisson(lam=200, size=len(date_range))

data = pd.DataFrame({'date': date_range, 'sales': sales_data})

# Split data into training and testing sets
train_data = data[data['date'] < '2023-01-01']
test_data = data[data['date'] >= '2023-01-01']

# Train the Exponential Smoothing model
model = ExponentialSmoothing(train_data['sales'], trend='add', seasonal='add', seasonal_periods=365)
fitted_model = model.fit()


# Predict future sales
future_dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
future_forecast = fitted_model.forecast(steps=len(future_dates))

# Plot future sales prediction
plt.figure(figsize=(10, 5))
plt.plot(data['date'], data['sales'], label='Historical Sales')
plt.plot(future_dates, future_forecast, label='Future Predicted Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Future Sales Prediction')
plt.legend()
plt.show()
