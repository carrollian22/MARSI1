# -*- coding: utf-8 -*-
"""MARSI.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1yU9UyGTaPBJfjXQOlsTlcaS-qpo_b1rF
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import TVDataFeed Package
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git

from tvDatafeed import TvDatafeed, Interval

tv = TvDatafeed()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data and Feature Engineering
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import pytz

# ~~~~~~~~~~~~~ Index Data ~~~~~~~~~~~~~~~~~~~~
data = tv.get_hist(symbol='BTCUSDT.P',exchange='MEXC',interval=Interval.in_15_minute,n_bars=5000)

# Reset the index to make 'time' a regular column if it was previously the index
data.reset_index(inplace=True)

#print(index_data)

# ~~~~~~~~~~~~~~~~~ Joining Data ~~~~~~~~~~~~~~~~~
# Ensure 'datetime' in both DataFrames is of datetime type
data['datetime'] = pd.to_datetime(data['datetime']).dt.tz_localize(None)

# ~~~~~~~~~~~~ Feature Engineering ~~~~~~~~~~~~~~~~~~
# Reset index to make 'Datetime' a separate column
data.reset_index(inplace=True)

# Convert 'Datetime' to datetime object (this might already be in datetime format after reset_index)
#data['Datetime'] = pd.to_datetime(data['datetime'])

# Convert 'Datetime' to datetime object explicitly using .loc
data.loc[:, 'datetime'] = pd.to_datetime(data['datetime'])

# Calculate EMA with a period of n = 17
data['EMA'] = data['close'].ewm(span=20, adjust=False).mean()

# Create 'EMAResult' column: 1 if 'open' is above EMA, else 0
data['EMAResult'] = data.apply(lambda row: 1 if row['open'] > row['EMA'] else 0, axis=1)

# Calculate the 192-period Simple Moving Average (SMA)
data['SMA'] = data['open'].rolling(window=192).mean()

# Calculate the SMASlope with an 8-period window
data['SMASlope'] = (data['SMA'] - data['SMA'].shift(8)) / 8

# Create the SMASlopeResult feature with conditions for >10, <-10, and else 0
data['SMASlopeResult'] = data['SMASlope'].apply(lambda x: 1 if x > 3 else (-1 if x < -3 else 0))

# Calculate SMASlopeP (percent change of the SMA over 8 periods)
data['SMASlopeP'] = ((data['SMA'] - data['SMA'].shift(8)) / data['SMA'].shift(8)) * 25

# RSI
# Exponential Moving Average (Wilder's smoothing) for RSI calculation
window_length = 16
delta = data['open'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

# Use exponential weighted mean for Wilder's smoothing
avg_gain = gain.ewm(alpha=1/window_length, min_periods=window_length).mean()
avg_loss = loss.ewm(alpha=1/window_length, min_periods=window_length).mean()

# Calculate Relative Strength (RS) and RSI
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# Calculate RSIScore as RSI * (1 - SMASlopeP)
data['RSIScore'] = data['RSI'] * (1 - data['SMASlopeP'])

# Create 'DayOfWeek' column
data['DayOfWeek'] = data['datetime'].dt.day_name()

# Calculate 'Change' and create 'Result' column
data['Change'] = ((data['close'] - data['open']) / data['open']) * 100
data['Result'] = data['Change'].apply(lambda x: 1 if x > 0 else 0)

# Extract 'Time' from 'Datetime' and add it as a separate column in 'time_data'
data['Time'] = pd.to_datetime(data['datetime'], format='%H:%M:%S').dt.time

# Define the timezone (Eastern Time, which observes daylight savings)
eastern = pytz.timezone('US/Eastern')
data['LocalTime'] = data['datetime'].dt.tz_localize('UTC').dt.tz_convert(eastern).dt.time

# Drop the 'index' column if it exists
data = data.drop(columns=['index', 'symbol'], errors='ignore')

# Format 'datetime' to remove seconds and keep only the date and hour:minute
data['datetime'] = data['datetime'].dt.strftime('%Y-%m-%d %H:%M')



# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# RSI Prediction
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
import numpy as np

# Define function to calculate RSI
def calculate_rsi_with_16th_value(last_15_open_values, value, window_length=16):
    # Insert the new value at the 16th position
    updated_values = np.append(last_15_open_values, value)  # Add the new 16th value

    # Convert to pandas Series for easier manipulation
    updated_values_series = pd.Series(updated_values)

    # Calculate the price changes (delta) for the updated values
    delta = updated_values_series.diff()

    # Calculate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Use exponential weighted mean for Wilder's smoothing (EWMA) over the entire series
    avg_gain = gain.ewm(alpha=1/window_length, min_periods=window_length).mean()
    avg_loss = loss.ewm(alpha=1/window_length, min_periods=window_length).mean()

    # Now, we use the final smoothed value of avg_gain and avg_loss to calculate RSI
    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1] if avg_loss.iloc[-1] != 0 else np.inf  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    return rsi

# Example of your code where you use this function
last_15_open_values = data['open'].iloc[-100:].values # Get More Values since EWM needs a bigger window in lead up.
last_open = last_15_open_values[-1]
#print(last_15_open_values)

if data['SMASlopeResult'].iloc[-1] > 0:
    # Range is 0% to -4% from last_open and target RSI is 30
    range_start = last_open * 0.92
    range_end = last_open
    target_rsi = 32.5
else:
    # Range is 0% to +4% from last_open and target RSI is 70
    range_start = last_open * .98
    range_end = last_open * 1.08
    target_rsi = 67.5

range_of_values = np.linspace(range_start, range_end, 400)  # Create 40 steps

best_value = None
min_diff = float('inf')

for value in range_of_values:
    # Insert the value as the 16th data point and calculate RSI
    rsi_calculated = calculate_rsi_with_16th_value(last_15_open_values, value)

    # Print the current value and its corresponding RSI
    #print(f"Value: {value}, Calculated RSI: {rsi_calculated}")

    # Calculate the difference from the target RSI
    diff = abs(rsi_calculated - target_rsi)

    # Update the best value if this RSI is closer to the target
    if diff < min_diff:
        min_diff = diff
        best_value = value




print(f"The best value for RSI {target_rsi} is: {best_value}")
print()
print(f"Latest RSI: {data['RSI'].iloc[-1]}")
print(f"Latest Time: {data['LocalTime'].iloc[-1]}")
print(f"Latest Price: {data['close'].iloc[-1]}")
print(f"Difference between Latest Price and Best Value: {data['close'].iloc[-1] - best_value}")
