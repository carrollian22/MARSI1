import os
import telegram
from tvDatafeed import TvDatafeed, Interval
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import timedelta
import pytz

# Initialize Telegram Bot
TELEGRAM_API_TOKEN = os.getenv('Telegram_API_Token')  # Get the Telegram API Token from environment variables
CHAT_ID = '8078910487'  # Replace this with your chat ID or use @username format for the bot to message you

bot = telegram.Bot(token=Telegram_API_Token)

# Send message function
def send_message(message):
    bot.send_message(chat_id=CHAT_ID, text=message)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import TVDataFeed Package
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
tv = TvDatafeed()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data and Feature Engineering
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Fetch data
data = tv.get_hist(symbol='BTCUSDT.P', exchange='MEXC', interval=Interval.in_15_minute, n_bars=5000)

# Reset the index to make 'time' a regular column if it was previously the index
data.reset_index(inplace=True)

# Join Data
data['datetime'] = pd.to_datetime(data['datetime']).dt.tz_localize(None)

# Feature Engineering
data.loc[:, 'datetime'] = pd.to_datetime(data['datetime'])
data['EMA'] = data['close'].ewm(span=20, adjust=False).mean()
data['EMAResult'] = data.apply(lambda row: 1 if row['open'] > row['EMA'] else 0, axis=1)
data['SMA'] = data['open'].rolling(window=192).mean()
data['SMASlope'] = (data['SMA'] - data['SMA'].shift(8)) / 8
data['SMASlopeResult'] = data['SMASlope'].apply(lambda x: 1 if x > 3 else (-1 if x < -3 else 0))
data['SMASlopeP'] = ((data['SMA'] - data['SMA'].shift(8)) / data['SMA'].shift(8)) * 25

# RSI
window_length = 16
delta = data['open'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.ewm(alpha=1/window_length, min_periods=window_length).mean()
avg_loss = loss.ewm(alpha=1/window_length, min_periods=window_length).mean()

rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))
data['RSIScore'] = data['RSI'] * (1 - data['SMASlopeP'])
data['DayOfWeek'] = data['datetime'].dt.day_name()
data['Change'] = ((data['close'] - data['open']) / data['open']) * 100
data['Result'] = data['Change'].apply(lambda x: 1 if x > 0 else 0)

# Time-related columns
data['Time'] = pd.to_datetime(data['datetime'], format='%H:%M:%S').dt.time
eastern = pytz.timezone('US/Eastern')
data['LocalTime'] = data['datetime'].dt.tz_localize('UTC').dt.tz_convert(eastern).dt.time

# Drop unnecessary columns
data = data.drop(columns=['index', 'symbol'], errors='ignore')
data['datetime'] = data['datetime'].dt.strftime('%Y-%m-%d %H:%M')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# RSI Prediction
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

# RSI Calculation
def calculate_rsi_with_16th_value(last_15_open_values, value, window_length=16):
    updated_values = np.append(last_15_open_values, value)  # Add the new 16th value
    updated_values_series = pd.Series(updated_values)

    delta = updated_values_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1/window_length, min_periods=window_length).mean()
    avg_loss = loss.ewm(alpha=1/window_length, min_periods=window_length).mean()

    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1] if avg_loss.iloc[-1] != 0 else np.inf  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    return rsi

# Define last 15 values and compute target range for RSI
last_15_open_values = data['open'].iloc[-100:].values  # Get More Values since EWM needs a bigger window in lead up.
last_open = last_15_open_values[-1]

if data['SMASlopeResult'].iloc[-1] > 0:
    range_start = last_open * 0.92
    range_end = last_open
    target_rsi = 32.5
else:
    range_start = last_open * .98
    range_end = last_open * 1.08
    target_rsi = 67.5

range_of_values = np.linspace(range_start, range_end, 400)  # Create 40 steps

best_value = None
min_diff = float('inf')

# Find the best value that minimizes the difference from the target RSI
for value in range_of_values:
    rsi_calculated = calculate_rsi_with_16th_value(last_15_open_values, value)

    diff = abs(rsi_calculated - target_rsi)

    if diff < min_diff:
        min_diff = diff
        best_value = value

# Prepare message to send to Telegram
message = f"""
The best value for RSI {target_rsi} is: {best_value}
Latest RSI: {data['RSI'].iloc[-1]}
Latest Time: {data['LocalTime'].iloc[-1]}
Latest Price: {data['close'].iloc[-1]}
Difference between Latest Price and Best Value: {data['close'].iloc[-1] - best_value}
"""

# Send the message to Telegram
send_message(message)

