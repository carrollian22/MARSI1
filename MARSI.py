import os
import telegram
from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import timedelta
import pytz
import sys
from io import StringIO
import asyncio
import io

# Initialize Telegram Bot
TELEGRAM_API_TOKEN = os.getenv('TELEGRAM_API_TOKEN')  # Get the Telegram API Token from environment variables
CHAT_ID = '8078910487'  # Replace this with your chat ID or use @username format for the bot to message you

bot = telegram.Bot(token=TELEGRAM_API_TOKEN)

# Send message function
async def send_message(message):
    await bot.send_message(chat_id=CHAT_ID, text=message)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tv = TvDatafeed()

# List of coins
coins = ['BTC', 'ETH', 'SOL', 'SUI', 'ADA', 'AVAX', 'LINK', 'DOT', 'UNI', 'DOGE', 'ICP']

# List to store summary data for each coin
coin_summary = []

# Function to calculate RSI with a 16th value
def calculate_rsi_with_16th_value(last_15_open_values, value, window_length=16):
    updated_values = np.append(last_15_open_values, value)  # Add the new 16th value
    updated_values_series = pd.Series(updated_values)
    delta = updated_values_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/window_length, min_periods=window_length).mean()
    avg_loss = loss.ewm(alpha=1/window_length, min_periods=window_length).mean()
    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1] if avg_loss.iloc[-1] != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Loop through each coin
for coin in coins:
    symbol = f'{coin}USDT.P'
    #print(f"Fetching data for {symbol}...")

    # Fetch data
    data = tv.get_hist(symbol=symbol, exchange='MEXC', interval=Interval.in_15_minute, n_bars=5000)

    if data is not None:
        # Process the data
        data.reset_index(inplace=True)
        data['datetime'] = pd.to_datetime(data['datetime']).dt.tz_localize(None)

        # Feature Engineering
        data['SMA'] = data['open'].rolling(window=192).mean()
        data['SMASlope'] = (data['SMA'] - data['SMA'].shift(8)) / 8
        data['SMASlopeResult'] = data['SMASlope'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        data['SMASlopeP'] = ((data['SMA'] - data['SMA'].shift(8)) / data['SMA'].shift(8)) * 25

        # RSI Calculation
        window_length = 16
        delta = data['open'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/window_length, min_periods=window_length).mean()
        avg_loss = loss.ewm(alpha=1/window_length, min_periods=window_length).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # Calculate the best RSI value
        last_15_open_values = data['open'].iloc[-100:].values
        last_open = last_15_open_values[-1]

        target_rsi = None

        if data['SMASlopeP'].iloc[-1] > 0:
            range_start = last_open * 0.92
            range_end = last_open
            target_rsi = 32.5

        elif data['SMASlopeP'].iloc[-1] < 0:
            range_start = last_open
            range_end = last_open * 1.08
            target_rsi = 67.5

        # Skip processing if target_rsi is None
        if target_rsi is not None:
            range_of_values = np.linspace(range_start, range_end, 400)
            best_value = None
            min_diff = float('inf')

            for value in range_of_values:
                rsi_calculated = calculate_rsi_with_16th_value(last_15_open_values, value)
                diff = abs(rsi_calculated - target_rsi)
                if diff < min_diff:
                    min_diff = diff
                    best_value = value

                # Add summary data for this coin
            coin_summary.append({
                'Coin': coin,
                'RSI': round(data['RSI'].iloc[-1],0),
                'TGT RSI': target_rsi,
                'Best Val': round(best_value, 3)
            })
                    # Print the last 5 rows of data for ETH
        #if coin == 'AVAX':
            #print(f"Last 5 rows of data for {coin}:")
            #print(data.tail(5))
    else:
        print(f"No data retrieved for {symbol}. Skipping.\n")

# Create a summary dataframe
coin_summary_df = pd.DataFrame(coin_summary)
coin_summary_df = coin_summary_df.sort_values(by=['TGT RSI', 'RSI'], ascending=[False, False])

# Display the summary dataframe
# Print the most recent datetime for the current coin
print(f"Most Recent Datetime: {data['datetime'].iloc[-1]}")
print()
print(coin_summary_df)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Initialize a StringIO object to capture output
new_stdout = io.StringIO()

# Redirect standard output to the StringIO object
old_stdout = sys.stdout
sys.stdout = new_stdout

# Display the summary dataframe
# Print the most recent datetime for the current coin
print(f"Most Recent Datetime: {data['datetime'].iloc[-1]}")
print()
print(coin_summary_df.to_string(index=False))

# Capture the output
output = new_stdout.getvalue()

# Reset the standard output to original
sys.stdout = old_stdout

# Send the captured output to Telegram
asyncio.run(send_message(output))









