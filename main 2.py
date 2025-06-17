import yfinance as yf
import pandas as pd
import numpy as np
import os
import datetime

from models.option_pricing_models import BlackScholesModel
from strategies.model_mispricing import generate_signals
from backtest.backtester import Backtester

# ----------------------------
# Config
# ----------------------------
TICKER = "AAPL"
RISK_FREE_RATE = 0.04
THRESHOLD = 0.2
MIN_BID = 0.5
STRIKE_WINDOW = 0.1
OPTION_TYPE = 'call'
POSITION_SIZE = 1
STARTING_CASH = 100_000

# ----------------------------
# Load spot and expiration
# ----------------------------
ticker = yf.Ticker(TICKER)
hist = ticker.history(period="5d")
if hist.empty:
    raise ValueError(f"No price data found for {TICKER}")

spot = hist['Close'].dropna().iloc[-1]
expirations = ticker.options
if not expirations:
    raise ValueError(f"No option expirations found for {TICKER}")

EXPIRATION = expirations[0]
T = (pd.to_datetime(EXPIRATION) - pd.Timestamp.today()).days / 365

# ----------------------------
# Load and clean options data
# ----------------------------
chain = ticker.option_chain(EXPIRATION)
calls = chain.calls[['strike', 'impliedVolatility', 'bid', 'ask']].dropna()
calls = calls[
    (calls['strike'] >= spot * (1 - STRIKE_WINDOW)) &
    (calls['strike'] <= spot * (1 + STRIKE_WINDOW)) &
    (calls['bid'] > MIN_BID) &
    (calls['impliedVolatility'] > 0)
]
calls['MidPrice'] = (calls['bid'] + calls['ask']) / 2

if calls.empty:
    raise ValueError("Filtered options DataFrame is empty. Try adjusting filters.")

# ----------------------------
# Apply Pricing Model and Generate Signals
# ----------------------------
model = BlackScholesModel(r=RISK_FREE_RATE)
signals = generate_signals(calls, model, spot, T, threshold=THRESHOLD, option_type=OPTION_TYPE)

# ----------------------------
# Backtest the Signals
# ----------------------------
bt = Backtester(starting_cash=STARTING_CASH, position_size=POSITION_SIZE)
bt.execute_signals(signals)

summary = bt.summary()
trade_log = bt.get_trade_log()
trade_df = pd.DataFrame(trade_log)

# ----------------------------
# Save Results to Excel
# ----------------------------
os.makedirs("results", exist_ok=True)
date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
filename = f"{TICKER}_signals_{EXPIRATION}_{date_str}.xlsx"
output_path = os.path.join("results", filename)

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    trade_df.to_excel(writer, sheet_name="Trades", index=False)
    pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)

print(f"✅ Signals generated: {len(signals)}")
print(f"✅ Results saved to: {output_path}")
