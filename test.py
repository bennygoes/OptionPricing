import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models.option_pricing_models import BlackScholesModel, BinomialTreeModel, SABRModel

# ----------------------------
# Configuration
# ----------------------------
TICKER = "AAPL"
RISK_FREE_RATE = 0.04
STRIKE_WINDOW = 0.10  # ±10% around spot
MIN_BID = 0.5
BINOMIAL_STEPS = 100
SABR_PARAMS = {'alpha': 0.3, 'beta': 0.5, 'rho': -0.3, 'nu': 0.5}

# ----------------------------
# Load Spot Price & Expiration
# ----------------------------
ticker = yf.Ticker(TICKER)

# Safe fallback for spot price
hist = ticker.history(period="5d")
if hist.empty:
    raise ValueError(f"No price data found for {TICKER}")
spot_price = hist['Close'].dropna().iloc[-1]

# Get the nearest available expiration
expirations = ticker.options
if not expirations:
    raise ValueError(f"No option expirations available for {TICKER}")
EXPIRATION = expirations[0]
T = (pd.to_datetime(EXPIRATION) - pd.Timestamp.today()).days / 365

# ----------------------------
# Load Option Chain and Clean
# ----------------------------
calls = ticker.option_chain(EXPIRATION).calls.copy()
calls = calls[['strike', 'impliedVolatility', 'bid', 'ask']].dropna()
calls = calls[
    (calls['strike'] >= spot_price * (1 - STRIKE_WINDOW)) &
    (calls['strike'] <= spot_price * (1 + STRIKE_WINDOW)) &
    (calls['bid'] > MIN_BID) &
    (calls['impliedVolatility'] > 0)
]
calls['MidPrice'] = (calls['bid'] + calls['ask']) / 2

# ----------------------------
# Initialize Models
# ----------------------------
bs_model = BlackScholesModel(r=RISK_FREE_RATE)
bt_model = BinomialTreeModel(r=RISK_FREE_RATE, steps=BINOMIAL_STEPS)
sabr_model = SABRModel(**SABR_PARAMS, r=RISK_FREE_RATE)

# ----------------------------
# Apply Models
# ----------------------------
bs_prices, bt_prices, sabr_prices = [], [], []
bs_errors, bt_errors, sabr_errors = [], [], []

for _, row in calls.iterrows():
    K = row['strike']
    sigma = row['impliedVolatility']
    market_price = row['MidPrice']

    # Price using each model
    bs_p = bs_model.price(spot_price, K, T, sigma, 'call')
    bt_p = bt_model.price(spot_price, K, T, sigma, 'call')
    sabr_p = sabr_model.price(spot_price, K, T, 'call')

    # Append prices
    bs_prices.append(bs_p)
    bt_prices.append(bt_p)
    sabr_prices.append(sabr_p)

    # Relative errors
    bs_errors.append(abs(bs_p - market_price) / market_price)
    bt_errors.append(abs(bt_p - market_price) / market_price)
    sabr_errors.append(abs(sabr_p - market_price) / market_price)

# Add results to DataFrame
calls['BSPrice'] = bs_prices
calls['BS_RelError'] = bs_errors

calls['BTPrice'] = bt_prices
calls['BT_RelError'] = bt_errors

calls['SABRPrice'] = sabr_prices
calls['SABR_RelError'] = sabr_errors

# ----------------------------
# Show Result Table
# ----------------------------
output_df = calls[['strike', 'MidPrice',
                   'BSPrice', 'BS_RelError',
                   'BTPrice', 'BT_RelError',
                   'SABRPrice', 'SABR_RelError']]

output_path = f"results/{TICKER}_option_model_errors_{EXPIRATION}.xlsx"
output_df.to_excel(output_path, index=False)

print(f"Saved results to {output_path}")

# ----------------------------
# Optional: Plot Error Comparison
# ----------------------------
plt.figure(figsize=(10, 6))
plt.plot(calls['strike'], calls['BS_RelError'], label='Black-Scholes', marker='o')
plt.plot(calls['strike'], calls['BT_RelError'], label='Binomial Tree', marker='x')
plt.plot(calls['strike'], calls['SABR_RelError'], label='SABR', marker='s')
plt.xlabel("Strike Price")
plt.ylabel("Relative Error")
plt.title(f"{TICKER} Option Model Pricing Errors ({EXPIRATION})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
