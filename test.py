import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models.option_pricing_models import BlackScholesModel, BinomialTreeModel, SABRModel

# ----------------------------
# Config
# ----------------------------
TICKER = "AAPL"
EXPIRATION = "2024-12-20"
RISK_FREE_RATE = 0.04
BINOMIAL_STEPS = 100
SABR_PARAMS = {'alpha': 0.3, 'beta': 0.5, 'rho': -0.3, 'nu': 0.5}
MIN_BID = 0.5  # Filter out low-liquidity options

# ----------------------------
# Load Option Chain
# ----------------------------
ticker = yf.Ticker(TICKER)
opt_chain = ticker.option_chain(EXPIRATION)
calls = opt_chain.calls.copy()

spot_price = ticker.history(period="1d")['Close'].iloc[-1]
T = (pd.to_datetime(EXPIRATION) - pd.Timestamp.today()).days / 365

calls = calls[['strike', 'impliedVolatility', 'bid', 'ask']].dropna()
calls = calls[(calls['impliedVolatility'] > 0) & (calls['bid'] > MIN_BID)]
calls['MidPrice'] = (calls['bid'] + calls['ask']) / 2

# ----------------------------
# Initialize Models
# ----------------------------
bs_model = BlackScholesModel(r=RISK_FREE_RATE)
bt_model = BinomialTreeModel(r=RISK_FREE_RATE, steps=BINOMIAL_STEPS)
sabr_model = SABRModel(**SABR_PARAMS, r=RISK_FREE_RATE)

# ----------------------------
# Compute Model Prices and Errors
# ----------------------------
bs_prices = []
bt_prices = []
sabr_prices = []

for _, row in calls.iterrows():
    K = row['strike']
    sigma = row['impliedVolatility']
    market_price = row['MidPrice']

    bs_price = bs_model.price(spot_price, K, T, sigma, 'call')
    bt_price = bt_model.price(spot_price, K, T, sigma, 'call')
    sabr_price = sabr_model.price(spot_price, K, T, 'call')

    bs_prices.append(bs_price)
    bt_prices.append(bt_price)
    sabr_prices.append(sabr_price)

calls['BSPrice'] = bs_prices
calls['BTPrice'] = bt_prices
calls['SABRPrice'] = sabr_prices

calls['BS_RelError'] = np.abs(calls['BSPrice'] - calls['MidPrice']) / calls['MidPrice']
calls['BT_RelError'] = np.abs(calls['BTPrice'] - calls['MidPrice']) / calls['MidPrice']
calls['SABR_RelError'] = np.abs(calls['SABRPrice'] - calls['MidPrice']) / calls['MidPrice']

# ----------------------------
# Display Error Summary
# ----------------------------
print(calls[['strike', 'MidPrice', 'BS_RelError', 'BT_RelError', 'SABR_RelError']].head(10))

# ----------------------------
# Plot Relative Errors
# ----------------------------
plt.figure(figsize=(10, 6))
plt.plot(calls['strike'], calls['BS_RelError'], label='Black-Scholes', marker='o')
plt.plot(calls['strike'], calls['BT_RelError'], label='Binomial Tree', marker='x')
plt.plot(calls['strike'], calls['SABR_RelError'], label='SABR', marker='s')
plt.xlabel("Strike Price")
plt.ylabel("Relative Error")
plt.title(f"{TICKER} Option Model Relative Pricing Errors")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
