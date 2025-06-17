# strategies/model_mispricing.py

def generate_signals(df, model, spot, T, threshold=0.2, option_type='call'):
    """
    Generate trading signals based on model-mispricing.
    
    Parameters:
    - df: DataFrame with 'strike', 'impliedVolatility', 'MidPrice'
    - model: A pricing model class with a `.price()` method
    - spot: Current spot price of the underlying asset
    - T: Time to expiration in years
    - threshold: relative error threshold for trading
    - option_type: 'call' or 'put'
    
    Returns:
    - List of signals: dicts with keys {action, strike, market, model, error}
    """
    signals = []

    for _, row in df.iterrows():
        K = row['strike']
        sigma = row['impliedVolatility']
        market = row['MidPrice']
        model_price = model.price(spot, K, T, sigma, option_type)

        rel_error = abs(model_price - market) / market

        if model_price < market * (1 - threshold):
            signals.append({
                "action": "sell",
                "strike": K,
                "market": market,
                "model": model_price,
                "rel_error": rel_error
            })
        elif model_price > market * (1 + threshold):
            signals.append({
                "action": "buy",
                "strike": K,
                "market": market,
                "model": model_price,
                "rel_error": rel_error
            })

    return signals
