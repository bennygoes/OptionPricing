import pandas as pd

class Backtester:
    def __init__(self, starting_cash=100000):
        self.cash = starting_cash
        self.trades = []

    def place_trade(self, option_price, direction, size):
        pnl = -option_price * size if direction == 'buy' else option_price * size
        self.trades.append({'price': option_price, 'side': direction, 'size': size, 'pnl': pnl})
        self.cash += pnl

    def summarize(self):
        df = pd.DataFrame(self.trades)
        return {
            'total_pnl': df['pnl'].sum(),
            'num_trades': len(df),
            'cash_remaining': self.cash
        }