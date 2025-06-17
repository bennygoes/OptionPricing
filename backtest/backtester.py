# backtest/backtester.py

class Backtester:
    def __init__(self, starting_cash=100_000, position_size=1):
        self.cash = starting_cash
        self.position_size = position_size
        self.trades = []

    def execute_signals(self, signals):
        for s in signals:
            trade_value = s['market'] * self.position_size
            pnl = 0

            if s['action'] == 'buy':
                pnl = -trade_value  # cost of entering
            elif s['action'] == 'sell':
                pnl = trade_value   # receive premium

            trade_record = {
                **s,  # copy all fields from signal
                'trade_value': trade_value,
                'pnl': pnl
            }
            self.trades.append(trade_record)

    def summary(self):
        total_pnl = sum(t['pnl'] for t in self.trades)
        return {
            'num_trades': len(self.trades),
            'total_pnl': total_pnl,
            'final_cash': self.cash + total_pnl
        }

    def get_trade_log(self):
        return self.trades
