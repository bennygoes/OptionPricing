import yfinance as yf

def get_option_chain(ticker, expiration):
    opt = yf.Ticker(ticker)
    chain = opt.option_chain(expiration)
    return chain.calls, chain.puts, opt.history(period="1d")['Close'].iloc[-1]
