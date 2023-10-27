import yfinance as yf

ticker_symbol = "^GSPC"

start_date = "1900-01-01"
end_date = "2023-07-09"

data = yf.download(ticker_symbol, start=start_date, end=end_date)
data.to_csv("/mnt/d/forex/sp500_daily_prices.csv")
