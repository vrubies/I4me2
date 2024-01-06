import yfinance as yf
import json
from datetime import datetime
import pandas as pd

def fetch_stock_history(stock, start_date, end_date):
    ticker = yf.Ticker(stock)
    hist = ticker.history(start=start_date, end=end_date)
    # Convert Timestamp to string for JSON serialization
    return {date.strftime('%Y-%m-%d'): price if not pd.isna(price) else 0 for date, price in hist['Close'].items()}

def save_stock_history_to_json(stock_list, start_date, end_date, output_file):
    stock_history = {}
    for stock in stock_list:
        print(f"Fetching history for {stock}...")
        stock_history[stock] = fetch_stock_history(stock, start_date, end_date)
    
    with open(output_file, 'w') as file:
        json.dump(stock_history, file)
    print(f"Stock history saved to {output_file}")

# Main process
output_file = 'data/fool_articles/stock_history.json'
stock_list = ["MSFT", "AAPL", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "TSLA", "UNH", 
              "LLY", "JPM", "AVGO", "XOM", "V", "JNJ", "PG", "MA", "HD", "MRK", "COST", "ABBV", 
              "CVX", "ADBE", "CRM", "PEP", "KO", "BAC", "WMT", "AMD", "MCD", "ACN", "NFLX", 
              "CSCO", "TMO", "INTC", "LIN", "ABT", "WFC", "CMCSA", "PFE", "DIS", "INTU", "VZ", 
              "ORCL", "AMGN", "QCOM", "DHR", "TXN", "PM", "UNP", "IBM", "CAT", "COP", "SPGI", 
              "BA", "NOW", "GE", "HON", "NKE", "NEE", "AMAT", "GS", "T", "RTX", "LOW", "PLD", 
              "UBER", "BKNG", "MS", "UPS", "ISRG", "ELV", "MDT", "BLK", "AXP", "SBUX", "VRTX", 
              "DE", "BMY", "TJX", "GILD", "CVS", "C", "LMT", "AMT", "SCHW", "MDLZ", "SYK", "REGN", 
              "LRCX", "ADP", "PGR", "MMC", "ADI", "ETN", "CB", "MU", "PANW", "CI"]  # Your list of stock tickers
start_date = '2000-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')  # Today's date
save_stock_history_to_json(stock_list, start_date, end_date, output_file)
