import yfinance as yf
import pandas as pd
import datetime
import os
import time
import numpy as np
import re

def save_sp500_tickers(filename="sp500_tickers.csv"):
    sp500_tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    tickers = sp500_tickers['Symbol'].tolist()
    pd.DataFrame(tickers, columns=["Ticker"]).to_csv(filename, index=False)
    print(f"Saved S&P 500 tickers to {filename}")

def load_sp500_tickers(filename="sp500_tickers.csv"):
    if os.path.exists(filename):
        return pd.read_csv(filename)['Ticker'].tolist()
    else:
        print(f"{filename} not found. Fetching tickers from Wikipedia...")
        save_sp500_tickers(filename)
        return load_sp500_tickers(filename)

def fetch_midday_prices_2_years(filename="sp500_tickers.csv", years=2):
    tickers = load_sp500_tickers(filename)
    start_date = (datetime.datetime.now() - datetime.timedelta(weeks=years * 52)).strftime('%Y-%m-%d')
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    midday_prices = []
    
    for ticker in tickers:
        try:
            print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
            data = yf.download(ticker, interval="60m", start=start_date, end=end_date)
            data.index = pd.to_datetime(data.index)
            if data.empty:
                print(f"No data for {ticker}, skipping...")
                continue
            tuesday_midday = data[(data.index.dayofweek == 1) & (data.index.hour == 12)]
            if tuesday_midday.empty:
                print(f"No midday prices for {ticker} on Tuesdays, skipping...")
                continue
            for _, row in tuesday_midday.iterrows():
                midday_prices.append({
                    "Ticker": ticker,
                    "Date": row.name.date(),
                    "Midday Price": row['Close']
                })
            time.sleep(0.5)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    
    midday_df = pd.DataFrame(midday_prices)
    midday_df.to_csv("midday_prices_2_years.csv", index=False)
    print(f"Saved midday prices to 'midday_prices_2_years.csv'")


def clean_midday_price(midday_price_str):
    match = re.search(r'(\d+\.\d+)', str(midday_price_str))
    if match:
        return float(match.group(1))
    else:
        return np.nan


def calculate_covariances(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'Midday Price', 'Ticker'])
    df['Midday Price'] = df['Midday Price'].apply(clean_midday_price)
    df = df.dropna(subset=['Midday Price'])
    df_cleaned = df.groupby(['Date', 'Ticker'], as_index=False)['Midday Price'].mean()
    df_pivot = df_cleaned.pivot_table(index='Date', columns='Ticker', values='Midday Price')
    df_returns = df_pivot.pct_change().dropna()
    cov_matrix = df_returns.cov()
    cov_pairs = []
    seen_pairs = set()
    
    for col in cov_matrix.columns:
        for row in cov_matrix.index:
            if col != row:
                pair = tuple(sorted([col, row]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    cov_pairs.append((col, row, cov_matrix.loc[row, col]))
    
    cov_df = pd.DataFrame(cov_pairs, columns=['Stock 1', 'Stock 2', 'Covariance'])
    avg_cov = cov_df.groupby(['Stock 1', 'Stock 2']).mean().reset_index()
    avg_cov_sorted = avg_cov.sort_values(by='Covariance', ascending=False)
    return avg_cov_sorted


def fetch_week_data(stock, start_date, end_date):
    print(f"Fetching data for {stock} from {start_date} to {end_date}...")
    data = yf.download(stock, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data found for {stock} from {start_date} to {end_date}.")
    return data


def execute_trade(top_stock, start_date, end_date, investment=100):
    try:
        week_data = fetch_week_data(top_stock, start_date, end_date)
        open_price = float(week_data['Open'].iloc[0])
        close_price = float(week_data['Close'].iloc[-1])
        shares = investment / open_price
        final_value = shares * close_price
        profit_loss = final_value - investment
        return {
            'Stock': top_stock,
            'Open Price': open_price,
            'Close Price': close_price,
            'Final Value': final_value,
            'Profit/Loss': profit_loss
        }
    except Exception as e:
        print(f"Error during trade execution: {e}")
        return None

if __name__ == '__main__':
    df = pd.read_csv('midday_prices_2_years.csv')
    avg_cov_sorted = calculate_covariances(df)
    avg_cov_sorted.to_csv("sorted_covariance_pairs.csv", index=False)
    print("Saved sorted covariance pairs to 'sorted_covariance_pairs.csv'")
    
    top_pair = avg_cov_sorted.iloc[0]
    top_stock = top_pair['Stock 1']
    
    results = []
    for i in range(30):
        start_date = (datetime.datetime(2024, 1, 1) + datetime.timedelta(weeks=i)).strftime('%Y-%m-%d')
        end_date = (datetime.datetime(2024, 1, 7) + datetime.timedelta(weeks=i)).strftime('%Y-%m-%d')
        result = execute_trade(top_stock, start_date, end_date, investment=100)
        if result:
            results.append(result)
    
    pd.DataFrame(results).to_csv('30_weeks_trade_results.csv', index=False)
    print("Saved 30 weeks trade results to '30_weeks_trade_results.csv'")
