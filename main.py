import yfinance as yf
import pandas as pd
import datetime
import time
import itertools

FAANG_TICKERS = ["META", "AAPL", "AMZN", "NFLX", "GOOGL"]

def fetch_faang_midday_prices(output_file, year):
    if year==2023:
        start_date = "2023-03-01"
        end_date = "2023-12-31"
    else:
        start_date = "2024-03-01"
        end_date = "2024-12-31"

    midday_prices = []
    
    for ticker in FAANG_TICKERS:
        try:
            #print(f"Fetching data for {ticker}...")
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
                    "Stock Ticker": ticker,
                    "Date": row.name.date(),
                    "Price": row['Close']
                })
            
            time.sleep(0.5)  # Avoid rate limits
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    
    midday_df = pd.DataFrame(midday_prices)
    midday_df.to_csv(output_file, index=False)
    #print(f"Saved FAANG midday prices to '{output_file}'")

def calculate_weekly_percent_change(input_filename, output_filename):
    df = pd.read_csv(input_filename)  

    if df.shape[1] < 3:
        print("Error: The file must have at least three columns (Stock Ticker, Date, Price).")
        return

    df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date column is in datetime format
    df = df.sort_values(by=['Stock Ticker', 'Date'])  # Sort by stock and date

    df['Change'] = 0.0  # Initialize percent change column

    prev_prices = {}  # Dictionary to store last week's price for each stock

    for i in range(len(df)):
        ticker = df.at[i, 'Stock Ticker']
        price = df.at[i, 'Price']
        
        if ticker in prev_prices:
            prev_price = prev_prices[ticker]
            df.at[i, 'Change'] = ((price - prev_price) / prev_price) if prev_price != 0 else 0
        else:
            df.at[i, 'Change'] = 0  # First week's change is 0

        prev_prices[ticker] = price  # Update previous week's price

    df.to_csv(output_filename, index=False)  
    #print(f"Saved percent change data to '{output_filename}'")

def compute_normalized_dot_products(input_filename, output_filename):
    df = pd.read_csv(input_filename)  

    if df.shape[1] < 4:
        print("Error: The file must have at least four columns (Stock Ticker, Date, Price, Percent Change).")
        return

    df = df.pivot(index="Date", columns="Stock Ticker", values="Change")  # Pivot for easy vectorized dot product
    df = df.dropna()  # Drop rows with missing values (weeks where a stock is missing)

    FAANG_TICKERS = df.columns.tolist()
    dot_products = {}

    for ticker1, ticker2 in itertools.combinations(FAANG_TICKERS, 2):  # Get all unique pairs
        dot_product = (df[ticker1] * df[ticker2]).sum()  # Compute normalized dot product
        dot_products[(ticker1, ticker2)] = dot_product

    # Convert results to a DataFrame and save
    result_df = pd.DataFrame(dot_products.items(), columns=["Ticker Pair", "Dot Product"])
    result_df = result_df.sort_values(by="Dot Product", ascending=False)

    
    result_df.to_csv(output_filename, index=False)

    #print(f"Saved dot products to '{output_filename}'")


def backtest_top_pairs(input_filename, dot_products_filename, start_date, end_date, initial_budget=10000):
    # Load percent change data and dot products
    df = pd.read_csv(input_filename)
    dot_products_df = pd.read_csv(dot_products_filename)
    
    # Sort by normalized dot product (highest first)
    top_pairs = dot_products_df.sort_values(by="Dot Product", ascending=False).head(2)
    
    # Extract top pairs
    ticker1, ticker2 = top_pairs.iloc[0]["Ticker Pair"][1:-1].split(", ")[0][1:-1], top_pairs.iloc[0]["Ticker Pair"][1:-1].split(", ")[1][1:-1]  # Properly split the tickers without quotes
    print(f"Top 2 most correlated pairs for backtesting: {ticker1} and {ticker2}")
    
    # Initialize portfolio: equal amount of money invested in both stocks
    portfolio = {ticker1: initial_budget / 2, ticker2: initial_budget / 2}
    cash = 0
    total_value = initial_budget
    prices = {ticker1: [], ticker2: []}

    # Read the percent change data and filter by the date range
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    # Debugging: Check the column names and initial data
    print(f"Columns in DataFrame: {df.columns}")
    print(f"Data from {start_date} to {end_date}: {df.head()}")

    # Track initial value
    start_value = total_value
    print(f"Starting portfolio value: ${start_value}")
    
    # Pivot the dataframe to get percent changes for each stock as columns
    df = df.pivot(index="Date", columns="Stock Ticker", values="Change")  # Pivot for easy vectorized dot product
    df = df.dropna()  # Drop rows with missing values (weeks where a stock is missing)

    # Loop through the data for each week
    for i in range(1, len(df)):
        current_date = df.index[i]  # Get the date from the DataFrame index
        previous_date = df.index[i - 1]  # Get the previous date

        # Ensure tickers exist in the columns before accessing
        if ticker1 not in df.columns or ticker2 not in df.columns:
            print(f"Warning: Missing data for {ticker1} or {ticker2} on {current_date}")
            continue

        # Access the percent change for each stock
        change_ticker1 = df.at[current_date, ticker1]  # Change for ticker1
        change_ticker2 = df.at[current_date, ticker2]  # Change for ticker2

        # Update the prices based on previous week's change (assuming an equal share of initial budget)
        price_ticker1 = portfolio[ticker1] * (1 + change_ticker1)
        price_ticker2 = portfolio[ticker2] * (1 + change_ticker2)
        
        # Track the new portfolio values
        portfolio[ticker1] = price_ticker1
        portfolio[ticker2] = price_ticker2
        
        # Calculate the cash value left
        total_value = portfolio[ticker1] + portfolio[ticker2] + cash

        # Print weekly portfolio value
        print(f"Date: {current_date}, Portfolio Value: ${total_value}")

        # Check if one stock outperformed the other by more than 5%
        if abs(change_ticker1 - change_ticker2) > 0.1:
            if change_ticker1 > change_ticker2:
                # Ticker1 outperforms, sell 2% of total portfolio and buy 2% more of Ticker2
                trade_value = total_value * 0.1  # 2% of total portfolio value
                sell_value = portfolio[ticker1] * 0.1  # 2% of Ticker1 portfolio value
                
                # Sell Ticker1, buy Ticker2
                if sell_value > portfolio[ticker1]:
                    # Not enough stock to sell, add cash to the bank
                    cash += sell_value
                    print(f"Not enough {ticker1} to sell, adding {sell_value} to cash")
                else:
                    # Perform the transaction
                    portfolio[ticker1] -= sell_value
                    buy_value = trade_value / price_ticker2
                    portfolio[ticker2] += buy_value
                    cash -= trade_value
            else:
                # Ticker2 outperforms, sell 2% of total portfolio and buy 2% more of Ticker1
                trade_value = total_value * 0.05  # 2% of total portfolio value
                sell_value = portfolio[ticker2] * 0.05  # 2% of Ticker2 portfolio value
                
                # Sell Ticker2, buy Ticker1
                if sell_value > portfolio[ticker2]:
                    # Not enough stock to sell, add cash to the bank
                    cash += sell_value
                    print(f"Not enough {ticker2} to sell, adding {sell_value} to cash")
                else:
                    # Perform the transaction
                    portfolio[ticker2] -= sell_value
                    buy_value = trade_value / price_ticker1
                    portfolio[ticker1] += buy_value
                    cash -= trade_value

        # Track the total value at each step for weekly reporting
        total_value = portfolio[ticker1] + portfolio[ticker2] + cash

    # Report final portfolio value
    end_value = total_value
    print(f"End of backtesting: ${end_value}")
    print(f"Start value: ${start_value}, End value: ${end_value}")
    return start_value, end_value








if __name__ == "__main__":
    fetch_faang_midday_prices("faang_midday_prices_2023.csv", 2023)
    fetch_faang_midday_prices("faang_midday_prices_2024.csv", 2024)

    #cleaning 2023 data
    df = pd.read_csv("faang_midday_prices_2023.csv", dtype=str)
    third_col_name = df.columns[2]
    print(third_col_name)
    
    for i in range(len(df)):
        
        df.at[i, third_col_name] = float((str(df.at[i, third_col_name])).split()[2])
    df.to_csv("faang_midday_prices_2023.csv", index=False)

    #cleaning 2024 data
    df = pd.read_csv("faang_midday_prices_2024.csv", dtype=str)
    third_col_name = df.columns[2]
    print(third_col_name)
    
    for i in range(len(df)):
        
        df.at[i, third_col_name] = float((str(df.at[i, third_col_name])).split()[2])
    df.to_csv("faang_midday_prices_2024.csv", index=False)
    calculate_weekly_percent_change("faang_midday_prices_2023.csv", "faang_weekly_percent_change2023.csv")
    calculate_weekly_percent_change("faang_midday_prices_2024.csv", "faang_weekly_percent_change2024.csv")

    compute_normalized_dot_products("faang_weekly_percent_change2023.csv", "faang_ticker_dot_products.csv")
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    input_filename = 'faang_weekly_percent_change2024.csv'
    dot_products_filename = 'faang_ticker_dot_products.csv'

    start_value, end_value = backtest_top_pairs(input_filename, dot_products_filename, start_date, end_date)



                
    

