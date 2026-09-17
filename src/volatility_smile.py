import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from implied_vol import implied_vol_call
from datetime import datetime

def get_plots_dir():
    """
    Returns the absolute path to the project's plots/ folder, regardless
    of which directory the script is run from.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, '..', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

def get_option_chain(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    current_price = ticker.history(period="1d")['Close'].iloc[-1]

    expiry_dates = ticker.options
    # Skip very-near-term expiries; pick one a few weeks/months out for stability
    chosen_expiry = expiry_dates[4]  # adjust index to pick a later date

    option_chain = ticker.option_chain(chosen_expiry)
    calls = option_chain.calls.copy()
    calls['mid_price'] = (calls['bid'] + calls['ask']) / 2

    return calls, current_price, chosen_expiry


def calculate_implied_vols(calls, S, r, expiry_date_str):
    """
    Computes implied volatility for each option in the chain using its
    mid-price. Filters out options with invalid/missing bid-ask data
    (illiquid options often show bid=0 or ask=0, which would break the
    calculation), and skips deep ITM/OTM options where implied vol
    calculations tend to be unstable due to very low or negligible
    time value.
    """
    expiry_date = datetime.strptime(expiry_date_str, "%Y-%m-%d")
    T = (expiry_date - datetime.now()).days / 365

    strikes = []
    implied_vols = []

    for _, row in calls.iterrows():
        K = row['strike']
        market_price = row['mid_price']

        if market_price <= 0 or row['bid'] <= 0:
            continue

        if T <= 0:
            continue

        iv = implied_vol_call(market_price, S, K, T, r)

        if iv is None:
            continue

        if 0.01 < iv < 3.0:
            strikes.append(K)
            implied_vols.append(iv)

    return strikes, implied_vols

import os

def plot_volatility_smile(strikes, implied_vols, ticker_symbol, expiry_date_str):
    plt.figure(figsize=(8, 5))
    plt.plot(strikes, implied_vols, marker='o', linestyle='-')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.title(f'{ticker_symbol} Volatility Smile (Expiry: {expiry_date_str})')
    plt.grid(True)

    save_path = os.path.join(get_plots_dir(), 'volatility_smile.png')
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    calls, S, expiry = get_option_chain("AAPL")
    print(f"Current AAPL price: {S:.2f}")
    print(f"Nearest expiry: {expiry}")
    print(calls[['strike', 'lastPrice', 'bid', 'ask']].head(10))

if __name__ == "__main__":
    ticker_symbol = "AAPL"
    r = 0.05

    calls, S, expiry = get_option_chain(ticker_symbol)
    print(f"Current {ticker_symbol} price: {S:.2f}")
    print(f"Nearest expiry: {expiry}")

    strikes, implied_vols = calculate_implied_vols(calls, S, r, expiry)
    print(f"Computed {len(strikes)} implied volatilities")

    plot_volatility_smile(strikes, implied_vols, ticker_symbol, expiry)