import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from implied_vol import implied_vol_call
from monte_carlo import get_plots_dir

RISK_FREE_RATE = 0.005


def get_option_chain(ticker="AAPL", expiry_index=3):
    """Download live call options for one expiry. Returns (calls, spot, expiry_date)."""
    stock = yf.Ticker(ticker)
    expiries = stock.options
    expiry = expiries[min(expiry_index, len(expiries) - 1)]
    calls = stock.option_chain(expiry).calls.copy()
    calls["mid_price"] = (calls["bid"] + calls["ask"]) / 2
    spot = float(stock.history(period="1d")["Close"].iloc[-1])
    return calls, spot, expiry


def time_to_expiry(expiry):
    """Years from today until the expiry date string (YYYY-MM-DD)."""
    days = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.now()).days
    return days / 365


def calculate_implied_vols(calls, S, T, r, moneyness_range=(0.7, 1.3)):
    """Implied vol for every usable call. Returns DataFrame with strike, mid_price, iv."""
    rows = []
    if T <= 0:
        return pd.DataFrame(columns=["strike", "mid_price", "iv"])
    with np.errstate(all="ignore"):  # solver can overflow on deep ITM strikes; those get filtered out
        for _, option in calls.iterrows():
            K = option["strike"]
            if not (moneyness_range[0] <= K / S <= moneyness_range[1]):
                continue
            if option["bid"] <= 0 or option["ask"] <= 0:
                continue
            iv = implied_vol_call(option["mid_price"], S, K, T, r)
            if iv is None or not (0.01 < iv < 3.0):
                continue
            rows.append({"strike": K, "mid_price": option["mid_price"], "iv": iv})
    return pd.DataFrame(rows)


def atm_implied_vol(iv_table, S):
    """Return (strike, iv) of the option whose strike is closest to the spot price."""
    idx = (iv_table["strike"] - S).abs().idxmin()
    return iv_table.loc[idx, "strike"], iv_table.loc[idx, "iv"]


def plot_volatility_smile(iv_table, S, ticker, expiry):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(iv_table["strike"], iv_table["iv"] * 100, marker="o", markersize=4,
            linewidth=1.5, color="#2a6fdb", label="Implied volatility")
    ax.axvline(S, color="#555555", linestyle="--", linewidth=1.2, label=f"Spot = {S:.2f}")
    ax.set_xlabel("Strike price")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title(f"{ticker} volatility smile (expiry {expiry}, data as of {datetime.now():%d %b %Y})")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)

    path = os.path.join(get_plots_dir(), f"volatility_smile_{ticker}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    return path


if __name__ == "__main__":
    ticker = "TSLA"   # change to "AAPL" etc.
    r = RISK_FREE_RATE

    calls, S, expiry = get_option_chain(ticker)
    T = time_to_expiry(expiry)
    iv_table = calculate_implied_vols(calls, S, T, r)

    print(f"{ticker}: spot {S:.2f}, expiry {expiry}, T = {T:.3f} years, {len(iv_table)} usable options")
    plot_volatility_smile(iv_table, S, ticker, expiry)

    strike, atm_iv = atm_implied_vol(iv_table, S)
    print(f"\nAt-the-money strike {strike:.2f}: implied vol = {atm_iv:.4f} ({atm_iv * 100:.1f}%)")
    print("\nAll implied vols:")
    print(iv_table.to_string(index=False))