import numpy as np

def binomial_tree_price(S, K, T, r, sigma, N, option_type="call", american=False):
    """
    Price an option on a Cox-Ross-Rubinstein binomial tree.

    option_type: "call" or "put"
    american:    if True, check early exercise at every node
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    dt = T/N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt)-d) / (u -d)
    discount = np.exp(-r*dt)

    def payoff(prices):
        if option_type == "call":
            return np.maximum(prices - K, 0)
        return np.maximum(K-prices, 0)
    
    # Stock prices at expiry: index j = number of up-moves (0 .. N)
    stock_prices = np.array([S * u**j * d**(N - j) for j in range(N + 1)])
    option_values = payoff(stock_prices)

    # Work backwards one step at a time
    for step in range(N - 1, -1, -1):
        # Prices one step earlier: drop the lowest node, divide by u
        stock_prices = stock_prices[1:] / u
        continuation = discount * (p * option_values[1:] + (1 - p) * option_values[:-1])

        if american:
            option_values = np.maximum(continuation, payoff(stock_prices))
        else:
            option_values = continuation

    return option_values[0]

def binomial_convergence(S, K, T, r, sigma, option_type = "call", american=False, steps=(1, 2, 5, 10, 25, 50, 100, 250, 500, 1000)):
    """Return (N, price) pairs to observe as N grows."""
    return [(n, binomial_tree_price(S, K, T, r, sigma, n, option_type, american)) for n in steps]


import os
import matplotlib.pyplot as plt

from black_scholes import black_scholes_put
from monte_carlo import get_plots_dir

def plot_binomial_convergence(S, K, T, r, sigma, max_steps=200):
    """Plot the European put from the tree against the Black-Scholes price as N grows."""
    steps = list(range(1, max_steps + 1))
    tree_prices = [binomial_tree_price(S, K, T, r, sigma, n, "put") for n in steps]
    bs_price = black_scholes_put(S, K, T, r, sigma)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, tree_prices, color="#2a6fdb", linewidth=1.5, label="Binomial tree (European put)")
    ax.axhline(bs_price, color="#555555", linewidth=1.5, linestyle="--",
               label=f"Black-Scholes = {bs_price:.4f}")
    ax.set_xlabel("Number of steps N")
    ax.set_ylabel("Put price")
    ax.set_title("Binomial tree converges to Black-Scholes")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)

    path = os.path.join(get_plots_dir(), "binomial_convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved plots/{os.path.basename(path)}")
    return tree_prices


if __name__ == "__main__":
    from black_scholes import black_scholes_call, black_scholes_put

    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    print("Hand-check (put): N=1 -> 7.29, N=2 European -> 4.66, N=2 American -> 5.74")
    print(f"N=1 Euro put: {binomial_tree_price(S, K, T, r, sigma, 1, 'put'):.2f}")
    print(f"N=2 Euro put: {binomial_tree_price(S, K, T, r, sigma, 2, 'put'):.2f}")
    print(f"N=2 Amer put: {binomial_tree_price(S, K, T, r, sigma, 2, 'put', american=True):.2f}")

    N = 1000
    print(f"\nBlack-Scholes call: {black_scholes_call(S, K, T, r, sigma):.4f}")
    print(f"Tree European call: {binomial_tree_price(S, K, T, r, sigma, N, 'call'):.4f}")
    print(f"Tree American call: {binomial_tree_price(S, K, T, r, sigma, N, 'call', True):.4f}")

    print(f"\nBlack-Scholes put:  {black_scholes_put(S, K, T, r, sigma):.4f}")
    print(f"Tree European put:  {binomial_tree_price(S, K, T, r, sigma, N, 'put'):.4f}")
    print(f"Tree American put:  {binomial_tree_price(S, K, T, r, sigma, N, 'put', True):.4f}")

    plot_binomial_convergence(S, K, T, r, sigma)