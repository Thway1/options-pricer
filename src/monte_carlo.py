import numpy as np
import matplotlib.pyplot as plt
from black_scholes import black_scholes_call
import os

def get_plots_dir():
    """
    Returns the absolute path to the project's plots/ folder, regardless
    of which directory the script is run from.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, '..', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

def simulated_stock_prices(S, T, r, sigma, num_sims):
    """
    Simulates possible stock prices at expiry using the closed-form
    solution to Geometric Brownian Motion under risk-neutral pricing:
    S_T = S * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
    where Z is drawn from a standard normal distribution.
    """
    Z = np.random.standard_normal(num_sims)
    S_T = S * np.exp(((r-0.5*sigma**2)*T) + (sigma*np.sqrt(T)*Z))
    return S_T

def monte_carlo_call_price(S, K, T, r, sigma, num_sims):
    """
    Prices a European call option via Monte Carlo simulation:
    1. Simulate many possible stock prices at expiry (S_T)
    2. Compute the call payoff for each: max(S_T - K, 0)
    3. Average all payoffs (expected payoff under risk-neutral measure)
    4. Discount that average back to today using e^(-rT)
    """
    S_T = simulated_stock_prices(S, T, r, sigma, num_sims)
    payoffs = np.maximum(S_T - K, 0)
    average_payoff = np.mean(payoffs)
    price = np.exp(-r*T) * average_payoff
    return price

def convergence_data(S, K, T, r, sigma, sim_counts):
    """
    Computes the Monte Carlo price at each simulation count in the given
    list, to show how the estimate stabilizes (converges) as the number
    of simulations increases.
    """
    prices = []
    for n in sim_counts:
        price = monte_carlo_call_price(S, K, T, r, sigma, n)
        prices.append(price)
    return prices

def simulate_price_paths(S, T, r, sigma, num_sims, num_steps):
    """
    Simulates full price paths (not just the final price) using GBM,
    stepping forward in small time increments. Unlike simulate_stock_prices
    (which jumps directly to S_T), this tracks every intermediate value --
    needed for visualizing the random walk, or for path-dependent options
    (not used in this project's pricing, but useful for illustration).
    """
    dt = T/num_steps
    paths = np.zeros((num_sims, num_steps+1))
    paths[:,0] = S

    for t in range(1, num_steps+1):
        Z = np.random.standard_normal(num_sims)
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    return paths

def plot_price_paths(S, T, r, sigma, num_sims=200, num_steps=250):
    paths = simulate_price_paths(S, T, r, sigma, num_sims, num_steps)

    plt.figure(figsize=(10, 6))
    plt.plot(paths.T, linewidth=0.8, alpha=0.7)
    plt.xlabel('Time Steps')
    plt.ylabel('Simulated Stock Price')
    plt.title('Simulated GBM Price Paths')

    save_path = os.path.join(get_plots_dir(), 'simulated_paths.png')
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    prices = simulated_stock_prices(100, 1, 0.05, 0.2, 10000)
    print(f"Mean of Simulated Prices: {prices.mean():.2f}")
    print(f"Min: {prices.min():.2f}, Max: {prices.max():.2f}")
    price = monte_carlo_call_price(100, 100, 1, 0.05, 0.2, 10000)
    print(f"Monte Carlo Call Price: {price:.4f}")

    print("\n --- Convergence Check ---")
    sim_counts = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
    mc_prices = convergence_data(100, 100, 1, 0.05, 0.2, sim_counts)
    bs_price = black_scholes_call(100, 100, 1, 0.05, 0.2)

    for n, price in zip(sim_counts, mc_prices):
        print(f"n={n:>7}: MC Price = {price:.4f}")
    print(f"Black-Scholes Price: {bs_price:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(sim_counts, mc_prices, marker='o', label='Monte Carlo Price')
    plt.axhline(y=bs_price, color='r', linestyle='--', label='Black-Scholes Price')
    plt.xscale('log')
    plt.xlabel('Number of Simulations (log scale)')
    plt.ylabel('Call Option Price')
    plt.title('Monte Carlo Convergence to Black-Scholes Price')
    plt.legend()
    plt.savefig('monte_carlo_convergence.png')
    plt.show()

    print("\n--- Simulated Price Paths ---")
    plot_price_paths(S=100, T=1, r=0.05, sigma=0.2, num_sims=200, num_steps=250)