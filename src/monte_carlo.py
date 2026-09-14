import numpy as np
import matplotlib.pyplot as plt
from black_scholes import black_scholes_call

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