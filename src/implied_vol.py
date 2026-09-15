import numpy as np
from black_scholes import black_scholes_call, vega

def implied_vol_call(market_price, S, K, T, r, initial_guess=0.2, tolerance=1e-6, max_iterations=100):
    """
    Solves for implied volatility using the Newton-Raphson method:
    repeatedly refines a volatility guess by using Vega (the price's
    sensitivity to volatility) as the local slope, until the model price
    matches the observed market price within tolerance.
    """
    sigma = initial_guess
    for i in range(max_iterations):
        price_estimate = black_scholes_call(S, K, T, r, sigma)
        price_diff = price_estimate - market_price

        if abs(price_diff) < tolerance:
            return sigma

        v = vega(S, K, T, r, sigma) / 0.01
        sigma = sigma - price_diff / v

    return None  # explicitly signal failure to converge, rather than crashing

if __name__ == "__main__":
    true_sigma = 0.25
    market_price = black_scholes_call(100, 100, 1, 0.05, true_sigma)
    print(f"True Sigma = {true_sigma}")
    print(f"Market Price generated with true sigma: {market_price:.4f}")

    recovered_sigma = implied_vol_call(market_price, 100, 100, 1, 0.05)
    print(f"Recovered implied volitility {recovered_sigma:.6f}")