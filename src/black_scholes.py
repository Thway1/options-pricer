import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + ((r+ 0.5 * sigma**2)*T)) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp((-r)*T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + ((r+ 0.5 * sigma**2)*T)) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    put_price = K * np.exp((-r)*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def check_call_put_parity(S, K, T, r, sigma):
    call = black_scholes_call(S, K, T, r, sigma)
    put = black_scholes_put(S, K, T, r, sigma)
    left_side = call - put
    right_side = S - K*np.exp(-r*T)
    difference = abs(left_side-right_side)

    print(f"Left Side: {left_side:.4f}")
    print(f"Right Side: {right_side:.4f}")
    print(f"Difference: {difference:.10f}")

    return difference < 1e-8

def delta_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + ((r+ 0.5 * sigma**2)*T)) / (sigma*np.sqrt(T))
    return norm.cdf(d1)

def delta_call_finite_diff(S, K, T, r, sigma, h=0.01):
    price_up = black_scholes_call(S + h, K, T, r, sigma)
    price_down= black_scholes_call(S - h, K, T, r, sigma)
    return (price_up-price_down)/(2*h)

if __name__ == "__main__":
    call_price = black_scholes_call(100, 100, 1, 0.05, 0.2)
    print(f"Call Price: {call_price:.2f}")
    put_price = black_scholes_put(100, 100, 1, 0.05, 0.2)
    print(f"Put Price: {put_price:.2f}")

    print("\n --- Put-Call Parity Check ---")
    is_valid = check_call_put_parity(100, 100, 1, 0.05, 0.2)
    print(f"Parity Holds: {is_valid}")

    print("\n---Data Check---")
    analytical_delta = delta_call(100, 100, 1, 0.05, 0.2)
    numerical_delta = delta_call_finite_diff(100, 100, 1, 0.05, 0.2)
    print(f"Analytical Delta: {analytical_delta:.6f}")
    print(f"Numerical Delta: {numerical_delta:.6f}")
    print(f"Differenc: {abs(analytical_delta-numerical_delta):.10f}")