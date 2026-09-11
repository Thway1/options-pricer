import numpy as np
from scipy.stats import norm

# --- Pricing functions ---

def black_scholes_call(S, K, T, r, sigma):
    """
    Prices a European call option using the Black-Scholes formula.
    S: current stock price
    K: strike price
    T: time to expiry (in years)
    r: risk-free interest rate (annualised)
    sigma: volatility of the underlying (annualised)
    """
    d1 = (np.log(S/K) + ((r+ 0.5 * sigma**2)*T)) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp((-r)*T) * norm.cdf(d2)
     # N(d1): risk-neutral-adjusted term for the stock received if exercised
    # N(d2): risk-neutral probability the option finishes in-the-money, used to weight the discounted strike price paid if exercised
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """
    Prices a European put option using the Black-Scholes formula.
    Same inputs as black_scholes_call. Uses N(-d1)/N(-d2) instead of
    N(d1)/N(d2) since the normal distribution's symmetry (1 - N(x) = N(-x))
    gives the complementary (below-strike) probability needed for a put.
    """
    d1 = (np.log(S/K) + ((r+ 0.5 * sigma**2)*T)) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    put_price = K * np.exp((-r)*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

# --- Model validation ---

def check_call_put_parity(S, K, T, r, sigma):
    """
    Verifies put-call parity: C - P = S - K*e^(-rT).
    This relationship holds independent of Black-Scholes (it follows from
    no-arbitrage payoff replication alone), making it a useful sanity check
    that the call and put pricing functions are mutually consistent.
    """
    call = black_scholes_call(S, K, T, r, sigma)
    put = black_scholes_put(S, K, T, r, sigma)
    left_side = call - put
    right_side = S - K*np.exp(-r*T)
    difference = abs(left_side-right_side)

    print(f"Left Side: {left_side:.4f}")
    print(f"Right Side: {right_side:.4f}")
    print(f"Difference: {difference:.10f}")

    # Tolerance accounts for floating-point rounding, not model error
    return difference < 1e-8

# --- Greeks: Delta ---
def delta_call(S, K, T, r, sigma):
    """
    Delta = N(d1), where d1 = [ln(S/K) + (r+(sigma^2)/2)*T] / [sigma*T**(-2)]
    Analytical Delta for a call option: dC/dS = N(d1).
    Measures the option price's sensitivity to a $1 move in the stock price;
    also the number of shares needed to hedge one short call (Delta hedging).
    """
    d1 = (np.log(S/K) + ((r+ 0.5 * sigma**2)*T)) / (sigma*np.sqrt(T))
    return norm.cdf(d1)

def delta_call_finite_diff(S, K, T, r, sigma, h=0.01):
    """
    Numerical Delta via central finite difference: (C(S+h) - C(S-h)) / 2h.
    Approximates the same derivative as delta_call() without using calculus,
    by directly measuring the price curve's slope at S. Used to validate
    the analytical formula above.
    """
    price_up = black_scholes_call(S + h, K, T, r, sigma)
    price_down= black_scholes_call(S - h, K, T, r, sigma)
    return (price_up-price_down)/(2*h)

# --- Greeks: Gamma ---
def gamma(S, K, T, r, sigma):
    """
    Gamma = N'(d1)/(S*sigma*T**(-2)), where d1 = [ln(S/K) + (r+(sigma^2)/2)*T] / [sigma*T**(-2)]
    Analytical Gamma (same formula for calls and puts): d(Delta)/dS.
    Measures the curvature of the option price curve, i.e. how fast Delta
    itself changes as the stock moves. This is why a Delta hedge needs
    periodic rebalancing rather than being set once and left alone.
    Uses the normal PDF (norm.pdf), not the CDF, since Gamma is the
    derivative of Delta's N(d1) term.
    """
    d1 = (np.log(S/K) + ((r+ 0.5 * sigma**2)*T)) / (sigma*np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def gamma_finite_diff(S, K, T, r, sigma, h=0.01):
    """
    Numerical Gamma via central finite difference for a second derivative:
    (C(S+h) - 2*C(S) + C(S-h)) / h^2. Validates the analytical Gamma above.
    """
    price_up = black_scholes_call(S+h, K, T, r, sigma)
    price_mid = black_scholes_call(S, K, T, r, sigma)
    price_down = black_scholes_call(S-h, K, T, r, sigma)
    return (price_up - 2 * price_mid + price_down) / (h**2)

# --- Tests / demonstration ---
if __name__ == "__main__":
    call_price = black_scholes_call(100, 100, 1, 0.05, 0.2)
    print(f"Call Price: {call_price:.2f}")
    put_price = black_scholes_put(100, 100, 1, 0.05, 0.2)
    print(f"Put Price: {put_price:.2f}")

    print("\n --- Put-Call Parity Check ---")
    is_valid = check_call_put_parity(100, 100, 1, 0.05, 0.2)
    print(f"Parity Holds: {is_valid}")

    print("\n---Delta Check---")
    analytical_delta = delta_call(100, 100, 1, 0.05, 0.2)
    numerical_delta = delta_call_finite_diff(100, 100, 1, 0.05, 0.2)
    print(f"Analytical Delta: {analytical_delta:.6f}")
    print(f"Numerical Delta: {numerical_delta:.6f}")
    print(f"Differenc: {abs(analytical_delta-numerical_delta):.10f}")

    print("\n---Gamma Check---")
    analytical_gamma = gamma(100, 100, 1, 0.05, 0.2)
    numerical_gamma = gamma_finite_diff(100, 100, 1, 0.05, 0.2)
    print (f"Analytical Gamma: {analytical_gamma:.6f}")
    print (f"Numerical Gamma: {numerical_gamma:.6f}")
    print (f"Difference; {abs(analytical_gamma - numerical_gamma):.10f}")