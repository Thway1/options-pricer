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

# --- Greeks : Vega ---
def vega(S, K, T, r, sigma):
    """
    Analytical Vega (same formula for calls and puts): dC/d(sigma).
    Measures the option price's sensitivity to a change in volatility.
    Uses the normal PDF, since it comes from differentiating N(d1) with
    respect to sigma (same underlying mechanism as Gamma's derivation,
    just differentiating with respect to a different variable).
    Conventionally scaled by 0.01 so the result represents the price
    change for a 1 percentage point move in volatility (e.g. 20% -> 21%),
    rather than a full 100 percentage point move.
    """
    d1 = (np.log(S/K) + ((r+ 0.5 * sigma**2)*T)) / (sigma*np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1) * 0.01

def vega_finite_diff(S, K, T, r, sigma, h=0.01):
    """
    Numerical Vega via central finite difference, bumping sigma instead of S:
    (C(sigma+h) - C(sigma-h)) / 2h. Validates the analytical Vega above.
    Note: h is much smaller here than for Delta/Gamma, since sigma is
    typically a small decimal (e.g. 0.2), so a proportionally smaller
    bump keeps the approximation accurate.
    """
    price_up = black_scholes_call(S, K, T, r, sigma+h)
    price_down = black_scholes_call(S, K, T, r, sigma-h)
    return (price_up-price_down)/(2*h) * 0.01

# --- Greeks : Vega ---
def theta_call(S, K, T, r, sigma):
    """
    Analytical Theta for a call option: how the option's fair value
    changes as time passes (T decreases), holding S, K, r, sigma fixed.
    Conventionally expressed per day (dividing the raw annual value by 365),
    since "value lost per year" isn't a practically useful number to reason
    about day-to-day.
    """
    d1 = (np.log(S/K) + ((r+ 0.5 * sigma**2)*T)) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    term1 = (-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2 = -r * K * (np.exp(-r * T)) * norm.cdf(d2)
    theta_annual = term1 + term2
    return theta_annual/365

def theta_call_finite_diff(S, K, T, r, sigma, h=0.0001):
    """
    Numerical Theta via finite difference, bumping T. Since Theta measures
    value change as time PASSES (T decreasing), this compares price at a
    slightly smaller T against the current price, then converts to a
    per-day figure to match the analytical convention above.
    """
    price_now = black_scholes_call(S, K, T, r, sigma)
    price_later = black_scholes_call(S, K, T - h, r, sigma)
    #foward difference (one directional)
    theta_approx = (price_later - price_now) / h 
    return theta_approx / 365

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

    print("\n--- Vega Check ---")
    analytical_vega = vega(100, 100, 1, 0.05, 0.2)
    numerical_vega = vega_finite_diff(100, 100, 1, 0.05, 0.2)
    print(f"Analytical Vega: {analytical_vega:.6f}")
    print(f"Finite Diff Vega: {numerical_vega:.6f}")
    print(f"Difference: {abs(analytical_vega - numerical_vega):.10f}")

    print("\n--- Theta Check ---")
    analytical_theta = theta_call(100, 100, 1, 0.05, 0.2)
    numerical_theta = theta_call_finite_diff(100, 100, 1, 0.05, 0.2)
    print(f"Analytical Theta: {analytical_theta:.6f}")
    print(f"Finite Diff Theta: {numerical_theta:.6f}")
    print(f"Difference: {abs(analytical_theta - numerical_theta):.10f}")