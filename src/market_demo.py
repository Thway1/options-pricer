from black_scholes import black_scholes_call, black_scholes_put
from monte_carlo import monte_carlo_call_price, monte_carlo_put_price
from binomial_tree import binomial_tree_price
from volatility_smile import (get_option_chain, time_to_expiry, calculate_implied_vols, atm_implied_vol, RISK_FREE_RATE)


def run_market_demo(ticker="TSLA", num_sims=500_000, tree_steps=1000):
    """Price the at-the-money option three ways, using the market's own implied vol."""
    r = RISK_FREE_RATE
    calls, S, expiry = get_option_chain(ticker)
    T = time_to_expiry(expiry)
    iv_table = calculate_implied_vols(calls, S, T, r)

    K, sigma = atm_implied_vol(iv_table, S)
    market_price = iv_table.loc[iv_table["strike"] == K, "mid_price"].iloc[0]

    print(f"{ticker}  S={S:.2f}  K={K:.2f}  T={T:.3f}  r={r}  sigma(ATM implied)={sigma:.4f}")
    print(f"Market mid price (call): {market_price:.2f}\n")

    print(f"{'Method':<22}{'Call':>10}{'Put':>10}")
    print(f"{'Black-Scholes':<22}{black_scholes_call(S, K, T, r, sigma):>10.4f}{black_scholes_put(S, K, T, r, sigma):>10.4f}")
    print(f"{'Binomial tree':<22}{binomial_tree_price(S, K, T, r, sigma, tree_steps, 'call'):>10.4f}{binomial_tree_price(S, K, T, r, sigma, tree_steps, 'put'):>10.4f}")
    print(f"{'Monte Carlo':<22}{monte_carlo_call_price(S, K, T, r, sigma, num_sims):>10.4f}{monte_carlo_put_price(S, K, T, r, sigma, num_sims):>10.4f}")
    print(f"{'Binomial (American)':<22}{binomial_tree_price(S, K, T, r, sigma, tree_steps, 'call', True):>10.4f}{binomial_tree_price(S, K, T, r, sigma, tree_steps, 'put', True):>10.4f}")


if __name__ == "__main__":
    run_market_demo("TSLA")