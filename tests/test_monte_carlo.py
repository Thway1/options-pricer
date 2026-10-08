import math
import pytest

from black_scholes import black_scholes_call
from monte_carlo import monte_carlo_call_price, simulated_stock_prices
from black_scholes import black_scholes_put
from monte_carlo import monte_carlo_put_price


S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2


def test_monte_carlo_call_close_to_black_scholes():
    # No seed needed: with 500,000 sims the standard error is ~0.02,
    # so a 0.15 tolerance is about 7 standard errors wide.
    mc = monte_carlo_call_price(S, K, T, r, sigma, 500_000)
    assert mc == pytest.approx(black_scholes_call(S, K, T, r, sigma), abs=0.15)

def test_monte_carlo_put_close_to_black_scholes():
    mc = monte_carlo_put_price(S, K, T, r, sigma, 500_000)
    assert mc == pytest.approx(black_scholes_put(S, K, T, r, sigma), abs=0.15)


def test_simulated_prices_are_positive():
    prices = simulated_stock_prices(S, T, r, sigma, 100_000)
    assert (prices > 0).all()


def test_simulated_mean_matches_risk_neutral_drift():
    # Under risk-neutral pricing, E[S_T] = S * e^(rT)
    prices = simulated_stock_prices(S, T, r, sigma, 500_000)
    assert prices.mean() == pytest.approx(S * math.exp(r * T), abs=0.5)

def test_monte_carlo_is_unbiased():
    errs = [monte_carlo_call_price(S, K, T, r, sigma, 100_000)
            - black_scholes_call(S, K, T, r, sigma) for _ in range(20)]
    assert abs(sum(errs) / len(errs)) < 0.3