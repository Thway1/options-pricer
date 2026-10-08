import pytest

from black_scholes import black_scholes_call, black_scholes_put
from binomial_tree import binomial_tree_price

S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2


# --- Hand-calculated values from the worked tree examples ---

def test_one_step_put():
    assert binomial_tree_price(S, K, T, r, sigma, 1, "put") == pytest.approx(7.29, abs=0.005)


def test_one_step_american_equals_european():
    euro = binomial_tree_price(S, K, T, r, sigma, 1, "put")
    amer = binomial_tree_price(S, K, T, r, sigma, 1, "put", american=True)
    assert amer == pytest.approx(euro)


def test_two_step_european_put():
    assert binomial_tree_price(S, K, T, r, sigma, 2, "put") == pytest.approx(4.66, abs=0.005)


def test_two_step_american_put():
    assert binomial_tree_price(S, K, T, r, sigma, 2, "put", american=True) == pytest.approx(5.74, abs=0.005)


# --- Convergence to Black-Scholes ---

def test_european_call_converges_to_black_scholes():
    tree = binomial_tree_price(S, K, T, r, sigma, 1000, "call")
    assert tree == pytest.approx(black_scholes_call(S, K, T, r, sigma), abs=0.005)


def test_european_put_converges_to_black_scholes():
    tree = binomial_tree_price(S, K, T, r, sigma, 1000, "put")
    assert tree == pytest.approx(black_scholes_put(S, K, T, r, sigma), abs=0.005)


# --- American vs European properties ---

def test_american_call_equals_european_call_without_dividends():
    euro = binomial_tree_price(S, K, T, r, sigma, 500, "call")
    amer = binomial_tree_price(S, K, T, r, sigma, 500, "call", american=True)
    assert amer == pytest.approx(euro, abs=1e-8)


def test_american_put_worth_more_than_european_put():
    euro = binomial_tree_price(S, K, T, r, sigma, 500, "put")
    amer = binomial_tree_price(S, K, T, r, sigma, 500, "put", american=True)
    assert amer > euro + 0.3


def test_american_put_at_least_intrinsic_value():
    # Deep in the money: you can always exercise now and get K - S
    amer = binomial_tree_price(60, K, T, r, sigma, 200, "put", american=True)
    assert amer >= K - 60


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        binomial_tree_price(S, K, T, r, sigma, 10, "straddle")