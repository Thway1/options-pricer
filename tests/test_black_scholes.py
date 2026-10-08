import math
import pytest

from black_scholes import (
    black_scholes_call,
    black_scholes_put,
    delta_call, delta_call_finite_diff,
    gamma, gamma_finite_diff,
    vega, vega_finite_diff,
    theta_call, theta_call_finite_diff,
)

# Standard test case
S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2


def test_call_known_value():
    assert black_scholes_call(S, K, T, r, sigma) == pytest.approx(10.4506, abs=1e-4)


def test_put_known_value():
    assert black_scholes_put(S, K, T, r, sigma) == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity():
    call = black_scholes_call(S, K, T, r, sigma)
    put = black_scholes_put(S, K, T, r, sigma)
    assert call - put == pytest.approx(S - K * math.exp(-r * T), abs=1e-8)


def test_call_increases_with_stock_price():
    assert black_scholes_call(110, K, T, r, sigma) > black_scholes_call(100, K, T, r, sigma)


def test_deep_itm_call_close_to_intrinsic():
    # S much larger than K: call is worth about S - K*e^(-rT)
    price = black_scholes_call(200, K, T, r, sigma)
    assert price == pytest.approx(200 - K * math.exp(-r * T), abs=0.01)


def test_delta_matches_finite_difference():
    assert delta_call(S, K, T, r, sigma) == pytest.approx(
        delta_call_finite_diff(S, K, T, r, sigma), abs=1e-4)


def test_gamma_matches_finite_difference():
    assert gamma(S, K, T, r, sigma) == pytest.approx(
        gamma_finite_diff(S, K, T, r, sigma), abs=1e-4)


def test_vega_matches_finite_difference():
    assert vega(S, K, T, r, sigma) == pytest.approx(
        vega_finite_diff(S, K, T, r, sigma), abs=1e-3)


def test_theta_matches_finite_difference():
    assert theta_call(S, K, T, r, sigma) == pytest.approx(
        theta_call_finite_diff(S, K, T, r, sigma), rel=1e-2)