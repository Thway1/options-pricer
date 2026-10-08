import pytest

from black_scholes import black_scholes_call
from implied_vol import implied_vol_call

S, K, T, r = 100, 100, 1, 0.05


@pytest.mark.parametrize("true_sigma", [0.1, 0.2, 0.3, 0.5])
def test_recovers_known_volatility(true_sigma):
    market_price = black_scholes_call(S, K, T, r, true_sigma)
    assert implied_vol_call(market_price, S, K, T, r) == pytest.approx(true_sigma, abs=1e-5)


@pytest.mark.parametrize("guess", [0.05, 0.2, 0.8])
def test_result_does_not_depend_on_initial_guess(guess):
    market_price = black_scholes_call(S, K, T, r, 0.3)
    result = implied_vol_call(market_price, S, K, T, r, initial_guess=guess)
    assert result == pytest.approx(0.3, abs=1e-5)