# Options Pricer

A Black-Scholes options pricing engine built in Python, with Monte Carlo validation, Greeks (analytical + numerical), and an empirical volatility smile derived from real market data.

**Status: ~65% complete** (core scope done; binomial tree, tests, and final write-up remaining)

---

## What this project does

Starting from the Black-Scholes-Merton model, this project prices European options, validates that pricing against an independent method (Monte Carlo simulation), computes and verifies the option Greeks two different ways, and finally uses real market data to show where the Black-Scholes model's core assumptions break down in practice (the volatility smile).

---

## Project structure

```
options-pricer/
├── src/
│   ├── black_scholes.py      # Core pricing, put-call parity, Greeks
│   ├── monte_carlo.py        # Monte Carlo pricing, convergence check, path simulation
│   ├── implied_vol.py        # Newton-Raphson implied volatility solver
│   └── volatility_smile.py   # Real option chain data + volatility smile plot
├── plots/                    # Saved output plots
├── tests/                    # (pending) pytest unit tests
├── notebooks/                # (pending) consolidated notebook with narrative
├── data/                     # (unused so far — data pulled live via yfinance)
└── requirements.txt
```

---

## 1. Black-Scholes pricing (`black_scholes.py`)

Implements the closed-form Black-Scholes formula for European calls and puts:

```
C = S·N(d1) − K·e^(−rT)·N(d2)
P = K·e^(−rT)·N(−d2) − S·N(−d1)
```

**Key understanding:**
- `N(d2)` = risk-neutral probability the option finishes in-the-money; weights the discounted strike (the "expected cost of paying K")
- `N(d1)` = the analogous weighting for the stock side; also equals the option's Delta
- `N(−x) = 1 − N(x)` (normal distribution symmetry) is why the put formula flips signs rather than needing a separate derivation
- `S` is always today's price — the formula never projects a future stock price; uncertainty is handled entirely through σ and the N() terms

### Put-call parity check
`C − P = S − K·e^(−rT)` — verified both numerically (in code) and derived algebraically from the payoff functions directly (no Black-Scholes assumptions needed, since it follows purely from no-arbitrage). Used as an internal consistency check between the call and put functions.

---

## 2. The Greeks (`black_scholes.py`)

Each Greek is implemented **two ways** — analytically (closed-form) and numerically (finite difference) — and checked against each other.

| Greek | Measures | Formula basis |
|---|---|---|
| Delta | Sensitivity to a $1 move in stock price | N(d1) |
| Gamma | Sensitivity of Delta itself (curvature) | Normal PDF, second derivative |
| Vega | Sensitivity to a 1% change in volatility | Normal PDF, scaled ×0.01 by convention |
| Theta | Value lost per day as time passes | Two-term formula, converted to per-day |

**Finite difference method:** central difference for first derivatives (`(f(x+h) − f(x−h)) / 2h`), three-point central difference for the second derivative (Gamma). Vega bumps σ instead of S, with a much smaller `h` since σ is a small decimal.

### Why this matters — Delta hedging
Delta tells a market maker who has sold ("written") an option how many shares to hold to offset the option's price sensitivity to the stock — going **long** stock to hedge a **short** call (opposite-direction exposure cancels out). Since Delta itself shifts as the stock moves (that's Gamma), a real hedge requires continuous rebalancing ("dynamic hedging"), gradually accumulating (or shedding) stock in step with the option's changing sensitivity — which, worked through with real numbers, shows how the hedge naturally prepares the position for the option's eventual payoff at expiry, rather than needing to be calculated separately.

---

## 3. Monte Carlo pricing (`monte_carlo.py`)

Prices the same options via simulation, as an independent check on Black-Scholes.

**Method:**
1. Simulate thousands of possible stock prices at expiry using the closed-form GBM solution:
   `S_T = S × exp((r − 0.5σ²)T + σ√T × Z)`, where Z ~ Standard Normal
2. Compute the payoff for each: `max(S_T − K, 0)`
3. Average the payoffs (Law of Large Numbers → this average converges to the true expected payoff)
4. Discount back to today: `× e^(−rT)`

**Why it works:** random samples drawn from the normal distribution naturally cluster according to the distribution's own shape — so averaging many simulated payoffs automatically reproduces the correct probability-weighted expectation, without explicitly calculating any probabilities by hand.

**Convergence plot:** shows the Monte Carlo price stabilizing toward the Black-Scholes price as simulation count increases from 100 to 500,000 — a direct visual demonstration of the Law of Large Numbers.

**Bonus: path simulation.** A separate function (`simulate_price_paths`) steps forward in small time increments (rather than jumping straight to expiry) to visualize the actual random walk — not used for pricing here (unnecessary for path-independent European options), but a useful illustration of what GBM looks like as a process, and relevant context for path-dependent (exotic) options.

**Key background concept — logs, e, and lognormality:**
- Percentage returns don't add across periods (+10% then −10% ≠ 0% overall); **log-returns do** (`ln(a/b) + ln(b/c) = ln(a/c)`), which is why the model is built on log-returns rather than raw returns
- Raw price ratios (S_T/S₀) are always positive and right-skewed (bounded below by 0, unbounded above) — this can't be normal
- The **log** of that ratio can range across negative and positive values, and *that* is what's assumed to be normally distributed — this is the actual meaning of "lognormal"
- **Important caveat for the limitations section:** this is a modeling assumption, not an empirical fact — real stock returns have fatter tails and more extreme moves than a true normal distribution predicts. The volatility smile (next section) is direct evidence of this.

---

## 4. Implied volatility & the volatility smile (`implied_vol.py`, `volatility_smile.py`)

**Implied volatility** inverts the usual pricing direction: given an observed market price, solve backward for the σ that Black-Scholes would need to produce it.

**Method — Newton-Raphson:**
```
new_σ = old_σ − (BS_price(old_σ) − market_price) / Vega(old_σ)
```
Iterates until the model price matches the market price within tolerance. Verified first by generating a price from a known σ and confirming the solver recovers it exactly, before applying to real data.

**Real data pipeline:**
1. Pull a live AAPL option chain via `yfinance` (a specific expiry date, not the nearest one — very short-dated options break the solver, since Vega becomes too small and the update step becomes unstable)
2. Use **mid-price** (average of bid/ask), not `lastPrice`, as the market price — more reliable for options that trade infrequently
3. Filter out bad data (zero bid/ask) and non-converged results (solver returns `None` after max iterations without meeting tolerance)
4. Solve for implied volatility at every strike in the chain

**The result:** plotting implied volatility against strike price produces a **smile/skew shape**, not a flat line. Since Black-Scholes assumes one constant σ applies across all strikes, a flat line is what the model predicts — the smile is real markets empirically disagreeing with that assumption. This is one of the most well-known and important limitations of Black-Scholes, and this project demonstrates it directly from live data rather than just stating it.

**Key terms clarified along the way:** bid (best current buy offer), ask (best current sell offer), spread (ask − bid, reflects liquidity), mid-price (the more reliable fair-value proxy for illiquid instruments).

---

## Model limitations (living section — expand as the project continues)

- **Constant volatility assumption is false in practice** — directly evidenced by the volatility smile above
- **Lognormal/normal log-return assumption is a simplification** — real markets show fat tails and skew
- **No dividends** — the base model assumes none; real stocks that pay dividends see a predictable price drop on the ex-dividend date, which this model doesn't account for
- **European exercise only** — Black-Scholes can't directly price American options (early exercise); addressed via binomial tree, pending
- *(To add once complete: binomial tree comparison, any findings from unit testing)*

---

## Key terms glossary (built up through the project)

- **ITM / OTM** — in/out of the money; call ITM when S > K, put ITM when S < K
- **Arbitrage** — risk-free profit from a price discrepancy; markets self-correct to eliminate it, which underlies why put-call parity must hold
- **Risk-neutral pricing** — a valuation trick assuming all assets grow at the risk-free rate; not a claim about real-world expected returns, just a mathematically valid simplification via a hedging/replication argument
- **Replicating portfolio** — a stock + risk-free cash combination proven to have identical payoffs to an option in every scenario; the logical basis for why Black-Scholes gives the "correct," arbitrage-free price
- **Delta-neutral** — a combined position (option + hedge) with ~zero sensitivity to small stock price moves
- **Long / short** — owning an asset (benefits from price rises) vs. having sold/borrowed it (benefits from price falls)

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Still to do

- [ ] Binomial tree pricer (Cox-Ross-Rubinstein), extended to American options
- [ ] Unit tests (pytest) — verify pricing functions against known reference values, put-call parity, Greeks consistency
- [ ] Consolidated Jupyter notebook — pull all modules together with narrative and plots
- [ ] Finalize model limitations write-up
- [ ] Volatility surface (multi-expiry extension), if time allows
