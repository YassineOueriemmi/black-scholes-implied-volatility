# This module computes a stable and arbitrage-safe implied volatility by combining Newton–Raphson and bisection.

import math
from typing import Literal, Optional
from pricer.black_scholes import black_scholes_price, black_scholes_vega

OptionType = Literal["call", "put"]

"""
Returns theoretical no-arbitrage bounds for a European option under continuous rates:
    Call lower bound: max(0, S e^{-qT} - K e^{-rT})
    Call upper bound: S e^{-qT}
    Put  lower bound: max(0, K e^{-rT} - S e^{-qT})
    Put  upper bound: K e^{-rT}
"""


def _no_arbitrage_bounds(option_type: OptionType, S: float, K: float, T: float, r: float, q: float) -> tuple[float, float]:

    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    if option_type == "call":
        lower = max(0.0, S * disc_q - K * disc_r)
        upper = S * disc_q
    else:
        lower = max(0.0, K * disc_r - S * disc_q)
        upper = K * disc_r

    return lower, upper


"""
Computes implied volatility using Newton-Raphson.
Returns: sigma (float). If convergence fails or price is invalid, returns NaN.
"""


def implied_vol_newton(option_type: OptionType, market_price: float, S: float, K: float, T: float, r: float, q: float = 0.0, initial_sigma: float = 0.2, tol: float = 1e-7, max_iter: int = 100, vega_floor: float = 1e-10, sigma_min: float = 1e-6, sigma_max: float = 5.0, enforce_bounds: bool = True) -> float:

    option_type = option_type.lower().strip()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
    if market_price < 0:
        return float("nan")
    if T <= 0:  # No time value at expiry: IV is undefined if market price != intrinsic.
        return float("nan")

    if enforce_bounds:
        lb, ub = _no_arbitrage_bounds(option_type, S, K, T, r, q)
        # Price inconsistent with simple European bounds => no stable IV
        if not (lb - 1e-12 <= market_price <= ub + 1e-12):
            return float("nan")

    sigma = float(initial_sigma)
    sigma = min(max(sigma, sigma_min), sigma_max)

    for _ in range(max_iter):
        price = black_scholes_price(option_type, S, K, T, r, q, sigma)
        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        vega = black_scholes_vega(S, K, T, r, q, sigma)
        # Vega too small => Newton unstable (deep ITM/OTM or near expiry)
        if vega < vega_floor:
            return float("nan")

        sigma -= diff / vega

        # Keep sigma within reasonable bounds
        if sigma < sigma_min or sigma > sigma_max:
            return float("nan")

    return float("nan")


"""
Robust fallback method: bisection on sigma.
Slower but far more stable than Newton.
"""


def implied_vol_bisection(option_type: OptionType, market_price: float, S: float, K: float, T: float, r: float, q: float = 0.0, tol: float = 1e-7, max_iter: int = 200, sigma_low: float = 1e-6, sigma_high: float = 5.0, enforce_bounds: bool = True) -> float:

    option_type = option_type.lower().strip()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
    if market_price < 0 or T <= 0:
        return float("nan")

    if enforce_bounds:
        lb, ub = _no_arbitrage_bounds(option_type, S, K, T, r, q)
        if not (lb - 1e-12 <= market_price <= ub + 1e-12):
            return float("nan")

    low, high = sigma_low, sigma_high
    f_low = black_scholes_price(option_type, S, K, T, r, q, low) - market_price
    f_high = black_scholes_price(
        option_type, S, K, T, r, q, high) - market_price

    # If signs are not opposite, bisection won't work
    if f_low * f_high > 0:
        return float("nan")

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = black_scholes_price(
            option_type, S, K, T, r, q, mid) - market_price

        if abs(f_mid) < tol or (high - low) < tol:
            return mid

        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    return float("nan")


"""
Convenience wrapper:
- try Newton first
- fallback to bisection if Newton fails
"""


def implied_vol(option_type: OptionType, market_price: float, S: float, K: float, T: float, r: float, q: float = 0.0, initial_sigma: float = 0.2,) -> float:

    sigma = implied_vol_newton(option_type=option_type, market_price=market_price,
                               S=S, K=K, T=T, r=r, q=q, initial_sigma=initial_sigma)
    if math.isnan(sigma):
        sigma = implied_vol_bisection(
            option_type=option_type, market_price=market_price, S=S, K=K, T=T, r=r, q=q)
    return sigma
