"""
This module implements the Black–Scholes framework to price European call and put options, compute Vega, 
and validate arbitrage consistency (via put–call parity), using numerically robust inputs.
"""

import math
from dataclasses import dataclass
from typing import Literal, Optional
from scipy.stats import norm

# Option type restricted to call or put
OptionType = Literal["call", "put"]

# Immutable data class to store Black–Scholes model parameters


@dataclass(frozen=True)
class BSInputs:
    S: float  # Spot
    K: float  # Strike
    T: float  # Time to maturity in years
    r: float  # Continuously compounded risk-free rate
    q: float = 0.0  # Continuous dividend yield (default value 0)
    sigma: float = 0.2  # Volatility (defaukt value 0.2)

 # Validates that option inputs are mathematically admissible


def _validate_inputs(S: float, K: float, T: float, sigma: float) -> None:
    if S <= 0:
        raise ValueError("Spot S must be > 0.")
    if K <= 0:
        raise ValueError("Strike K must be > 0.")
    if T < 0:
        raise ValueError("Time to maturity T must be >= 0.")
    if sigma < 0:
        raise ValueError("Volatility sigma must be >= 0.")


def black_scholes_price(option_type: OptionType, S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:

    option_type = option_type.lower().strip()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    _validate_inputs(S, K, T, sigma)

# If expired (T=0): return intrinsic value (European payoff at maturity)
    if T == 0:
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

# If sigma==0: deterministic forward under risk-neutral measure
# ST = S * exp((r-q)T) and payoff discounted at r.
# Price = exp(-rT) * payoff(ST)
    if sigma == 0:
        forward_T = S * math.exp((r - q) * T)
        disc = math.exp(-r * T)
        if option_type == "call":
            return disc * max(forward_T - K, 0.0)
        else:
            return disc * max(K - forward_T, 0.0)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    if option_type == "call":
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    else:
        return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)


"""
Vega = dPrice/dSigma (per 1.0 volatility unit, i.e., not per 1%).
Useful for Newton-Raphson implied volatility.
"""


def black_scholes_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    _validate_inputs(S, K, T, sigma)

    if T == 0 or sigma == 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return S * math.exp(-q * T) * norm.pdf(d1) * sqrtT


"""
Put Call Parity check
Returns True if put-call parity holds within tolerance: C - P = S e^{-qT} - K e^{-rT}
"""


def put_call_parity_check(S: float, K: float, T: float, r: float, q: float, call_price: float, put_price: float, tol: float = 1e-6) -> bool:

    lhs = call_price - put_price
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    return abs(lhs - rhs) <= tol
