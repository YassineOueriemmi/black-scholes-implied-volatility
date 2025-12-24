# Implied Volatility Analysis – Black–Scholes

This project implements a Black–Scholes option pricer and an implied volatility solver,
and applies them to real market data obtained from Yahoo Finance.

The main focus of the project is the analysis of implied volatility dynamics for
equity options.

---

## Project Overview

The project provides:
- Pricing of European call and put options using the Black–Scholes model
- Computation of implied volatility via numerical root-finding methods
- Empirical analysis based on listed equity options

---

## Notebook

The full analysis, including data retrieval, visualizations, and empirical results,
is presented in the Jupyter notebook:

- `black_scholes_implied_volatility.ipynb`

This notebook contains the complete workflow and should be consulted to fully
understand the project.

---

## Volatility Analysis

The analysis includes:
- **Volatility skew** across strikes
- **Term structure of implied volatility** across maturities
- **Implied volatility surface** (strike × maturity)

These results highlight the limitations of the constant-volatility assumption
embedded in the Black–Scholes framework.

---

## Model vs Market Comparison

The project compares:
- Market-implied volatility (Yahoo Finance) vs model-implied volatility
- Theoretical Black–Scholes prices vs observed market prices

The discrepancies illustrate the impact of volatility smiles, skew effects,
and market frictions.

---

## Files

- `black_scholes.py` — Black–Scholes pricing functions
- `implied_vol.py` — Implied volatility solver
- `black_scholes_implied_volatility.ipynb` — Data analysis and visualizations
- `requirements.txt` — Project dependencies

---

## Conclusion

Implied volatility emerges as the key market variable driving option prices.
While the Black–Scholes model provides a useful benchmark, market data clearly
exhibit volatility skew and term structure effects that go beyond the model’s
assumptions.
