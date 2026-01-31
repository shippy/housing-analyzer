# Prague Housing Analyzer 🏠

Bayesian buy-vs-rent analysis for Prague real estate using PyMC probabilistic modeling.

## Overview

This tool helps answer: *"Should I buy this apartment now, or rent and keep my money invested?"*

It models uncertainty in:
- Prague real estate appreciation
- Mortgage interest rates (CNB)
- Rent inflation
- USD/CZK exchange rate volatility
- Investment returns (US & CZ equities)

## Quick Start

```bash
# Install dependencies
uv sync

# Run the Marimo app
uv run marimo run app.py
```

## Configuration

Default scenario (editable in the app):
- Property: 25,000,000 CZK (Vinohrady, 190m²)
- Down payment: 5,000,000 CZK
- USD holdings to liquidate: $200,000
- Time horizon: 10 years

## Project Structure

```
├── app.py              # Main Marimo dashboard
├── models/
│   ├── __init__.py
│   ├── appreciation.py # RE price model
│   ├── mortgage.py     # Interest rate model
│   ├── currency.py     # USD/CZK model
│   └── simulation.py   # Monte Carlo orchestration
├── data/
│   ├── __init__.py
│   └── fetch.py        # Data fetching from ČSÚ, CNB, etc.
└── tests/
    └── test_models.py
```

## Data Sources

- **ČSÚ** - Czech Statistical Office (housing prices)
- **CNB** - Czech National Bank (rates, FX)
- **Yahoo Finance** - Historical returns
