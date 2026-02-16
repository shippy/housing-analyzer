# Prague Housing Analyzer

Bayesian buy-vs-rent analysis for Prague real estate using Monte Carlo simulation with PyMC probabilistic modeling.

## Overview

This tool helps answer: *"Should I buy this apartment now, or rent and keep my money invested?"*

It compares two strategies:
- **Buy**: Purchase property with mortgage, liquidating USD investments for down payment
- **Rent + Invest**: Continue renting, keep money invested in equities

The model accounts for uncertainty in:
- Prague real estate appreciation (district-specific)
- Mortgage interest rates (Vasicek mean-reversion model)
- Rent inflation
- USD/CZK exchange rate volatility
- Investment returns (US equities)

## Quick Start

```bash
# Install dependencies
uv sync

# Copy and configure your personal defaults
cp .env.example .env
# Edit .env with your property price, savings, rent, etc.

# Run the interactive Marimo dashboard
uv run marimo run app.py
```

## Configuration (.env)

Create a `.env` file with your personal scenario. All values have sensible defaults, but you should customize them:

```bash
# Property you're considering (CZK)
PROPERTY_PRICE=15000000
DOWN_PAYMENT=3000000

# Your current rent as a tenant (CZK/month)
MONTHLY_RENT=25000

# Your USD savings that would be liquidated for down payment
USD_HOLDINGS=100000

# Analysis horizon
YEARS=10

# Expected rent growth (3% = 0.03)
RENT_INFLATION=0.03

# Prague district (affects appreciation rate)
# Options: prague_avg, prague_1 through prague_10
DISTRICT=prague_avg

# Buy-to-let mode: buy property, rent it out, live in cheaper rental
BUY_TO_LET=false
RENTAL_YIELD=0.025        # Gross annual yield (2.5%)
BTL_EXPENSE_RATE=0.30     # Landlord expenses: vacancy, maintenance, tax (30%)

# FX outlook (bearish USD view)
FX_MIXTURE_ENABLED=true
FX_WEAK_DOLLAR_PROB=0.6   # 60% chance of weak dollar regime
FX_WEAK_DOLLAR_DRIFT=-0.04  # -4%/year in weak regime
FX_STABLE_DRIFT=0.01      # +1%/year in stable regime
```

## Usage

### Interactive Dashboard (Marimo)

The main interface with visualizations, sensitivity analysis, and detailed breakdowns:

```bash
uv run marimo run app.py
```

### Command-Line Interface

For scripting, automation, or quick checks:

```bash
# Run with defaults from .env
uv run python cli.py run

# Override specific parameters
uv run python cli.py run --property-price 20000000 --years 15

# Enable bearish USD outlook
uv run python cli.py run --fx-bearish

# Buy-to-let mode
uv run python cli.py run --buy-to-let --rental-yield 0.03

# JSON output for scripting
uv run python cli.py run --output json > results.json

# Compare neutral vs bearish FX scenarios
uv run python cli.py compare

# Parameter sweep (vary rent from 20k to 40k)
uv run python cli.py sweep --vary monthly_rent --from 20000 --to 40000 --step 5000
```

### Grid Search

Explore which combinations of parameters favor buying vs renting:

```bash
# Run grid search across rent levels, time horizons, and FX scenarios
uv run python grid_search.py

# With more samples per scenario (slower but more accurate)
uv run python grid_search.py 1000
```

Outputs:
- `grid_search_results.json` - Raw results
- `grid_search_results.png` - Heatmaps of P(buy wins)
- `tradeoff_analysis.png` - Risk vs reward charts

## Key Concepts

### Buy-to-Let Mode

Instead of living in the purchased property, you:
1. Buy an investment property and rent it out
2. Continue living in a (potentially cheaper) rental
3. Rental income offsets mortgage costs

Configure with:
- `RENTAL_YIELD`: Gross annual rent / property value (Prague avg: 2-3%)
- `BTL_EXPENSE_RATE`: Vacancy, maintenance, management, taxes (typically 25-35%)

### FX Mixture Model

If you hold USD and are bearish on the dollar, enable the mixture model:
- Each simulation path is assigned to "weak dollar" or "stable" regime
- Weak regime: USD depreciates against CZK
- This affects the value of your USD holdings over time

### Model Features

- **Refinancing**: Mortgages refinanced every 5 years at prevailing rates
- **Tax benefits**: Czech mortgage interest deduction (max 150k CZK/year)
- **Correlations**: Inflation, stocks, FX, and property returns are correlated
- **Inflation adjustment**: Results shown in real (today's) CZK

## Project Structure

```
housing-analyzer/
├── app.py                 # Marimo interactive dashboard
├── cli.py                 # Command-line interface
├── grid_search.py         # Parameter grid search
├── settings.py            # Load defaults from .env
├── core/
│   ├── config.py          # ScenarioConfig dataclass
│   └── runner.py          # High-level simulation runner
├── models/
│   ├── appreciation.py    # Real estate price model (GBM)
│   ├── mortgage.py        # Interest rate model (Vasicek)
│   ├── currency.py        # USD/CZK model (GBM)
│   └── simulation.py      # Monte Carlo orchestration
├── data/
│   └── fetch.py           # Data from CSU, CNB, Yahoo Finance
└── tests/
    └── ...
```

## Data Sources

- **CSU** (Czech Statistical Office) - Housing price indices
- **CNB** (Czech National Bank) - Mortgage rates, FX rates
- **Yahoo Finance** - Historical equity returns

## Running Tests

```bash
uv run pytest
```

## License

MIT
