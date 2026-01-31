"""Monte Carlo simulation with PyMC Bayesian models."""

import numpy as np
import pymc as pm
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Any

from .appreciation import AppreciationModel, fit_from_hpi_data
from .mortgage import MortgageModel
from .currency import CurrencyModel, FXMixtureConfig


# Prague district appreciation multipliers (relative to city average)
# Based on historical price growth differentials
DISTRICT_MULTIPLIERS = {
    "prague_avg": 1.0,
    "prague_1": 0.95,   # Center - already expensive, slower growth
    "prague_2": 1.05,   # Vinohrady - high demand
    "prague_3": 1.08,   # Žižkov - gentrifying
    "prague_4": 1.02,   # Nusle/Podolí - solid
    "prague_5": 1.00,   # Smíchov - average
    "prague_6": 0.98,   # Dejvice - established
    "prague_7": 1.06,   # Holešovice - trendy
    "prague_8": 1.04,   # Karlín - redeveloped
    "prague_9": 1.10,   # Vysočany - catching up
    "prague_10": 1.03,  # Vršovice - popular
}


@dataclass
class _InternalSimConfig:
    """Internal configuration for simulation calculations.

    Note: For external use, import ScenarioConfig from core.config instead.
    This internal class holds fixed model parameters not exposed to CLI/UI.
    """

    # Property parameters
    property_price: float  # CZK
    down_payment: float  # CZK
    monthly_rent: float  # CZK

    # Investment parameters
    usd_holdings: float  # USD
    stock_return_mean: float = 0.07  # Expected annual stock return
    stock_return_std: float = 0.18  # Stock return volatility

    # Mortgage parameters
    mortgage_term_years: int = 30
    initial_mortgage_rate: float = 0.055
    refinance_interval_years: int = 5  # Check refinancing every N years
    refinance_cost_pct: float = 0.005  # 0.5% of remaining balance

    # Rent parameters
    rent_growth_rate: float = 0.03

    # Property costs (as fraction of property value)
    property_tax_rate: float = 0.001
    maintenance_rate: float = 0.01
    insurance_rate: float = 0.002

    # Transaction costs
    buying_costs_rate: float = 0.04
    selling_costs_rate: float = 0.03

    # Tax benefits (Czech specific)
    mortgage_interest_deduction_limit: float = 300_000  # CZK/year max
    income_tax_rate: float = 0.15  # 15% income tax
    capital_gains_exempt_years: int = 5  # Years of residence for exemption
    capital_gains_tax_rate: float = 0.15  # If not exempt

    # Inflation (AR(1) process with persistence)
    inflation_mean: float = 0.025  # 2.5% annual long-run mean
    inflation_std: float = 0.015   # Innovation standard deviation
    inflation_persistence: float = 0.7  # AR(1) coefficient (0=iid, 1=random walk)

    # Location
    district: str = "prague_avg"


def calculate_monthly_payment(principal: float, annual_rate: float, months: int) -> float:
    """Calculate fixed monthly mortgage payment."""
    if annual_rate <= 0:
        return principal / months
    monthly_rate = annual_rate / 12
    payment = principal * (monthly_rate * (1 + monthly_rate) ** months) / (
        (1 + monthly_rate) ** months - 1
    )
    return payment


def calculate_remaining_balance(principal: float, annual_rate: float, 
                                total_months: int, months_paid: int) -> float:
    """Calculate remaining mortgage balance after N months."""
    if months_paid >= total_months:
        return 0.0
    if annual_rate <= 0:
        return principal * (1 - months_paid / total_months)
    
    monthly_rate = annual_rate / 12
    monthly_payment = calculate_monthly_payment(principal, annual_rate, total_months)
    
    # Remaining balance formula
    balance = principal * (1 + monthly_rate) ** months_paid - \
              monthly_payment * ((1 + monthly_rate) ** months_paid - 1) / monthly_rate
    return max(0, balance)


def calculate_interest_paid_year(principal: float, annual_rate: float,
                                  total_months: int, year: int) -> float:
    """Calculate interest paid in a specific year of the mortgage."""
    if annual_rate <= 0:
        return 0.0
    
    monthly_rate = annual_rate / 12
    monthly_payment = calculate_monthly_payment(principal, annual_rate, total_months)
    
    interest_paid = 0.0
    for month in range(year * 12, min((year + 1) * 12, total_months)):
        balance = calculate_remaining_balance(principal, annual_rate, total_months, month)
        interest_this_month = balance * monthly_rate
        interest_paid += interest_this_month
    
    return interest_paid


# Global model cache to avoid refitting on every simulation
_model_cache: dict[str, Any] = {}


# Correlation matrix for risk factors
# Order: [property_appreciation, stock_returns, fx_changes, inflation]
#
# Key relationships modeled:
# - Property/inflation: +0.50 (real assets hedge inflation well)
# - Stocks/inflation: -0.25 (unexpected inflation hurts stock valuations)
# - Stocks/FX: +0.30 (in USD crisis, both stocks and dollar fall together;
#                     this captures stagflation/weak-dollar scenarios that
#                     hurt USD-denominated portfolios)
# - Property/FX: -0.15 (CZK assets benefit when USD weakens)
#
# Note: "FX" here is USD/CZK, so positive FX = stronger dollar.
# A weak dollar (negative FX shock) combined with weak stocks (negative stock shock)
# requires POSITIVE correlation to occur together.
RISK_FACTOR_CORRELATION = np.array([
    [1.00,  0.25, -0.15,  0.50],  # Property appreciation
    [0.25,  1.00,  0.30, -0.25],  # Stock returns
    [-0.15, 0.30,  1.00,  0.10],  # FX changes (USD/CZK)
    [0.50, -0.25,  0.10,  1.00],  # Inflation
])


def generate_correlated_shocks(
    n_samples: int,
    n_years: int,
    correlation_matrix: NDArray[np.float64] = RISK_FACTOR_CORRELATION,
) -> NDArray[np.float64]:
    """Generate correlated standard normal shocks using Cholesky decomposition.

    Args:
        n_samples: Number of Monte Carlo samples.
        n_years: Number of years to simulate.
        correlation_matrix: Correlation matrix for risk factors.

    Returns:
        Array of shape (n_factors, n_samples, n_years) with correlated shocks.
    """
    n_factors = correlation_matrix.shape[0]

    # Cholesky decomposition: Σ = L @ L.T
    # To generate correlated samples: X = L @ Z where Z ~ N(0, I)
    L = np.linalg.cholesky(correlation_matrix)

    # Generate independent standard normal shocks
    Z = np.random.standard_normal((n_factors, n_samples, n_years))

    # Apply Cholesky to introduce correlation
    # For each (sample, year), multiply the factor vector by L
    correlated = np.zeros_like(Z)
    for s in range(n_samples):
        for y in range(n_years):
            correlated[:, s, y] = L @ Z[:, s, y]

    return correlated


def get_fitted_models(
    use_cache: bool = True,
    draws: int = 1000,
    tune: int = 500,
) -> tuple[AppreciationModel, MortgageModel, CurrencyModel]:
    """Get fitted PyMC models, using cache if available."""
    cache_key = f"models_{draws}_{tune}"
    
    if use_cache and cache_key in _model_cache:
        return _model_cache[cache_key]
    
    print("Fitting Bayesian models to historical data...")
    
    try:
        from data.fetch import fetch_housing_prices, fetch_cnb_rates, fetch_fx_rates
        
        hp_data = fetch_housing_prices()
        hpi_values = hp_data["hpi_index"].to_numpy()
        appreciation_model = fit_from_hpi_data(hpi_values)
        print(f"  Appreciation: μ={appreciation_model.summary()['mu_mean']:.1%} ± {appreciation_model.summary()['mu_std']:.1%}")
        
        rates_data = fetch_cnb_rates()
        mortgage_rates = rates_data["mortgage_rate_pct"].to_numpy() / 100
        current_mortgage_rate = float(mortgage_rates[-1]) if len(mortgage_rates) > 0 else 0.055
        mortgage_model = MortgageModel(
            current_rate=current_mortgage_rate,
            historical_rates=mortgage_rates,
            draws=draws,
            tune=tune,
        )
        print(f"  Mortgage: θ={mortgage_model.summary()['theta_mean']:.1%}, current={current_mortgage_rate:.1%}")
        
        fx_data = fetch_fx_rates()
        fx_rates = fx_data["usd_czk"].to_numpy()
        current_fx = float(fx_rates[-1]) if len(fx_rates) > 0 else 24.0
        currency_model = CurrencyModel(
            current_rate=current_fx,
            historical_rates=fx_rates,
            draws=draws,
            tune=tune,
        )
        print(f"  FX: σ={currency_model.summary()['sigma_mean']:.1%}, current={current_fx:.1f}")
        
    except Exception as e:
        print(f"  Warning: Could not fetch data ({e}), using priors only")
        appreciation_model = AppreciationModel(draws=draws, tune=tune)
        mortgage_model = MortgageModel(draws=draws, tune=tune)
        currency_model = CurrencyModel(draws=draws, tune=tune)
    
    models = (appreciation_model, mortgage_model, currency_model)
    
    if use_cache:
        _model_cache[cache_key] = models
    
    return models


def sample_stock_returns(
    n_samples: int,
    years: int,
    mean: float = 0.07,
    std: float = 0.18,
    correlated_shocks: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Sample stock market returns with parameter uncertainty.

    Args:
        n_samples: Number of Monte Carlo samples.
        years: Number of years to simulate.
        mean: Expected annual return.
        std: Annual return volatility.
        correlated_shocks: Optional pre-generated correlated shocks (n_samples, years).
                          If provided, these are used instead of independent sampling.

    Returns:
        Cumulative return factors of shape (n_samples, years).
    """
    with pm.Model():
        mu = pm.Normal("mu", mu=mean, sigma=0.02)
        sigma = pm.HalfNormal("sigma", sigma=std)
        trace = pm.sample(draws=500, tune=200, cores=1, progressbar=False)

    mu_samples = trace.posterior["mu"].values.flatten()
    sigma_samples = trace.posterior["sigma"].values.flatten()

    idx = np.random.choice(len(mu_samples), size=n_samples)
    mus = mu_samples[idx]
    sigmas = sigma_samples[idx]

    annual_returns = np.zeros((n_samples, years))
    for t in range(years):
        if correlated_shocks is not None:
            # Use correlated shocks: transform from standard normal to actual returns
            annual_returns[:, t] = mus + sigmas * correlated_shocks[:, t]
        else:
            annual_returns[:, t] = np.random.normal(mus, sigmas)

    return np.cumprod(1 + annual_returns, axis=1)


def run_simulation(
    property_price: float,
    down_payment: float,
    usd_holdings: float,
    monthly_rent: float,
    years: int,
    renovation_cost: float = 0,
    n_samples: int = 5000,
    seed: int | None = None,
    rent_growth_rate: float = 0.03,
    use_cached_models: bool = True,
    pymc_draws: int = 1000,
    pymc_tune: int = 500,
    # FX mixture model parameters
    fx_mixture_enabled: bool = False,
    fx_weak_dollar_prob: float = 0.6,
    fx_weak_dollar_drift: float = -0.04,
    fx_stable_drift: float = 0.01,
    # Model options
    district: str = "prague_avg",
    enable_refinancing: bool = True,
    enable_tax_benefits: bool = True,
    inflation_adjust: bool = True,
    enable_correlations: bool = True,
    # Buy-to-let mode
    buy_to_let: bool = False,
    rental_yield: float = 0.025,
    btl_expense_rate: float = 0.30,  # Landlord expenses (vacancy, maintenance, mgmt, tax)
) -> dict[str, Any]:
    """Run Monte Carlo simulation with PyMC Bayesian models.

    Features:
    - Yearly wealth tracking for breakeven analysis
    - Mortgage refinancing every 5 years if rates drop
    - Czech tax benefits (interest deduction, capital gains exemption)
    - District-specific appreciation
    - Inflation-adjusted wealth
    - Downside risk metrics
    - Buy-to-let mode: rent out property while living in cheaper rental
    """
    if seed is not None:
        np.random.seed(seed)
    
    config = _InternalSimConfig(
        property_price=property_price,
        down_payment=down_payment,
        monthly_rent=monthly_rent,
        usd_holdings=usd_holdings,
        district=district,
        rent_growth_rate=rent_growth_rate,
    )
    
    # Get district multiplier
    district_mult = DISTRICT_MULTIPLIERS.get(district, 1.0)
    
    # Get fitted models
    appreciation_model, mortgage_model, currency_model = get_fitted_models(
        use_cache=use_cached_models,
        draws=pymc_draws,
        tune=pymc_tune,
    )
    
    # Apply FX mixture config
    fx_mixture_config = FXMixtureConfig(
        enabled=fx_mixture_enabled,
        weak_dollar_prob=fx_weak_dollar_prob,
        weak_dollar_drift=fx_weak_dollar_drift,
        stable_drift=fx_stable_drift,
    )
    currency_model.mixture_config = fx_mixture_config
    
    # Sample paths
    print(f"Sampling {n_samples} Monte Carlo paths...")
    if buy_to_let:
        gross_monthly = property_price * rental_yield / 12
        net_yield = rental_yield * (1 - btl_expense_rate)
        net_monthly = gross_monthly * (1 - btl_expense_rate)
        print(f"  Buy-to-let: {rental_yield:.1%} gross, {net_yield:.1%} net (after {btl_expense_rate:.0%} expenses)")
        print(f"    Net income: {net_monthly:,.0f} CZK/mo, you pay: {monthly_rent:,.0f} CZK/mo")
    if fx_mixture_enabled:
        print(f"  FX mixture: {fx_weak_dollar_prob:.0%} weak (drift={fx_weak_dollar_drift:.1%}), "
              f"{1-fx_weak_dollar_prob:.0%} stable (drift={fx_stable_drift:.1%})")
    if district != "prague_avg":
        print(f"  District: {district} (multiplier: {district_mult:.2f})")
    if not enable_correlations:
        print("  Correlations: DISABLED (independent risk factors)")

    # Generate shocks for risk factors
    # Order: [property_appreciation, stock_returns, fx_changes, inflation]
    if enable_correlations:
        correlated_shocks = generate_correlated_shocks(n_samples, years)
    else:
        # Independent shocks (no correlation structure)
        correlated_shocks = np.random.standard_normal((4, n_samples, years))
    property_shocks = correlated_shocks[0]  # (n_samples, years)
    stock_shocks = correlated_shocks[1]     # (n_samples, years)
    # fx_shocks = correlated_shocks[2]      # Not used directly (model has own dynamics)
    inflation_shocks = correlated_shocks[3] # (n_samples, years)

    # Base appreciation adjusted by district
    # Scale the returns (not the factors) by district multiplier
    # If base grew 50% (factor 1.50) and district_mult is 1.05,
    # we want 52.5% growth (factor 1.525), not 1.50^1.05
    base_appreciation = appreciation_model.sample_paths(years, n_samples)
    base_returns = base_appreciation - 1  # Convert factors to cumulative returns
    property_appreciation = 1 + base_returns * district_mult  # Scale returns, convert back

    # Apply property shocks to adjust appreciation correlation with other factors
    # This post-hoc adjustment preserves the Bayesian parameter uncertainty
    # while introducing correlation with stocks/inflation
    appr_mu = np.mean(base_returns, axis=1, keepdims=True)
    appr_std = np.std(base_returns, axis=1, keepdims=True) + 1e-8
    standardized_returns = (base_returns - appr_mu) / appr_std
    # Blend original with correlated shocks (50% blend to preserve Bayesian uncertainty)
    blended_returns = 0.5 * standardized_returns + 0.5 * property_shocks
    property_appreciation = 1 + (appr_mu + appr_std * blended_returns) * district_mult

    fx_rates = currency_model.sample_paths_annual(years, n_samples)
    mortgage_rates = mortgage_model.sample_paths_annual(years, n_samples)

    # Use correlated shocks for stock returns
    stock_cumulative = sample_stock_returns(
        n_samples, years, correlated_shocks=stock_shocks
    )

    # Sample inflation paths using AR(1) process with correlated shocks
    # AR(1): inflation_t = rho * inflation_{t-1} + (1-rho) * mu + sigma_innov * shock
    # where sigma_innov = sigma * sqrt(1 - rho^2) to preserve unconditional variance
    rho = config.inflation_persistence
    sigma_innov = config.inflation_std * np.sqrt(1 - rho**2) if rho < 1 else config.inflation_std

    inflation_annual = np.zeros((n_samples, years))
    # Initialize at long-run mean
    inflation_annual[:, 0] = config.inflation_mean + sigma_innov * inflation_shocks[:, 0]

    for y in range(1, years):
        inflation_annual[:, y] = (
            rho * inflation_annual[:, y - 1]
            + (1 - rho) * config.inflation_mean
            + sigma_innov * inflation_shocks[:, y]
        )

    # Floor inflation at a reasonable minimum (deflation limit)
    inflation_annual = np.maximum(inflation_annual, -0.05)
    inflation_cumulative = np.cumprod(1 + inflation_annual, axis=1)
    
    # ============================================================
    # INITIALIZE TRACKING ARRAYS
    # ============================================================

    buy_wealth_yearly = np.zeros((n_samples, years))
    rent_wealth_yearly = np.zeros((n_samples, years))
    remaining_balance_yearly = np.zeros((n_samples, years))  # For underwater calculation
    
    # ============================================================
    # SCENARIO 1: BUY (with refinancing and tax benefits)
    # ============================================================
    
    loan_amount = property_price - down_payment
    buying_costs = property_price * config.buying_costs_rate
    total_initial_buy = down_payment + buying_costs + renovation_cost
    
    initial_usd_in_czk = usd_holdings * currency_model.current_rate
    remaining_cash_buy = np.maximum(0, initial_usd_in_czk - total_initial_buy)
    
    # Track mortgage state per sample
    current_principal = np.full(n_samples, loan_amount)
    current_rate = np.full(n_samples, config.initial_mortgage_rate)
    months_into_mortgage = np.zeros(n_samples)
    remaining_term_months = np.full(n_samples, config.mortgage_term_years * 12)
    
    # Track cumulative tax savings
    cumulative_tax_savings = np.zeros(n_samples)
    
    # Property values over time
    property_values = property_price * property_appreciation
    
    # Simulate year by year
    invested_cash = remaining_cash_buy.copy()
    
    for y in range(years):
        # Monthly payment for this year
        monthly_payment = np.array([
            calculate_monthly_payment(p, r, int(m))
            for p, r, m in zip(current_principal, current_rate, remaining_term_months)
        ])
        annual_mortgage = monthly_payment * 12

        # Property costs
        annual_property_costs = (
            config.property_tax_rate + config.maintenance_rate + config.insurance_rate
        ) * property_values[:, y]

        # Buy-to-let cash flow adjustment
        # In owner-occupied mode: you pay mortgage+costs but save on rent (implicit)
        # In buy-to-let mode: you pay mortgage+costs+your_rent but receive rental_income
        # The NET difference vs owner-occupied = rental_income - your_rent
        # (mortgage and property costs are same in both modes)
        if buy_to_let:
            # Rental income from property (yield on current property value, minus landlord expenses)
            gross_rental_income = rental_yield * property_values[:, y]
            annual_rental_income = gross_rental_income * (1 - btl_expense_rate)
            # Your own rent (grows at rent_growth_rate)
            own_rent_this_year = monthly_rent * 12 * (1 + config.rent_growth_rate) ** y
            # Net difference vs owner-occupied: you get rental income but pay your own rent
            btl_net_vs_owner_occupied = annual_rental_income - own_rent_this_year
            # Add to invested cash (will grow with stocks)
            invested_cash += btl_net_vs_owner_occupied

        # Update mortgage state
        start_principal = current_principal.copy()  # Balance at start of year
        months_into_mortgage += 12
        remaining_term_months = np.maximum(0, remaining_term_months - 12)

        # Calculate remaining balance at end of year
        remaining_balance = np.array([
            calculate_remaining_balance(loan_amount, r, config.mortgage_term_years * 12, int(m))
            for r, m in zip(current_rate, months_into_mortgage)
        ])
        current_principal = remaining_balance

        # Interest paid this year (for tax deduction)
        # Interest = Total payments - Principal reduction
        # This is more accurate than start_principal * rate which overestimates
        if enable_tax_benefits:
            principal_reduction = start_principal - remaining_balance
            interest_paid = annual_mortgage - principal_reduction
            # Ensure non-negative (can happen with rounding or paid-off mortgages)
            interest_paid = np.maximum(0, interest_paid)
            deductible_interest = np.minimum(interest_paid, config.mortgage_interest_deduction_limit)
            tax_savings = deductible_interest * config.income_tax_rate
            cumulative_tax_savings += tax_savings
        remaining_balance_yearly[:, y] = remaining_balance  # Track for underwater calculation

        # Refinancing check every N years
        if enable_refinancing and (y + 1) % config.refinance_interval_years == 0:
            new_market_rate = mortgage_rates[:, y]
            # Refinance if new rate is at least 0.5% lower
            should_refinance = (new_market_rate < current_rate - 0.005) & (remaining_balance > 0)

            # Apply refinancing costs and reset terms
            refinance_cost = remaining_balance * config.refinance_cost_pct * should_refinance
            current_rate = np.where(should_refinance, new_market_rate, current_rate)
            remaining_term_months = np.where(
                should_refinance,
                (config.mortgage_term_years - y - 1) * 12,  # New term
                remaining_term_months
            )
            invested_cash -= refinance_cost  # Pay from savings

        # Invested cash grows
        if y > 0:
            invested_cash *= (stock_cumulative[:, y] / stock_cumulative[:, y-1])
        else:
            invested_cash *= stock_cumulative[:, 0]

        # Calculate wealth at year y
        selling_costs = property_values[:, y] * config.selling_costs_rate

        # Capital gains tax (if selling before exemption period)
        if enable_tax_benefits and y < config.capital_gains_exempt_years:
            capital_gain = np.maximum(0, property_values[:, y] - property_price - buying_costs)
            capital_gains_tax = capital_gain * config.capital_gains_tax_rate
        else:
            capital_gains_tax = 0

        equity = property_values[:, y] - remaining_balance - selling_costs - capital_gains_tax
        buy_wealth_yearly[:, y] = equity + invested_cash + cumulative_tax_savings
    
    # ============================================================
    # SCENARIO 2: RENT + INVEST
    # ============================================================

    # USD holdings grow with stocks, converted at year-end FX
    usd_value_yearly = usd_holdings * stock_cumulative * fx_rates

    # Savings accumulation - renter saves difference between buyer's costs and rent
    cumulative_savings = np.zeros(n_samples)

    # Track buyer's mortgage state for fair comparison (same as buy scenario)
    rent_comp_principal = np.full(n_samples, loan_amount)
    rent_comp_rate = np.full(n_samples, config.initial_mortgage_rate)
    rent_comp_term_months = np.full(n_samples, config.mortgage_term_years * 12)
    rent_comp_months_paid = np.zeros(n_samples)

    for y in range(years):
        # Rent for this year (same for renter and buy-to-let buyer's own housing)
        rent_this_year = monthly_rent * 12 * (1 + config.rent_growth_rate) ** y

        # What would buyer pay THIS year (using same rate dynamics as buy scenario)
        buyer_monthly_payment = np.array([
            calculate_monthly_payment(p, r, int(m))
            for p, r, m in zip(rent_comp_principal, rent_comp_rate, rent_comp_term_months)
        ])
        buyer_annual_mortgage = buyer_monthly_payment * 12

        # Property costs buyer would pay (using simulated appreciation)
        buyer_property_costs = (
            config.property_tax_rate + config.maintenance_rate + config.insurance_rate
        ) * property_values[:, y]

        # Renter saves the difference between owner-occupied buyer's costs and rent
        # This comparison is consistent regardless of buy_to_let mode because:
        # - In owner-occupied: btl_net advantage is zero (no rental income, no own rent)
        # - In buy-to-let: btl_net advantage is captured in the BUY scenario's invested_cash
        # Using owner-occupied as baseline avoids double-counting the btl_net advantage
        annual_savings = buyer_annual_mortgage + buyer_property_costs - rent_this_year

        # Compound existing savings using stock returns (correlated with other factors)
        if y > 0:
            cumulative_savings *= (stock_cumulative[:, y] / stock_cumulative[:, y - 1])
        else:
            cumulative_savings *= stock_cumulative[:, 0]
        cumulative_savings += annual_savings

        rent_wealth_yearly[:, y] = usd_value_yearly[:, y] + cumulative_savings

        # Update buyer's mortgage state (same logic as buy scenario for fair comparison)
        rent_comp_months_paid += 12
        rent_comp_term_months = np.maximum(0, rent_comp_term_months - 12)

        rent_comp_principal = np.array([
            calculate_remaining_balance(loan_amount, r, config.mortgage_term_years * 12, int(m))
            for r, m in zip(rent_comp_rate, rent_comp_months_paid)
        ])

        # Apply same refinancing logic as buyer
        if enable_refinancing and (y + 1) % config.refinance_interval_years == 0:
            new_market_rate = mortgage_rates[:, y]
            should_refinance = (new_market_rate < rent_comp_rate - 0.005) & (rent_comp_principal > 0)
            rent_comp_rate = np.where(should_refinance, new_market_rate, rent_comp_rate)
            rent_comp_term_months = np.where(
                should_refinance,
                (config.mortgage_term_years - y - 1) * 12,
                rent_comp_term_months
            )
    
    # ============================================================
    # INFLATION ADJUSTMENT
    # ============================================================
    
    if inflation_adjust:
        buy_wealth_yearly_real = buy_wealth_yearly / inflation_cumulative
        rent_wealth_yearly_real = rent_wealth_yearly / inflation_cumulative
    else:
        buy_wealth_yearly_real = buy_wealth_yearly
        rent_wealth_yearly_real = rent_wealth_yearly
    
    # Final wealth (nominal and real)
    buy_wealth = buy_wealth_yearly[:, -1]
    rent_wealth = rent_wealth_yearly[:, -1]
    buy_wealth_real = buy_wealth_yearly_real[:, -1]
    rent_wealth_real = rent_wealth_yearly_real[:, -1]
    
    # ============================================================
    # ANALYSIS
    # ============================================================
    
    def percentiles(arr: NDArray) -> dict[str, float]:
        return {
            "p5": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }
    
    # Breakeven timeline: P(buy wins) at each year
    buy_wins_by_year = np.mean(buy_wealth_yearly_real > rent_wealth_yearly_real, axis=0)
    
    # Final probabilities
    buy_wins_prob = float(np.mean(buy_wealth_real > rent_wealth_real))
    
    # Downside risk metrics
    initial_capital = initial_usd_in_czk
    
    # P(underwater): property value < remaining mortgage
    # Uses per-sample remaining balances that account for refinancing
    p_underwater_by_year = []
    for y in range(years):
        underwater = property_values[:, y] < remaining_balance_yearly[:, y]
        p_underwater_by_year.append(float(np.mean(underwater)))
    
    # P(rent wealth < initial capital)
    p_rent_loss = float(np.mean(rent_wealth_real < initial_capital / inflation_cumulative[:, -1]))
    p_buy_loss = float(np.mean(buy_wealth_real < initial_capital / inflation_cumulative[:, -1]))
    
    # Worst 5% scenarios
    buy_worst_5pct = float(np.percentile(buy_wealth_real, 5))
    rent_worst_5pct = float(np.percentile(rent_wealth_real, 5))
    
    wealth_diff = buy_wealth_real - rent_wealth_real
    
    summary_stats = {
        "buy": percentiles(buy_wealth),
        "rent": percentiles(rent_wealth),
        "buy_real": percentiles(buy_wealth_real),
        "rent_real": percentiles(rent_wealth_real),
        "difference": percentiles(wealth_diff),
        "buy_wins_prob": buy_wins_prob,
        "expected_advantage_buy": float(np.mean(wealth_diff)),
        "median_advantage_buy": float(np.median(wealth_diff)),
    }
    
    downside_metrics = {
        "p_underwater_final": p_underwater_by_year[-1] if p_underwater_by_year else 0,
        "p_underwater_by_year": p_underwater_by_year,
        "p_rent_loss_real": p_rent_loss,
        "p_buy_loss_real": p_buy_loss,
        "buy_worst_5pct_real": buy_worst_5pct,
        "rent_worst_5pct_real": rent_worst_5pct,
        "buy_var_95": float(initial_capital - np.percentile(buy_wealth_real, 5)),
        "rent_var_95": float(initial_capital - np.percentile(rent_wealth_real, 5)),
    }
    
    model_summaries = {
        "appreciation": appreciation_model.summary(),
        "mortgage": mortgage_model.summary(),
        "currency": currency_model.summary(),
    }
    
    return {
        "buy_wealth": buy_wealth,
        "rent_wealth": rent_wealth,
        "buy_wealth_real": buy_wealth_real,
        "rent_wealth_real": rent_wealth_real,
        "buy_wealth_yearly": buy_wealth_yearly,
        "rent_wealth_yearly": rent_wealth_yearly,
        "buy_wealth_yearly_real": buy_wealth_yearly_real,
        "rent_wealth_yearly_real": rent_wealth_yearly_real,
        "buy_wins_prob": buy_wins_prob,
        "buy_wins_by_year": buy_wins_by_year.tolist(),
        "summary_stats": summary_stats,
        "downside_metrics": downside_metrics,
        "model_summaries": model_summaries,
        "config": config,
        "district_multiplier": district_mult,
        "inflation_cumulative_median": float(np.median(inflation_cumulative[:, -1])),
        "paths": {
            "property_appreciation": property_appreciation,
            "fx_rates": fx_rates,
            "stock_cumulative": stock_cumulative,
            "mortgage_rates": mortgage_rates,
            "inflation_cumulative": inflation_cumulative,
        },
    }


def print_summary(results: dict[str, Any]) -> None:
    """Print a human-readable summary.

    Deprecated: Use core.runner.format_results() instead.
    """
    from core.runner import format_results
    print(format_results(results))
