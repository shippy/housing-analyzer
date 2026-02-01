"""Scenario configuration for housing analysis simulations."""

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ScenarioConfig:
    """Complete configuration for a buy vs rent simulation scenario.

    This dataclass consolidates all parameters needed to run a simulation,
    enabling reproducible experiments and CLI-based analysis.
    """

    # Property parameters
    property_price: float  # CZK
    down_payment: float  # CZK
    monthly_rent: float  # CZK

    # Investment parameters
    usd_holdings: float  # USD

    # Time horizon
    years: int = 10
    renovation_cost: float = 0  # CZK - additional upfront cost for older properties
    rent_growth_rate: float = 0.03

    # FX mixture model parameters
    fx_mixture_enabled: bool = False
    fx_weak_dollar_prob: float = 0.6
    fx_weak_dollar_drift: float = -0.04
    fx_stable_drift: float = 0.01

    # Model options
    district: str = "prague_avg"
    enable_refinancing: bool = True
    enable_tax_benefits: bool = True
    inflation_adjust: bool = True
    enable_correlations: bool = True

    # Buy-to-let mode: buy property, rent it out, live in smaller rental
    buy_to_let: bool = False
    rental_yield: float = 0.025  # Gross annual yield on property (2.5% default for Prague)
    btl_expense_rate: float = 0.30  # Landlord expenses as fraction of gross rent (vacancy, maintenance, mgmt, tax)

    # Simulation parameters
    n_samples: int = 5000
    seed: int | None = None

    # PyMC parameters (advanced)
    pymc_draws: int = 1000
    pymc_tune: int = 500
    use_cached_models: bool = True

    @classmethod
    def from_defaults(cls) -> "ScenarioConfig":
        """Create a config from environment defaults in settings.py."""
        from settings import defaults

        return cls(
            property_price=float(defaults.property_price),
            down_payment=float(defaults.down_payment),
            monthly_rent=float(defaults.monthly_rent),
            usd_holdings=float(defaults.usd_holdings),
            years=defaults.years,
            rent_growth_rate=defaults.rent_inflation,
            district=defaults.district,
            # Buy-to-let
            buy_to_let=defaults.buy_to_let,
            rental_yield=defaults.rental_yield,
            btl_expense_rate=defaults.btl_expense_rate,
            # FX mixture
            fx_mixture_enabled=defaults.fx_mixture_enabled,
            fx_weak_dollar_prob=defaults.fx_weak_dollar_prob,
            fx_weak_dollar_drift=defaults.fx_weak_dollar_drift,
            fx_stable_drift=defaults.fx_stable_drift,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return asdict(self)

    def to_simulation_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for run_simulation() function."""
        return {
            "property_price": self.property_price,
            "down_payment": self.down_payment,
            "monthly_rent": self.monthly_rent,
            "renovation_cost": self.renovation_cost,
            "usd_holdings": self.usd_holdings,
            "years": self.years,
            "rent_growth_rate": self.rent_growth_rate,
            "n_samples": self.n_samples,
            "seed": self.seed,
            "use_cached_models": self.use_cached_models,
            "pymc_draws": self.pymc_draws,
            "pymc_tune": self.pymc_tune,
            "fx_mixture_enabled": self.fx_mixture_enabled,
            "fx_weak_dollar_prob": self.fx_weak_dollar_prob,
            "fx_weak_dollar_drift": self.fx_weak_dollar_drift,
            "fx_stable_drift": self.fx_stable_drift,
            "district": self.district,
            "enable_refinancing": self.enable_refinancing,
            "enable_tax_benefits": self.enable_tax_benefits,
            "inflation_adjust": self.inflation_adjust,
            "enable_correlations": self.enable_correlations,
            "buy_to_let": self.buy_to_let,
            "rental_yield": self.rental_yield,
            "btl_expense_rate": self.btl_expense_rate,
        }

    def with_fx_bearish(self) -> "ScenarioConfig":
        """Return a copy with bearish USD outlook."""
        return ScenarioConfig(
            **{
                **self.to_dict(),
                "fx_mixture_enabled": True,
                "fx_weak_dollar_prob": 0.8,
                "fx_weak_dollar_drift": -0.04,
                "fx_stable_drift": 0.01,
            }
        )

    def with_fx_neutral(self) -> "ScenarioConfig":
        """Return a copy with neutral FX outlook (use historical)."""
        return ScenarioConfig(
            **{
                **self.to_dict(),
                "fx_mixture_enabled": False,
            }
        )

    def summary(self) -> str:
        """Return a human-readable summary of the configuration."""
        mode = "Buy-to-let (landlord)" if self.buy_to_let else "Owner-occupied"
        lines = [
            "Scenario Configuration",
            "=" * 40,
            f"Mode: {mode}",
            f"Property price: {self.property_price:,.0f} CZK",
            f"Down payment: {self.down_payment:,.0f} CZK",
        ]
        if self.renovation_cost > 0:
            lines.append(f"Renovation cost: {self.renovation_cost:,.0f} CZK")
            total_upfront = self.down_payment + self.renovation_cost
            lines.append(f"Total upfront: {total_upfront:,.0f} CZK (excl. closing costs)")
        lines.extend([
            f"Monthly rent (your housing): {self.monthly_rent:,.0f} CZK",
            f"USD holdings: ${self.usd_holdings:,.0f}",
            f"Time horizon: {self.years} years",
            f"Rent growth: {self.rent_growth_rate:.1%}/year",
            f"District: {self.district}",
        ])

        if self.buy_to_let:
            gross_monthly = self.property_price * self.rental_yield / 12
            net_yield = self.rental_yield * (1 - self.btl_expense_rate)
            net_monthly = gross_monthly * (1 - self.btl_expense_rate)
            lines.extend([
                "",
                "Buy-to-let Details:",
                f"  Gross rental yield: {self.rental_yield:.1%}",
                f"  Landlord expenses: {self.btl_expense_rate:.0%} (vacancy, maintenance, mgmt, tax)",
                f"  Net rental yield: {net_yield:.1%}",
                f"  Net rental income: {net_monthly:,.0f} CZK/month",
                f"  You pay rent: {self.monthly_rent:,.0f} CZK/month",
            ])

        lines.extend([
            "",
            "Model Options:",
            f"  Refinancing: {'enabled' if self.enable_refinancing else 'disabled'}",
            f"  Tax benefits: {'enabled' if self.enable_tax_benefits else 'disabled'}",
            f"  Inflation adjust: {'enabled' if self.inflation_adjust else 'disabled'}",
            f"  Correlations: {'enabled' if self.enable_correlations else 'disabled'}",
        ])

        if self.fx_mixture_enabled:
            expected_drift = (
                self.fx_weak_dollar_prob * self.fx_weak_dollar_drift
                + (1 - self.fx_weak_dollar_prob) * self.fx_stable_drift
            )
            lines.extend([
                "",
                "FX Mixture Model:",
                f"  P(weak dollar): {self.fx_weak_dollar_prob:.0%}",
                f"  Weak drift: {self.fx_weak_dollar_drift:.1%}/year",
                f"  Stable drift: {self.fx_stable_drift:.1%}/year",
                f"  Expected drift: {expected_drift:.1%}/year",
            ])
        else:
            lines.append("\nFX: Using historical drift")

        lines.extend([
            "",
            f"Simulation: {self.n_samples} samples" + (f", seed={self.seed}" if self.seed else ""),
        ])

        return "\n".join(lines)
