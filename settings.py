"""Application settings loaded from environment variables.

Create a .env file in the project root with your personal values.
See .env.example for the template.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class ScenarioDefaults(BaseSettings):
    """Default scenario parameters loaded from environment.

    These can be overridden in the app UI, but the defaults
    come from .env to avoid committing personal financial data.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Property parameters
    property_price: int = 15_000_000  # CZK
    down_payment: int = 3_000_000  # CZK
    monthly_rent: int = 25_000  # CZK

    # Investment parameters
    usd_holdings: int = 100_000  # USD

    # Time horizon
    years: int = 10

    # Rent inflation expectation
    rent_inflation: float = 0.03

    # Default district
    district: str = "prague_avg"

    # Buy-to-let mode
    buy_to_let: bool = False
    rental_yield: float = 0.025  # 2.5% gross
    btl_expense_rate: float = 0.30  # 30% expenses

    # FX mixture model (bearish USD view)
    fx_mixture_enabled: bool = False
    fx_weak_dollar_prob: float = 0.6
    fx_weak_dollar_drift: float = -0.04
    fx_stable_drift: float = 0.01


# Singleton instance
defaults = ScenarioDefaults()
