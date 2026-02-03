import numpy as np

from core.time_grid import aggregate_to_annual


def test_aggregate_to_annual_includes_partial_year():
    # 14 months: two full years would be 24 months, so 2 months partial
    monthly = np.arange(1, 15, dtype=float)
    annual = aggregate_to_annual(monthly, months_per_year=12, mode="mean", include_partial=True)
    # First year mean of 1..12 is 6.5, partial mean of 13..14 is 13.5
    assert np.allclose(annual, np.array([6.5, 13.5]))


def test_mortgage_annual_includes_partial_year(monkeypatch):
    from models.mortgage import MortgageModel

    model = MortgageModel(current_rate=0.05, historical_rates=np.array([0.05, 0.05]))

    def fake_paths(years, n_samples, dt=1 / 12):
        return np.tile(np.arange(1, 15, dtype=float), (n_samples, 1))

    monkeypatch.setattr(model, "sample_paths", fake_paths)
    annual = model.sample_paths_annual(years=2, n_samples=1)
    assert annual.shape == (1, 2)
    assert np.allclose(annual[0], np.array([6.5, 13.5]))


def test_fx_annualization_uses_trading_days():
    from models.currency import CurrencyModel

    rates = np.linspace(20.0, 21.0, 260)
    model_252 = CurrencyModel(current_rate=21.0, historical_rates=rates, trading_days_per_year=252)
    model_365 = CurrencyModel(current_rate=21.0, historical_rates=rates, trading_days_per_year=365)
    assert model_252.summary()["sigma_mean"] != model_365.summary()["sigma_mean"]


def test_fx_default_trading_days_is_252():
    from models.currency import CurrencyModel

    model = CurrencyModel(current_rate=21.0, historical_rates=np.linspace(20.0, 21.0, 260))
    assert model.trading_days_per_year == 252
