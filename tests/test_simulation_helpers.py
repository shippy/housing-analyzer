import numpy as np

from models import simulation


def test_buy_cashflow_owner_occupied() -> None:
    compute_delta = getattr(simulation, "compute_buy_cashflow_delta", None)
    assert callable(compute_delta)

    delta = compute_delta(
        annual_mortgage=120.0,
        annual_property_costs=30.0,
        buy_to_let=False,
        annual_rental_income=0.0,
        own_rent=0.0,
    )

    assert np.isclose(delta, -150.0)


def test_buy_cashflow_buy_to_let() -> None:
    compute_delta = getattr(simulation, "compute_buy_cashflow_delta", None)
    assert callable(compute_delta)

    delta = compute_delta(
        annual_mortgage=120.0,
        annual_property_costs=30.0,
        buy_to_let=True,
        annual_rental_income=40.0,
        own_rent=50.0,
    )

    assert np.isclose(delta, -160.0)


def test_remaining_balance_uses_current_state() -> None:
    remaining_balance_after_year = getattr(simulation, "remaining_balance_after_months", None)
    assert callable(remaining_balance_after_year)

    current_principal = 800.0
    annual_rate = 0.06
    remaining_term_months = 300
    months_paid = 12

    expected = simulation.calculate_remaining_balance(
        current_principal,
        annual_rate,
        remaining_term_months,
        months_paid,
    )
    actual = remaining_balance_after_year(
        current_principal,
        annual_rate,
        remaining_term_months,
        months_paid,
    )

    assert np.isclose(actual, expected)


def test_simulation_returns_components() -> None:
    results = simulation.run_simulation(
        property_price=5_000_000,
        down_payment=1_000_000,
        usd_holdings=100_000,
        monthly_rent=20_000,
        years=2,
        n_samples=10,
        seed=1,
        use_cached_models=True,
    )
    assert "components" in results
    assert "equity_yearly" in results["components"]
