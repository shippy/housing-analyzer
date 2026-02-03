# Time-Grid Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify time aggregation and remove remaining calculation issues by introducing a shared time-grid helper, updating FX and mortgage annualization, and wiring real simulation components into the UI.

**Architecture:** Add a small `core/time_grid.py` helper for annual aggregation and use it from models and simulation. Track simulation component arrays and surface them to the UI so composition uses real values.

**Tech Stack:** Python, NumPy, PyMC, pytest, Marimo UI.
---

### Task 1: Add time-grid aggregation helper

**Files:**
- Create: `core/time_grid.py`
- Test: `tests/test_time_grid.py`

**Step 1: Write the failing test**
```python
import numpy as np
from core.time_grid import aggregate_to_annual

def test_aggregate_to_annual_includes_partial_year():
    # 14 months: two full years would be 24 months, so 2 months partial
    monthly = np.arange(1, 15, dtype=float)
    annual = aggregate_to_annual(monthly, months_per_year=12, mode="mean", include_partial=True)
    # First year mean of 1..12 is 6.5, partial mean of 13..14 is 13.5
    assert np.allclose(annual, np.array([6.5, 13.5]))
```

**Step 2: Run test to verify it fails**
Run: `uv run pytest tests/test_time_grid.py::test_aggregate_to_annual_includes_partial_year -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'core.time_grid'`

**Step 3: Write minimal implementation**
```python
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

def aggregate_to_annual(
    values: NDArray[np.float64],
    months_per_year: int = 12,
    mode: str = "mean",
    include_partial: bool = True,
) -> NDArray[np.float64]:
    if months_per_year <= 0:
        raise ValueError("months_per_year must be positive")
    if values.ndim != 1:
        raise ValueError("values must be a 1D array")

    n_months = values.shape[0]
    n_full_years = n_months // months_per_year
    full = values[: n_full_years * months_per_year]
    annual = []

    if n_full_years > 0:
        reshaped = full.reshape(n_full_years, months_per_year)
        if mode == "mean":
            annual.extend(np.mean(reshaped, axis=1))
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    remainder = values[n_full_years * months_per_year :]
    if include_partial and remainder.size > 0:
        annual.append(float(np.mean(remainder)))

    return np.array(annual, dtype=float)
```

**Step 4: Run test to verify it passes**
Run: `uv run pytest tests/test_time_grid.py::test_aggregate_to_annual_includes_partial_year -v`  
Expected: PASS

**Step 5: Commit**
```bash
git add core/time_grid.py tests/test_time_grid.py
git commit -m "feat: add annual aggregation helper"
```

---

### Task 2: Update mortgage annualization to use helper

**Files:**
- Modify: `models/mortgage.py`
- Test: `tests/test_time_grid.py`

**Step 1: Write the failing test**
```python
import numpy as np
from models.mortgage import MortgageModel

def test_mortgage_annual_includes_partial_year(monkeypatch):
    model = MortgageModel(current_rate=0.05, historical_rates=np.array([0.05, 0.05]))

    # Stub sample_paths to return 14 months
    def fake_paths(years, n_samples, dt=1/12):
        return np.tile(np.arange(1, 15, dtype=float), (n_samples, 1))

    monkeypatch.setattr(model, "sample_paths", fake_paths)
    annual = model.sample_paths_annual(years=2, n_samples=1)
    assert annual.shape == (1, 2)
    assert np.allclose(annual[0], np.array([6.5, 13.5]))
```

**Step 2: Run test to verify it fails**
Run: `uv run pytest tests/test_time_grid.py::test_mortgage_annual_includes_partial_year -v`  
Expected: FAIL because current annualization truncates remainder.

**Step 3: Write minimal implementation**
```python
from core.time_grid import aggregate_to_annual

monthly = self.sample_paths(years, n_samples, dt=1/12)
annual = np.array([aggregate_to_annual(row) for row in monthly])
return annual[:, :years]
```

**Step 4: Run test to verify it passes**
Run: `uv run pytest tests/test_time_grid.py::test_mortgage_annual_includes_partial_year -v`  
Expected: PASS

**Step 5: Commit**
```bash
git add models/mortgage.py tests/test_time_grid.py
git commit -m "fix: handle partial-year mortgage averages"
```

---

### Task 3: Make FX annualization explicit and consistent

**Files:**
- Modify: `models/currency.py`
- Test: `tests/test_time_grid.py`

**Step 1: Write the failing test**
```python
import numpy as np
from models.currency import CurrencyModel

def test_fx_annualization_uses_trading_days():
    rates = np.linspace(20.0, 21.0, 260)
    model_252 = CurrencyModel(current_rate=21.0, historical_rates=rates, trading_days_per_year=252)
    model_365 = CurrencyModel(current_rate=21.0, historical_rates=rates, trading_days_per_year=365)
    # Expect different daily sigma based on annualization assumption
    assert model_252.summary()["sigma_mean"] != model_365.summary()["sigma_mean"]
```

**Step 2: Run test to verify it fails**
Run: `uv run pytest tests/test_time_grid.py::test_fx_annualization_uses_trading_days -v`  
Expected: FAIL because the parameter doesn't exist yet.

**Step 3: Write minimal implementation**
```python
# CurrencyModel.__init__
def __init__(..., trading_days_per_year: int = 365, ...):
    self.trading_days_per_year = trading_days_per_year

# In _fit
daily_mu = mu / self.trading_days_per_year
daily_sigma = sigma / np.sqrt(self.trading_days_per_year)

# In sample_paths
def sample_paths(..., dt: float | None = None):
    if dt is None:
        dt = 1 / self.trading_days_per_year
```

**Step 4: Run test to verify it passes**
Run: `uv run pytest tests/test_time_grid.py::test_fx_annualization_uses_trading_days -v`  
Expected: PASS

**Step 5: Commit**
```bash
git add models/currency.py tests/test_time_grid.py
git commit -m "fix: make FX annualization explicit"
```

---

### Task 4: Track simulation-derived components for UI

**Files:**
- Modify: `models/simulation.py`
- Modify: `app.py`
- Test: `tests/test_simulation_helpers.py`

**Step 1: Write the failing test**
```python
from models.simulation import run_simulation

def test_simulation_returns_components():
    results = run_simulation(
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
```

**Step 2: Run test to verify it fails**
Run: `uv run pytest tests/test_simulation_helpers.py::test_simulation_returns_components -v`  
Expected: FAIL because components are not returned.

**Step 3: Write minimal implementation**
```python
# In run_simulation, track:
equity_yearly = np.zeros((n_samples, years))
invested_cash_yearly = np.zeros((n_samples, years))
usd_value_yearly = ...
savings_yearly = ...

equity_yearly[:, y] = equity
invested_cash_yearly[:, y] = invested_cash
...

return {
  ...
  "components": {
    "equity_yearly": equity_yearly,
    "invested_cash_yearly": invested_cash_yearly,
    "usd_value_yearly": usd_value_yearly,
    "savings_yearly": savings_yearly,
  },
}
```

**Step 4: Run test to verify it passes**
Run: `uv run pytest tests/test_simulation_helpers.py::test_simulation_returns_components -v`  
Expected: PASS

**Step 5: Update UI to use components**
- Replace rough approximations with medians derived from `components`.
- Use `np.median(components["equity_yearly"][:, -1])`, etc.

**Step 6: Run UI-related tests**
Run: `uv run pytest tests/test_cli.py -v`  
Expected: PASS

**Step 7: Commit**
```bash
git add models/simulation.py app.py tests/test_simulation_helpers.py
git commit -m "feat: use simulation components for composition"
```

---

### Task 5: Full test pass

**Step 1: Run all tests**
Run: `uv run pytest`  
Expected: PASS

**Step 2: Commit**
```bash
git add -A
git commit -m "test: ensure full suite passes after time-grid refactor"
```
