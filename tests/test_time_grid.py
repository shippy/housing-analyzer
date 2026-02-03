import numpy as np

from core.time_grid import aggregate_to_annual


def test_aggregate_to_annual_includes_partial_year():
    # 14 months: two full years would be 24 months, so 2 months partial
    monthly = np.arange(1, 15, dtype=float)
    annual = aggregate_to_annual(monthly, months_per_year=12, mode="mean", include_partial=True)
    # First year mean of 1..12 is 6.5, partial mean of 13..14 is 13.5
    assert np.allclose(annual, np.array([6.5, 13.5]))
