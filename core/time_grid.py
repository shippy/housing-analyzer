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
    annual: list[float] = []

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
