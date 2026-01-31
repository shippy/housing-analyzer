"""Real estate appreciation model using PyMC Bayesian inference."""

import numpy as np
import pymc as pm
import arviz as az
from numpy.typing import NDArray


class AppreciationModel:
    """Bayesian model for Prague real estate appreciation.
    
    Fits a hierarchical model to historical housing price index data,
    estimating the mean and volatility of annual returns with uncertainty.
    """
    
    def __init__(
        self,
        historical_returns: NDArray[np.float64] | None = None,
        draws: int = 1000,
        tune: int = 500,
    ):
        """Initialize and fit the model.
        
        Args:
            historical_returns: Array of historical annual returns (e.g., 0.05 for 5%).
                               If None, uses weakly informative priors.
            draws: Number of posterior draws.
            tune: Number of tuning steps.
        """
        self.historical_returns = historical_returns
        self.draws = draws
        self.tune = tune
        self.trace = None
        self._fit()
    
    def _fit(self) -> None:
        """Fit the Bayesian model to historical data."""
        with pm.Model() as model:
            # Priors for mean annual return and volatility
            # Based on long-run real estate expectations
            mu = pm.Normal("mu", mu=0.05, sigma=0.03)  # ~5% expected, wide prior
            sigma = pm.HalfNormal("sigma", sigma=0.10)  # Volatility prior
            
            if self.historical_returns is not None and len(self.historical_returns) > 0:
                # Likelihood: observed returns are normally distributed
                # (log-returns are approximately normal for small values)
                obs = pm.Normal(
                    "returns",
                    mu=mu,
                    sigma=sigma,
                    observed=self.historical_returns,
                )
            
            # Sample from posterior
            self.trace = pm.sample(
                draws=self.draws,
                tune=self.tune,
                cores=1,  # Safer in containerized environments
                progressbar=False,
                return_inferencedata=True,
            )
        
        self.model = model
        
        # Extract posterior means for quick access
        self.mu_samples = self.trace.posterior["mu"].values.flatten()
        self.sigma_samples = self.trace.posterior["sigma"].values.flatten()
    
    def sample_paths(self, years: int, n_samples: int) -> NDArray[np.float64]:
        """Sample future appreciation paths from the posterior predictive.
        
        Args:
            years: Number of years to simulate.
            n_samples: Number of Monte Carlo paths.
            
        Returns:
            Array of shape (n_samples, years) with cumulative appreciation factors.
            E.g., 1.05 means property is worth 105% of initial value.
        """
        # Sample parameters from posterior
        idx = np.random.choice(len(self.mu_samples), size=n_samples)
        mus = self.mu_samples[idx]
        sigmas = self.sigma_samples[idx]
        
        # Generate paths: each sample uses its own posterior parameter draw
        annual_returns = np.zeros((n_samples, years))
        for t in range(years):
            annual_returns[:, t] = np.random.normal(mus, sigmas)
        
        # Convert to cumulative factors (1 + r1) * (1 + r2) * ...
        cumulative = np.cumprod(1 + annual_returns, axis=1)
        
        return cumulative
    
    def summary(self) -> dict:
        """Return summary statistics of the fitted model."""
        return {
            "mu_mean": float(np.mean(self.mu_samples)),
            "mu_std": float(np.std(self.mu_samples)),
            "mu_hdi_95": (
                float(np.percentile(self.mu_samples, 2.5)),
                float(np.percentile(self.mu_samples, 97.5)),
            ),
            "sigma_mean": float(np.mean(self.sigma_samples)),
            "sigma_std": float(np.std(self.sigma_samples)),
        }


def fit_from_hpi_data(hpi_series: NDArray[np.float64]) -> AppreciationModel:
    """Convenience function to fit model from HPI index series.

    Args:
        hpi_series: Time series of house price index values (assumed quarterly).

    Returns:
        Fitted AppreciationModel.
    """
    if len(hpi_series) < 8:
        # Not enough data, use priors only
        return AppreciationModel(historical_returns=None)

    # Take non-overlapping annual observations to ensure independence
    # Use every 4th quarterly observation (e.g., Q1 of each year)
    annual_values = hpi_series[::4]  # Q1, Q5, Q9, ... (indices 0, 4, 8, ...)

    if len(annual_values) < 2:
        return AppreciationModel(historical_returns=None)

    # Calculate year-over-year returns from non-overlapping annual data
    # These observations are now independent
    yoy_returns = (annual_values[1:] / annual_values[:-1]) - 1

    return AppreciationModel(historical_returns=yoy_returns)
