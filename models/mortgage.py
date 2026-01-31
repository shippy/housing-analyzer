"""Mortgage rate model using PyMC Bayesian inference.

Models Czech mortgage rates as a mean-reverting (Vasicek) process
with Bayesian estimation of parameters.
"""

import numpy as np
import pymc as pm
from numpy.typing import NDArray


class MortgageModel:
    """Bayesian Vasicek model for Czech mortgage rates.
    
    The Vasicek model: dr = κ(θ - r)dt + σdW
    - κ (kappa): speed of mean reversion
    - θ (theta): long-term mean rate
    - σ (sigma): volatility
    """
    
    def __init__(
        self,
        current_rate: float = 0.055,
        historical_rates: NDArray[np.float64] | None = None,
        draws: int = 1000,
        tune: int = 500,
    ):
        """Initialize and fit the model.
        
        Args:
            current_rate: Current mortgage rate (e.g., 0.055 for 5.5%).
            historical_rates: Array of historical rates for fitting.
            draws: Number of posterior draws.
            tune: Number of tuning steps.
        """
        self.current_rate = current_rate
        self.historical_rates = historical_rates
        self.draws = draws
        self.tune = tune
        self.trace = None
        self._fit()
    
    def _fit(self) -> None:
        """Fit the Bayesian Vasicek model."""
        with pm.Model() as model:
            # Priors for Vasicek parameters
            # Long-term mean: expect around 4-5% based on historical Czech rates
            theta = pm.Normal("theta", mu=0.045, sigma=0.015)
            
            # Mean reversion speed: how fast rates return to mean
            # Higher = faster reversion. Typical values 0.1-0.5 per year
            kappa = pm.HalfNormal("kappa", sigma=0.3)
            
            # Volatility of rate changes
            sigma = pm.HalfNormal("sigma", sigma=0.02)
            
            if self.historical_rates is not None and len(self.historical_rates) > 12:
                # Fit to observed rate changes
                # Discretized Vasicek: r(t+1) - r(t) = κ(θ - r(t))Δt + σ√Δt ε
                dt = 1/12  # Monthly data
                r_t = self.historical_rates[:-1]
                r_next = self.historical_rates[1:]
                dr = r_next - r_t
                
                # Expected change under Vasicek
                expected_dr = kappa * (theta - r_t) * dt
                std_dr = sigma * np.sqrt(dt)
                
                obs = pm.Normal(
                    "rate_changes",
                    mu=expected_dr,
                    sigma=std_dr,
                    observed=dr,
                )
            
            self.trace = pm.sample(
                draws=self.draws,
                tune=self.tune,
                cores=1,
                progressbar=False,
                return_inferencedata=True,
            )
        
        self.model = model
        
        # Extract posterior samples
        self.theta_samples = self.trace.posterior["theta"].values.flatten()
        self.kappa_samples = self.trace.posterior["kappa"].values.flatten()
        self.sigma_samples = self.trace.posterior["sigma"].values.flatten()
    
    def sample_paths(self, years: int, n_samples: int, dt: float = 1/12) -> NDArray[np.float64]:
        """Sample future rate paths from posterior predictive.
        
        Args:
            years: Number of years to simulate.
            n_samples: Number of paths.
            dt: Time step (default monthly = 1/12).
            
        Returns:
            Array of shape (n_samples, n_steps) with simulated rates.
        """
        n_steps = int(years / dt)
        
        # Sample parameters from posterior
        idx = np.random.choice(len(self.theta_samples), size=n_samples)
        thetas = self.theta_samples[idx]
        kappas = self.kappa_samples[idx]
        sigmas = self.sigma_samples[idx]
        
        # Simulate Vasicek paths
        rates = np.zeros((n_samples, n_steps + 1))
        rates[:, 0] = self.current_rate
        
        sqrt_dt = np.sqrt(dt)
        for t in range(n_steps):
            dW = np.random.normal(0, 1, n_samples)
            dr = kappas * (thetas - rates[:, t]) * dt + sigmas * sqrt_dt * dW
            rates[:, t + 1] = np.maximum(0.001, rates[:, t] + dr)  # Floor at 0.1%
        
        return rates[:, 1:]  # Exclude initial rate
    
    def sample_paths_annual(self, years: int, n_samples: int) -> NDArray[np.float64]:
        """Sample annual average rates.
        
        Args:
            years: Number of years.
            n_samples: Number of paths.
            
        Returns:
            Array of shape (n_samples, years) with annual average rates.
        """
        monthly = self.sample_paths(years, n_samples, dt=1/12)
        # Reshape to (n_samples, years, 12) and take mean over months
        n_months = monthly.shape[1]
        n_full_years = n_months // 12
        trimmed = monthly[:, :n_full_years * 12]
        annual = trimmed.reshape(n_samples, n_full_years, 12).mean(axis=2)
        
        # Pad if needed
        if annual.shape[1] < years:
            pad = np.tile(annual[:, -1:], (1, years - annual.shape[1]))
            annual = np.hstack([annual, pad])
        
        return annual[:, :years]
    
    def summary(self) -> dict:
        """Return summary statistics of the fitted model."""
        return {
            "theta_mean": float(np.mean(self.theta_samples)),
            "theta_hdi_95": (
                float(np.percentile(self.theta_samples, 2.5)),
                float(np.percentile(self.theta_samples, 97.5)),
            ),
            "kappa_mean": float(np.mean(self.kappa_samples)),
            "sigma_mean": float(np.mean(self.sigma_samples)),
            "current_rate": self.current_rate,
        }
