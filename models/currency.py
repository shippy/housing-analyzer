"""Currency (USD/CZK) model using PyMC Bayesian inference.

Models exchange rate as Geometric Brownian Motion with
Bayesian estimation of drift and volatility.

Supports mixture model for asymmetric FX views (e.g., bearish USD).
"""

import numpy as np
import pymc as pm
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass
class FXMixtureConfig:
    """Configuration for FX regime mixture model.

    Attributes:
        weak_dollar_prob: Probability of weak dollar regime (0-1).
        weak_dollar_drift: Annual drift in weak regime (e.g., -0.04 = -4%).
        stable_drift: Annual drift in stable/strong regime (e.g., 0.01 = +1%).
        regime_persistence_years: Expected duration of each regime in years.
                                  Lower values = more frequent regime switches.
        enabled: Whether to use mixture model (False = standard GBM from data).
    """
    weak_dollar_prob: float = 0.6
    weak_dollar_drift: float = -0.04
    stable_drift: float = 0.01
    regime_persistence_years: float = 3.0  # Average regime lasts ~3 years
    enabled: bool = False


class CurrencyModel:
    """Bayesian GBM model for USD/CZK exchange rate.
    
    GBM model: dS = μS dt + σS dW
    - μ (mu): drift (expected annual return)
    - σ (sigma): volatility
    
    For FX, drift is often close to zero (no arbitrage),
    but we estimate it from data.
    
    Optionally supports a mixture model for incorporating views
    about dollar strength/weakness.
    """
    
    def __init__(
        self,
        current_rate: float = 24.0,
        historical_rates: NDArray[np.float64] | None = None,
        draws: int = 1000,
        tune: int = 500,
        mixture_config: FXMixtureConfig | None = None,
    ):
        """Initialize and fit the model.
        
        Args:
            current_rate: Current USD/CZK rate.
            historical_rates: Array of historical daily rates for fitting.
            draws: Number of posterior draws.
            tune: Number of tuning steps.
            mixture_config: Optional config for regime mixture model.
        """
        self.current_rate = current_rate
        self.historical_rates = historical_rates
        self.draws = draws
        self.tune = tune
        self.trace = None
        self.mixture_config = mixture_config or FXMixtureConfig()
        self._fit()
    
    def _fit(self) -> None:
        """Fit the Bayesian GBM model."""
        with pm.Model() as model:
            # Priors
            # Drift: centered around 0 for FX (interest rate parity)
            # But CZK has been strengthening, so allow negative drift
            mu = pm.Normal("mu", mu=0.0, sigma=0.05)
            
            # Volatility: FX typically 8-15% annual
            sigma = pm.HalfNormal("sigma", sigma=0.15)
            
            if self.historical_rates is not None and len(self.historical_rates) > 100:
                # Calculate log returns (daily)
                log_returns = np.diff(np.log(self.historical_rates))
                
                # Remove any NaN or inf
                log_returns = log_returns[np.isfinite(log_returns)]
                
                if len(log_returns) > 50:
                    # Annualize: daily returns, ~252 trading days
                    # Daily drift = annual_mu / 252
                    # Daily vol = annual_sigma / sqrt(252)
                    daily_mu = mu / 252
                    daily_sigma = sigma / np.sqrt(252)
                    
                    obs = pm.Normal(
                        "log_returns",
                        mu=daily_mu,
                        sigma=daily_sigma,
                        observed=log_returns,
                    )
            
            self.trace = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=4,
                cores=1,
                progressbar=False,
                return_inferencedata=True,
            )
        
        self.model = model
        
        # Extract posterior samples
        self.mu_samples = self.trace.posterior["mu"].values.flatten()
        self.sigma_samples = self.trace.posterior["sigma"].values.flatten()
    
    def sample_paths(
        self,
        years: int,
        n_samples: int,
        dt: float = 1/252,
    ) -> NDArray[np.float64]:
        """Sample future exchange rate paths from posterior predictive.
        
        Args:
            years: Number of years to simulate.
            n_samples: Number of paths.
            dt: Time step (default daily = 1/252).
            
        Returns:
            Array of shape (n_samples, n_steps) with simulated rates.
        """
        n_steps = int(years / dt)
        
        # Sample volatility from posterior
        idx = np.random.choice(len(self.sigma_samples), size=n_samples)
        sigmas = self.sigma_samples[idx]

        # GBM simulation
        # S(t+dt) = S(t) * exp((μ - σ²/2)dt + σ√dt * Z)
        rates = np.zeros((n_samples, n_steps + 1))
        rates[:, 0] = self.current_rate

        sqrt_dt = np.sqrt(dt)

        if self.mixture_config.enabled:
            # Regime-switching mixture model
            # Calculate switching probability per step from persistence parameter
            # If regime lasts ~P years on average, switch prob per step = dt/P
            switch_prob = dt / self.mixture_config.regime_persistence_years

            # Initialize regimes based on stationary distribution
            in_weak_regime = np.random.binomial(
                1, self.mixture_config.weak_dollar_prob, n_samples
            ).astype(bool)

            for t in range(n_steps):
                # Determine drift based on current regime
                mus = np.where(
                    in_weak_regime,
                    self.mixture_config.weak_dollar_drift,
                    self.mixture_config.stable_drift,
                )

                Z = np.random.normal(0, 1, n_samples)
                drift = (mus - 0.5 * sigmas**2) * dt
                diffusion = sigmas * sqrt_dt * Z
                rates[:, t + 1] = rates[:, t] * np.exp(drift + diffusion)

                # Regime switching: with prob switch_prob, consider switching
                switches = np.random.binomial(1, switch_prob, n_samples).astype(bool)
                # When switching occurs, new regime drawn from stationary distribution
                new_regime_weak = np.random.binomial(
                    1, self.mixture_config.weak_dollar_prob, n_samples
                ).astype(bool)
                in_weak_regime = np.where(switches, new_regime_weak, in_weak_regime)
        else:
            # Standard GBM: constant drift from posterior
            mus = self.mu_samples[idx]

            for t in range(n_steps):
                Z = np.random.normal(0, 1, n_samples)
                drift = (mus - 0.5 * sigmas**2) * dt
                diffusion = sigmas * sqrt_dt * Z
                rates[:, t + 1] = rates[:, t] * np.exp(drift + diffusion)

        return rates[:, 1:]
    
    def sample_paths_annual(self, years: int, n_samples: int) -> NDArray[np.float64]:
        """Sample year-end exchange rates.
        
        Args:
            years: Number of years.
            n_samples: Number of paths.
            
        Returns:
            Array of shape (n_samples, years) with year-end rates.
        """
        daily = self.sample_paths(years, n_samples, dt=1/252)
        # Take every 252nd value (year-end)
        indices = np.arange(252, daily.shape[1] + 1, 252) - 1
        indices = indices[:years]
        
        if len(indices) < years:
            # Pad with last value
            annual = daily[:, indices] if len(indices) > 0 else np.full((n_samples, years), self.current_rate)
            if annual.shape[1] < years:
                pad_val = annual[:, -1:] if annual.shape[1] > 0 else np.full((n_samples, 1), self.current_rate)
                pad = np.tile(pad_val, (1, years - annual.shape[1]))
                annual = np.hstack([annual, pad])
            return annual
        
        return daily[:, indices]
    
    def summary(self) -> dict:
        """Return summary statistics of the fitted model."""
        result = {
            "mu_mean": float(np.mean(self.mu_samples)),
            "mu_std": float(np.std(self.mu_samples)),
            "mu_hdi_95": (
                float(np.percentile(self.mu_samples, 2.5)),
                float(np.percentile(self.mu_samples, 97.5)),
            ),
            "sigma_mean": float(np.mean(self.sigma_samples)),
            "sigma_std": float(np.std(self.sigma_samples)),
            "sigma_hdi_95": (
                float(np.percentile(self.sigma_samples, 2.5)),
                float(np.percentile(self.sigma_samples, 97.5)),
            ),
            "current_rate": self.current_rate,
            "mixture_enabled": self.mixture_config.enabled,
        }
        
        if self.mixture_config.enabled:
            result.update({
                "weak_dollar_prob": self.mixture_config.weak_dollar_prob,
                "weak_dollar_drift": self.mixture_config.weak_dollar_drift,
                "stable_drift": self.mixture_config.stable_drift,
            })
        
        return result
