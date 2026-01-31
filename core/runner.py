"""High-level simulation runner with output formatting."""

import json
import sys
from contextlib import redirect_stdout
from io import StringIO
from typing import Any

import numpy as np

from .config import ScenarioConfig


def run_scenario(config: ScenarioConfig, quiet: bool = False) -> dict[str, Any]:
    """Run a simulation scenario with the given configuration.

    Args:
        config: Complete scenario configuration.
        quiet: If True, suppress stdout output from simulation.

    Returns:
        Dictionary containing simulation results.
    """
    from models.simulation import run_simulation

    if quiet:
        # Redirect stdout to suppress print statements
        with redirect_stdout(StringIO()):
            return run_simulation(**config.to_simulation_kwargs())
    else:
        return run_simulation(**config.to_simulation_kwargs())


def format_results(results: dict[str, Any], config: ScenarioConfig | None = None) -> str:
    """Format simulation results as human-readable text.

    Args:
        results: Results dictionary from run_scenario().
        config: Optional config for additional context.

    Returns:
        Formatted text summary.
    """
    stats = results["summary_stats"]
    downside = results.get("downside_metrics", {})

    lines = [
        "=" * 60,
        "BUY vs RENT+INVEST Monte Carlo Simulation",
        "=" * 60,
    ]

    if config:
        lines.extend([
            "",
            f"Property: {config.property_price:,.0f} CZK | "
            f"Down: {config.down_payment:,.0f} CZK | "
            f"Rent: {config.monthly_rent:,.0f} CZK/mo",
            f"USD Holdings: ${config.usd_holdings:,.0f} | "
            f"Horizon: {config.years} years | "
            f"Samples: {config.n_samples}",
        ])

    lines.extend([
        "",
        "--- Results (inflation-adjusted) ---",
        f"Probability that BUYING wins: {stats['buy_wins_prob']:.1%}",
        f"Expected advantage of buying: {stats['expected_advantage_buy']:,.0f} CZK",
        "",
        "--- Final Wealth Distribution (real CZK) ---",
        f"                 5th %ile      Median      95th %ile",
        f"Buy:        {stats['buy_real']['p5']:>12,.0f} {stats['buy_real']['p50']:>12,.0f} {stats['buy_real']['p95']:>12,.0f}",
        f"Rent:       {stats['rent_real']['p5']:>12,.0f} {stats['rent_real']['p50']:>12,.0f} {stats['rent_real']['p95']:>12,.0f}",
    ])

    if downside:
        lines.extend([
            "",
            "--- Downside Risk ---",
            f"P(underwater on mortgage): {downside['p_underwater_final']:.1%}",
            f"P(buy wealth < initial): {downside['p_buy_loss_real']:.1%}",
            f"P(rent wealth < initial): {downside['p_rent_loss_real']:.1%}",
            f"Worst 5% buy outcome: {downside['buy_worst_5pct_real']:,.0f} CZK",
            f"Worst 5% rent outcome: {downside['rent_worst_5pct_real']:,.0f} CZK",
        ])

    # Recommendation
    prob = stats["buy_wins_prob"]
    if prob > 0.6:
        recommendation = "BUY (strong)"
    elif prob > 0.5:
        recommendation = "BUY (marginal)"
    elif prob > 0.4:
        recommendation = "RENT (marginal)"
    else:
        recommendation = "RENT (strong)"

    lines.extend([
        "",
        "=" * 60,
        f"RECOMMENDATION: {recommendation}",
        "=" * 60,
    ])

    return "\n".join(lines)


def results_to_json(results: dict[str, Any], config: ScenarioConfig | None = None) -> str:
    """Convert simulation results to JSON for programmatic use.

    Args:
        results: Results dictionary from run_scenario().
        config: Optional config to include in output.

    Returns:
        JSON string with summary statistics (excludes large arrays).
    """
    output = {
        "summary_stats": results["summary_stats"],
        "downside_metrics": results.get("downside_metrics", {}),
        "buy_wins_prob": results["buy_wins_prob"],
        "buy_wins_by_year": results.get("buy_wins_by_year", []),
        "district_multiplier": results.get("district_multiplier", 1.0),
        "inflation_cumulative_median": results.get("inflation_cumulative_median", 1.0),
    }

    if config:
        output["config"] = config.to_dict()

    # Include model summaries if available
    if "model_summaries" in results:
        output["model_summaries"] = results["model_summaries"]

    return json.dumps(output, indent=2, default=_json_serializer)


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def compare_scenarios(
    config_a: ScenarioConfig,
    config_b: ScenarioConfig,
    label_a: str = "Scenario A",
    label_b: str = "Scenario B",
    quiet: bool = False,
) -> str:
    """Run and compare two scenarios.

    Args:
        config_a: First scenario configuration.
        config_b: Second scenario configuration.
        label_a: Label for first scenario.
        label_b: Label for second scenario.
        quiet: If True, suppress simulation progress output.

    Returns:
        Formatted comparison text.
    """
    if not quiet:
        print(f"Running {label_a}...")
    results_a = run_scenario(config_a, quiet=quiet)

    if not quiet:
        print(f"Running {label_b}...")
    results_b = run_scenario(config_b, quiet=quiet)

    stats_a = results_a["summary_stats"]
    stats_b = results_b["summary_stats"]

    lines = [
        "=" * 70,
        "SCENARIO COMPARISON",
        "=" * 70,
        "",
        f"{'Metric':<30} {label_a:>18} {label_b:>18}",
        "-" * 70,
        f"{'P(Buy Wins)':<30} {stats_a['buy_wins_prob']:>17.1%} {stats_b['buy_wins_prob']:>17.1%}",
        f"{'Buy Median (real)':<30} {stats_a['buy_real']['p50']:>14,.0f} CZK {stats_b['buy_real']['p50']:>14,.0f} CZK",
        f"{'Rent Median (real)':<30} {stats_a['rent_real']['p50']:>14,.0f} CZK {stats_b['rent_real']['p50']:>14,.0f} CZK",
        f"{'Expected Advantage':<30} {stats_a['expected_advantage_buy']:>14,.0f} CZK {stats_b['expected_advantage_buy']:>14,.0f} CZK",
    ]

    # Add downside comparison if available
    down_a = results_a.get("downside_metrics", {})
    down_b = results_b.get("downside_metrics", {})
    if down_a and down_b:
        lines.extend([
            "",
            "Downside Risk:",
            f"{'P(underwater)':<30} {down_a['p_underwater_final']:>17.1%} {down_b['p_underwater_final']:>17.1%}",
            f"{'Worst 5% (buy)':<30} {down_a['buy_worst_5pct_real']:>14,.0f} CZK {down_b['buy_worst_5pct_real']:>14,.0f} CZK",
        ])

    lines.append("=" * 70)

    return "\n".join(lines)


def parameter_sweep(
    base_config: ScenarioConfig,
    param_name: str,
    values: list[float],
    quiet: bool = False,
) -> list[dict[str, Any]]:
    """Run simulations across a range of parameter values.

    Args:
        base_config: Base configuration to modify.
        param_name: Parameter name to vary.
        values: List of values to test.
        quiet: If True, suppress simulation progress output.

    Returns:
        List of dicts with value and key results.
    """
    results = []

    for val in values:
        config_dict = base_config.to_dict()
        config_dict[param_name] = val
        config = ScenarioConfig(**config_dict)

        if not quiet:
            print(f"Running {param_name}={val}...")
        sim_results = run_scenario(config, quiet=quiet)

        stats = sim_results["summary_stats"]
        results.append({
            param_name: val,
            "buy_wins_prob": stats["buy_wins_prob"],
            "buy_median_real": stats["buy_real"]["p50"],
            "rent_median_real": stats["rent_real"]["p50"],
            "expected_advantage": stats["expected_advantage_buy"],
        })

    return results


def format_sweep_results(
    sweep_results: list[dict[str, Any]],
    param_name: str,
) -> str:
    """Format parameter sweep results as a table.

    Args:
        sweep_results: Results from parameter_sweep().
        param_name: Name of the varied parameter.

    Returns:
        Formatted table string.
    """
    lines = [
        "=" * 70,
        f"PARAMETER SWEEP: {param_name}",
        "=" * 70,
        "",
        f"{param_name:>15}  {'P(Buy Wins)':>12}  {'Exp. Advantage':>15}  {'Recommendation':>15}",
        "-" * 70,
    ]

    for r in sweep_results:
        prob = r["buy_wins_prob"]
        if prob > 0.5:
            rec = "BUY"
        else:
            rec = "RENT"

        val = r[param_name]
        if isinstance(val, float) and val > 1000:
            val_str = f"{val:,.0f}"
        elif isinstance(val, float):
            val_str = f"{val:.2%}" if val < 1 else f"{val:.2f}"
        else:
            val_str = str(val)

        lines.append(
            f"{val_str:>15}  {prob:>11.1%}  {r['expected_advantage']:>12,.0f} CZK  {rec:>15}"
        )

    lines.append("=" * 70)

    return "\n".join(lines)
