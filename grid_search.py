#!/usr/bin/env python3
"""Grid search to find conditions where buying is most favorable."""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from core.config import ScenarioConfig
from core.runner import run_scenario


def run_grid_search(
    samples_per_scenario: int = 500,
    seed: int = 42,
) -> list[dict]:
    """Run grid search across key parameters.

    Varies: monthly_rent, years, fx outlook
    Fixed: property_price, down_payment, usd_holdings from defaults
    """
    base_config = ScenarioConfig.from_defaults()

    # Parameter ranges (realistic for Prague)
    monthly_rents = [20_000, 23_000, 26_000, 29_000, 32_000, 35_000]
    years_range = [5, 7, 10, 12, 15, 20]
    fx_scenarios = ["neutral", "bearish"]

    results = []
    total = len(monthly_rents) * len(years_range) * len(fx_scenarios)

    print(f"Running grid search: {total} scenarios")
    print(f"  Rent: {monthly_rents[0]:,} - {monthly_rents[-1]:,} CZK")
    print(f"  Years: {years_range[0]} - {years_range[-1]}")
    print(f"  FX: {fx_scenarios}")
    print(f"  Samples per scenario: {samples_per_scenario}")
    print()
    print("Settings from .env:")
    print(f"  Property: {base_config.property_price:,.0f} CZK")
    print(f"  Down payment: {base_config.down_payment:,.0f} CZK")
    print(f"  Buy-to-let: {base_config.buy_to_let}")
    if base_config.buy_to_let:
        net_yield = base_config.rental_yield * (1 - base_config.btl_expense_rate)
        print(f"    Gross yield: {base_config.rental_yield:.1%}, Expenses: {base_config.btl_expense_rate:.0%}, Net yield: {net_yield:.1%}")
    print()

    for i, (rent, years, fx) in enumerate(itertools.product(monthly_rents, years_range, fx_scenarios)):
        # Build config - inherit all settings from defaults (including buy_to_let, btl_expense_rate, etc.)
        config = ScenarioConfig(
            property_price=base_config.property_price,
            down_payment=base_config.down_payment,
            monthly_rent=rent,
            usd_holdings=base_config.usd_holdings,
            years=years,
            rent_growth_rate=base_config.rent_growth_rate,
            district=base_config.district,
            # Inherit BTL settings from defaults
            buy_to_let=base_config.buy_to_let,
            rental_yield=base_config.rental_yield,
            btl_expense_rate=base_config.btl_expense_rate,
            # Inherit model options
            enable_refinancing=base_config.enable_refinancing,
            enable_tax_benefits=base_config.enable_tax_benefits,
            inflation_adjust=base_config.inflation_adjust,
            enable_correlations=base_config.enable_correlations,
            # Simulation
            n_samples=samples_per_scenario,
            seed=seed + i,  # Different seed for each scenario
        )

        if fx == "bearish":
            config = config.with_fx_bearish()

        # Run simulation (quiet mode)
        print(f"[{i+1}/{total}] rent={rent:,}, years={years}, fx={fx}...", end=" ", flush=True)
        sim_results = run_scenario(config, quiet=True)

        stats = sim_results["summary_stats"]
        downside = sim_results.get("downside_metrics", {})

        result = {
            "monthly_rent": rent,
            "years": years,
            "fx_scenario": fx,
            "buy_wins_prob": stats["buy_wins_prob"],
            "expected_advantage": stats["expected_advantage_buy"],
            "buy_median_real": stats["buy_real"]["p50"],
            "rent_median_real": stats["rent_real"]["p50"],
            "buy_p5_real": stats["buy_real"]["p5"],
            "rent_p5_real": stats["rent_real"]["p5"],
            "p_underwater": downside.get("p_underwater_final", 0),
        }
        results.append(result)
        print(f"P(buy)={result['buy_wins_prob']:.1%}")

    return results


def find_best_scenarios(results: list[dict], top_n: int = 10) -> None:
    """Print the best scenarios for buying."""
    sorted_results = sorted(results, key=lambda x: x["buy_wins_prob"], reverse=True)

    print("\n" + "=" * 70)
    print("TOP SCENARIOS FOR BUYING")
    print("=" * 70)
    print(f"{'Rent':>10} {'Years':>6} {'FX':>10} {'P(Buy Wins)':>12} {'Exp. Adv.':>15}")
    print("-" * 70)

    for r in sorted_results[:top_n]:
        print(f"{r['monthly_rent']:>10,} {r['years']:>6} {r['fx_scenario']:>10} "
              f"{r['buy_wins_prob']:>11.1%} {r['expected_advantage']:>12,.0f} CZK")

    # Also show worst scenarios
    print("\n" + "=" * 70)
    print("WORST SCENARIOS FOR BUYING (best for renting)")
    print("=" * 70)
    print(f"{'Rent':>10} {'Years':>6} {'FX':>10} {'P(Buy Wins)':>12} {'Exp. Adv.':>15}")
    print("-" * 70)

    for r in sorted_results[-top_n:]:
        print(f"{r['monthly_rent']:>10,} {r['years']:>6} {r['fx_scenario']:>10} "
              f"{r['buy_wins_prob']:>11.1%} {r['expected_advantage']:>12,.0f} CZK")


def create_heatmaps(results: list[dict], output_path: str = "grid_search_results.png") -> None:
    """Create heatmaps showing buy probability across parameters."""
    # Separate by FX scenario
    neutral = [r for r in results if r["fx_scenario"] == "neutral"]
    bearish = [r for r in results if r["fx_scenario"] == "bearish"]

    # Get unique values
    rents = sorted(set(r["monthly_rent"] for r in results))
    years = sorted(set(r["years"] for r in results))

    # Create matrices
    def make_matrix(data, key):
        matrix = np.zeros((len(rents), len(years)))
        for r in data:
            i = rents.index(r["monthly_rent"])
            j = years.index(r["years"])
            matrix[i, j] = r[key]
        return matrix

    prob_neutral = make_matrix(neutral, "buy_wins_prob")
    prob_bearish = make_matrix(bearish, "buy_wins_prob")
    adv_neutral = make_matrix(neutral, "expected_advantage") / 1e6  # In millions
    adv_bearish = make_matrix(bearish, "expected_advantage") / 1e6

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Buy vs Rent Analysis: Grid Search Results", fontsize=14, fontweight="bold")

    rent_labels = [f"{r//1000}k" for r in rents]

    # Plot 1: P(Buy Wins) - Neutral FX
    im1 = axes[0, 0].imshow(prob_neutral, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    axes[0, 0].set_title("P(Buy Wins) - FX Neutral")
    axes[0, 0].set_xlabel("Time Horizon (years)")
    axes[0, 0].set_ylabel("Monthly Rent (CZK)")
    axes[0, 0].set_xticks(range(len(years)))
    axes[0, 0].set_xticklabels(years)
    axes[0, 0].set_yticks(range(len(rents)))
    axes[0, 0].set_yticklabels(rent_labels)

    # Add text annotations
    for i in range(len(rents)):
        for j in range(len(years)):
            val = prob_neutral[i, j]
            color = "white" if val < 0.3 or val > 0.7 else "black"
            axes[0, 0].text(j, i, f"{val:.0%}", ha="center", va="center", color=color, fontsize=8)

    plt.colorbar(im1, ax=axes[0, 0], label="Probability")

    # Plot 2: P(Buy Wins) - Bearish FX
    im2 = axes[0, 1].imshow(prob_bearish, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    axes[0, 1].set_title("P(Buy Wins) - FX Bearish (weak USD)")
    axes[0, 1].set_xlabel("Time Horizon (years)")
    axes[0, 1].set_ylabel("Monthly Rent (CZK)")
    axes[0, 1].set_xticks(range(len(years)))
    axes[0, 1].set_xticklabels(years)
    axes[0, 1].set_yticks(range(len(rents)))
    axes[0, 1].set_yticklabels(rent_labels)

    for i in range(len(rents)):
        for j in range(len(years)):
            val = prob_bearish[i, j]
            color = "white" if val < 0.3 or val > 0.7 else "black"
            axes[0, 1].text(j, i, f"{val:.0%}", ha="center", va="center", color=color, fontsize=8)

    plt.colorbar(im2, ax=axes[0, 1], label="Probability")

    # Plot 3: Expected Advantage - Neutral
    vmax_adv = max(abs(adv_neutral).max(), abs(adv_bearish).max())
    im3 = axes[1, 0].imshow(adv_neutral, cmap="RdYlGn", aspect="auto", vmin=-vmax_adv, vmax=vmax_adv)
    axes[1, 0].set_title("Expected Advantage of Buying (M CZK) - FX Neutral")
    axes[1, 0].set_xlabel("Time Horizon (years)")
    axes[1, 0].set_ylabel("Monthly Rent (CZK)")
    axes[1, 0].set_xticks(range(len(years)))
    axes[1, 0].set_xticklabels(years)
    axes[1, 0].set_yticks(range(len(rents)))
    axes[1, 0].set_yticklabels(rent_labels)

    for i in range(len(rents)):
        for j in range(len(years)):
            val = adv_neutral[i, j]
            color = "white" if abs(val) > vmax_adv * 0.5 else "black"
            axes[1, 0].text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=8)

    plt.colorbar(im3, ax=axes[1, 0], label="Million CZK")

    # Plot 4: Expected Advantage - Bearish
    im4 = axes[1, 1].imshow(adv_bearish, cmap="RdYlGn", aspect="auto", vmin=-vmax_adv, vmax=vmax_adv)
    axes[1, 1].set_title("Expected Advantage of Buying (M CZK) - FX Bearish")
    axes[1, 1].set_xlabel("Time Horizon (years)")
    axes[1, 1].set_ylabel("Monthly Rent (CZK)")
    axes[1, 1].set_xticks(range(len(years)))
    axes[1, 1].set_xticklabels(years)
    axes[1, 1].set_yticks(range(len(rents)))
    axes[1, 1].set_yticklabels(rent_labels)

    for i in range(len(rents)):
        for j in range(len(years)):
            val = adv_bearish[i, j]
            color = "white" if abs(val) > vmax_adv * 0.5 else "black"
            axes[1, 1].text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=8)

    plt.colorbar(im4, ax=axes[1, 1], label="Million CZK")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nHeatmap saved to: {output_path}")

    return fig


def create_tradeoff_chart(results: list[dict], output_path: str = "tradeoff_analysis.png") -> None:
    """Create a chart showing risk vs reward tradeoffs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Risk vs Reward Tradeoffs: Buy vs Rent", fontsize=14, fontweight="bold")

    neutral = [r for r in results if r["fx_scenario"] == "neutral"]
    bearish = [r for r in results if r["fx_scenario"] == "bearish"]

    # Plot 1: Expected advantage vs P(buy wins)
    ax1 = axes[0]

    for r in neutral:
        color = plt.cm.viridis(r["years"] / 20)
        size = r["monthly_rent"] / 1000
        ax1.scatter(
            r["expected_advantage"] / 1e6,
            r["buy_wins_prob"] * 100,
            c=[color],
            s=size * 3,
            alpha=0.7,
            marker="o",
            edgecolors="black",
            linewidths=0.5,
        )

    ax1.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50% threshold")
    ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Expected Advantage of Buying (Million CZK)")
    ax1.set_ylabel("P(Buy Wins) %")
    ax1.set_title("FX Neutral: Each point = (rent, years) combination\n(color=years, size=rent)")
    ax1.grid(True, alpha=0.3)

    # Add colorbar for years
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(5, 20))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label("Years")

    # Plot 2: Downside comparison (worst 5% outcomes)
    ax2 = axes[1]

    years_list = sorted(set(r["years"] for r in neutral))
    rents_list = sorted(set(r["monthly_rent"] for r in neutral))

    for years in years_list:
        subset = [r for r in neutral if r["years"] == years]
        subset = sorted(subset, key=lambda x: x["monthly_rent"])

        buy_p5 = [r["buy_p5_real"] / 1e6 for r in subset]
        rent_p5 = [r["rent_p5_real"] / 1e6 for r in subset]
        x = [r["monthly_rent"] / 1000 for r in subset]

        ax2.plot(x, buy_p5, "o-", label=f"Buy {years}yr", alpha=0.7)
        ax2.plot(x, rent_p5, "s--", label=f"Rent {years}yr", alpha=0.5)

    ax2.set_xlabel("Monthly Rent (thousand CZK)")
    ax2.set_ylabel("Worst 5% Outcome (Million CZK, real)")
    ax2.set_title("Downside Risk: 5th Percentile Wealth")
    ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Tradeoff chart saved to: {output_path}")

    return fig


def main():
    """Run the grid search and generate visualizations."""
    # Parse arguments
    samples = 500
    if len(sys.argv) > 1:
        try:
            samples = int(sys.argv[1])
        except ValueError:
            print(f"Usage: python {sys.argv[0]} [samples_per_scenario]")
            sys.exit(1)

    # Run grid search
    results = run_grid_search(samples_per_scenario=samples)

    # Save raw results
    output_json = Path("grid_search_results.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_json}")

    # Find best scenarios
    find_best_scenarios(results)

    # Create visualizations
    create_heatmaps(results)
    create_tradeoff_chart(results)

    print("\nDone!")


if __name__ == "__main__":
    main()
