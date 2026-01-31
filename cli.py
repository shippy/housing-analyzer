#!/usr/bin/env python3
"""Command-line interface for housing analysis simulations.

Usage:
    python cli.py run                          # Run with defaults
    python cli.py run --property-price 20000000
    python cli.py run --fx-bearish             # USD weakness scenario
    python cli.py run --output json            # JSON output
    python cli.py compare --fx-bearish         # Compare neutral vs bearish
    python cli.py sweep --vary monthly_rent --from 20000 --to 60000 --step 5000
"""

import argparse
import sys

from core.config import ScenarioConfig
from core.runner import (
    run_scenario,
    format_results,
    results_to_json,
    compare_scenarios,
    parameter_sweep,
    format_sweep_results,
)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common simulation arguments to a parser."""
    # Property parameters
    parser.add_argument(
        "--property-price",
        type=float,
        help="Property price in CZK (default: from .env)",
    )
    parser.add_argument(
        "--down-payment",
        type=float,
        help="Down payment in CZK (default: from .env)",
    )
    parser.add_argument(
        "--monthly-rent",
        type=float,
        help="Monthly rent in CZK (default: from .env)",
    )
    parser.add_argument(
        "--renovation-cost",
        type=float,
        default=0,
        help="Renovation cost in CZK - additional upfront cost (default: 0)",
    )
    parser.add_argument(
        "--usd-holdings",
        type=float,
        help="USD holdings to invest (default: from .env)",
    )

    # Time parameters
    parser.add_argument(
        "--years",
        type=int,
        help="Time horizon in years (default: from .env)",
    )
    parser.add_argument(
        "--rent-growth",
        type=float,
        help="Annual rent growth rate (default: 0.03)",
    )

    # FX presets
    fx_group = parser.add_mutually_exclusive_group()
    fx_group.add_argument(
        "--fx-bearish",
        action="store_true",
        help="Enable bearish USD outlook (80%% weak dollar probability)",
    )
    fx_group.add_argument(
        "--fx-neutral",
        action="store_true",
        help="Use historical FX drift (default)",
    )

    # Detailed FX controls
    parser.add_argument(
        "--fx-weak-prob",
        type=float,
        help="P(weak dollar regime) for mixture model",
    )
    parser.add_argument(
        "--fx-weak-drift",
        type=float,
        help="Annual drift in weak dollar regime",
    )
    parser.add_argument(
        "--fx-stable-drift",
        type=float,
        help="Annual drift in stable regime",
    )

    # Model options
    parser.add_argument(
        "--district",
        type=str,
        choices=[
            "prague_avg", "prague_1", "prague_2", "prague_3", "prague_4",
            "prague_5", "prague_6", "prague_7", "prague_8", "prague_9", "prague_10",
        ],
        help="Prague district for appreciation adjustment",
    )
    parser.add_argument(
        "--no-refinancing",
        action="store_true",
        help="Disable mortgage refinancing",
    )
    parser.add_argument(
        "--no-tax-benefits",
        action="store_true",
        help="Disable Czech tax benefits",
    )
    parser.add_argument(
        "--no-inflation-adjust",
        action="store_true",
        help="Show nominal (not inflation-adjusted) wealth",
    )
    parser.add_argument(
        "--no-correlations",
        action="store_true",
        help="Disable risk factor correlations",
    )

    # Buy-to-let mode
    parser.add_argument(
        "--buy-to-let",
        action="store_true",
        help="Buy property to rent out while living in cheaper rental",
    )
    parser.add_argument(
        "--rental-yield",
        type=float,
        default=0.025,
        help="Annual rental yield on property (default: 0.025 = 2.5%%)",
    )

    # Simulation parameters
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of Monte Carlo samples (default: 5000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )


def build_config_from_args(args: argparse.Namespace) -> ScenarioConfig:
    """Build a ScenarioConfig from parsed command-line arguments."""
    config = ScenarioConfig.from_defaults()

    # Override with CLI arguments
    if args.property_price is not None:
        config.property_price = args.property_price
    if args.down_payment is not None:
        config.down_payment = args.down_payment
    if args.monthly_rent is not None:
        config.monthly_rent = args.monthly_rent
    if args.renovation_cost:
        config.renovation_cost = args.renovation_cost
    if args.usd_holdings is not None:
        config.usd_holdings = args.usd_holdings
    if args.years is not None:
        config.years = args.years
    if args.rent_growth is not None:
        config.rent_growth_rate = args.rent_growth
    if args.district is not None:
        config.district = args.district

    # Model options
    if args.no_refinancing:
        config.enable_refinancing = False
    if args.no_tax_benefits:
        config.enable_tax_benefits = False
    if args.no_inflation_adjust:
        config.inflation_adjust = False
    if args.no_correlations:
        config.enable_correlations = False

    # FX settings
    if args.fx_bearish:
        config.fx_mixture_enabled = True
        config.fx_weak_dollar_prob = 0.8
        config.fx_weak_dollar_drift = -0.04
        config.fx_stable_drift = 0.01
    elif args.fx_neutral:
        config.fx_mixture_enabled = False

    # Detailed FX overrides
    if args.fx_weak_prob is not None:
        config.fx_mixture_enabled = True
        config.fx_weak_dollar_prob = args.fx_weak_prob
    if args.fx_weak_drift is not None:
        config.fx_mixture_enabled = True
        config.fx_weak_dollar_drift = args.fx_weak_drift
    if args.fx_stable_drift is not None:
        config.fx_mixture_enabled = True
        config.fx_stable_drift = args.fx_stable_drift

    # Buy-to-let mode
    if args.buy_to_let:
        config.buy_to_let = True
    config.rental_yield = args.rental_yield

    # Simulation parameters
    config.n_samples = args.samples
    if args.seed is not None:
        config.seed = args.seed

    return config


def cmd_run(args: argparse.Namespace) -> int:
    """Run a single scenario."""
    config = build_config_from_args(args)

    if args.output != "json":
        print(config.summary())
        print()

    # Suppress stdout in JSON mode for clean output
    results = run_scenario(config, quiet=(args.output == "json"))

    if args.output == "json":
        print(results_to_json(results, config))
    else:
        print(format_results(results, config))

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare two scenarios (neutral vs bearish FX by default)."""
    base_config = build_config_from_args(args)

    # Create two scenarios: neutral and bearish
    config_neutral = base_config.with_fx_neutral()
    config_bearish = base_config.with_fx_bearish()

    comparison = compare_scenarios(
        config_neutral,
        config_bearish,
        label_a="FX Neutral",
        label_b="FX Bearish",
        quiet=(args.output == "json"),
    )

    print(comparison)
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Run a parameter sweep."""
    base_config = build_config_from_args(args)

    # Build value range
    values = []
    current = args.sweep_from
    while current <= args.sweep_to:
        values.append(current)
        current += args.step

    if not values:
        print(f"Error: No values in range [{args.sweep_from}, {args.sweep_to}]", file=sys.stderr)
        return 1

    quiet = args.output == "json"
    if not quiet:
        print(f"Running sweep: {args.vary} from {args.sweep_from} to {args.sweep_to} (step {args.step})")
        print(f"Total scenarios: {len(values)}")
        print()

    results = parameter_sweep(base_config, args.vary, values, quiet=quiet)

    if args.output == "json":
        import json
        print(json.dumps(results, indent=2))
    else:
        print()
        print(format_sweep_results(results, args.vary))

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prague Housing Analyzer CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a single simulation scenario")
    add_common_args(run_parser)
    run_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare neutral vs bearish FX scenarios")
    add_common_args(compare_parser)
    compare_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    # Sweep command
    sweep_parser = subparsers.add_parser("sweep", help="Run parameter sweep")
    add_common_args(sweep_parser)
    sweep_parser.add_argument(
        "--vary",
        type=str,
        required=True,
        help="Parameter to vary (e.g., monthly_rent, property_price)",
    )
    sweep_parser.add_argument(
        "--from",
        dest="sweep_from",
        type=float,
        required=True,
        help="Start value for sweep",
    )
    sweep_parser.add_argument(
        "--to",
        dest="sweep_to",
        type=float,
        required=True,
        help="End value for sweep",
    )
    sweep_parser.add_argument(
        "--step",
        type=float,
        required=True,
        help="Step size for sweep",
    )
    sweep_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "compare":
        return cmd_compare(args)
    elif args.command == "sweep":
        return cmd_sweep(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
