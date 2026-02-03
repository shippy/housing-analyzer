# Time-Grid Refactor Design

## Goal
Unify time-step aggregation across models so FX annualization, mortgage averaging,
and UI composition use consistent, simulation-derived data without ad-hoc
assumptions.

## Context
Recent fixes removed several calculation errors, but a few remaining issues were
rooted in inconsistent time aggregation and approximate UI reporting. This
design introduces a shared time-grid/aggregation layer used across models and
simulation outputs.

## Proposed Architecture
- Add a small time-grid helper module that converts monthly series to annual
  series with explicit handling of partial years.
- Use this helper in `MortgageModel.sample_paths_annual` and (where applicable)
  in FX annualization so assumptions are centralized.
- Track simulation-derived wealth components (equity, invested cash, USD value,
  savings) so the UI composition chart can use actual values.

## Components
1. **Time aggregation helper**
   - API: `aggregate_to_annual(values, months_per_year=12, mode="mean",
     include_partial=True)`
   - Handles full years via reshape/mean.
   - For partial years, computes mean over remaining months and appends.
   - Validates shape and raises a `ValueError` for invalid inputs.

2. **FX annualization**
   - Make trading-days-per-year explicit and configurable.
   - Ensure daily `dt` and annualization both use the same value.

3. **Mortgage annual averages**
   - Use the helper to avoid truncation/padding artifacts.
   - Preserve requested `years` length while honoring partial-year data.

4. **Simulation components**
   - Track per-year equity, invested cash, USD value, and savings directly in
     `run_simulation`.
   - Return these under a new `components` key for the UI to consume.

## Data Flow
1. Simulation generates monthly/annual series.
2. Aggregation helper produces annual averages where needed.
3. Simulation stores per-year component arrays.
4. UI consumes components directly for composition chart.

## Error Handling
- Validate aggregation inputs and provide actionable errors.
- Validate trading-days-per-year range to avoid nonsensical inputs.

## Testing Strategy
- Unit tests for aggregation helper with known sequences and partial-year cases.
- FX annualization test to verify daily sigma changes with trading-days-per-year.
- Mortgage annual sampling test to verify partial-year handling and output length.
- UI composition test to verify it uses simulation-derived components.

## Non-Goals
- Redesign the simulation model or correlations.
- Change user-facing CLI options beyond exposing explicit configuration where
  needed.
