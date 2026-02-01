"""Prague Housing Analyzer - Buy vs Rent+Invest Dashboard."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="Prague Housing Analyzer")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from settings import defaults
    return defaults, go, make_subplots, mo, np


@app.cell
def _(mo):
    mo.md(
        """
        # 🏠 Prague Housing Analyzer
        
        **Buy vs Rent+Invest Analysis with Uncertainty Quantification**
        
        This tool uses Bayesian modeling to compare:
        - **Buying** a property (with mortgage, liquidating investments)
        - **Renting** an equivalent property and keeping money invested
        
        All projections include uncertainty from:
        - Real estate appreciation volatility
        - Interest rate changes
        - Currency fluctuations (USD/CZK)
        - Investment return variability
        """
    )
    return ()


@app.cell
def _(mo):
    mo.md("## 📊 Scenario Configuration")
    return ()


@app.cell
def _(mo):
    # Scenario presets as reference
    preset_info = mo.accordion({
        "📋 Scenario Presets (reference)": mo.md("""
| Preset | Property | Down Payment | Rent | Years | FX | Notes |
|--------|----------|--------------|------|-------|-----|-------|
| **Conservative** | 25M | 5M (20%) | 30k | 10 | Bearish | Assumes weak USD, shorter horizon |
| **Optimistic** | 25M | 7.5M (30%) | 25k | 15 | Neutral | Higher down payment, longer horizon |
| **First-time Buyer** | 15M | 3M (20%) | 25k | 10 | Neutral | Smaller property, typical starter |
| **Buy-to-Let** | 25M | 5M | 30k | 15 | Neutral | Enable BTL mode, 2.5% yield |

*Adjust the sliders below to match your scenario.*
""")
    })
    preset_info
    return ()


@app.cell
def _(defaults, mo):
    # Property inputs - defaults loaded from .env via settings.py
    property_price = mo.ui.slider(
        start=10_000_000,
        stop=50_000_000,
        step=1_000_000,
        value=defaults.property_price,
        label="Property Price (CZK)",
        show_value=True,
    )

    down_payment = mo.ui.slider(
        start=1_000_000,
        stop=25_000_000,
        step=500_000,
        value=defaults.down_payment,
        label="Down Payment (CZK)",
        show_value=True,
    )

    usd_holdings = mo.ui.slider(
        start=0,
        stop=500_000,
        step=10_000,
        value=defaults.usd_holdings,
        label="USD Holdings to Liquidate ($)",
        show_value=True,
    )

    monthly_rent = mo.ui.slider(
        start=15_000,
        stop=60_000,
        step=1_000,
        value=defaults.monthly_rent,
        label="Monthly Rent (CZK)",
        show_value=True,
    )

    renovation_cost = mo.ui.slider(
        start=0,
        stop=5_000_000,
        step=100_000,
        value=0,
        label="Renovation Cost (CZK)",
        show_value=True,
    )

    years = mo.ui.slider(
        start=5,
        stop=30,
        step=1,
        value=defaults.years,
        label="Time Horizon (years)",
        show_value=True,
    )

    rent_inflation = mo.ui.slider(
        start=0.01,
        stop=0.10,
        step=0.01,
        value=defaults.rent_inflation,
        label="Annual Rent Inflation",
        show_value=True,
    )
    
    # District selector - map internal values to display names
    _district_options = {
        "Prague Average": "prague_avg",
        "Prague 1 (Center)": "prague_1",
        "Prague 2 (Vinohrady)": "prague_2",
        "Prague 3 (Žižkov)": "prague_3",
        "Prague 4 (Nusle/Podolí)": "prague_4",
        "Prague 5 (Smíchov)": "prague_5",
        "Prague 6 (Dejvice)": "prague_6",
        "Prague 7 (Holešovice)": "prague_7",
        "Prague 8 (Karlín)": "prague_8",
        "Prague 9 (Vysočany)": "prague_9",
        "Prague 10 (Vršovice)": "prague_10",
    }
    # Find display name matching the default internal value
    _default_district_display = next(
        (k for k, v in _district_options.items() if v == defaults.district),
        "Prague Average"
    )
    district = mo.ui.dropdown(
        options=_district_options,
        value=_default_district_display,
        label="District",
    )
    
    # Model options
    enable_refinancing = mo.ui.checkbox(value=True, label="Enable mortgage refinancing (every 5 years)")
    enable_tax_benefits = mo.ui.checkbox(value=True, label="Include Czech tax benefits")
    inflation_adjust = mo.ui.checkbox(value=True, label="Show inflation-adjusted (real) wealth")
    enable_correlations = mo.ui.checkbox(value=True, label="Correlate risk factors (inflation/stocks/FX/property)")

    # Buy-to-let mode
    buy_to_let = mo.ui.checkbox(value=defaults.buy_to_let, label="Buy-to-let mode (rent out property, live in smaller rental)")
    rental_yield = mo.ui.slider(
        start=0.015,
        stop=0.05,
        step=0.005,
        value=defaults.rental_yield,
        label="Gross Rental Yield",
        show_value=True,
    )
    btl_expense_rate = mo.ui.slider(
        start=0.10,
        stop=0.50,
        step=0.05,
        value=defaults.btl_expense_rate,
        label="Landlord Expenses (%)",
        show_value=True,
    )

    # Use vstack with individual rows for clean alignment
    mo.vstack([
        mo.hstack([
            mo.vstack([property_price, usd_holdings, years], align="stretch"),
            mo.vstack([down_payment, monthly_rent, rent_inflation], align="stretch"),
            mo.vstack([renovation_cost, district], align="stretch"),
        ], widths=[1, 1, 1], gap=2),
        mo.md("**Model Options:**"),
        mo.hstack([enable_refinancing, enable_tax_benefits], gap=1),
        mo.hstack([inflation_adjust, enable_correlations], gap=1),
    ])
    return btl_expense_rate, buy_to_let, district, down_payment, enable_correlations, enable_refinancing, enable_tax_benefits, inflation_adjust, monthly_rent, property_price, renovation_cost, rent_inflation, rental_yield, usd_holdings, years


@app.cell
def _(btl_expense_rate, buy_to_let, mo, property_price, rental_yield):
    # Show BTL controls conditionally with live net yield
    if buy_to_let.value:
        net_yield = rental_yield.value * (1 - btl_expense_rate.value)
        net_monthly = property_price.value * net_yield / 12
        btl_display = mo.vstack([
            mo.md("**Buy-to-Let Mode:**"),
            buy_to_let,
            mo.hstack([rental_yield, btl_expense_rate], gap=1),
            mo.callout(
                mo.md(f"**Net yield: {net_yield:.1%}** → {net_monthly:,.0f} CZK/month after expenses"),
                kind="info"
            ),
        ])
    else:
        btl_display = mo.vstack([
            mo.md("**Buy-to-Let Mode:**"),
            buy_to_let,
            mo.md("*Enable to configure rental yield and landlord expenses*"),
        ])
    btl_display
    return ()


@app.cell
def _(down_payment, mo, property_price, renovation_cost, usd_holdings):
    # Show funding warning if down payment exceeds USD holdings
    # Assume ~20.5 CZK/USD as approximate current rate
    fx_rate_approx = 20.5
    closing_costs_rate = 0.04

    initial_usd_czk = usd_holdings.value * fx_rate_approx
    total_upfront = down_payment.value + (property_price.value * closing_costs_rate) + renovation_cost.value
    shortfall = total_upfront - initial_usd_czk

    if shortfall > 0:
        funding_warning = mo.callout(
            mo.md(f"""
**Funding note:** Down payment + costs exceed USD holdings by **{shortfall:,.0f} CZK**

The model assumes you have additional CZK savings for this. These funds are *not* counted
in the rent scenario (they wouldn't be invested in stocks). If you would invest this CZK
in stocks when renting, the comparison is asymmetric.
"""),
            kind="warn"
        )
    else:
        funding_warning = mo.md("")  # Empty when no shortfall

    funding_warning
    return ()


@app.cell
def _(mo):
    mo.md(
        """
        ### 💱 FX Outlook (USD/CZK)

        Enable the mixture model to incorporate your view on dollar strength.
        Each simulation path is assigned to either a "weak dollar" or "stable" regime.
        """
    )
    return ()


@app.cell
def _(defaults, mo):
    # FX mixture model controls
    fx_mixture_enabled = mo.ui.checkbox(
        value=defaults.fx_mixture_enabled,
        label="Enable FX mixture model (override historical drift)",
    )

    fx_weak_dollar_prob = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=defaults.fx_weak_dollar_prob,
        label="P(Weak Dollar Regime)",
        show_value=True,
    )

    fx_weak_dollar_drift = mo.ui.slider(
        start=-0.10,
        stop=0.0,
        step=0.01,
        value=defaults.fx_weak_dollar_drift,
        label="Weak Dollar Annual Drift",
        show_value=True,
    )

    fx_stable_drift = mo.ui.slider(
        start=-0.02,
        stop=0.05,
        step=0.01,
        value=defaults.fx_stable_drift,
        label="Stable/Strong Dollar Annual Drift",
        show_value=True,
    )
    
    mo.vstack([
        fx_mixture_enabled,
        mo.hstack([
            mo.vstack([fx_weak_dollar_prob, fx_stable_drift], align="stretch"),
            mo.vstack([fx_weak_dollar_drift, mo.md("")], align="stretch"),  # Empty placeholder for alignment
        ], widths=[1, 1], gap=2),
    ])
    return fx_mixture_enabled, fx_stable_drift, fx_weak_dollar_drift, fx_weak_dollar_prob


@app.cell
def _(fx_mixture_enabled, fx_stable_drift, fx_weak_dollar_drift, fx_weak_dollar_prob, mo):
    # Show expected drift summary
    if fx_mixture_enabled.value:
        _expected_drift = (fx_weak_dollar_prob.value * fx_weak_dollar_drift.value + 
                          (1 - fx_weak_dollar_prob.value) * fx_stable_drift.value)
        _fx_summary = mo.callout(
            mo.md(f"**Expected FX drift:** {_expected_drift:.1%}/year — *Negative = CZK strengthens*"),
            kind="info",
        )
    else:
        _fx_summary = mo.md("*Mixture model disabled — using historical FX drift from data*")
    _fx_summary
    return ()


@app.cell
def _(mo):
    mo.md("## 🎲 Run Simulation")
    return ()


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run Monte Carlo Simulation (5000 samples)")
    run_button
    return (run_button,)


@app.cell
def _(btl_expense_rate, buy_to_let, district, down_payment, enable_correlations, enable_refinancing, enable_tax_benefits, fx_mixture_enabled, fx_stable_drift, fx_weak_dollar_drift, fx_weak_dollar_prob, inflation_adjust, mo, monthly_rent, np, property_price, renovation_cost, rent_inflation, rental_yield, run_button, usd_holdings, years):
    # Only run when button is clicked
    mo.stop(not run_button.value)

    # Import simulation (uses same run_simulation as core.runner)
    try:
        from models.simulation import run_simulation

        results = run_simulation(
            property_price=property_price.value,
            down_payment=down_payment.value,
            usd_holdings=usd_holdings.value,
            monthly_rent=monthly_rent.value,
            years=years.value,
            renovation_cost=renovation_cost.value,
            n_samples=5000,
            rent_growth_rate=rent_inflation.value,
            # FX mixture model
            fx_mixture_enabled=fx_mixture_enabled.value,
            fx_weak_dollar_prob=fx_weak_dollar_prob.value,
            fx_weak_dollar_drift=fx_weak_dollar_drift.value,
            fx_stable_drift=fx_stable_drift.value,
            # Model options
            district=district.value,
            enable_refinancing=enable_refinancing.value,
            enable_tax_benefits=enable_tax_benefits.value,
            inflation_adjust=inflation_adjust.value,
            enable_correlations=enable_correlations.value,
            # Buy-to-let mode
            buy_to_let=buy_to_let.value,
            rental_yield=rental_yield.value,
            btl_expense_rate=btl_expense_rate.value,
        )
    except ImportError:
        # Fallback for testing UI before models are ready
        results = {
            "buy_wealth": np.random.lognormal(17, 0.3, 5000),
            "rent_wealth": np.random.lognormal(17, 0.4, 5000),
            "buy_wins_prob": 0.55,
            "summary_stats": {
                "buy": {"p5": 18_000_000, "p50": 25_000_000, "p95": 35_000_000},
                "rent": {"p5": 15_000_000, "p50": 23_000_000, "p95": 38_000_000},
            }
        }
    return (results,)


@app.cell
def _(mo, results):
    prob = results["buy_wins_prob"]
    stats = results["summary_stats"]

    # Use inflation-adjusted (real) wealth to match the recommendation logic
    buy_stats = stats.get("buy_real", stats.get("buy", {}))
    rent_stats = stats.get("rent_real", stats.get("rent", {}))

    # Handle both old mock format and new nested format
    buy_median = buy_stats.get("p50", stats.get("buy_median", 0))
    rent_median = rent_stats.get("p50", stats.get("rent_median", 0))
    buy_p5 = buy_stats.get("p5", stats.get("buy_p10", 0))
    rent_p5 = rent_stats.get("p5", stats.get("rent_p10", 0))
    buy_p95 = buy_stats.get("p95", stats.get("buy_p90", 0))
    rent_p95 = rent_stats.get("p95", stats.get("rent_p90", 0))

    # Determine verdict and confidence
    if prob > 0.6:
        verdict = "BUY"
        verdict_icon = "🏠"
        callout_kind = "success"
        confidence = "High confidence"
    elif prob > 0.5:
        verdict = "BUY"
        verdict_icon = "🏠"
        callout_kind = "warn"
        confidence = "Marginal"
    elif prob > 0.4:
        verdict = "RENT + INVEST"
        verdict_icon = "📈"
        callout_kind = "warn"
        confidence = "Marginal"
    else:
        verdict = "RENT + INVEST"
        verdict_icon = "📈"
        callout_kind = "info"
        confidence = "High confidence"

    expected_advantage = stats.get("expected_advantage_buy", 0)
    adv_sign = "+" if expected_advantage > 0 else ""

    mo.md("## 🎯 Result Summary")
    return buy_median, buy_p5, buy_p95, buy_stats, callout_kind, confidence, expected_advantage, adv_sign, prob, rent_median, rent_p5, rent_p95, rent_stats, stats, verdict, verdict_icon


@app.cell
def _(adv_sign, buy_median, buy_p5, buy_p95, callout_kind, confidence, expected_advantage, mo, prob, rent_median, rent_p5, rent_p95, verdict, verdict_icon):
    # Prominent verdict callout
    verdict_callout = mo.callout(
        mo.md(f"""
### {verdict_icon} Recommendation: **{verdict}**

**P(Buy Wins): {prob:.1%}** · {confidence} · Expected advantage: {adv_sign}{expected_advantage:,.0f} CZK
"""),
        kind=callout_kind,
    )

    results_table = mo.md(f"""
| Metric | Buy | Rent+Invest | Difference |
|--------|-----|-------------|------------|
| **Median Wealth** | {buy_median:,.0f} | {rent_median:,.0f} | {buy_median - rent_median:+,.0f} |
| 5th Pctl (downside) | {buy_p5:,.0f} | {rent_p5:,.0f} | {buy_p5 - rent_p5:+,.0f} |
| 95th Pctl (upside) | {buy_p95:,.0f} | {rent_p95:,.0f} | {buy_p95 - rent_p95:+,.0f} |

*All values in CZK, inflation-adjusted*
""")

    mo.vstack([verdict_callout, results_table])
    return results_table, verdict_callout


@app.cell
def _(go, inflation_adjust, make_subplots, np, results):
    # Use inflation-adjusted values if enabled (consistent with recommendation)
    _buy_wealth = results.get("buy_wealth_real", results["buy_wealth"]) if inflation_adjust.value else results["buy_wealth"]
    _rent_wealth = results.get("rent_wealth_real", results["rent_wealth"]) if inflation_adjust.value else results["rent_wealth"]
    _suffix = " (real)" if inflation_adjust.value else " (nominal)"

    # Wealth distribution comparison
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Final Wealth Distribution{_suffix}", "Buy vs Rent Scatter"),
        horizontal_spacing=0.1,
    )

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=_buy_wealth / 1e6,
            name="Buy",
            opacity=0.7,
            marker_color="green",
            nbinsx=50,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=_rent_wealth / 1e6,
            name="Rent+Invest",
            opacity=0.7,
            marker_color="blue",
            nbinsx=50,
        ),
        row=1, col=1,
    )

    # Scatter (subsample for performance)
    idx = np.random.choice(len(_buy_wealth), size=min(500, len(_buy_wealth)), replace=False)
    fig.add_trace(
        go.Scatter(
            x=_buy_wealth[idx] / 1e6,
            y=_rent_wealth[idx] / 1e6,
            mode="markers",
            marker=dict(size=4, opacity=0.5),
            name="Scenarios",
            showlegend=False,
        ),
        row=1, col=2,
    )
    
    # Add diagonal line (break-even)
    max_val = max(_buy_wealth.max(), _rent_wealth.max()) / 1e6
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Break-even",
            showlegend=False,
        ),
        row=1, col=2,
    )
    
    fig.update_layout(
        height=400,
        barmode="overlay",
        title_text="Monte Carlo Simulation Results",
    )
    fig.update_xaxes(title_text="Final Wealth (M CZK)", row=1, col=1)
    fig.update_xaxes(title_text="Buy Wealth (M CZK)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Rent Wealth (M CZK)", row=1, col=2)
    
    fig
    return fig, idx, max_val


@app.cell
def _(mo):
    mo.md("## 📈 Breakeven Timeline & Wealth Paths")
    return ()


@app.cell
def _(go, np, results, years):
    """Breakeven timeline - when does buying start winning?"""
    _buy_wins_by_year = results.get("buy_wins_by_year", [])
    
    if not _buy_wins_by_year:
        breakeven_fig = go.Figure()
        breakeven_fig.add_annotation(text="Run simulation to see breakeven analysis", showarrow=False)
    else:
        _years_arr = np.arange(1, len(_buy_wins_by_year) + 1)
        
        breakeven_fig = go.Figure()
        
        breakeven_fig.add_trace(go.Scatter(
            x=_years_arr,
            y=[p * 100 for p in _buy_wins_by_year],
            mode="lines+markers",
            name="P(Buy Wins)",
            line=dict(color="green", width=2),
            marker=dict(size=8),
        ))
        
        # Add 50% reference line
        breakeven_fig.add_hline(y=50, line_dash="dash", line_color="gray",
                               annotation_text="50% (indifferent)")
        
        breakeven_fig.update_layout(
            title="Breakeven Timeline: When Does Buying Start Winning?",
            xaxis_title="Year",
            yaxis_title="P(Buy Wins) %",
            yaxis=dict(range=[0, 100]),
            height=350,
        )
    
    breakeven_fig
    return (breakeven_fig,)


@app.cell
def _(go, np, results):
    """Sample wealth paths over time"""
    _buy_yearly = results.get("buy_wealth_yearly_real")
    _rent_yearly = results.get("rent_wealth_yearly_real")
    
    if _buy_yearly is None or _rent_yearly is None:
        paths_fig = go.Figure()
        paths_fig.add_annotation(text="Run simulation to see wealth paths", showarrow=False)
    else:
        _n_samples = min(50, _buy_yearly.shape[0])
        _idx = np.random.choice(_buy_yearly.shape[0], size=_n_samples, replace=False)
        _years_arr = np.arange(1, _buy_yearly.shape[1] + 1)
        
        paths_fig = go.Figure()
        
        # Plot individual paths (semi-transparent)
        for i in _idx[:30]:
            paths_fig.add_trace(go.Scatter(
                x=_years_arr, y=_buy_yearly[i] / 1e6,
                mode="lines", line=dict(color="green", width=0.5),
                opacity=0.2, showlegend=False,
            ))
            paths_fig.add_trace(go.Scatter(
                x=_years_arr, y=_rent_yearly[i] / 1e6,
                mode="lines", line=dict(color="blue", width=0.5),
                opacity=0.2, showlegend=False,
            ))
        
        # Add median paths
        paths_fig.add_trace(go.Scatter(
            x=_years_arr, y=np.median(_buy_yearly, axis=0) / 1e6,
            mode="lines", name="Buy (median)",
            line=dict(color="darkgreen", width=3),
        ))
        paths_fig.add_trace(go.Scatter(
            x=_years_arr, y=np.median(_rent_yearly, axis=0) / 1e6,
            mode="lines", name="Rent (median)",
            line=dict(color="darkblue", width=3),
        ))
        
        # Add P5/P95 bands
        paths_fig.add_trace(go.Scatter(
            x=_years_arr, y=np.percentile(_buy_yearly, 95, axis=0) / 1e6,
            mode="lines", line=dict(width=0), showlegend=False,
        ))
        paths_fig.add_trace(go.Scatter(
            x=_years_arr, y=np.percentile(_buy_yearly, 5, axis=0) / 1e6,
            mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(0,128,0,0.1)", name="Buy 90% CI",
        ))
        
        paths_fig.update_layout(
            title="Wealth Paths Over Time (Inflation-Adjusted)",
            xaxis_title="Year",
            yaxis_title="Wealth (M CZK, real)",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
    
    paths_fig
    return (paths_fig,)


@app.cell
def _(mo):
    mo.md("## ⚠️ Downside Risk Analysis")
    return ()


@app.cell
def _(mo, results):
    """Downside risk metrics display"""
    _downside = results.get("downside_metrics", {})
    
    if not _downside:
        _risk_output = mo.md("*Run simulation to see risk metrics*")
    else:
        _p_underwater = _downside.get("p_underwater_final", 0)
        _p_buy_loss = _downside.get("p_buy_loss_real", 0)
        _p_rent_loss = _downside.get("p_rent_loss_real", 0)
        _buy_worst = _downside.get("buy_worst_5pct_real", 0)
        _rent_worst = _downside.get("rent_worst_5pct_real", 0)
        
        _risk_output = mo.md(
            f"""
            ### Risk Metrics (inflation-adjusted)
            
            | Metric | Buy | Rent+Invest |
            |--------|-----|-------------|
            | P(final wealth < initial capital) | {_p_buy_loss:.1%} | {_p_rent_loss:.1%} |
            | Worst 5% outcome | {_buy_worst:,.0f} CZK | {_rent_worst:,.0f} CZK |
            
            **Mortgage-specific risks:**
            - P(underwater at end of horizon): **{_p_underwater:.1%}**
            
            *"Underwater" = property value < remaining mortgage balance*
            """
        )
    _risk_output
    return ()


@app.cell
def _(go, results):
    """Underwater probability over time"""
    _downside = results.get("downside_metrics", {})
    _p_underwater_by_year = _downside.get("p_underwater_by_year", [])
    
    if not _p_underwater_by_year:
        underwater_fig = go.Figure()
        underwater_fig.add_annotation(text="Run simulation to see underwater risk", showarrow=False)
    else:
        _years_arr = list(range(1, len(_p_underwater_by_year) + 1))
        
        underwater_fig = go.Figure()
        underwater_fig.add_trace(go.Bar(
            x=_years_arr,
            y=[p * 100 for p in _p_underwater_by_year],
            marker_color="firebrick",
            name="P(Underwater)",
        ))
        
        underwater_fig.update_layout(
            title="Probability of Being Underwater on Mortgage",
            xaxis_title="Year",
            yaxis_title="P(Underwater) %",
            height=300,
        )
    
    underwater_fig
    return (underwater_fig,)


@app.cell
def _(mo):
    mo.md("## 🔍 Diagnostics & Sensitivity Analysis")
    return ()


@app.cell
def _(mo, np, results):
    """Variance decomposition - what drives the uncertainty?"""
    _paths = results.get("paths", {})
    
    if not _paths:
        mo.stop(True, mo.md("*Run simulation first to see diagnostics*"))
    
    # Extract final values
    _property_growth = _paths["property_appreciation"][:, -1]
    _fx_final = _paths["fx_rates"][:, -1]
    _stock_final = _paths["stock_cumulative"][:, -1]
    
    _buy_wealth = results["buy_wealth"]
    _rent_wealth = results["rent_wealth"]
    
    # Calculate correlations with outcomes
    variance_correlations = {
        "Property Appreciation": {
            "buy": np.corrcoef(_property_growth, _buy_wealth)[0, 1],
            "rent": np.corrcoef(_property_growth, _rent_wealth)[0, 1],
        },
        "FX Rate (USD/CZK)": {
            "buy": np.corrcoef(_fx_final, _buy_wealth)[0, 1],
            "rent": np.corrcoef(_fx_final, _rent_wealth)[0, 1],
        },
        "Stock Returns": {
            "buy": np.corrcoef(_stock_final, _buy_wealth)[0, 1],
            "rent": np.corrcoef(_stock_final, _rent_wealth)[0, 1],
        },
    }
    
    # Build table
    _rows = []
    for _factor, _corrs in variance_correlations.items():
        _rows.append(f"| {_factor} | {_corrs['buy']:.2f} | {_corrs['rent']:.2f} |")
    
    _table = "\n".join(_rows)
    
    mo.md(
        f"""### 📊 Variance Drivers (Correlation with Final Wealth)

| Factor | Buy Correlation | Rent Correlation |
|--------|-----------------|------------------|
{_table}

*Higher absolute correlation = factor explains more variance in outcome.*

**Interpretation:**
- Property appreciation drives **buying** outcomes (you own the asset)
- FX rates drive **renting** outcomes (your USD holdings fluctuate)
- Stock returns affect both (savings/remaining cash get invested)"""
    )
    return (variance_correlations,)


@app.cell
def _(go, np, results):
    """Wealth component breakdown"""
    _paths = results.get("paths", {})
    _stats = results["summary_stats"]
    
    # For buy scenario: property equity vs invested cash
    # For rent scenario: USD holdings vs accumulated savings
    
    _buy_median = _stats["buy"]["p50"]
    _rent_median = _stats["rent"]["p50"]
    
    # Approximate component breakdown (median scenario)
    # This is illustrative - actual would need tracking from simulation
    _property_growth_median = np.median(_paths["property_appreciation"][:, -1])
    _fx_median = np.median(_paths["fx_rates"][:, -1])
    _stock_median = np.median(_paths["stock_cumulative"][:, -1])
    
    _config = results.get("config")
    if _config:
        _property_value_end = _config.property_price * _property_growth_median
        _loan_remaining = _config.property_price - _config.down_payment  # simplified
        _equity_approx = _property_value_end * 0.97 - _loan_remaining * 0.5  # rough estimate
        _invested_cash_approx = _buy_median - _equity_approx
        
        _usd_value_end = _config.usd_holdings * _stock_median * _fx_median
        _savings_approx = _rent_median - _usd_value_end
    else:
        _equity_approx = _buy_median * 0.7
        _invested_cash_approx = _buy_median * 0.3
        _usd_value_end = _rent_median * 0.6
        _savings_approx = _rent_median * 0.4
    
    composition_fig = go.Figure()
    
    composition_fig.add_trace(go.Bar(
        name="Property Equity",
        x=["Buy"],
        y=[max(0, _equity_approx) / 1e6],
        marker_color="darkgreen",
    ))
    composition_fig.add_trace(go.Bar(
        name="Invested Cash",
        x=["Buy"],
        y=[max(0, _invested_cash_approx) / 1e6],
        marker_color="lightgreen",
    ))
    composition_fig.add_trace(go.Bar(
        name="USD Holdings (converted)",
        x=["Rent+Invest"],
        y=[max(0, _usd_value_end) / 1e6],
        marker_color="darkblue",
    ))
    composition_fig.add_trace(go.Bar(
        name="Accumulated Savings",
        x=["Rent+Invest"],
        y=[max(0, _savings_approx) / 1e6],
        marker_color="lightblue",
    ))
    
    composition_fig.update_layout(
        barmode="stack",
        title="Median Wealth Composition",
        yaxis_title="Wealth (M CZK)",
        height=350,
    )
    
    composition_fig
    return (composition_fig,)


@app.cell
def _(mo):
    mo.md("### 🌪️ Sensitivity Analysis")
    return ()


@app.cell
def _(mo):
    sensitivity_button = mo.ui.run_button(label="Run Sensitivity Analysis (±20% on each parameter)")
    sensitivity_button
    return (sensitivity_button,)


@app.cell
def _(btl_expense_rate, buy_to_let, down_payment, fx_mixture_enabled, fx_stable_drift, fx_weak_dollar_drift, fx_weak_dollar_prob, mo, monthly_rent, np, property_price, renovation_cost, rent_inflation, rental_yield, sensitivity_button, usd_holdings, years):
    """One-at-a-time sensitivity analysis (tornado chart)"""
    mo.stop(not sensitivity_button.value, mo.md("*Click the button above to run sensitivity analysis*"))

    from models.simulation import run_simulation as _run_sim

    _base_params = {
        "property_price": property_price.value,
        "down_payment": down_payment.value,
        "usd_holdings": usd_holdings.value,
        "monthly_rent": monthly_rent.value,
        "years": years.value,
        "renovation_cost": renovation_cost.value,
        "rent_growth_rate": rent_inflation.value,
        "fx_mixture_enabled": fx_mixture_enabled.value,
        "fx_weak_dollar_prob": fx_weak_dollar_prob.value,
        "fx_weak_dollar_drift": fx_weak_dollar_drift.value,
        "fx_stable_drift": fx_stable_drift.value,
        "buy_to_let": buy_to_let.value,
        "rental_yield": rental_yield.value,
        "btl_expense_rate": btl_expense_rate.value,
    }
    
    # Run baseline with fewer samples for speed
    _baseline = _run_sim(**_base_params, n_samples=1000, seed=42)
    sensitivity_baseline_prob = _baseline["buy_wins_prob"]
    
    sensitivities = {}
    _param_labels = {
        "property_price": "Property Price",
        "down_payment": "Down Payment", 
        "usd_holdings": "USD Holdings",
        "monthly_rent": "Monthly Rent",
        "rent_growth_rate": "Rent Inflation",
    }
    
    for _param, _label in _param_labels.items():
        _params_low = _base_params.copy()
        _params_high = _base_params.copy()
        
        _base_val = _base_params[_param]
        _params_low[_param] = _base_val * 0.8
        _params_high[_param] = _base_val * 1.2
        
        _result_low = _run_sim(**_params_low, n_samples=1000, seed=42)
        _result_high = _run_sim(**_params_high, n_samples=1000, seed=42)
        
        sensitivities[_label] = {
            "low": _result_low["buy_wins_prob"] - sensitivity_baseline_prob,
            "high": _result_high["buy_wins_prob"] - sensitivity_baseline_prob,
        }
    
    sensitivities
    return sensitivity_baseline_prob, sensitivities


@app.cell
def _(go, np, sensitivities, sensitivity_baseline_prob):
    """Tornado chart visualization"""
    _labels = list(sensitivities.keys())
    _low_vals = [sensitivities[l]["low"] * 100 for l in _labels]  # Convert to percentage points
    _high_vals = [sensitivities[l]["high"] * 100 for l in _labels]
    
    # Sort by total range
    _ranges = [abs(h - l) for h, l in zip(_high_vals, _low_vals)]
    _sorted_idx = np.argsort(_ranges)[::-1]
    
    _labels = [_labels[i] for i in _sorted_idx]
    _low_vals = [_low_vals[i] for i in _sorted_idx]
    _high_vals = [_high_vals[i] for i in _sorted_idx]
    
    tornado_fig = go.Figure()
    
    tornado_fig.add_trace(go.Bar(
        y=_labels,
        x=_low_vals,
        name="-20%",
        orientation="h",
        marker_color="steelblue",
    ))
    
    tornado_fig.add_trace(go.Bar(
        y=_labels,
        x=_high_vals,
        name="+20%",
        orientation="h",
        marker_color="firebrick",
    ))
    
    tornado_fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    tornado_fig.update_layout(
        title=f"Sensitivity: Change in P(Buy Wins) from Baseline ({sensitivity_baseline_prob:.1%})",
        xaxis_title="Change in P(Buy Wins) (percentage points)",
        barmode="overlay",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    
    tornado_fig
    return (tornado_fig,)


@app.cell
def _(mo, sensitivities):
    """Sensitivity summary text"""
    most_sensitive = max(sensitivities.items(), key=lambda x: abs(x[1]["high"] - x[1]["low"]))
    
    mo.md(
        f"""
        **Key Finding:** The model is most sensitive to **{most_sensitive[0]}**.
        
        - Decreasing it by 20% changes P(Buy Wins) by {most_sensitive[1]['low']:+.1%}
        - Increasing it by 20% changes P(Buy Wins) by {most_sensitive[1]['high']:+.1%}
        
        *This tells you which assumptions to scrutinize most carefully.*
        """
    )
    return (most_sensitive,)


@app.cell
def _(mo):
    mo.md("### 🔬 Model Diagnostics")
    return ()


@app.cell
def _(mo, results):
    """Display model parameter estimates"""
    _model_summaries = results.get("model_summaries", {})
    
    if not _model_summaries:
        mo.stop(True, mo.md("*Model summaries not available*"))
    
    _rows = []
    
    if "appreciation" in _model_summaries:
        _a = _model_summaries["appreciation"]
        _rows.append(f"| RE Appreciation (μ) | {_a.get('mu_mean', 0):.2%} | {_a.get('mu_std', 0):.2%} |")
        _rows.append(f"| RE Volatility (σ) | {_a.get('sigma_mean', 0):.2%} | {_a.get('sigma_std', 0):.2%} |")
    
    if "mortgage" in _model_summaries:
        _m = _model_summaries["mortgage"]
        _rows.append(f"| Mortgage LT Mean (θ) | {_m.get('theta_mean', 0):.2%} | {_m.get('theta_std', 0):.2%} |")
        _rows.append(f"| Mortgage Speed (κ) | {_m.get('kappa_mean', 0):.2f} | {_m.get('kappa_std', 0):.2f} |")
    
    if "currency" in _model_summaries:
        _c = _model_summaries["currency"]
        _rows.append(f"| FX Drift (μ) | {_c.get('mu_mean', 0):.2%} | {_c.get('mu_std', 0):.2%} |")
        _rows.append(f"| FX Volatility (σ) | {_c.get('sigma_mean', 0):.2%} | {_c.get('sigma_std', 0):.2%} |")
    
    _table = "\n".join(_rows)
    
    mo.md(
        f"""### Fitted Model Parameters (Posterior Means ± Std)

| Parameter | Mean | Std |
|-----------|------|-----|
{_table}

*Parameters estimated from Czech historical data via Bayesian inference.*"""
    )
    return ()


@app.cell
def _(go, results):
    """Convergence diagnostic - distribution shapes"""
    _paths = results.get("paths", {})
    
    if not _paths:
        diagnostic_fig = go.Figure()
        diagnostic_fig.add_annotation(text="Run simulation to see diagnostics", showarrow=False)
    else:
        _property_final = _paths["property_appreciation"][:, -1]
        
        diagnostic_fig = go.Figure()
        
        diagnostic_fig.add_trace(go.Histogram(
            x=_property_final,
            name="Property Appreciation Factor",
            opacity=0.7,
            nbinsx=40,
        ))
        
        diagnostic_fig.update_layout(
            title="Distribution of Sampled Appreciation Factors",
            xaxis_title="Cumulative Appreciation Factor",
            yaxis_title="Count",
            height=300,
        )
    
    diagnostic_fig
    return (diagnostic_fig,)


@app.cell
def _(mo):
    mo.md("## 📈 Model Assumptions & Decision Summary")
    return ()


@app.cell
def _(district, enable_refinancing, enable_tax_benefits, fx_mixture_enabled, fx_stable_drift, fx_weak_dollar_drift, fx_weak_dollar_prob, inflation_adjust, mo, results):
    """Dynamic assumptions table with defaults vs configured values"""
    _models = results.get("model_summaries", {})
    _config = results.get("config")
    _stats = results.get("summary_stats", {})
    
    # Get fitted values
    _appr = _models.get("appreciation", {})
    _mort = _models.get("mortgage", {})
    _curr = _models.get("currency", {})
    
    # Build comparison table
    _rows = []
    
    # Real Estate
    _re_default = "μ=5%, σ=8%"
    _re_fitted = f"μ={_appr.get('mu_mean', 0.05):.1%}, σ={_appr.get('sigma_mean', 0.08):.1%}"
    _rows.append(f"| RE Appreciation | {_re_default} | {_re_fitted} |")
    
    # District
    _dist_mult = results.get("district_multiplier", 1.0)
    _rows.append(f"| District Multiplier | 1.0× (avg) | {_dist_mult:.2f}× ({district.value}) |")
    
    # Mortgage
    _mort_default = "θ=4.5%, κ=0.3"
    _mort_fitted = f"θ={_mort.get('theta_mean', 0.045):.1%}, κ={_mort.get('kappa_mean', 0.3):.2f}"
    _rows.append(f"| Mortgage (Vasicek) | {_mort_default} | {_mort_fitted} |")
    
    # FX
    if fx_mixture_enabled.value:
        _fx_config = f"Mixture: {fx_weak_dollar_prob.value:.0%}@{fx_weak_dollar_drift.value:.0%}, {1-fx_weak_dollar_prob.value:.0%}@{fx_stable_drift.value:.0%}"
    else:
        _fx_config = f"μ={_curr.get('mu_mean', 0):.1%}, σ={_curr.get('sigma_mean', 0.1):.1%}"
    _rows.append(f"| USD/CZK | Historical GBM | {_fx_config} |")
    
    # Stock returns
    _stock_default = "μ=7%, σ=18%"
    _stock_config = f"μ={_config.stock_return_mean:.0%}, σ={_config.stock_return_std:.0%}" if _config else _stock_default
    _rows.append(f"| Equity Returns | {_stock_default} | {_stock_config} |")
    
    # Rent inflation
    _rent_default = "3%/yr"
    _rent_config = f"{_config.rent_growth_rate:.0%}/yr" if _config else _rent_default
    _rows.append(f"| Rent Inflation | {_rent_default} | {_rent_config} |")
    
    _table = "\n".join(_rows)
    
    # Features enabled
    _features = []
    if enable_refinancing.value:
        _features.append("✓ Refinancing (5yr)")
    if enable_tax_benefits.value:
        _features.append("✓ Czech tax benefits")
    if inflation_adjust.value:
        _features.append("✓ Inflation-adjusted")
    if fx_mixture_enabled.value:
        _features.append("✓ FX mixture model")
    _features_str = " | ".join(_features) if _features else "None"
    
    # Decision summary
    _prob = _stats.get("buy_wins_prob", 0.5)
    _verdict = "**BUY**" if _prob > 0.5 else "**RENT + INVEST**"
    _confidence = abs(_prob - 0.5) * 2  # 0-1 scale
    if _confidence < 0.1:
        _confidence_str = "Low confidence (near 50/50)"
    elif _confidence < 0.3:
        _confidence_str = "Moderate confidence"
    else:
        _confidence_str = "High confidence"
    
    mo.md(
        f"""### Configured vs Default Parameters

| Factor | Default | Current Config |
|--------|---------|----------------|
{_table}

**Features enabled:** {_features_str}

---

### 🎯 Decision Summary

Based on the current configuration:
- **Recommendation:** {_verdict}
- **P(Buy Wins):** {_prob:.1%}
- **Confidence:** {_confidence_str}

*{_confidence_str} — {"consider both options carefully" if _confidence < 0.2 else "the model shows a clear preference"}.*"""
    )
    return ()


@app.cell
def _(mo):
    mo.md(
        """
        ---
        
        ### 🔧 Technical Details
        
        - **Monte Carlo samples**: 5,000 independent scenarios
        - **Data sources**: ČSÚ, CNB, Yahoo Finance
        - **Built with**: PyMC, Marimo, Plotly
        
        *Disclaimer: This is a decision-support tool, not financial advice.*
        """
    )
    return ()


if __name__ == "__main__":
    app.run()
