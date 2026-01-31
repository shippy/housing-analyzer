# Housing Analyzer - Development Guidelines

## Model Scrutiny

When working with simulation models, unexpected results should trigger investigation of **both** the model and the implications:

1. **Check model logic first**: Surprising outputs often indicate bugs
   - Example: Buy-to-let initially showed P(win)=9% even with +37k/mo net income
   - Root cause: Double-counting mortgage costs that were already handled implicitly

2. **Verify assumptions match reality**:
   - Prague's 2-3% rental yields are real (confirmed via CNB, Global Property Guide)
   - Price-to-rent ratios >40 indicate overvalued market

3. **Ensure apples-to-apples comparisons**:
   - Both scenarios must use consistent cost treatment
   - Cash flows should reflect actual differences, not re-count shared expenses

4. **When results seem wrong**:
   - Trace through the math manually with simple numbers
   - Compare intermediate values, not just final outputs
   - Ask: "What would make this result true?" - if the answer is absurd, there's likely a bug

## Project Structure

- `core/` - Shared logic (ScenarioConfig, runner functions)
- `models/` - Bayesian simulation models (PyMC)
- `cli.py` - Command-line interface
- `app.py` - Marimo notebook UI
- `grid_search.py` - Parameter sweep analysis
