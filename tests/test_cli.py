import argparse

from cli import add_common_args, build_config_from_args
from core.config import ScenarioConfig


def test_cli_ignores_btl_params_when_disabled(monkeypatch) -> None:
    defaults = ScenarioConfig.from_defaults()
    defaults.buy_to_let = False
    defaults.rental_yield = 0.025
    defaults.btl_expense_rate = 0.30
    monkeypatch.setattr(ScenarioConfig, "from_defaults", lambda: defaults)

    parser = argparse.ArgumentParser()
    add_common_args(parser)

    args = parser.parse_args(["--rental-yield", "0.05", "--btl-expense-rate", "0.40"])
    config = build_config_from_args(args)

    assert config.buy_to_let is False
    assert config.rental_yield == defaults.rental_yield
    assert config.btl_expense_rate == defaults.btl_expense_rate
