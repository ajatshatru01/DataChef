"""Headless smoke tests for the real Streamlit entry point."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from datachef import config


def test_app_starts_empty_and_loads_sample(monkeypatch) -> None:
    monkeypatch.setattr(config, "validate", lambda: [])
    app_path = Path(__file__).parents[2] / "datachef" / "ui" / "app.py"
    app = AppTest.from_file(app_path)

    app.run(timeout=20)
    assert not app.exception

    sample_button = next(button for button in app.button if "Ecommerce" in button.label)
    sample_button.click()
    app.run(timeout=20)

    assert not app.exception
    assert any("sample_ecommerce_orders.csv" in item.value for item in app.success)
