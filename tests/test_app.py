"""Smoke test: the Streamlit app runs end to end, every tab included."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
TAB_PREFIXES = ["Live ", "Backtest ", "Teams", "How it works"]


@pytest.fixture
def in_repo_root():
    """The app reads data/ and models/ relative to the working directory."""
    previous = Path.cwd()
    os.chdir(ROOT)
    yield
    os.chdir(previous)


@pytest.mark.parametrize("tab", ["", "teams"])
def test_app_runs_without_errors(in_repo_root, tab):
    # Streamlit runs every tab's code on each run, whichever tab is open.
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=180)
    if tab:
        app.query_params["tab"] = tab
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    labels = [t.label for t in app.tabs][: len(TAB_PREFIXES)]
    assert all(label.startswith(p) for label, p in zip(labels, TAB_PREFIXES, strict=True))
