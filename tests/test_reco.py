"""Tests for the data, feature and model layers.

Run with pytest from the repository root (see README).
"""

from __future__ import annotations

import pandas as pd
import pytest

from reco.config import EVENT_GRADE, EVENT_WEIGHT, POSITIVE_EVENTS
from reco.data import DataQualityReport, validate
from reco.features import FeatureBuilder
from reco.model import AppRecommender


@pytest.fixture
def small_users() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": ["u1", "u2", "u3"],
        "platform": ["ios", "android", "ios"],
        "age_bucket": ["18-24", "25-34", "35-44"],
        "country_code": ["US", "FR", None],
        "signup_date": ["2026-01-01", "2026-02-01", "2026-03-01"],
    })


@pytest.fixture
def small_apps() -> pd.DataFrame:
    return pd.DataFrame({
        "app_id": ["a1", "a2", "a3", "a4"],
        "platform": ["ios", "android", "ios", "ios"],
        "category": ["Games", "Games", "Shopping", "Travel"],
        "age_rating": ["4+", "12+", "9+", "17+"],
        "status": ["active", "active", "active", "paused"],
        "size_mb": [10.0, 20.0, 30.0, 40.0],
        "avg_rating": [4.0, 3.5, 4.5, 3.0],
        "total_ratings": [100, 200, 300, 400],
        "app_date_inserted": ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"],
    })


@pytest.fixture
def small_events() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": ["u1", "u1", "u1", "u2", "u2", "u3"],
        "app_id": ["a1", "a1", "a3", "a2", "a2", "a3"],
        "event_time": pd.to_datetime([
            "2026-04-01", "2026-04-02", "2026-04-03",
            "2026-04-01", "2026-04-02", "2026-04-01",
        ]),
        "event_name": ["af_view", "af_install", "af_open", "af_view", "af_click", "af_install"],
        "interaction_type": ["view", "install", "open", "view", "click", "install"],
        "duration_seconds": [5.0, 10.0, 20.0, 5.5, 6.0, 10.0],
    })


def _empty_builder():
    return FeatureBuilder(
        pd.DataFrame({"user_id": [], "platform": [], "age_bucket": [],
                      "country_code": [], "signup_date": []}),
        pd.DataFrame({"app_id": [], "platform": [], "category": [], "age_rating": [],
                      "status": [], "size_mb": [], "avg_rating": [],
                      "total_ratings": [], "app_date_inserted": []}),
    )


def test_funnel_order() -> None:
    grades = [EVENT_GRADE[e] for e in
              ["af_view", "af_click", "af_install", "af_open",
               "af_complete_registration", "af_level_achieved", "af_purchase"]]
    assert grades == sorted(grades) and len(set(grades)) == len(grades)


def test_positive_events_include_open_and_beyond() -> None:
    assert "af_open" in POSITIVE_EVENTS
    assert "af_install" in POSITIVE_EVENTS
    assert "af_view" not in POSITIVE_EVENTS
    assert "af_click" not in POSITIVE_EVENTS


def test_weights_strictly_increasing_along_funnel() -> None:
    ws = [EVENT_WEIGHT[e] for e in
          ["af_install", "af_open", "af_complete_registration",
           "af_level_achieved", "af_purchase"]]
    assert ws == sorted(ws)


def test_collapse_pairs_keeps_strongest_event(small_events) -> None:
    pairs = _empty_builder().collapse_pairs(small_events)
    row = pairs[pairs.user_id == "u1"].sort_values("app_id")
    # u1 has (a1: view+install -> install) and (a3: open)
    assert set(row["strongest_event"]) == {"af_install", "af_open"}


def test_pointwise_label_boundary(small_events) -> None:
    app_cols = ["app_id", "platform", "category", "age_rating", "status",
                "size_mb", "avg_rating", "total_ratings", "app_date_inserted"]
    builder = FeatureBuilder(
        pd.DataFrame({"user_id": ["u1"], "platform": ["ios"], "age_bucket": ["18-24"],
                      "country_code": ["US"], "signup_date": ["2026-01-01"]}),
        pd.DataFrame({"app_id": ["a1", "a2", "a3", "a4"],
                      "platform": ["ios", "android", "ios", "ios"],
                      "category": ["Games", "Games", "Shopping", "Travel"],
                      "age_rating": ["4+", "12+", "9+", "17+"],
                      "status": ["active"] * 4,
                      "size_mb": [10, 20, 30, 40],
                      "avg_rating": [4, 3.5, 4.5, 3],
                      "total_ratings": [100, 200, 300, 400],
                      "app_date_inserted": ["2025-01-01"] * 4}),
    )
    pairs = builder.collapse_pairs(small_events)
    inst = pairs[pairs.strongest_event == "af_install"]
    assert inst.is_positive.all()
    view = pairs[pairs.strongest_event == "af_view"]
    assert (~view.is_positive.astype(bool)).all()


def test_validate_detects_orphans(small_users, small_apps, small_events) -> None:
    events = small_events.copy()
    events = pd.concat([events, pd.DataFrame({
        "user_id": ["u_orphan"], "app_id": ["a1"],
        "event_time": pd.to_datetime(["2026-04-01"]),
        "event_name": ["af_view"], "interaction_type": ["view"],
    })])
    rep: DataQualityReport = validate(small_users, small_apps, events)
    codes = {f.code for f in rep.findings}
    assert "FK_USER_MISSING" in codes


def test_model_cold_user_never_crashes() -> None:
    m = AppRecommender(objective="listwise")
    # never fitted -> recommend must return a list, not raise
    out = m.recommend("u_zzz", n=5)
    assert isinstance(out, list)