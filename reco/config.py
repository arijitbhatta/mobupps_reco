"""Shared constants for the recommender: funnel order, weights, features, objectives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data"


@dataclass(frozen=True)
class Paths:
    """Paths to the raw input files."""

    data_dir: Path = DEFAULT_DATA_DIR

    @property
    def users(self) -> Path:
        return self.data_dir / "users.csv"

    @property
    def apps(self) -> Path:
        return self.data_dir / "apps.csv"

    @property
    def interactions(self) -> Path:
        return self.data_dir / "interactions.csv"


# Funnel order, weakest -> strongest. af_open is post-install (launching an
# already-installed app), so it sits after af_install.
EVENT_NAMES = (
    "af_view",
    "af_click",
    "af_install",
    "af_open",
    "af_complete_registration",
    "af_level_achieved",
    "af_purchase",
)

# Pointwise positive label: anything at or past install.
POSITIVE_EVENTS = frozenset(
    {"af_install", "af_open", "af_complete_registration", "af_level_achieved", "af_purchase"}
)

# Pointwise sample_weight, increasing along the funnel. Negatives (view/click/
# unobserved) use a floor of 1. The -1 values are never used directly.
EVENT_WEIGHT = {
    "af_view": -1.0,
    "af_click": -1.0,
    "af_install": 2.0,
    "af_open": 3.0,
    "af_complete_registration": 4.0,
    "af_level_achieved": 5.0,
    "af_purchase": 6.0,
}

# Graded relevance for ranking objectives. 0 is reserved for unobserved negatives.
EVENT_GRADE = {
    "af_view": 1,
    "af_click": 2,
    "af_install": 3,
    "af_open": 4,
    "af_complete_registration": 5,
    "af_level_achieved": 6,
    "af_purchase": 7,
}

# 2^i - 1 ladder so each grade strictly outranks all weaker grades combined.
LABEL_GAIN = [0, 1, 3, 7, 15, 31, 63, 127]

OBJECTIVES = {
    "pointwise": "binary",
    "pairwise": "rank_xendcg",
    "listwise": "lambdarank",
}

CATEGORICAL_FEATURES = [
    "user_platform",
    "user_age_bucket",
    "user_country",
    "app_category",
    "app_age_rating",
    "app_status",
    "user_pref_category",
]

NUMERIC_FEATURES = [
    "is_same_platform",
    "has_metadata",
    "app_size_mb",
    "app_avg_rating",
    "app_log_total_ratings",
    "app_age_days",
    "user_signup_age_days",
    "user_n_events",
    "user_n_distinct_apps",
    "user_n_installs",
    "user_install_rate",
    "user_recency_days",
    "user_cat_affinity",
    "app_pop_log1p",
    "seg_pop_log1p",
    "seg_age_pop_log1p",
    "seg_cap_pop_log1p",
]