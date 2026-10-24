"""Data loading, validation and the data-quality report."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import Paths


def _read(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, **kw)


def load_tables(data_dir: Path | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load (users, apps, interactions) with timestamps and numerics coerced."""
    p = Paths(Path(data_dir)) if data_dir else Paths()

    users = _read(p.users)
    apps = _read(p.apps)
    interactions = _read(p.interactions)

    # The log has two timestamp formats. format="mixed" parses both; the default
    # fast parser silently drops rows without microseconds.
    interactions["event_time"] = pd.to_datetime(
        interactions["event_time"], errors="coerce", format="mixed"
    )

    for col in ("size_mb", "avg_rating", "total_ratings"):
        if col in apps.columns:
            apps[col] = pd.to_numeric(apps[col], errors="coerce")

    return users, apps, interactions


@dataclass
class DQFinding:
    code: str
    severity: str  # "info" | "warning" | "error"
    table: str
    description: str
    n_affected: int = 0
    action: str = ""


@dataclass
class DataQualityReport:
    findings: list[DQFinding] = field(default_factory=list)

    def add(self, code: str, severity: str, table: str, description: str,
            n_affected: int = 0, action: str = "") -> None:
        self.findings.append(
            DQFinding(code, severity, table, description, n_affected, action)
        )

    def summary(self) -> str:
        sev = {"error": 0, "warning": 0, "info": 0}
        for f in self.findings:
            sev[f.severity] += 1
        return (f"{len(self.findings)} findings: "
                f"{sev['error']} errors, {sev['warning']} warnings, {sev['info']} info")

    def __iter__(self):
        return iter(self.findings)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame([f.__dict__ for f in self.findings])


def validate(users: pd.DataFrame, apps: pd.DataFrame, interactions: pd.DataFrame) -> DataQualityReport:
    """Schema and referential-integrity checks over the three tables."""
    rep = DataQualityReport()

    for key, df, name in (
        ("user_id", users, "users"),
        ("app_id", apps, "apps"),
    ):
        dups = int(df[key].duplicated().sum())
        if dups:
            rep.add("PK_DUPLICATES", "error", name,
                    f"Duplicate `{key}` rows", dups, "Drop duplicates in loader.")

    user_ids = set(users["user_id"])
    app_ids = set(apps["app_id"])

    orphan_users = set(interactions["user_id"].unique()) - user_ids
    if orphan_users:
        rep.add(
            "FK_USER_MISSING", "warning", "interactions",
            "interactions reference user_id absent from users.csv (cold users with no "
            "profile, each observed once).",
            len(orphan_users),
            "Flag + keep: real traffic, the model must not crash on them.",
        )

    orphan_apps = set(interactions["app_id"].unique()) - app_ids
    if orphan_apps:
        rep.add(
            "FK_APP_MISSING", "warning", "interactions",
            "interactions reference app_id absent from apps.csv (cold items with no "
            "metadata, each touched by one user).",
            len(orphan_apps),
            "Flag + keep; they are a hard recall ceiling.",
        )

    n_nat = int(interactions["event_time"].isna().sum())
    if n_nat:
        rep.add("TIME_UNPARSABLE", "warning", "interactions",
                "event_time could not be parsed (coerced to NaT).", n_nat,
                "Exclude from the split/training table.")

    known_events = {"af_view", "af_click", "af_install", "af_open",
                    "af_complete_registration", "af_level_achieved", "af_purchase"}
    unknown_events = set(interactions["event_name"].unique()) - known_events
    if unknown_events:
        rep.add("EVENT_UNKNOWN", "warning", "interactions",
                f"Unrecognized event_name values {sorted(map(str, unknown_events))}.",
                len(unknown_events), "Treat as lowest grade.")

    post_install = {"af_purchase", "af_complete_registration", "af_level_achieved"}
    mis = int(((interactions["event_name"].isin(post_install)) &
               (interactions["interaction_type"] != "open")).sum())
    rep.add(
        "EVENTTYPE_COLLAPSED", "info", "interactions",
        "af_purchase / af_complete_registration / af_level_achieved all carry "
        "interaction_type == 'open'; depth only recoverable from event_name.",
        mis,
        "Don't fix: read strength from event_name.",
    )

    if "duration_seconds" in interactions.columns:
        n_dur_null = int(interactions["duration_seconds"].isna().sum())
        if n_dur_null:
            rep.add(
                "DURATION_NULL", "info", "interactions",
                "duration_seconds is null (largely on open/post-install events).",
                n_dur_null,
                "Ignore as a feature.",
            )

    app_platform = apps.set_index("app_id")["platform"].rename("app_platform")
    merged = interactions[["user_id", "app_id"]].merge(
        users[["user_id", "platform"]], on="user_id", how="left"
    ).merge(app_platform, on="app_id", how="left")
    cross = int((merged["platform"].notna() & merged["app_platform"].notna() &
                 (merged["platform"] != merged["app_platform"])).sum())
    if cross:
        rep.add(
            "CROSS_PLATFORM", "info", "interactions",
            "interactions whose app platform differs from the user's platform.", cross,
            "Keep; encode is_same_platform instead of deleting.",
        )

    n_country_null = int(users["country_code"].isna().sum())
    if n_country_null:
        rep.add("USER_COUNTRY_NULL", "info", "users", "Null country_code.",
                n_country_null, "Map to an 'unknown' sentinel.")

    seen_apps = set(interactions["app_id"].unique())
    never_seen = app_ids - seen_apps
    if never_seen:
        rep.add("APP_NEVER_SEEN", "info", "apps",
                "Apps in the catalog never present in interactions.", len(never_seen),
                "Keep: legitimate cold-start candidates for inference.")

    t = interactions["event_time"]
    if t.notna().any():
        lo, hi = t.min(), t.max()
        span = (hi - lo).days
        rep.add(
            "WINDOW_NOTE", "info", "interactions",
            f"Documented '~90 days' but interaction span is ~{span} days "
            f"({lo.date()} -> {hi.date()}).",
            int(span),
            "Flag only.",
        )

    return rep


def interaction_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return interaction rows with required columns present and NaT dropped."""
    required = {"user_id", "app_id", "event_time", "event_name", "interaction_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"interactions missing required columns: {sorted(missing)}")
    return df.dropna(subset=["event_time"])