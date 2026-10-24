"""Feature construction for (user, app) pairs.

One row per (user, app) pair is the shared train/serve grain. Counts and
popularity priors are computed on train events only; time features are measured
from the train horizon (latest train timestamp).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CATEGORICAL_FEATURES, EVENT_WEIGHT, NUMERIC_FEATURES, POSITIVE_EVENTS

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

UNK = "<UNK>"


def event_weight(event_name: str) -> float:
    return EVENT_WEIGHT.get(event_name, 1.0)


def _categorical(f: pd.DataFrame) -> pd.DataFrame:
    """Normalize present categorical columns to str with an unknown sentinel."""
    for c in CATEGORICAL_FEATURES:
        if c not in f.columns:
            continue
        if isinstance(f[c].dtype, pd.CategoricalDtype):
            f[c] = f[c].astype(str)
        f[c] = f[c].where(f[c].notna(), UNK).astype(str)
    return f


class FeatureBuilder:
    """Builds (user, app) feature frames from users, apps and train events."""

    def __init__(self, users: pd.DataFrame, apps: pd.DataFrame) -> None:
        app = apps[
            ["app_id", "platform", "category", "age_rating", "status",
             "size_mb", "avg_rating", "total_ratings", "app_date_inserted"]
        ].copy()
        app = app.rename(columns={
            "platform": "app_platform",
            "category": "app_category",
            "age_rating": "app_age_rating",
            "status": "app_status",
            "size_mb": "app_size_mb",
            "avg_rating": "app_avg_rating",
        })
        app["has_metadata"] = 1
        app["app_log_total_ratings"] = np.log1p(
            pd.to_numeric(app["total_ratings"], errors="coerce").fillna(0.0))
        for c in ("app_size_mb", "app_avg_rating"):
            app[c] = pd.to_numeric(app[c], errors="coerce")
        app["app_date_inserted"] = pd.to_datetime(app["app_date_inserted"], errors="coerce")
        self._app_features = app.set_index("app_id")

        user = users[
            ["user_id", "platform", "age_bucket", "country_code", "signup_date"]
        ].copy()
        user = user.rename(columns={
            "platform": "user_platform",
            "age_bucket": "user_age_bucket",
            "country_code": "user_country",
        })
        user["user_country"] = user["user_country"].fillna(UNK).astype(str)
        user["user_age_bucket"] = user["user_age_bucket"].fillna(UNK).astype(str)
        user["user_platform"] = user["user_platform"].astype(str)
        user["signup_date"] = pd.to_datetime(user["signup_date"], errors="coerce")
        self._user_static = user.drop_duplicates("user_id").set_index("user_id")

    def fit(self, train_events: pd.DataFrame) -> "FeatureBuilder":
        ev = train_events[train_events["event_time"].notna()].copy()
        if ev.empty:
            raise ValueError("train events are empty after dropping NaT timestamps")

        self.horizon = ev["event_time"].max()

        agg = ev.groupby("user_id", sort=False)
        user_agg = pd.DataFrame({
            "user_n_events": agg.size(),
            "user_n_installs": agg["event_name"].apply(lambda s: s.isin(POSITIVE_EVENTS).sum()),
            "user_n_distinct_apps": agg["app_id"].nunique(),
            "user_last_event": agg["event_time"].max(),
        })
        user_agg["user_install_rate"] = (
            user_agg["user_n_installs"] / user_agg["user_n_events"].clip(lower=1))
        user_agg["user_recency_days"] = (
            (self.horizon - user_agg["user_last_event"]).dt.total_seconds() / 86400.0
        ).clip(lower=0.0)
        user_agg = user_agg.drop(columns="user_last_event")

        # mode category over the user's train history
        ev_cat = ev[["user_id", "app_id"]].merge(
            self._app_features[["app_category"]], left_on="app_id", right_index=True, how="left")
        ev_cat["app_category"] = ev_cat["app_category"].fillna(UNK).astype(str)
        pref = (ev_cat.groupby(["user_id", "app_category"], sort=False).size()
                .reset_index(name="_c")
                .sort_values(["user_id", "_c"], ascending=[True, False])
                .drop_duplicates("user_id")
                .set_index("user_id")["app_category"])
        user_agg["user_pref_category"] = pref.reindex(user_agg.index).fillna(UNK).astype(str)
        self._user_agg = user_agg

        cat_counts = ev_cat.groupby(["user_id", "app_category"], sort=False).size()
        total = ev_cat.groupby("user_id", sort=False).size()
        share = cat_counts.astype(float) / total.reindex(
            cat_counts.index.get_level_values("user_id")).values
        self._user_cat_share = share.rename("share").reset_index()

        self._app_pop = np.log1p(ev.groupby("app_id", sort=False).size().astype(float))

        seg = ev[["user_id", "app_id"]].merge(
            self._user_static[["user_platform", "user_age_bucket", "user_country"]],
            left_on="user_id", right_index=True, how="left")
        seg = _categorical(seg[["user_platform", "user_age_bucket", "user_country"]])
        seg["app_id"] = ev[["user_id", "app_id"]].reset_index(drop=True)["app_id"]
        self._seg_pop = {
            "seg_pop_log1p": self._seg_table(seg, ["user_country", "user_platform"]),
            "seg_age_pop_log1p": self._seg_table(seg, ["user_age_bucket", "user_platform"]),
            "seg_cap_pop_log1p": self._seg_table(
                seg, ["user_country", "user_age_bucket", "user_platform"]),
        }
        return self

    @staticmethod
    def _seg_table(seg: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Flat (key, app_id) -> log1p(count) table for a segment."""
        key = seg[cols[0]].astype(str)
        for c in cols[1:]:
            key = key + "|" + seg[c].astype(str)
        tmp = pd.DataFrame({"key": key.values, "app_id": seg["app_id"].values})
        counts = tmp.groupby(["key", "app_id"], sort=False).size().reset_index(name="v")
        counts["v"] = np.log1p(counts["v"].astype(float))
        return counts

    @staticmethod
    def _resolve_seg(f: pd.DataFrame, *cols: str, table: pd.DataFrame) -> np.ndarray:
        key = f[cols[0]].astype(str)
        for c in cols[1:]:
            key = key + "|" + f[c].astype(str)
        df = pd.DataFrame({"key": key.values, "app_id": f["app_id"].values})
        merged = df.merge(table, on=["key", "app_id"], how="left")
        return merged["v"].fillna(0.0).values

    def _join_static(self, pairs: pd.DataFrame) -> pd.DataFrame:
        f = pairs[["user_id", "app_id"]].copy()
        f = f.join(self._user_static, on="user_id", how="left")
        f = f.join(self._app_features, on="app_id", how="left")

        f["user_signup_age_days"] = (
            (self.horizon - pd.to_datetime(f["signup_date"])).dt.total_seconds() / 86400.0
        ).clip(lower=0.0)
        f["app_age_days"] = (
            (self.horizon - pd.to_datetime(f["app_date_inserted"])).dt.total_seconds() / 86400.0
        ).clip(lower=0.0)
        f["is_same_platform"] = (
            f["user_platform"].astype(str) == f["app_platform"].astype(str)).astype(np.int64)
        return f

    def _attach_user_agg_and_pop(self, f: pd.DataFrame) -> pd.DataFrame:
        f = f.join(self._user_agg, on="user_id", how="left")

        keys = pd.DataFrame({
            "user_id": f["user_id"].values,
            "app_category": f["app_category"].fillna(UNK).astype(str).values,
        })
        merged = keys.merge(
            self._user_cat_share.rename(columns={"app_category": "app_category"}),
            on=["user_id", "app_category"], how="left")
        f["user_cat_affinity"] = merged["share"].fillna(0.0).values

        f["app_pop_log1p"] = self._app_pop.reindex(f["app_id"]).fillna(0.0).values
        f["seg_pop_log1p"] = self._resolve_seg(
            f, "user_country", "user_platform", table=self._seg_pop["seg_pop_log1p"])
        f["seg_age_pop_log1p"] = self._resolve_seg(
            f, "user_age_bucket", "user_platform", table=self._seg_pop["seg_age_pop_log1p"])
        f["seg_cap_pop_log1p"] = self._resolve_seg(
            f, "user_country", "user_age_bucket", "user_platform",
            table=self._seg_pop["seg_cap_pop_log1p"])
        return f

    def collapse_pairs(self, events: pd.DataFrame) -> pd.DataFrame:
        """Event log -> one row per (user, app), labelled by the strongest event."""
        ev = events[events["event_time"].notna()].copy()
        ev["_weight"] = ev["event_name"].map(event_weight).fillna(1.0).clip(lower=1.0)

        idx = ev.groupby(["user_id", "app_id"], sort=False)["_weight"].idxmax()
        out = ev.loc[idx, ["user_id", "app_id", "event_name"]].rename(
            columns={"event_name": "strongest_event"}).reset_index(drop=True)
        out["weight"] = ev.loc[idx, "_weight"].reset_index(drop=True)
        out["n_events"] = ev.groupby(["user_id", "app_id"], sort=False).size().values
        out["is_positive"] = out["strongest_event"].isin(POSITIVE_EVENTS).astype(np.int64)
        return out

    def build_train_frame(self, train_events: pd.DataFrame) -> pd.DataFrame:
        """Feature frame plus labels for observed (user, app) pairs."""
        pairs = self.collapse_pairs(train_events)
        f = self._join_static(pairs)
        f = self._attach_user_agg_and_pop(f)
        f = _categorical(f)
        f["strongest_event"] = pairs["strongest_event"].values
        f["weight"] = pairs["weight"].values
        f["is_positive"] = pairs["is_positive"].values
        f["n_events"] = pairs["n_events"].values
        return f

    def candidates_for_user(self, user_id: str) -> pd.DataFrame:
        """Feature frame of all eligible candidates for one user.

        Eligible = apps with metadata on the user's platform, excluding blocked
        apps. Paused apps stay eligible (a campaign state, not usability). Falls
        back to any-platform active/paused if the platform slice is empty.
        """
        if user_id not in self._user_static.index:
            return pd.DataFrame(columns=list(FEATURE_COLUMNS) + ["app_id", "user_id"])

        u = self._user_static.loc[user_id]
        app_mask = self._app_features["app_status"].astype(str) != "blocked"
        app_ids = self._app_features[app_mask & (
            self._app_features["app_platform"].astype(str) == str(u["user_platform"]))].index
        if len(app_ids) == 0:
            app_ids = self._app_features[app_mask].index

        pairs = pd.DataFrame({"user_id": [user_id] * len(app_ids), "app_id": app_ids.tolist()})
        f = self._join_static(pairs)
        f = self._attach_user_agg_and_pop(f)
        f = _categorical(f)
        return f