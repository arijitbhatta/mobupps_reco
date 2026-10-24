"""The recommender: a LightGBM model over (user, app) pairs.

Three training objectives, chosen via the ``objective`` argument:

* ``pointwise`` - binary classifier, positive = event >= install, weighted by
  funnel position. af_view/af_click are negatives.
* ``pairwise`` - rank_xendcg, graded relevance, af_view down-weighted.
* ``listwise`` - lambdarank with label_gain, same graded relevance.

Cold users fall back to country x age x platform segment popularity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from .config import (CATEGORICAL_FEATURES, EVENT_GRADE, EVENT_WEIGHT, LABEL_GAIN,
                     OBJECTIVES, POSITIVE_EVENTS)
from .data import load_tables
from .features import FEATURE_COLUMNS, FeatureBuilder

MIN_HISTORY_FOR_MODEL = 5


def _grade_of(event_name: str) -> int:
    return EVENT_GRADE.get(event_name, 0)


class AppRecommender:
    """LightGBM (user, app) recommender with a segment-popularity cold start."""

    def __init__(self, objective: str = "listwise", data_dir=None,
                 n_estimators: int = 250, num_leaves: int = 63,
                 learning_rate: float = 0.1, n_neg_per_user: int = 5,
                 seed: int = 42) -> None:
        if objective not in OBJECTIVES:
            raise ValueError(f"unknown objective {objective!r}; choose from {sorted(OBJECTIVES)}")
        self.objective = objective
        self.data_dir = data_dir
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_neg_per_user = n_neg_per_user
        self.seed = seed

        self._users = None
        self._apps = None
        self._builder = None
        self._model = None
        self._is_classifier = False
        self._train = None
        self._user_seen_apps: dict[str, set[str]] = {}
        self._user_has_history: set[str] = set()
        self._cold_top: dict[tuple, list[str]] = {}
        self._global_top: list[str] = []

    def _load_side_tables(self) -> None:
        if self._users is None or self._apps is None:
            self._users, self._apps, _ = load_tables(self.data_dir)

    def fit(self, train_df: pd.DataFrame) -> "AppRecommender":
        """Train on the interaction train slice."""
        self._load_side_tables()
        train = train_df[train_df["event_time"].notna()].copy()
        if train.empty:
            raise ValueError("empty train slice after dropping NaT timestamps")

        self._builder = FeatureBuilder(self._users, self._apps).fit(train)
        self._train = train
        self._user_seen_apps = self._collect_seen(train)
        self._user_has_history = set(train["user_id"].unique())

        frame = self._build_train_frame(train)
        self._model = self._fit_lgb(frame)
        self._fit_cold()
        return self

    @staticmethod
    def _collect_seen(train: pd.DataFrame) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for uid in train["user_id"].unique():
            out[uid] = set(train.loc[train["user_id"] == uid, "app_id"])
        return out

    def _build_train_frame(self, train: pd.DataFrame) -> pd.DataFrame:
        """Observed pairs plus sampled unobserved negatives."""
        obs = self._builder.build_train_frame(train)
        neg = self._sample_unobserved(train)

        if self.objective == "pointwise":
            pos = obs[obs["is_positive"] == 1].copy()
            weak = obs[obs["is_positive"] == 0].copy()
            pos["label"] = 1
            weak["label"] = 0
            pos["sample_weight"] = pos["strongest_event"].map(EVENT_WEIGHT).fillna(2.0)
            weak["sample_weight"] = 1.0
            neg["label"] = 0
            neg["sample_weight"] = 1.0
            keep = FEATURE_COLUMNS + ["user_id", "app_id", "label", "sample_weight"]
            return pd.concat([pos[keep], weak[keep], neg[keep]], ignore_index=True)

        obs["label"] = obs["strongest_event"].map(_grade_of).fillna(0).astype(int)
        neg["label"] = 0
        keep = FEATURE_COLUMNS + ["user_id", "app_id", "label"]
        return pd.concat([obs[keep], neg[keep]], ignore_index=True)

    def _sample_unobserved(self, train: pd.DataFrame) -> pd.DataFrame:
        """Sample unobserved (user, app) negatives, excluding seen apps.

        The pool is restricted to same-platform, non-blocked apps the user never
        touched, so a held-out install can't become a sampled negative.
        """
        rng = np.random.default_rng(self.seed)
        app_features = self._builder._app_features
        blocked = app_features["app_status"].astype(str) == "blocked"
        eligible_by_platform = {
            plat: app_features[(~blocked) & (app_features["app_platform"].astype(str) == plat)]
            .index.tolist()
            for plat in app_features["app_platform"].astype(str).unique()
        }
        user_platform = self._builder._user_static["user_platform"].astype(str)

        rows = []
        for uid in self._user_seen_apps:
            seen = self._user_seen_apps[uid]
            plat = user_platform.get(uid, "<UNK>")
            eligible = eligible_by_platform.get(plat) or [
                a for a in app_features.index if a not in set(blocked[blocked].index)
            ]
            pool = [a for a in eligible if a not in seen]
            if not pool:
                continue
            k = min(self.n_neg_per_user, len(pool))
            chosen = rng.choice(pool, size=k, replace=False)
            rows.append(pd.DataFrame({"user_id": [uid] * k, "app_id": chosen}))

        if not rows:
            return pd.DataFrame(columns=FEATURE_COLUMNS + ["user_id", "app_id", "label"])

        pairs = pd.concat(rows, ignore_index=True)
        f = self._builder._join_static(pairs)
        f = self._builder._attach_user_agg_and_pop(f)
        f = self._categorize(f)
        return f

    def _categorize(self, f: pd.DataFrame) -> pd.DataFrame:
        for c in CATEGORICAL_FEATURES:
            f[c] = f[c].fillna("<UNK>").astype(str)
        return f

    def _fit_lgb(self, frame: pd.DataFrame):
        X = frame[FEATURE_COLUMNS].copy()
        for c in CATEGORICAL_FEATURES:
            X[c] = X[c].astype("category")

        common = dict(
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            random_state=self.seed,
            verbose=-1,
            n_jobs=-1,
        )

        if self.objective == "pointwise":
            model = lgb.LGBMClassifier(objective="binary", **common)
            model.fit(X, frame["label"].astype(int), sample_weight=frame["sample_weight"])
            self._is_classifier = True
            return model

        # LightGBM requires rows grouped by query (user) for ranking objectives.
        frame = frame.sort_values("user_id", kind="stable").reset_index(drop=True)
        X = frame[FEATURE_COLUMNS].copy()
        for c in CATEGORICAL_FEATURES:
            X[c] = X[c].astype("category")
        group = frame.groupby("user_id", sort=False).size().tolist()

        params = {
            "objective": OBJECTIVES[self.objective],
            "label_gain": LABEL_GAIN,
            **common,
        }
        if self.objective == "listwise":
            params["metric"] = "ndcg"
        model = lgb.LGBMRanker(**params)
        model.fit(X, frame["label"].astype(int), group=group)
        self._is_classifier = False
        return model

    def _fit_cold(self) -> None:
        """Segment (country x age x platform) and global popularity over train."""
        train = self._train
        us = self._builder._user_static

        self._global_top = (train.groupby("app_id", sort=False).size()
                            .sort_values(ascending=False).index.tolist())

        seg = train[["user_id", "app_id"]].join(
            us[["user_platform", "user_age_bucket", "user_country"]], on="user_id", how="left")
        seg["user_platform"] = seg["user_platform"].fillna("<UNK>").astype(str)
        seg["user_age_bucket"] = seg["user_age_bucket"].fillna("<UNK>").astype(str)
        seg["user_country"] = seg["user_country"].fillna("<UNK>").astype(str)
        seg["_cap"] = seg["user_country"] + "|" + seg["user_age_bucket"] + "|" + seg["user_platform"]

        cold: dict[tuple, list[str]] = {}
        for key, grp in seg.groupby("_cap", sort=False):
            order = grp.groupby("app_id", sort=False).size().sort_values(ascending=False).index
            a, b, c = key.split("|")
            cold[(a, b, c)] = order.tolist()
        self._cold_top = cold

    def recommend(self, user_id: str, n: int = 10) -> list[str]:
        """Ranked app_ids for a user; never raises, never empty."""
        try:
            return self._recommend_inner(user_id, n)
        except Exception:
            return self._cold_recommend(user_id, n)

    def _recommend_inner(self, user_id: str, n: int) -> list[str]:
        if self._model is None or self._builder is None:
            return self._cold_recommend(user_id, n)
        hist = self._user_seen_apps.get(user_id, set())
        if user_id not in self._user_has_history or len(hist) < MIN_HISTORY_FOR_MODEL:
            return self._cold_recommend(user_id, n)

        cands = self._builder.candidates_for_user(user_id)
        if cands.empty or self._model is None:
            return self._cold_recommend(user_id, n)

        X = cands[FEATURE_COLUMNS].copy()
        for c in CATEGORICAL_FEATURES:
            X[c] = X[c].astype("category")

        if self._is_classifier:
            scores = self._model.predict_proba(X)[:, 1]
        else:
            scores = self._model.predict(X)

        order = np.argsort(-scores, kind="stable")
        # pure score ranking: re-engaging an installed app is a valid outcome.
        ranked = cands["app_id"].iloc[order].tolist()
        return ranked[:n]

    def _cold_recommend(self, user_id: str, n: int) -> list[str]:
        top: list[str] = self._global_top
        if self._builder is not None:
            seg = self._segment_key(user_id)
            top = self._cold_top.get(seg) or self._global_top
        hist = self._user_seen_apps.get(user_id, set())
        top = [a for a in top if a not in hist] or top
        return top[:n]

    def _segment_key(self, user_id: str) -> tuple:
        us = self._builder._user_static
        if us is not None and user_id in us.index:
            u = us.loc[user_id]
            return (str(u["user_country"]), str(u["user_age_bucket"]), str(u["user_platform"]))
        return ()