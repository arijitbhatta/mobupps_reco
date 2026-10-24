"""Evaluation harness and baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from evaluate import evaluate, load_test_split, recall_at_k, ndcg_at_k  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data import load_tables  # noqa: E402
from .model import AppRecommender  # noqa: E402


class PopularityBaseline:
    def __init__(self) -> None:
        self.top_apps: list[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        self.top_apps = df["app_id"].value_counts().index.tolist()

    def recommend(self, user_id: str, n: int) -> list[str]:
        return self.top_apps[:n]


class SegmentPopularityBaseline:
    def __init__(self, data_dir=None) -> None:
        self.data_dir = data_dir
        self._seg: dict[tuple, list[str]] = {}
        self._user_seg: dict[str, tuple] = {}
        self._global: list[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        users, _, _ = load_tables(self.data_dir)
        us = users[["user_id", "platform", "age_bucket", "country_code"]].copy()
        us["country_code"] = us["country_code"].fillna("<UNK>").astype(str)
        us["age_bucket"] = us["age_bucket"].fillna("<UNK>").astype(str)
        us["platform"] = us["platform"].astype(str)
        us = us.drop_duplicates("user_id").set_index("user_id")
        self._user_seg = {
            uid: (str(r["country_code"]), str(r["age_bucket"]), str(r["platform"]))
            for uid, r in us.iterrows()
        }

        self._global = df["app_id"].value_counts().index.tolist()

        seg = df[["user_id", "app_id"]].join(us, on="user_id", how="left")
        seg["platform"] = seg["platform"].fillna("<UNK>").astype(str)
        seg["age_bucket"] = seg["age_bucket"].fillna("<UNK>").astype(str)
        seg["country_code"] = seg["country_code"].fillna("<UNK>").astype(str)
        seg["_cap"] = seg["country_code"] + "|" + seg["age_bucket"] + "|" + seg["platform"]
        for key, grp in seg.groupby("_cap", sort=False):
            order = grp.groupby("app_id", sort=False).size().sort_values(ascending=False).index
            a, b, c = key.split("|")
            self._seg[(a, b, c)] = order.tolist()

    def recommend(self, user_id: str, n: int) -> list[str]:
        seg = self._user_seg.get(user_id)
        top = self._seg.get(seg) if seg else None
        return (top or self._global)[:n]


class RandomBaseline:
    def __init__(self, data_dir=None) -> None:
        self.data_dir = data_dir
        self.apps: list[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        _, apps, _ = load_tables(self.data_dir)
        self.apps = apps["app_id"].tolist()

    def recommend(self, user_id: str, n: int) -> list[str]:
        rng = np.random.default_rng(hash(user_id) % (2**32))
        return rng.choice(self.apps, size=min(n, len(self.apps)), replace=False).tolist()


def global_time_split(interactions_path: str, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strict global time split: train < cutoff <= test."""
    df = pd.read_csv(interactions_path)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", format="mixed")
    df = df.dropna(subset=["event_time"])
    cutoff_ts = pd.Timestamp(cutoff)
    train = df[df["event_time"] < cutoff_ts]
    test = df[df["event_time"] >= cutoff_ts]
    return train, test


INSTALL_ISH = {"af_install", "af_open", "af_complete_registration",
               "af_level_achieved", "af_purchase"}


def acquisition_recall_at_k(model, test_df: pd.DataFrame, k: int = 10) -> dict[str, float]:
    """Recall@k over install-or-beyond held-out apps only."""
    acq = test_df[test_df["event_name"].isin(INSTALL_ISH)]
    actuals = acq.groupby("user_id")["app_id"].apply(set).to_dict()

    recalls = []
    n_relevant_events = 0
    for uid, actual in actuals.items():
        if not actual:
            continue
        try:
            pred = model.recommend(uid, k)
        except Exception:
            pred = []
        n_relevant_events += len(actual)
        hits = sum(1 for a in pred[:k] if a in actual)
        recalls.append(hits / len(actual))

    return {
        "acquisition_recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "n_users_with_acquisition_signal": len(recalls),
    }


def slice_metrics(model, test_df: pd.DataFrame, k: int = 10,
                  user_feature_fn=None) -> dict[str, dict[str, float]]:
    """Per-slice Recall@k / NDCG@k.

    Slices by held-out app count (a proxy for engagement). If user_feature_fn is
    given, also slices by platform.
    """
    user_actuals = test_df.groupby("user_id")["app_id"].apply(set).to_dict()

    slices: dict[str, dict[str, list[float]]] = {}
    for uid, actual in user_actuals.items():
        try:
            pred = model.recommend(uid, k)
        except Exception:
            pred = []
        r = recall_at_k(pred, actual, k)
        n = ndcg_at_k(pred, actual, k)

        bucket = "1-2 apps" if len(actual) <= 2 else ("3-5 apps" if len(actual) <= 5 else "6+ apps")
        slices.setdefault(f"activity={bucket}", {"r": [], "n": []})
        slices[f"activity={bucket}"]["r"].append(r)
        slices[f"activity={bucket}"]["n"].append(n)

        if user_feature_fn is not None:
            info = user_feature_fn(uid) or {}
            plat = info.get("platform", "<UNK>")
            slices.setdefault(f"platform={plat}", {"r": [], "n": []})
            slices[f"platform={plat}"]["r"].append(r)
            slices[f"platform={plat}"]["n"].append(n)

    out = {}
    for name, d in slices.items():
        out[name] = {
            "recall": float(np.mean(d["r"])) if d["r"] else 0.0,
            "ndcg": float(np.mean(d["n"])) if d["n"] else 0.0,
            "n_users": len(d["r"]),
        }
    return dict(sorted(out.items()))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Evaluate the recommender variants.")
    p.add_argument("--objective", choices=["pointwise", "pairwise", "listwise", "all"],
                   default="listwise")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--interactions", default="data/interactions.csv")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--n-neg-per-user", type=int, default=5)
    p.add_argument("--fast", action="store_true", help="reduced trees for a quick run")
    args = p.parse_args(argv)

    train_df, test_df = load_test_split(args.interactions)
    print(f"Train interactions: {len(train_df):,}   Test: {len(test_df):,}")

    pop = PopularityBaseline(); pop.fit(train_df)
    seg = SegmentPopularityBaseline(args.data_dir); seg.fit(train_df)
    rnd = RandomBaseline(args.data_dir); rnd.fit(train_df)
    for name, b in [("popularity", pop), ("segment-pop", seg), ("random", rnd)]:
        r = evaluate(b, test_df, k=args.k, verbose=False)
        print(f"[baseline] {name:12s} Recall@{args.k}={r['recall_at_k']:.4f} "
              f"NDCG@{args.k}={r['ndcg_at_k']:.4f}")

    objs = ["pointwise", "pairwise", "listwise"] if args.objective == "all" else [args.objective]
    for obj in objs:
        m = AppRecommender(
            objective=obj, data_dir=args.data_dir,
            n_estimators=50 if args.fast else args.n_estimators,
            num_leaves=15 if args.fast else args.num_leaves,
            n_neg_per_user=args.n_neg_per_user,
        )
        m.fit(train_df)
        r = evaluate(m, test_df, k=args.k, verbose=False)
        a = acquisition_recall_at_k(m, test_df, k=args.k)
        print(f"[model]     {obj:12s} Recall@{args.k}={r['recall_at_k']:.4f} "
              f"NDCG@{args.k}={r['ndcg_at_k']:.4f}  "
              f"AcqRecall@{args.k}={a['acquisition_recall_at_k']:.4f}  "
              f"eval={r['n_users_evaluated']:,} empty={r['n_users_with_empty_predictions']:,}")


if __name__ == "__main__":
    main()