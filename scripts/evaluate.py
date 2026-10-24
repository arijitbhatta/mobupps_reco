"""
Adaxon Take-Home — Evaluation Script

This script is provided to candidates. Use it to evaluate your recommender.

USAGE:
    1. Implement a class with a `recommend(user_id: str, n: int) -> list[str]` method.
       The method must return a list of n app_ids ordered by predicted relevance
       (most relevant first).

    2. Pass an instance of your recommender to `evaluate()`:

        from evaluate import evaluate, load_test_split
        from your_module import YourRecommender

        train_df, test_df = load_test_split("data/interactions.csv")
        model = YourRecommender()
        model.fit(train_df)  # however your model is trained

        results = evaluate(model, test_df, k=10)
        print(results)

USING THIS SCRIPT IS OPTIONAL — you can build your own evaluation harness if you
prefer. We provide it so you don't have to spend time on this. We are NOT
evaluating you on your offline metrics. We DO want to see that you understand
what these metrics mean and what their limitations are.

The split is leave-last-N-out per user: for each user with >= 5 interactions,
the last 20% of their interactions (by timestamp) become the test set.
"""
from __future__ import annotations

import math
from typing import Protocol

import pandas as pd


class Recommender(Protocol):
    """Minimal interface your recommender must satisfy."""

    def recommend(self, user_id: str, n: int) -> list[str]:
        """Return a ranked list of n app_ids (most relevant first)."""
        ...


def load_test_split(
    interactions_path: str = "data/interactions.csv",
    test_fraction: float = 0.2,
    min_interactions: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based leave-last-N-out split per user.

    Returns (train_df, test_df).

    Users with fewer than `min_interactions` are kept entirely in train and
    excluded from the test set (they are cold-start cases — handle separately).

    NOTE: This split is intentionally simple. In production you would want a
    fixed time cutoff (e.g., everything before date X is train) to avoid
    target leakage. We accept the simpler per-user split for this exercise.
    """
    df = pd.read_csv(interactions_path)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df.dropna(subset=["event_time"])
    df = df.sort_values(["user_id", "event_time"])

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for user_id, group in df.groupby("user_id", sort=False):
        if len(group) < min_interactions:
            train_parts.append(group)
            continue
        n_test = max(1, int(len(group) * test_fraction))
        train_parts.append(group.iloc[:-n_test])
        test_parts.append(group.iloc[-n_test:])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True) if test_parts else pd.DataFrame(columns=df.columns)

    return train_df, test_df


def recall_at_k(predicted: list[str], actual: set[str], k: int) -> float:
    """Fraction of held-out items that appear in the top-k predictions."""
    if not actual:
        return 0.0
    top_k = predicted[:k]
    hits = sum(1 for item in top_k if item in actual)
    return hits / len(actual)


def ndcg_at_k(predicted: list[str], actual: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k. Binary relevance."""
    if not actual:
        return 0.0
    dcg = 0.0
    for i, item in enumerate(predicted[:k]):
        if item in actual:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because positions are 1-indexed and log2(1)=0
    ideal_hits = min(len(actual), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(
    model: Recommender,
    test_df: pd.DataFrame,
    k: int = 10,
    verbose: bool = True,
) -> dict[str, float]:
    """Evaluate a recommender on the test set.

    Computes Recall@k and NDCG@k, averaged over users that have at least one
    held-out interaction. Users for which the model returns an empty list are
    counted (with score 0) — handling unknown users is part of the model's job.

    Returns:
        Dict with keys:
            - recall_at_k
            - ndcg_at_k
            - n_users_evaluated
            - n_users_with_empty_predictions
    """
    user_actuals: dict[str, set[str]] = (
        test_df.groupby("user_id")["app_id"].apply(set).to_dict()
    )

    recalls: list[float] = []
    ndcgs: list[float] = []
    empty_preds = 0

    for user_id, actual in user_actuals.items():
        try:
            predicted = model.recommend(user_id, k)
        except Exception as exc:
            if verbose:
                print(f"  [warn] recommend() raised for {user_id}: {exc!r}")
            predicted = []

        if not predicted:
            empty_preds += 1

        recalls.append(recall_at_k(predicted, actual, k))
        ndcgs.append(ndcg_at_k(predicted, actual, k))

    n = len(recalls)
    results = {
        "recall_at_k": sum(recalls) / n if n else 0.0,
        "ndcg_at_k": sum(ndcgs) / n if n else 0.0,
        "n_users_evaluated": n,
        "n_users_with_empty_predictions": empty_preds,
    }

    if verbose:
        print("=" * 50)
        print(f"Evaluation results (k={k})")
        print("=" * 50)
        print(f"  Recall@{k}:  {results['recall_at_k']:.4f}")
        print(f"  NDCG@{k}:    {results['ndcg_at_k']:.4f}")
        print(f"  Users evaluated:           {results['n_users_evaluated']:,}")
        print(f"  Users with empty preds:    {results['n_users_with_empty_predictions']:,}")
        print("=" * 50)

    return results


if __name__ == "__main__":
    # Smoke test: a popularity baseline
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions", default="data/interactions.csv")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    print(f"Loading and splitting {args.interactions}...")
    train_df, test_df = load_test_split(args.interactions)
    print(f"  Train: {len(train_df):,} interactions")
    print(f"  Test:  {len(test_df):,} interactions")
    print(f"  Test users: {test_df['user_id'].nunique():,}")

    class PopularityBaseline:
        """Returns the globally most popular apps. A weak baseline; beat it."""

        def __init__(self) -> None:
            self.top_apps: list[str] = []

        def fit(self, df: pd.DataFrame) -> None:
            self.top_apps = df["app_id"].value_counts().index.tolist()

        def recommend(self, user_id: str, n: int) -> list[str]:
            return self.top_apps[:n]

    baseline = PopularityBaseline()
    baseline.fit(train_df)
    evaluate(baseline, test_df, k=args.k)
