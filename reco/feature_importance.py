"""Feature-importance report for the fitted LightGBM variants.

Prints gain (loss reduction per feature) and split (usage count) per variant.

Run: ``python -m reco.feature_importance --objective all --top 15``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from evaluate import load_test_split  # noqa: E402
import pandas as pd  # noqa: E402

from .features import FEATURE_COLUMNS  # noqa: E402
from .model import AppRecommender  # noqa: E402


def _importance(model: AppRecommender, importance_type: str) -> pd.DataFrame:
    imp = model._model.booster_.feature_importance(importance_type=importance_type)
    return pd.DataFrame({"feature": FEATURE_COLUMNS, importance_type: imp}).sort_values(
        importance_type, ascending=False, ignore_index=True
    )


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Print LightGBM feature importances.")
    p.add_argument("--objective", choices=["pointwise", "pairwise", "listwise", "all"],
                   default="pointwise")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--interactions", default="data/interactions.csv")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--top", type=int, default=15)
    args = p.parse_args(argv)

    train_df, _ = load_test_split(args.interactions)
    objs = ["pointwise", "pairwise", "listwise"] if args.objective == "all" else [args.objective]

    for obj in objs:
        m = AppRecommender(
            objective=obj, data_dir=args.data_dir,
            n_estimators=args.n_estimators, num_leaves=args.num_leaves,
        )
        m.fit(train_df)
        gain = _importance(m, "gain")
        split = _importance(m, "split")
        print(f"\n========== {obj} (n_estimators={args.n_estimators}, num_leaves={args.num_leaves}) ==========")
        top = gain.head(args.top).merge(split.head(args.top), on="feature", how="left")
        top["split"] = top["split"].fillna(0).astype(int)
        top["gain"] = top["gain"].round(0).astype(int)
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()