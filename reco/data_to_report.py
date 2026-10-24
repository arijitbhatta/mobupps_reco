"""CLI: print the data-quality report.

Usage: ``python -m reco.data_to_report [--data-dir PATH]``
"""

from __future__ import annotations

import argparse

from .data import load_tables, validate


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Print the data-quality report.")
    p.add_argument("--data-dir", default=None)
    args = p.parse_args(argv)

    users, apps, interactions = load_tables(args.data_dir)
    rep = validate(users, apps, interactions)

    print("=" * 70)
    print(f"Data-quality report — {rep.summary()}")
    print("=" * 70)
    for f in rep.findings:
        tag = f"{f.severity.upper():8s} [{f.code}] ({f.table})"
        print(f"\n{tag}")
        print(f"  {f.description}")
        if f.n_affected:
            print(f"  affected: {f.n_affected:,}")
        if f.action:
            print(f"  decision: {f.action}")


if __name__ == "__main__":
    main()