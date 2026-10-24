"""Convenience entry point: ``python -m reco`` dispatches to the evaluator."""

from __future__ import annotations

from .evaluate import main

if __name__ == "__main__":
    main()