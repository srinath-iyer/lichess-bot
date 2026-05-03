"""Evaluation helpers for homemade engines."""

from __future__ import annotations

from dataclasses import dataclass

import chess


@dataclass(slots=True)
class EvaluatorConfig:
    """Configuration for a future Texel-tuned evaluator."""

    random_seed: int | None = None


class PositionEvaluator:
    """Placeholder for a future custom evaluator."""

    def score(self, board: chess.Board) -> int:  # noqa: ARG002
        """Return a centipawn score for the current position."""
        return 0


