"""Time allocation helpers for homemade engines."""

from __future__ import annotations

import chess
import chess.engine


class TimeManager:
    """Placeholder for future style-aware time management."""

    def allocate(self, board: chess.Board, time_limit: chess.engine.Limit) -> chess.engine.Limit:  # noqa: ARG002
        """Return the time limit that should be used for the current move."""
        return time_limit
