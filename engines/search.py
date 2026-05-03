"""Search components for homemade engines."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import cast

import chess
import chess.engine
from chess.engine import PlayResult

from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import MOVE

from .evaluator import PositionEvaluator
from .time_manager import TimeManager


@dataclass(slots=True)
class SearchComponents:
    """Bundled dependencies for the homemade engine."""

    evaluator: PositionEvaluator = field(default_factory=PositionEvaluator)
    time_manager: TimeManager = field(default_factory=TimeManager)


class RandomMoveSearch:
    """Pick a legal move at random."""

    def choose(self, board: chess.Board, root_moves: MOVE | object) -> chess.Move:
        possible_moves = root_moves if isinstance(root_moves, list) and root_moves else list(board.legal_moves)
        if not possible_moves:
            raise chess.engine.EngineError("No legal moves are available.")
        return random.choice(possible_moves)


class RandomMove(MinimalEngine):
    """A thin homemade engine wrapper that can grow into a full searcher."""

    def __init__(self, commands, options, stderr, draw_or_resign, game, debug, **popen_args: str) -> None:
        super().__init__(commands, options, stderr, draw_or_resign, game, debug, **popen_args)
        self.components = SearchComponents()
        self.searcher = RandomMoveSearch()

    def search(self, board: chess.Board, *args: object) -> PlayResult:
        """Choose a legal move using the reusable search components."""
        if len(args) == 4:
            time_limit = cast(chess.engine.Limit, args[0])
            draw_offered = cast(bool, args[2])
            root_moves = cast(MOVE | object, args[3])
        else:
            time_limit = chess.engine.Limit()
            draw_offered = False
            root_moves = []

        _ = self.components.time_manager.allocate(board, time_limit)
        move = self.searcher.choose(board, root_moves)

        _ = self.components.evaluator.score(board)

        return PlayResult(move, None, draw_offered=draw_offered)
