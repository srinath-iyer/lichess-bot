"""Search components for homemade engines."""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import cast

import chess
import chess.engine
from chess.engine import PlayResult

from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import MOVE

from .evaluator import PIECE_VALUES, PositionEvaluator
from .policy_model import PolicyModelRuntime
from .time_manager import TimeManager
from typing import Optional
import math


logger = logging.getLogger(__name__)


class SearchTimeout(Exception):
    """Raised when the search budget is exhausted."""


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


class PolicyMoveSearch:
    """Select moves using model probabilities, with a weighted-random fallback."""

    def __init__(self, model_path: str | Path, threshold: float = 0.5) -> None:
        self.runtime = PolicyModelRuntime(model_path=model_path, threshold=threshold)
        self.rng = random.Random()

    def choose(self, board: chess.Board, root_moves: MOVE | object) -> chess.Move:
        possible_moves = root_moves if isinstance(root_moves, list) and root_moves else list(board.legal_moves)
        if not possible_moves:
            raise chess.engine.EngineError("No legal moves are available.")
        return self.runtime.choose_move(board, cast(list[chess.Move], possible_moves), rng=self.rng)


class AlphaBetaSearch:
    """Simple negamax/alpha-beta search with optional policy-based move ordering.

    This is intentionally small and safe: it performs a fixed-depth search and
    uses the provided `PolicyModelRuntime` to order moves (higher prob first).
    """

    def __init__(
        self,
        evaluator: PositionEvaluator,
        depth: int = 5,
        policy_runtime: Optional[PolicyModelRuntime] = None,
        use_policy_ordering: bool = True,
        time_limit_seconds: float = 2.75,
    ) -> None:
        self.evaluator = evaluator
        self.depth = int(depth)
        self.policy = policy_runtime
        self.use_policy_ordering = bool(use_policy_ordering)
        self.time_limit_seconds = float(time_limit_seconds)

    def choose(self, board: chess.Board, root_moves: MOVE | object) -> chess.Move:
        # Always regenerate legal moves from the board to ensure consistency
        # (ignore the root_moves parameter to avoid state corruption issues)
        root_turn = board.turn
        possible_moves = list(board.legal_moves)
        if not possible_moves:
            raise chess.engine.EngineError("No legal moves are available.")

        moves = self._order_moves(board, possible_moves, use_policy=self.policy is not None and self.use_policy_ordering)

        deadline = time.monotonic() + self.time_limit_seconds
        best_move = moves[0]
        best_score = -math.inf
        final_depth_results: list[tuple[chess.Move, float, float]] = []

        try:
            for search_depth in range(1, max(self.depth, 1) + 1):
                depth_best_move = best_move
                depth_best_score = -math.inf
                alpha = -math.inf
                beta = math.inf
                depth_results: list[tuple[chess.Move, float, float]] = []

                for mv in moves:
                    self._check_timeout(deadline)
                    board.push(mv)
                    try:
                        score = -self._negamax(board, search_depth - 1, -beta, -alpha, deadline)
                    finally:
                        board.pop()

                    depth_results.append((mv, score, 0))

                    if score > depth_best_score:
                        depth_best_score = score
                        depth_best_move = mv

                    alpha = max(alpha, score)

                best_move = depth_best_move
                best_score = depth_best_score
                final_depth_results = depth_results
        except SearchTimeout:
            pass

        if final_depth_results:
            ranked = sorted(final_depth_results, key=lambda item: item[1], reverse=True)[:3]

            def perspective_scores(score: float) -> tuple[float, float]:
                if root_turn == chess.WHITE:
                    return score, -score
                return -score, score

            move_summaries = []
            for move, score, _static_eval in ranked:
                white_eval, black_eval = perspective_scores(score)
                move_summaries.append(f"{move.uci()} White={white_eval:.1f} Black={black_eval:.1f}")

            logger.warning(
                "Move %s top 3: %s | best=%s White=%.1f Black=%.1f",
                "White" if root_turn == chess.WHITE else "Black",
                ", ".join(move_summaries),
                best_move.uci() if best_move is not None else "none",
                perspective_scores(best_score)[0],
                perspective_scores(best_score)[1],
            )

        _ = best_score
        return best_move if best_move is not None else random.choice(possible_moves)

    def analyze_root(
        self,
        board: chess.Board,
        max_depth: int = 3,
        breadth: int = 3,
        time_limit_seconds: float | None = None,
    ) -> dict[str, object]:
        """Return a compact analysis tree for the current root position.

        The result is JSON-serializable and intended for a UI that wants to show
        the top candidate moves plus a hoverable reply tree.
        """
        analysis_depth = max(1, int(max_depth))
        analysis_breadth = max(1, int(breadth))
        budget = self.time_limit_seconds if time_limit_seconds is None else min(self.time_limit_seconds, float(time_limit_seconds))
        deadline = time.monotonic() + max(budget, 0.1)
        try:
            return self._analyze_node(board, analysis_depth, analysis_breadth, deadline, use_policy=self.policy is not None and self.use_policy_ordering)
        except SearchTimeout:
            return self._node_snapshot(board)

    def _check_timeout(self, deadline: float) -> None:
        if time.monotonic() >= deadline:
            raise SearchTimeout()

    def _node_snapshot(self, board: chess.Board) -> dict[str, object]:
        score_for_side_to_move = float(self.evaluator.score(board))
        if board.turn == chess.WHITE:
            white_eval = score_for_side_to_move
            black_eval = -score_for_side_to_move
        else:
            white_eval = -score_for_side_to_move
            black_eval = score_for_side_to_move

        return {
            "fen": board.fen(),
            "turn": "white" if board.turn == chess.WHITE else "black",
            "score_for_side_to_move": score_for_side_to_move,
            "white_eval": white_eval,
            "black_eval": black_eval,
            "children": [],
        }

    def _analyze_node(
        self,
        board: chess.Board,
        depth: int,
        breadth: int,
        deadline: float,
        use_policy: bool,
    ) -> dict[str, object]:
        self._check_timeout(deadline)

        node = self._node_snapshot(board)
        if depth <= 0 or board.is_game_over():
            return node

        moves = self._order_moves(board, list(board.legal_moves), use_policy=use_policy)
        children: list[dict[str, object]] = []

        for mv in moves[:breadth]:
            self._check_timeout(deadline)
            san = board.san(mv)
            board.push(mv)
            try:
                child_node = self._analyze_node(board, depth - 1, breadth, deadline, use_policy=False)
            finally:
                board.pop()

            move_score = -float(cast(float, child_node["score_for_side_to_move"]))
            if board.turn == chess.WHITE:
                white_eval = move_score
                black_eval = -move_score
            else:
                white_eval = -move_score
                black_eval = move_score

            children.append({
                "uci": mv.uci(),
                "san": san,
                "score_for_side_to_move": move_score,
                "white_eval": white_eval,
                "black_eval": black_eval,
                "child": child_node,
            })

        children.sort(key=lambda item: cast(float, item["score_for_side_to_move"]), reverse=True)
        node["children"] = children
        if children:
            node["best_move"] = children[0]["uci"]
            node["best_score_for_side_to_move"] = children[0]["score_for_side_to_move"]
        return node

    def _order_moves(self, board: chess.Board, moves: list[chess.Move], use_policy: bool) -> list[chess.Move]:
        if not moves:
            return moves

        if use_policy:
            try:
                probs = self.policy.get_move_probabilities(board, moves) if self.policy else []
                if probs:
                    return [move for move, _ in sorted(zip(moves, probs), key=lambda item: item[1], reverse=True)]
            except Exception:
                pass

        def move_key(move: chess.Move) -> tuple[int, int, int]:
            gives_check = 1 if board.gives_check(move) else 0
            is_capture = 1 if board.is_capture(move) else 0
            capture_value = 0
            if is_capture:
                captured_piece = board.piece_at(move.to_square)
                capture_value = 0 if captured_piece is None else PIECE_VALUES.get(captured_piece.piece_type, 0)
            return (gives_check, is_capture, capture_value)

        return sorted(moves, key=move_key, reverse=True)

    def _negamax(self, board, depth, alpha, beta, deadline):
        self._check_timeout(deadline)
        if depth <= 0 or board.is_game_over():
            return float(self.evaluator.score(board))  # already side-to-move perspective

        moves = self._order_moves(board, list(board.legal_moves), use_policy=False)

        value = -math.inf
        for mv in moves:
            self._check_timeout(deadline)
            board.push(mv)
            try:
                score = -self._negamax(board, depth - 1, -beta, -alpha, deadline)
            finally:
                board.pop()

            value = max(value, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        return value


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

        return PlayResult(move, None, draw_offered=draw_offered)


class PolicyMove(MinimalEngine):
    """Homemade engine backed by a trained move-classification policy model."""

    def __init__(self, commands, options, stderr, draw_or_resign, game, debug, **popen_args: str) -> None:
        super().__init__(commands, options, stderr, draw_or_resign, game, debug, **popen_args)
        self.components = SearchComponents()

        model_path_raw = options.pop("model_path", "engines/train_eval/model_artifacts/policy_xgb.joblib")
        model_path = model_path_raw if isinstance(model_path_raw, (str, Path)) else "engines/train_eval/model_artifacts/policy_xgb.joblib"
        threshold_raw = options.pop("threshold", 0.5)
        try:
            threshold = float(threshold_raw) if isinstance(threshold_raw, (str, int, float)) else 0.5
        except (TypeError, ValueError):
            threshold = 0.5

        # Build a policy runtime and an alpha-beta searcher that uses it for ordering
        policy_runtime = PolicyModelRuntime(model_path=model_path, threshold=threshold)

        # Allow an optional search depth override from engine options
        depth_raw = options.pop("search_depth", 3)
        try:
            depth = int(str(depth_raw)) if not isinstance(depth_raw, (int, float, str)) else int(depth_raw)
        except (TypeError, ValueError):
            depth = 3

        timeout_raw = options.pop("search_timeout_seconds", 2.75)
        try:
            search_timeout_seconds = float(str(timeout_raw)) if not isinstance(timeout_raw, (int, float, str)) else float(timeout_raw)
        except (TypeError, ValueError):
            search_timeout_seconds = 2.75

        use_ordering_raw = options.pop("use_policy_ordering", True)
        use_ordering = bool(use_ordering_raw)

        self.searcher = AlphaBetaSearch(
            evaluator=self.components.evaluator,
            depth=depth,
            policy_runtime=policy_runtime,
            use_policy_ordering=use_ordering,
            time_limit_seconds=search_timeout_seconds,
        )

    def search(self, board: chess.Board, *args: object) -> PlayResult:
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

        return PlayResult(move, None, draw_offered=draw_offered)
