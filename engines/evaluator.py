"""Evaluation helpers for homemade engines."""

from __future__ import annotations

from dataclasses import dataclass

import chess


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

CHECK_BONUS = 15
CHECKMATE_SCORE = 100_000


@dataclass(slots=True)
class EvaluatorConfig:
    """Configuration for a future Texel-tuned evaluator."""

    random_seed: int | None = None


class PositionEvaluator:
    """Simple material-plus-tactics evaluator."""

    def score_components(self, board: chess.Board) -> tuple[int, int, int]:
        """Return the material and check components separately (third is unused)."""
        if board.is_checkmate():
            mate_score = CHECKMATE_SCORE if board.turn == chess.WHITE else -CHECKMATE_SCORE
            return (mate_score, 0, 0)

        if board.is_stalemate() or board.is_insufficient_material():
            return (0, 0, 0)

        material_score = 0
        for piece_type, value in PIECE_VALUES.items():
            if piece_type == chess.KING:
                continue
            material_score += len(board.pieces(piece_type, chess.WHITE)) * value
            material_score -= len(board.pieces(piece_type, chess.BLACK)) * value

        check_score = 0
        if board.is_check():
            check_score = -CHECK_BONUS

        return material_score, check_score, 0

    def score(self, board: chess.Board) -> int:
        """Return a centipawn score from the side-to-move's perspective.
        
        Positive = side-to-move is winning.
        Negative = side-to-move is losing.
        """
        material_score, check_score, capture_score = self.score_components(board)
        white_perspective_score = material_score + check_score + capture_score
        
        # Convert from White's perspective to side-to-move's perspective
        if board.turn == chess.WHITE:
            return white_perspective_score
        else:
            return -white_perspective_score


