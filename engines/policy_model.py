"""Inference helpers for the trained move-selection policy model."""

from __future__ import annotations

import random
import logging
from pathlib import Path
from typing import Any

import chess
import numpy as np

from .policy_features import extract_position_features, get_detailed_move_properties, numeric_delta


logger = logging.getLogger(__name__)

try:
    import joblib
except Exception:  # pragma: no cover - handled via runtime fallback in caller
    joblib = None

try:
    import pickle
except Exception:
    pickle = None


PIECE_TYPE_MAP = {
    "p": "pawn",
    "n": "knight",
    "b": "bishop",
    "r": "rook",
    "q": "queen",
    "k": "king",
}
EXPECTED_PIECE_COLUMNS = ["pawn", "knight", "bishop", "rook", "queen", "king"]


class PolicyModelRuntime:
    """Load a persisted model bundle and run per-move inference."""

    def __init__(self, model_path: str | Path, threshold: float = 0.5) -> None:
        self.model_path = Path(model_path)
        self.threshold = float(threshold)
        self.available = False
        self.error: str | None = None

        self.model: Any = None
        self.scaler: Any = None
        self.feature_columns: list[str] = []
        self.numeric_columns: list[str] = []

        self._load()

    def _load(self) -> None:
        import os
        logger.warning("CWD during model load: %s", os.getcwd())
        logger.warning("Resolved model path: %s", self.model_path.resolve())
        if not self.model_path.exists():
            self.error = f"Model file does not exist: {self.model_path}"
            return

        # Check for cached pickle file first (much faster loading)
        cache_path = self.model_path.with_suffix(".pkl")
        if cache_path.exists() and pickle:
            try:
                logger.info("Loading policy model from cache: %s", cache_path)
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                self.model = data["model"]
                self.scaler = data["scaler"]
                self.feature_columns = list(data["feature_columns"])
                self.numeric_columns = list(data["numeric_columns"])
                self.available = True
                logger.warning(
                    "Loaded policy model from cache with %d features (%d numeric).",
                    len(self.feature_columns),
                    len(self.numeric_columns),
                )
                return
            except Exception as exc:
                logger.warning("Failed to load from cache: %s; trying joblib", exc)

        # Load from joblib and cache for next time
        if joblib is None:
            self.error = "joblib is not installed in this environment."
            return

        try:
            logger.info("Loading policy model from joblib: %s", self.model_path)
            bundle = joblib.load(self.model_path)
            self.model = bundle["model"]
            self.scaler = bundle["scaler"]
            self.feature_columns = list(bundle["feature_columns"])
            self.numeric_columns = list(bundle["numeric_columns"])
            self.available = True
            
            # Try to save as pickle cache for faster loading next time
            if pickle:
                try:
                    cache_data = {
                        "model": self.model,
                        "scaler": self.scaler,
                        "feature_columns": self.feature_columns,
                        "numeric_columns": self.numeric_columns,
                    }
                    with open(cache_path, "wb") as f:
                        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.info("Cached policy model to %s for faster future loading", cache_path)
                except Exception as exc:
                    logger.warning("Failed to cache model: %s", exc)
            
            logger.warning(
                "Loaded policy model from joblib with %d features (%d numeric).",
                len(self.feature_columns),
                len(self.numeric_columns),
            )
        except Exception as exc:  # pragma: no cover - runtime fallback path
            self.error = f"Failed to load model bundle: {exc}"
            self.available = False

    def _features_for_move(self, board: chess.Board, move: chess.Move) -> dict[str, Any]:
        move_props = get_detailed_move_properties(board, move, is_chosen=False)

        current_position = extract_position_features(board)
        next_board = board.copy(stack=False)
        next_board.push(move)
        next_position = extract_position_features(next_board)
        delta_position = numeric_delta(current_position, next_position)

        row = {
            **move_props,
            **{f"position_{k}": v for k, v in current_position.items()},
            **delta_position,
        }

        raw_piece_type = str(row.get("piece_type", "")).lower().strip()
        normalized_piece = PIECE_TYPE_MAP.get(raw_piece_type, raw_piece_type)
        for piece in EXPECTED_PIECE_COLUMNS:
            row[f"is_{piece}_move"] = 1 if normalized_piece == piece else 0
        row.pop("piece_type", None)
        row.pop("is_chosen_move", None)

        return row

    def _prepare_batch(self, board: chess.Board, candidate_moves: list[chess.Move]) -> np.ndarray:
        """Prepare feature vectors for candidate moves as a numpy array."""
        rows = [self._features_for_move(board, mv) for mv in candidate_moves]
        
        # Create a list of vectors with features in the correct order
        feature_vectors = []
        for row in rows:
            vector = []
            for col in self.feature_columns:
                value = row.get(col, 0)
                vector.append(value)
            feature_vectors.append(vector)
        
        features_array = np.array(feature_vectors, dtype=object)
        
        # Convert numeric columns to float
        if self.numeric_columns:
            cols_to_scale = [i for i, c in enumerate(self.feature_columns) if c in self.numeric_columns]
            if cols_to_scale:
                # Extract numeric columns, convert to float, and scale
                numeric_data = []
                for row_idx, row in enumerate(feature_vectors):
                    numeric_row = []
                    for col_idx in cols_to_scale:
                        try:
                            val = float(row[col_idx]) if row[col_idx] is not None else 0.0
                        except (TypeError, ValueError):
                            val = 0.0
                        numeric_row.append(val)
                    numeric_data.append(numeric_row)
                
                numeric_array = np.array(numeric_data, dtype=float)
                scaled_values = self.scaler.transform(numeric_array)
                
                # Put scaled values back
                for row_idx, scaled_row in enumerate(scaled_values):
                    for i, col_idx in enumerate(cols_to_scale):
                        feature_vectors[row_idx][col_idx] = scaled_row[i]
                
                features_array = np.array(feature_vectors, dtype=float)
            else:
                features_array = np.array(feature_vectors, dtype=float)
        else:
            features_array = np.array(feature_vectors, dtype=float)
        
        return features_array

    def choose_move(self, board: chess.Board, candidate_moves: list[chess.Move], rng: random.Random | None = None) -> chess.Move:
        if not candidate_moves:
            raise chess.engine.EngineError("No legal moves are available.")

        random_source = rng if rng is not None else random

        if not self.available:
            return random_source.choice(candidate_moves)

        features_df = self._prepare_batch(board, candidate_moves)
        probabilities = self.model.predict_proba(features_df)[:, 1]

        ranked_moves = sorted(
            zip(candidate_moves, probabilities),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        top_five = ranked_moves[:5]
        logger.warning(
            "Top 5 moves by probability for position %s: %s",
            board.fen(),
            ", ".join(f"{move.uci()}={float(probability):.4f}" for move, probability in top_five),
        )

        weighted_candidates = [
            (move, max(float(probability), 0.0))
            for move, probability in zip(candidate_moves, probabilities)
        ]
        positive_candidates = [
            (move, weight)
            for move, weight in weighted_candidates
            if weight >= self.threshold
        ]

        selection_pool = positive_candidates if positive_candidates else weighted_candidates
        moves = [move for move, _ in selection_pool]
        weights = [weight for _, weight in selection_pool]

        if not any(weights):
            return random_source.choice(candidate_moves)

        return random_source.choices(moves, weights=weights, k=1)[0]

    def get_move_probabilities(self, board: chess.Board, candidate_moves: list[chess.Move]) -> list[float]:
        """Return the model probability for each candidate move.

        If the model is unavailable or an error occurs, return uniform weights (1.0).
        """
        if not candidate_moves:
            return []

        if not self.available:
            return [1.0 for _ in candidate_moves]

        try:
            features_df = self._prepare_batch(board, candidate_moves)
            probs = self.model.predict_proba(features_df)[:, 1]
            return [float(p) for p in probs]
        except Exception:
            return [1.0 for _ in candidate_moves]
