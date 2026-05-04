"""Self-play arena: two PolicyMove engines play each other and save PGN."""

import logging
import time
from pathlib import Path

import chess
import chess.pgn

from engines.search import AlphaBetaSearch
from engines.policy_model import PolicyModelRuntime
from engines.evaluator import PositionEvaluator


logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
logger = logging.getLogger(__name__)


def play_game(
    white_engine: AlphaBetaSearch,
    black_engine: AlphaBetaSearch,
    max_moves: int = 200,
    game_number: int = 1,
) -> chess.pgn.Game:
    """Play a single game between two engines.
    
    Args:
        white_engine: AlphaBetaSearch for White
        black_engine: AlphaBetaSearch for Black
        max_moves: Maximum number of moves (ply) before draw
        game_number: Game number for labeling
        
    Returns:
        A chess.pgn.Game object with the completed game
    """
    board = chess.Board()
    game = chess.pgn.Game()
    
    game.headers["Event"] = "Self-Play Arena"
    game.headers["Site"] = "LocalHost"
    game.headers["Date"] = time.strftime("%Y.%m.%d")
    game.headers["White"] = "PolicyMove (White)"
    game.headers["Black"] = "PolicyMove (Black)"
    game.headers["GameNumber"] = str(game_number)
    
    node = game
    move_count = 0
    
    logger.info(f"Starting game {game_number}...")
    
    while not board.is_game_over() and move_count < max_moves:
        engine = white_engine if board.turn == chess.WHITE else black_engine
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            break
        
        move = engine.choose(board, legal_moves)
        
        # Strict validation: check if the move is actually legal
        if move is None or not isinstance(move, chess.Move):
            logger.warning(f"Engine returned invalid move object: {move}, using random fallback")
            import random
            move = random.choice(legal_moves)
        elif move not in legal_moves:
            logger.warning(
                f"Engine returned illegal move {move.uci()} (from_sq={move.from_square}, to_sq={move.to_square}) "
                f"from {len(legal_moves)} legal moves, using random fallback"
            )
            import random
            move = random.choice(legal_moves)
        elif not board.is_pseudo_legal(move):
            logger.warning(
                f"Move {move.uci()} is in legal_moves but NOT pseudo-legal on current board, using random fallback"
            )
            import random
            move = random.choice(legal_moves)
        
        board.push(move)
        move_count += 1
        
        node = node.add_variation(move)
        
        logger.info(f"Move {(move_count + 1) // 2}: {move.uci()}")
    
    # Determine result
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            game.headers["Result"] = "0-1"
            game.headers["Termination"] = "Checkmate"
            logger.info(f"Game {game_number} result: Black wins by checkmate")
        else:
            game.headers["Result"] = "1-0"
            game.headers["Termination"] = "Checkmate"
            logger.info(f"Game {game_number} result: White wins by checkmate")
    elif board.is_stalemate():
        game.headers["Result"] = "1/2-1/2"
        game.headers["Termination"] = "Stalemate"
        logger.info(f"Game {game_number} result: Draw by stalemate")
    elif board.is_insufficient_material():
        game.headers["Result"] = "1/2-1/2"
        game.headers["Termination"] = "Insufficient material"
        logger.info(f"Game {game_number} result: Draw by insufficient material")
    elif move_count >= max_moves:
        game.headers["Result"] = "1/2-1/2"
        game.headers["Termination"] = "Max moves reached"
        logger.info(f"Game {game_number} result: Draw by move limit")
    else:
        game.headers["Result"] = "*"
        game.headers["Termination"] = "Incomplete"
        logger.info(f"Game {game_number} incomplete")
    
    game.headers["PlyCount"] = str(move_count)
    
    return game


def run_arena(num_games: int = 1, output_file: str = "self_play_games.pgn") -> None:
    """Run multiple self-play games and save to PGN.
    
    Args:
        num_games: Number of games to play
        output_file: Path to output PGN file
    """
    model_path = Path("engines/train_eval/model_artifacts/policy_xgb.joblib")
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
    
    logger.info(f"Loading policy model from {model_path}...")
    policy = PolicyModelRuntime(model_path=str(model_path), threshold=0.5)
    
    evaluator = PositionEvaluator()
    
    white_engine = AlphaBetaSearch(
        evaluator=evaluator,
        depth=4,
        policy_runtime=policy,
        use_policy_ordering=True,
        time_limit_seconds=0.1,
    )
    
    black_engine = AlphaBetaSearch(
        evaluator=evaluator,
        depth=4,
        policy_runtime=policy,
        use_policy_ordering=True,
        time_limit_seconds=0.1,
    )
    
    games = []
    
    for game_num in range(1, num_games + 1):
        game = play_game(white_engine, black_engine, game_number=game_num)
        games.append(game)
        logger.info(f"Game {game_num}/{num_games} completed\n")
    
    # Write all games to PGN file
    output_path = Path(output_file)
    with open(output_path, "w") as f:
        for game in games:
            f.write(str(game))
            f.write("\n\n")
    
    logger.info(f"Saved {num_games} game(s) to {output_path}")


if __name__ == "__main__":
    import sys
    
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    output_file = sys.argv[2] if len(sys.argv) > 2 else "self_play_games.pgn"
    
    run_arena(num_games=num_games, output_file=output_file)
