"""Build training examples from positions JSONL (produced by parse_pgns.py).

Produces a compressed JSONL where each line is a candidate move example with
position, move, features, delta features, and a binary label (1 if played).

Usage:
  python -m tools.build_training_examples --in game-data/positions.jsonl.gz --out game-data/training_examples.jsonl.gz --negatives 7
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import Dict, List

import chess


PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}


def material_count(board: chess.Board) -> Dict[str, float]:
    white = 0.0
    black = 0.0
    for piece_type, value in PIECE_VALUES.items():
        white += len(board.pieces(piece_type, chess.WHITE)) * value
        black += len(board.pieces(piece_type, chess.BLACK)) * value
    return {'white': white, 'black': black, 'diff': white - black, 'total': white + black}


def phase_score(mat_total: float) -> float:
    # approximate max material (both sides): 39 (9+5+3+3+1)*2 = 38? We'll use 39 for margin
    return float(mat_total) / 39.0


def move_features(board: chess.Board, move: chess.Move) -> Dict:
    mf = {
        'uci': move.uci(),
        'is_capture': board.is_capture(move),
        'is_check': board.gives_check(move),
        'is_promotion': move.promotion is not None,
        'promotion': chess.piece_name(move.promotion) if move.promotion else None,
        'from_file': move.from_square % 8,
        'from_rank': move.from_square // 8,
        'to_file': move.to_square % 8,
        'to_rank': move.to_square // 8,
        'piece': chess.piece_name(board.piece_type_at(move.from_square)) if board.piece_type_at(move.from_square) else None,
    }
    return mf


def delta_features(board: chess.Board, move: chess.Move) -> Dict:
    before = material_count(board)
    board_push = board.copy()
    board_push.push(move)
    after = material_count(board_push)
    return {
        'white_after': after['white'],
        'black_after': after['black'],
        'diff_after': after['diff'],
        'white_before': before['white'],
        'black_before': before['black'],
        'diff_before': before['diff'],
        'diff_delta': after['diff'] - before['diff'],
    }


def process_positions(inp: Path, outp: Path, negatives: int) -> int:
    count = 0
    with gzip.open(inp, 'rt', encoding='utf-8') as inf, gzip.open(outp, 'wt', encoding='utf-8') as outf:
        for line in inf:
            rec = json.loads(line)
            chosen = rec.get('chosen_move')
            legal = rec.get('legal_moves') or []
            if not chosen:
                continue
            if chosen not in legal:
                # Skip inconsistent position
                continue

            # sample negatives
            negatives_pool = [m for m in legal if m != chosen]
            sample_neg = random.sample(negatives_pool, k=min(len(negatives_pool), negatives))
            candidates = [chosen] + sample_neg

            board = chess.Board(rec['fen'])
            pos_mat = material_count(board)
            pos_phase = phase_score(pos_mat['total'])

            for mv in candidates:
                move = chess.Move.from_uci(mv)
                if move not in board.legal_moves:
                    continue
                mf = move_features(board, move)
                df = delta_features(board, move)
                label = 1 if mv == chosen else 0
                out = {
                    'game_id': rec.get('game_id'),
                    'ply': rec.get('ply'),
                    'side_to_move': rec.get('side_to_move'),
                    'fen': rec.get('fen'),
                    'move': mv,
                    'label': label,
                    'position_features': pos_mat,
                    'move_features': mf,
                    'delta_features': df,
                    'phase': pos_phase,
                }
                outf.write(json.dumps(out) + '\n')
                count += 1
    return count


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='inp', default='game-data/positions.jsonl.gz')
    p.add_argument('--out', dest='out', default='game-data/training_examples.jsonl.gz')
    p.add_argument('--negatives', type=int, default=7)
    args = p.parse_args()

    inp = Path(args.inp)
    outp = Path(args.out)
    if not inp.exists():
        print('Input positions file not found:', inp)
        return 1

    print('Building training examples from', inp)
    n = process_positions(inp, outp, args.negatives)
    print('Wrote', n, 'examples to', outp)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
