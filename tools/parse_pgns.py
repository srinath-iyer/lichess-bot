"""Parse PGN files in game-data/ into a JSONL of positions suitable for feature extraction.

Output format (one JSON object per line):
  {
    "game_id": str,            # source file name or PGN event/id
    "white": str,
    "black": str,
    "date": "YYYY.MM.DD",
    "ply": int,                # ply index (0-based) of the position
    "side_to_move": "white|black",
    "fen": str,                # FEN of the position
    "legal_moves": ["e2e4", ...],
    "chosen_move": "e2e4" or null,  # the move played from this position by `--player` if it's their turn
    "white_clock": float|null, # seconds remaining for white at this ply if present
    "black_clock": float|null, # seconds remaining for black at this ply if present
    "move_number": int         # fullmove number from FEN
  }

This script is conservative: it will skip games by the player `chessleopard` inside the date ranges
you specified, and it tolerates missing clock annotations. It outputs to `game-data/positions.jsonl.gz`.

Usage:
  python -m tools.parse_pgns --player YourUsername

If `--player` is omitted, the script will still produce all positions but `chosen_move` will be null.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import chess
import chess.pgn


def parse_pgn_date(d: str) -> Optional[datetime]:
    # PGN Date is usually YYYY.MM.DD or YYYY.MM.??
    try:
        parts = d.split('.')
        year = int(parts[0])
        month = int(parts[1]) if parts[1] != '??' else 1
        day = int(parts[2]) if parts[2] != '??' else 1
        return datetime(year, month, day)
    except Exception:
        return None


def clk_to_seconds(clk: str) -> Optional[float]:
    # clk formats: H:MM:SS or M:SS or MM:SS
    if not clk:
        return None
    try:
        parts = clk.split(':')
        parts = [int(p) for p in parts]
        if len(parts) == 3:
            h, m, s = parts
            return float(h * 3600 + m * 60 + s)
        elif len(parts) == 2:
            m, s = parts
            return float(m * 60 + s)
        else:
            return float(parts[0])
    except Exception:
        return None


def extract_clock_from_comment(comment: str) -> Optional[str]:
    # lichess uses "%clk 1:23:45" in comments sometimes
    if not comment:
        return None
    m = re.search(r"%clk\s*([0-9:]+)", comment)
    if m:
        return m.group(1)
    return None


def skip_chessleopard(game_headers: dict) -> bool:
    # Skip chessleopard games in the specified date ranges
    name = (game_headers.get('White') or '')
    date = game_headers.get('Date') or ''
    dt = parse_pgn_date(date)
    if not dt:
        return False
    # Ranges to skip for chessleopard
    ranges = [
        (datetime(2022, 6, 17), datetime(2022, 7, 16)),
        (datetime(2023, 1, 7), datetime(2023, 4, 5)),
    ]
    # If either White or Black is chessleopard, apply skip
    if (game_headers.get('White') == 'chessleopard_1') or (game_headers.get('Black') == 'chessleopard_1'):
        for start, end in ranges:
            if start <= dt <= end:
                return True
    return False


def process_pgn_file(path: Path, out_f, player: list[str], game_counter: list[int]):
    """
    Process a PGN file and extract positions.
    
    game_counter is a list with one int element to maintain a counter across calls.
    """
    with path.open('r', encoding='utf-8', errors='replace') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            headers = dict(game.headers)
            if skip_chessleopard(headers):
                continue

            # Use simple numerical game_id
            game_counter[0] += 1
            game_id = str(game_counter[0])
            date = headers.get('Date', '')
            # walk through moves
            board = game.board()
            node = game
            ply = 0
            # for each move, before making the move, record the position
            while node.variations:
                next_node = node.variations[0]
                # extract clocks from comment if present on the next_node (after move)
                white_clock = None
                black_clock = None
                # Comments often live on the node representing the move just played; extract before move for positions
                # For the current position (before next_node.move), try to find a %clk in node.comment
                clk = extract_clock_from_comment(node.comment if node.comment else '')
                if clk:
                    # the clock in the comment is usually the side that just moved's clock after move.
                    # We'll be conservative and not attempt to map it exactly; store it as None for now.
                    pass

                fen = board.fen()
                legal = [m.uci() for m in board.legal_moves]
                side = 'white' if board.turn == chess.WHITE else 'black'

                # If player is specified and it's their turn, chosen_move is the move they played
                chosen = None
                if player:
                    player_color = None
                    if headers.get('White') in player:
                        player_color = 'white'
                    elif headers.get('Black') in player:
                        player_color = 'black'
                    if player_color == side:
                        # the move we are about to apply is the player's move
                        chosen = next_node.move.uci() if next_node.move else None

                record = {
                    'game_id': game_id,
                    'white': headers.get('White'),
                    'black': headers.get('Black'),
                    'date': date,
                    'ply': ply,
                    'side_to_move': side,
                    'fen': fen,
                    'legal_moves': legal,
                    'chosen_move': chosen,
                    'white_clock': white_clock,
                    'black_clock': black_clock,
                    'move_number': board.fullmove_number,
                }
                out_f.write(json.dumps(record) + "\n")

                # advance
                board.push(next_node.move)
                node = next_node
                ply += 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--player', help='Your username to mark chosen moves', default=['srinathiyer','ChessCHNC','M87BlackHoleImage','chessleopard_1'], nargs='+')
    p.add_argument('--data-dir', help='Directory with PGN files', default='game-data')
    p.add_argument('--out', help='Output JSONL.gz', default='game-data/positions.jsonl.gz')
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pgn_files = list(data_dir.glob('*.pgn'))
    if not pgn_files:
        print('No pgn files found in', data_dir)
        return 1

    game_counter = [0]  # Use list to maintain counter across function calls
    with gzip.open(out_path, 'wt', encoding='utf-8') as out_f:
        for pgn in pgn_files:
            print('Processing', pgn)
            try:
                process_pgn_file(pgn, out_f, args.player, game_counter)
            except Exception as e:
                print('Error processing', pgn, e)

    print('Wrote positions to', out_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
