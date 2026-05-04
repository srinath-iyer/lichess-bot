"""Feature extraction helpers shared between training and online policy inference."""

from __future__ import annotations

import math
from typing import Any

import chess

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def get_detailed_move_properties(board: chess.Board, move: chess.Move, is_chosen: bool) -> dict[str, Any]:
    """Extract move-level properties used by the policy model."""
    from_sq = move.from_square
    to_sq = move.to_square
    piece_on_from = board.piece_at(from_sq)
    piece_on_to = board.piece_at(to_sq)

    piece_type_char = piece_on_from.symbol().lower() if piece_on_from else "?"
    moving_piece_value = PIECE_VALUES.get(piece_on_from.piece_type, 0) if piece_on_from else 0

    is_capture = board.is_capture(move)
    is_check = board.gives_check(move)
    is_castling = board.is_castling(move)

    from_file = chess.square_file(from_sq)
    from_rank = chess.square_rank(from_sq)
    to_file = chess.square_file(to_sq)
    to_rank = chess.square_rank(to_sq)

    file_delta = to_file - from_file
    rank_delta = to_rank - from_rank
    distance = math.sqrt(file_delta**2 + rank_delta**2)

    if board.turn == chess.WHITE:
        direction_sign = 1 if rank_delta > 0 else (-1 if rank_delta < 0 else 0)
    else:
        direction_sign = -1 if rank_delta > 0 else (1 if rank_delta < 0 else 0)

    signed_distance = distance * direction_sign if direction_sign != 0 else distance

    all_moves_from_square = [m for m in board.legal_moves if m.from_square == from_sq]
    capture_moves_from_square = [m for m in all_moves_from_square if board.is_capture(m)]
    equal_or_better_captures: list[chess.Move] = []
    capturable_pieces: list[int] = []

    for cap_move in capture_moves_from_square:
        target_piece = board.piece_at(cap_move.to_square)
        if target_piece:
            target_value = PIECE_VALUES.get(target_piece.piece_type, 0)
            capturable_pieces.append(target_value)
            if target_value >= moving_piece_value:
                equal_or_better_captures.append(cap_move)

    max_capturable = max(capturable_pieces) if capturable_pieces else 0
    piece_avoided_capture = (not is_capture) and len(equal_or_better_captures) > 0

    next_board = board.copy(stack=False)
    next_board.push(move)
    delivers_checkmate = int(next_board.is_checkmate())

    return {
        "is_chosen_move": int(is_chosen),
        "piece_type": piece_type_char,
        "is_capture": int(is_capture),
        "is_check": int(is_check),
        "is_castling": int(is_castling),
        "delivers_checkmate": delivers_checkmate,
        "distance": float(distance),
        "signed_distance": float(signed_distance),
        "from_square": int(from_sq),
        "to_square": int(to_sq),
        "from_file": int(from_file),
        "from_rank": int(from_rank),
        "to_file": int(to_file),
        "to_rank": int(to_rank),
        "num_alt_captures_from_piece": int(len(capture_moves_from_square)),
        "num_equal_or_better_captures": int(len(equal_or_better_captures)),
        "max_capturable_value": int(max_capturable),
        "avoids_capture": int(piece_avoided_capture),
        "captured_piece_value": int(PIECE_VALUES.get(piece_on_to.piece_type, 0)) if piece_on_to else 0,
    }


def extract_position_features(board: chess.Board) -> dict[str, float | int]:
    """Extract position-level features compatible with the training pipeline."""
    color = board.turn
    enemy_color = not color

    own_pieces = chess.popcount(board.occupied_co[color])
    enemy_pieces = chess.popcount(board.occupied_co[enemy_color])
    own_material = sum(len(board.pieces(pt, color)) * PIECE_VALUES[pt] for pt in PIECE_VALUES)
    enemy_material = sum(len(board.pieces(pt, enemy_color)) * PIECE_VALUES[pt] for pt in PIECE_VALUES)
    material_balance = own_material - enemy_material

    own_pawns = list(board.pieces(chess.PAWN, color))
    own_pawns_set = set(own_pawns)
    enemy_pawns = list(board.pieces(chess.PAWN, enemy_color))

    def pawn_is_passed(sq: int, pawn_color: bool, enemy_pawns_list: list[int]) -> bool:
        pawn_file = chess.square_file(sq)
        pawn_rank = chess.square_rank(sq)
        for ep in enemy_pawns_list:
            ep_file = chess.square_file(ep)
            ep_rank = chess.square_rank(ep)
            if abs(ep_file - pawn_file) <= 1:
                if pawn_color == chess.WHITE and ep_rank > pawn_rank:
                    return False
                if pawn_color == chess.BLACK and ep_rank < pawn_rank:
                    return False
        return True

    def pawn_is_backward(sq: int, pawn_color: bool) -> bool:
        pawn_file = chess.square_file(sq)
        pawn_rank = chess.square_rank(sq)
        forward_offset = 8 if pawn_color == chess.WHITE else -8
        forward_sq = sq + forward_offset
        if not (0 <= forward_sq <= 63):
            return False

        if board.piece_at(forward_sq) is not None:
            blocked = True
        else:
            attacked = False
            for ep in enemy_pawns:
                ep_file = chess.square_file(ep)
                if abs(ep_file - pawn_file) == 1:
                    ep_rank = chess.square_rank(ep)
                    attack_offset = 8 if not pawn_color else -8
                    if ep + attack_offset == forward_sq or ep + attack_offset + (1 if ep_file < pawn_file else -1) == forward_sq:
                        attacked = True
                        break
            blocked = attacked or board.piece_at(forward_sq) is not None

        if not blocked:
            return False

        support_left = pawn_file > 0 and chess.square(pawn_file - 1, pawn_rank) in own_pawns
        support_right = pawn_file < 7 and chess.square(pawn_file + 1, pawn_rank) in own_pawns
        return not (support_left or support_right)

    def pawns_connected(sq: int, all_own_pawns: list[int]) -> bool:
        pawn_file = chess.square_file(sq)
        pawn_rank = chess.square_rank(sq)
        left = chess.square(pawn_file - 1, pawn_rank) if pawn_file > 0 else None
        right = chess.square(pawn_file + 1, pawn_rank) if pawn_file < 7 else None
        return (left in all_own_pawns) or (right in all_own_pawns)

    def pawn_is_isolated(sq: int, all_own_pawns: list[int]) -> bool:
        pawn_file = chess.square_file(sq)
        left_file = pawn_file - 1
        right_file = pawn_file + 1
        has_left_neighbor = any(chess.square_file(p) == left_file for p in all_own_pawns)
        has_right_neighbor = any(chess.square_file(p) == right_file for p in all_own_pawns)
        return not (has_left_neighbor or has_right_neighbor)

    def pawn_in_chain(sq: int, all_own_pawns_set: set[int]) -> bool:
        pawn_file = chess.square_file(sq)
        pawn_rank = chess.square_rank(sq)
        diagonal_neighbors = [
            (pawn_file - 1, pawn_rank + 1),
            (pawn_file + 1, pawn_rank + 1),
            (pawn_file - 1, pawn_rank - 1),
            (pawn_file + 1, pawn_rank - 1),
        ]
        for next_file, next_rank in diagonal_neighbors:
            if 0 <= next_file <= 7 and 0 <= next_rank <= 7:
                if chess.square(next_file, next_rank) in all_own_pawns_set:
                    return True
        return False

    passed_pawns = sum(1 for sq in own_pawns if pawn_is_passed(sq, color, enemy_pawns))
    doubled_pawns = sum(1 for file_idx in range(8) if sum(1 for sq in own_pawns if chess.square_file(sq) == file_idx) >= 2)
    isolated_pawns = sum(1 for sq in own_pawns if pawn_is_isolated(sq, own_pawns))
    backward_pawns = sum(1 for sq in own_pawns if pawn_is_backward(sq, color))

    pawn_files = sorted(set(chess.square_file(sq) for sq in own_pawns))
    pawn_islands = 1 if pawn_files else 0
    for i in range(1, len(pawn_files)):
        if pawn_files[i] - pawn_files[i - 1] > 1:
            pawn_islands += 1

    connected_pawns = sum(1 for sq in own_pawns if pawns_connected(sq, own_pawns))
    chain_pawns = sum(1 for sq in own_pawns if pawn_in_chain(sq, own_pawns_set))

    mobility_by_type: dict[str, int] = {}
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
        total_attacks = 0
        for sq in board.pieces(piece_type, color):
            attacks = set(board.attacks(sq))
            if piece_type == chess.PAWN:
                attacks = {s for s in attacks if board.piece_at(s) is None or board.piece_at(s).color == enemy_color}
            else:
                attacks = {s for s in attacks if board.piece_at(s) is None or board.piece_at(s).color != color}
            total_attacks += len(attacks)
        piece_name = {
            chess.PAWN: "pawn",
            chess.KNIGHT: "knight",
            chess.BISHOP: "bishop",
            chess.ROOK: "rook",
            chess.QUEEN: "queen",
            chess.KING: "king",
        }[piece_type]
        mobility_by_type[piece_name] = total_attacks

    own_rooks = [sq for sq in board.pieces(chess.ROOK, color)]
    rooks_on_open_files = 0
    rooks_on_semi_open = 0
    rooks_on_seventh = 0
    seventh_rank = 6 if color == chess.WHITE else 1
    for rook_sq in own_rooks:
        rook_file = chess.square_file(rook_sq)
        own_pawns_on_file = sum(1 for sq in own_pawns if chess.square_file(sq) == rook_file)
        enemy_pawns_on_file = sum(1 for sq in enemy_pawns if chess.square_file(sq) == rook_file)
        if own_pawns_on_file == 0 and enemy_pawns_on_file == 0:
            rooks_on_open_files += 1
        elif own_pawns_on_file == 0:
            rooks_on_semi_open += 1
        if chess.square_rank(rook_sq) == seventh_rank:
            rooks_on_seventh += 1

    knight_outposts = 0
    for knight_sq in board.pieces(chess.KNIGHT, color):
        rank = chess.square_rank(knight_sq)
        if (color == chess.WHITE and rank >= 5) or (color == chess.BLACK and rank <= 2):
            attacked_by_pawn = False
            for ep in enemy_pawns:
                ep_file = chess.square_file(ep)
                knight_file = chess.square_file(knight_sq)
                if abs(knight_file - ep_file) == 1:
                    ep_rank = chess.square_rank(ep)
                    attack_rank = ep_rank + (8 if not color else -8)
                    if attack_rank == rank:
                        attacked_by_pawn = True
                        break
            if not attacked_by_pawn:
                knight_outposts += 1

    king_sq = board.king(color)
    pawn_shield = 0
    if king_sq is not None:
        king_file = chess.square_file(king_sq)
        king_rank = chess.square_rank(king_sq)
        shield_rank = king_rank + (1 if color == chess.WHITE else -1)
        if 0 <= shield_rank <= 7:
            for file_idx in range(max(0, king_file - 1), min(8, king_file + 2)):
                if chess.square(file_idx, shield_rank) in own_pawns:
                    pawn_shield += 1

    king_center_distance = 999
    if king_sq is not None:
        king_file = chess.square_file(king_sq)
        king_rank = chess.square_rank(king_sq)
        center_squares = [27, 28, 35, 36]
        for c_sq in center_squares:
            c_file = chess.square_file(c_sq)
            c_rank = chess.square_rank(c_sq)
            dist = max(abs(king_file - c_file), abs(king_rank - c_rank))
            king_center_distance = min(king_center_distance, dist)

    king_near_open = 0
    king_near_semi = 0
    if king_sq is not None:
        king_file = chess.square_file(king_sq)
        for file_idx in range(max(0, king_file - 2), min(8, king_file + 3)):
            own_on_f = sum(1 for sq in own_pawns if chess.square_file(sq) == file_idx)
            enemy_on_f = sum(1 for sq in enemy_pawns if chess.square_file(sq) == file_idx)
            if own_on_f == 0 and enemy_on_f == 0:
                king_near_open += 1
            elif own_on_f == 0:
                king_near_semi += 1

    if king_sq is not None:
        king_file = chess.square_file(king_sq)
        king_rank = chess.square_rank(king_sq)
        king_zone = {
            chess.square(file_idx, rank_idx)
            for file_idx in range(max(0, king_file - 1), min(8, king_file + 2))
            for rank_idx in range(max(0, king_rank - 1), min(8, king_rank + 2))
        }

        zone_attackers = 0
        zone_pawn_attackers = 0
        zone_knight_attackers = 0
        zone_bishop_attackers = 0
        zone_rook_attackers = 0
        zone_queen_attackers = 0

        for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
            for enemy_sq in board.pieces(pt, enemy_color):
                piece = board.piece_at(enemy_sq)
                if piece is None:
                    continue
                attacks = set(board.attacks(enemy_sq))
                if attacks & king_zone:
                    zone_attackers += 1
                    if piece.piece_type == chess.PAWN:
                        zone_pawn_attackers += 1
                    elif piece.piece_type == chess.KNIGHT:
                        zone_knight_attackers += 1
                    elif piece.piece_type == chess.BISHOP:
                        zone_bishop_attackers += 1
                    elif piece.piece_type == chess.ROOK:
                        zone_rook_attackers += 1
                    elif piece.piece_type == chess.QUEEN:
                        zone_queen_attackers += 1
    else:
        zone_attackers = 0
        zone_pawn_attackers = 0
        zone_knight_attackers = 0
        zone_bishop_attackers = 0
        zone_rook_attackers = 0
        zone_queen_attackers = 0

    king_to_enemy_king_dist = 999
    enemy_king_sq = board.king(enemy_color)
    if king_sq is not None and enemy_king_sq is not None:
        king_file = chess.square_file(king_sq)
        king_rank = chess.square_rank(king_sq)
        enemy_king_file = chess.square_file(enemy_king_sq)
        enemy_king_rank = chess.square_rank(enemy_king_sq)
        king_to_enemy_king_dist = max(abs(king_file - enemy_king_file), abs(king_rank - enemy_king_rank))

    king_to_friend_pawn_dist = 999
    king_to_enemy_pawn_dist = 999
    if king_sq is not None:
        king_file = chess.square_file(king_sq)
        king_rank = chess.square_rank(king_sq)
        for sq in own_pawns:
            sq_file = chess.square_file(sq)
            sq_rank = chess.square_rank(sq)
            dist = max(abs(king_file - sq_file), abs(king_rank - sq_rank))
            king_to_friend_pawn_dist = min(king_to_friend_pawn_dist, dist)
        for sq in enemy_pawns:
            sq_file = chess.square_file(sq)
            sq_rank = chess.square_rank(sq)
            dist = max(abs(king_file - sq_file), abs(king_rank - sq_rank))
            king_to_enemy_pawn_dist = min(king_to_enemy_pawn_dist, dist)

    king_tropism = 0.0
    if king_sq is not None:
        king_file = chess.square_file(king_sq)
        king_rank = chess.square_rank(king_sq)
        for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
            for enemy_sq in board.pieces(pt, enemy_color):
                sq_file = chess.square_file(enemy_sq)
                sq_rank = chess.square_rank(enemy_sq)
                dist = max(abs(king_file - sq_file), abs(king_rank - sq_rank))
                if dist > 0:
                    king_tropism += 1.0 / dist

    controlled_squares: set[int] = set()
    space_in_enemy_half_squares: set[int] = set()
    center_control_squares: set[int] = set()
    enemy_controlled_squares: set[int] = set()
    center_squares = {27, 28, 35, 36}

    for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
        for own_sq in board.pieces(pt, color):
            attacks = set(board.attacks(own_sq))
            controlled_squares.update(attacks)
            for attacked_sq in attacks:
                if (color == chess.WHITE and chess.square_rank(attacked_sq) >= 4) or (
                    color == chess.BLACK and chess.square_rank(attacked_sq) <= 3
                ):
                    space_in_enemy_half_squares.add(attacked_sq)
                if attacked_sq in center_squares:
                    center_control_squares.add(attacked_sq)

    for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
        for enemy_sq in board.pieces(pt, enemy_color):
            attacks = set(board.attacks(enemy_sq))
            enemy_controlled_squares.update(attacks)

    return {
        "own_piece_count": own_pieces,
        "enemy_piece_count": enemy_pieces,
        "material_balance": material_balance,
        "turn_white": int(color),
        "pawn_mobility": mobility_by_type.get("pawn", 0),
        "knight_mobility": mobility_by_type.get("knight", 0),
        "bishop_mobility": mobility_by_type.get("bishop", 0),
        "rook_mobility": mobility_by_type.get("rook", 0),
        "queen_mobility": mobility_by_type.get("queen", 0),
        "king_mobility": mobility_by_type.get("king", 0),
        "rook_open_file_count": rooks_on_open_files,
        "rook_semi_open_file_count": rooks_on_semi_open,
        "rook_seventh_rank_count": rooks_on_seventh,
        "knight_outpost_count": knight_outposts,
        "passed_pawn_count": passed_pawns,
        "doubled_pawn_file_count": doubled_pawns,
        "isolated_pawn_count": isolated_pawns,
        "backward_pawn_count": backward_pawns,
        "pawn_island_count": pawn_islands,
        "connected_pawn_count": connected_pawns,
        "pawn_chain_count": chain_pawns,
        "pawn_shield_count": pawn_shield,
        "king_center_distance": king_center_distance,
        "king_near_open_file_count": king_near_open,
        "king_near_semi_open_file_count": king_near_semi,
        "enemy_king_zone_attackers": zone_attackers,
        "enemy_king_zone_pawn_attackers": zone_pawn_attackers,
        "enemy_king_zone_knight_attackers": zone_knight_attackers,
        "enemy_king_zone_bishop_attackers": zone_bishop_attackers,
        "enemy_king_zone_rook_attackers": zone_rook_attackers,
        "enemy_king_zone_queen_attackers": zone_queen_attackers,
        "king_to_enemy_king_distance": king_to_enemy_king_dist,
        "king_to_friendly_pawn_distance": king_to_friend_pawn_dist,
        "king_to_enemy_pawn_distance": king_to_enemy_pawn_dist,
        "king_tropism": king_tropism,
        "controlled_square_count": len(controlled_squares),
        "space_in_enemy_half": len(space_in_enemy_half_squares),
        "center_control_count": len(center_control_squares),
        "enemy_controlled_square_count": len(enemy_controlled_squares),
    }


def numeric_delta(before_dict: dict[str, Any], after_dict: dict[str, Any]) -> dict[str, float | int]:
    """Compute after-minus-before deltas for numeric position features."""
    return {
        f"delta_position_{key}": after_dict[key] - before_dict[key]
        for key in before_dict
        if isinstance(before_dict[key], (int, float))
    }
