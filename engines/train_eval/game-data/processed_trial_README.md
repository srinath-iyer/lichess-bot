# `processed_trial.csv` Column Guide

This file documents the columns in `processed_trial.csv` and how each derived feature is computed.

## Row Layout

Each row is one candidate move from one chess position.

The CSV is built from the board before the move, the move itself, and the board after the move. It keeps:

- base identifiers for the position and move
- move properties
- position features for the current board state
- delta features computed as `next_position - position`

## Base Columns

- `game_id`: Sequential numeric ID for the source game, starting at `1`.
- `ply`: 0-based ply index inside the game.
- `move_uci`: Candidate move in UCI notation, such as `e2e4`.
- `current_fen`: FEN string for the position before the move.
- `next_fen`: FEN string after the move is applied.

## Move Feature Columns

These describe the move itself, relative to the current board.

- `is_chosen_move`: `1` if this is the actual move played in the game, else `0`.
- `piece_type`: Type of the moving piece, stored as a lowercase symbol-like label such as `p`, `n`, `b`, `r`, `q`, or `k`.
- `is_capture`: `1` if the move captures a piece.
- `is_check`: `1` if the move gives check.
- `is_castling`: `1` if the move is castling.
- `delivers_checkmate`: `1` if the move ends the game by checkmate.
- `signed_distance`: Euclidean distance from the source square to the destination square, signed by direction. Forward moves are positive from the mover’s perspective; backward moves are negative.
- `avoids_capture`: `1` if the move is a non-capture while there exists at least one legal capture from the same piece onto an equal-or-higher-value target.
- `captured_piece_value`: Material value of the piece captured by this move, or `0` if none.

## Current Position Features

All `position_*` columns describe the board before the move, from the side to move’s perspective.

### Material and piece counts

- `position_own_piece_count`: Number of pieces owned by the side to move.
- `position_enemy_piece_count`: Number of opponent pieces.
- `position_material_balance`: Own material minus enemy material, using standard piece values.
- `position_turn_white`: `1` if the side to move is White, else `0`.

### Piece mobility

These use attack masks from `python-chess`, not only legal moves, so pinned pieces still contribute their raw control.

- `position_pawn_mobility`: Total attack squares for own pawns.
- `position_knight_mobility`: Total attack squares for own knights.
- `position_bishop_mobility`: Total attack squares for own bishops.
- `position_rook_mobility`: Total attack squares for own rooks.
- `position_queen_mobility`: Total attack squares for own queens.
- `position_king_mobility`: Total attack squares for own king.

### Rook structure

- `position_rook_open_file_count`: Number of own rooks on files with no pawns from either side.
- `position_rook_semi_open_file_count`: Number of own rooks on files with no own pawns, but at least one enemy pawn.
- `position_rook_seventh_rank_count`: Number of own rooks on the advanced rank for that side.

### Knight structure

- `position_knight_outpost_count`: Number of own knights placed on advanced squares that are not attacked by enemy pawns.

### Pawn structure overview

These are heuristic counts based on pawn file/rank placement and pawn support patterns.

- `position_passed_pawn_count`: Number of own passed pawns.
- `position_doubled_pawn_file_count`: Number of files containing 2 or more own pawns.
- `position_isolated_pawn_count`: Number of own isolated pawns.
- `position_backward_pawn_count`: Number of own backward pawns.
- `position_pawn_island_count`: Number of pawn islands, meaning contiguous file groups containing own pawns.
- `position_connected_pawn_count`: Number of own pawns that have a friendly pawn on an adjacent file nearby.
- `position_pawn_chain_count`: Number of own pawns that are part of a diagonal pawn chain.
- `position_pawn_shield_count`: Number of own pawns in front of the king on nearby files.

### King safety and king activity

- `position_king_center_distance`: Chebyshev distance from the king to the nearest center square.
- `position_king_near_open_file_count`: Count of open files near the king’s file.
- `position_king_near_semi_open_file_count`: Count of semi-open files near the king’s file.
- `position_enemy_king_zone_attackers`: Number of enemy pieces attacking the king zone.
- `position_enemy_king_zone_pawn_attackers`: Enemy pawn attackers on the king zone.
- `position_enemy_king_zone_knight_attackers`: Enemy knight attackers on the king zone.
- `position_enemy_king_zone_bishop_attackers`: Enemy bishop attackers on the king zone.
- `position_enemy_king_zone_rook_attackers`: Enemy rook attackers on the king zone.
- `position_enemy_king_zone_queen_attackers`: Enemy queen attackers on the king zone.
- `position_king_to_enemy_king_distance`: Chebyshev distance from the king to the enemy king.
- `position_king_to_friendly_pawn_distance`: Chebyshev distance from the king to the nearest friendly pawn.
- `position_king_to_enemy_pawn_distance`: Chebyshev distance from the king to the nearest enemy pawn.
- `position_king_tropism`: Closeness of enemy pieces to the king, computed as a sum of inverse distances.

### Control and space

- `position_controlled_square_count`: Number of squares controlled by the side to move.
- `position_space_in_enemy_half`: Number of controlled squares in the opponent’s half of the board.
- `position_center_control_count`: Number of controlled center squares.
- `position_enemy_controlled_square_count`: Number of squares controlled by the opponent.

## Pawn Structure Definitions

The pawn features are calculated by scanning each pawn individually and applying a local structural rule. The process is intentionally explicit so the counts are reproducible and easy to reason about.

### Backward Pawns

A backward pawn is a pawn that:

- cannot safely advance one square, and
- has no friendly pawn support on adjacent files at the same rank

The process is:

1. Take a pawn on its current square.
2. Look one square forward from that pawn’s perspective.
3. If the forward square is occupied, or would be attacked in a way that makes the advance unsafe, treat the pawn as blocked.
4. Check the same-rank adjacent files for friendly pawn support.
5. If the pawn is blocked and unsupported, count it as backward.

This matches the intended definition for the notebook and README.

### Pawn Chains

A pawn chain is counted per pawn, not just per linked pair.

A pawn belongs to a chain if it has a friendly pawn on one of the four diagonal neighbor squares around it, using the pawn’s perspective:

- front-left
- front-right
- back-left
- back-right

The process is:

1. Take each pawn individually.
2. Check the four diagonal neighboring squares around it.
3. If any of those squares contain a friendly pawn, count the pawn as part of a chain.
4. Sum the pawns that satisfy that rule.

This is why a structure like the one below should count all six black pawns in the chain groups `b7-c6-d5` and `f7-g7-h7`:

```text
r n b q . r k .
p p . . p p b p
. . p . . n p .
. . . p . . . .
. . P P . B . .
. . N . P N . .
P P . . . P P P
R . . Q K B . R
```

The key point is that each pawn is evaluated on its own; the feature does not just count isolated links between two pawns.

### Connected Pawns

A connected pawn is a pawn with a friendly pawn on an adjacent file at the same rank.

The process is:

1. Take each pawn individually.
2. Check the left neighbor on the same rank.
3. Check the right neighbor on the same rank.
4. If either neighbor exists, count the pawn as connected.

Connected pawns are mutually defensive in the horizontal direction. This is different from pawn chains, which are diagonal.

### Pawn Islands

A pawn island is a maximal contiguous group of files containing at least one friendly pawn.

The process is:

1. Collect the files that contain friendly pawns.
2. Sort those files.
3. Count gaps where file numbers jump by more than 1.
4. The number of islands is the number of gaps plus 1.

Pawn islands measure how fragmented a pawn structure is. Fewer islands generally means a more cohesive structure.

### Isolated Pawns

An isolated pawn has no friendly pawn on either adjacent file at the same rank.

The process is:

1. Take each pawn individually.
2. Check the adjacent file to the left.
3. Check the adjacent file to the right.
4. If neither file has a friendly pawn on the same rank, count the pawn as isolated.

### Passed Pawns

A passed pawn is a pawn with no enemy pawn that can stop its advance on the same file or adjacent files ahead of it.

The process is:

1. Take each pawn individually.
2. Look for enemy pawns on the same file or adjacent files.
3. If an enemy pawn exists ahead of the pawn and can block its route to promotion, it is not passed.
4. If no such enemy pawn exists, count it as passed.

### Doubled Pawn Files

A doubled pawn file is a file containing two or more friendly pawns.

The process is:

1. Count friendly pawns on each file.
2. If a file contains at least two pawns, count that file as doubled.
3. Sum the doubled files.

## Delta Features

All `delta_position_*` columns are computed as:

`delta_position_feature = next_position_feature - position_feature`

So they measure how the move changes the position.

Examples:

- `delta_position_material_balance`: Change in material balance.
- `delta_position_pawn_chain_count`: Change in pawn-chain count.
- `delta_position_backward_pawn_count`: Change in backward-pawn count.
- `delta_position_king_tropism`: Change in king tropism.

## Notes on Calculation

- Position features are extracted from the board using `python-chess`.
- Pawn structure features are heuristic counts based on pawn location and support patterns.
- King safety features measure nearby pawn cover, file openness, attackers near the king, and proximity to opposing pieces.
- Delta features are a simple numeric difference between the post-move and pre-move feature values.

## Practical Interpretation

- Use the `position_*` columns to describe the current board state.
- Use the `delta_position_*` columns to describe how the candidate move changes that state.
- Use the move feature columns to describe tactical intent and move shape.

This CSV is intended for supervised learning on move choice and style imitation.
