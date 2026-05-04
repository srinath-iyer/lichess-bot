#!/usr/bin/env python3
"""Lichess-style analysis UI for chess engines - pluggable architecture."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import chess

from engines.search import AlphaBetaSearch
from engines.evaluator import PositionEvaluator
from engines.policy_model import PolicyModelRuntime


logger = logging.getLogger(__name__)


def _load_piece_svgs() -> dict[str, str]:
    """Load SVG pieces and convert to data URIs."""
    pieces = {}
    piece_dir = Path(__file__).parent / "piece set"
    
    if not piece_dir.exists():
        logger.warning("Piece set directory not found, using Unicode pieces")
        return pieces
    
    for svg_file in piece_dir.glob("*.svg"):
        try:
            with open(svg_file, "r") as f:
                svg_content = f.read()
            # Convert to data URI
            import base64
            svg_b64 = base64.b64encode(svg_content.encode()).decode()
            data_uri = f"data:image/svg+xml;base64,{svg_b64}"
            pieces[svg_file.stem.replace("Chess_", "")] = data_uri
        except Exception as e:
            logger.warning(f"Failed to load {svg_file}: {e}")
    
    return pieces


def _get_pieces_json(pieces_dict: dict[str, str]) -> str:
    """Convert pieces dict to JSON for JavaScript."""
    import json
    return json.dumps(pieces_dict)


@dataclass(slots=True)
class AnalysisSettings:
    """Configuration for analysis engine."""

    analysis_depth: int = 3
    analysis_breadth: int = 5
    analysis_timeout_seconds: float = 2.75
    use_policy_ordering: bool = True


class BackgroundModelLoader:
    """Load model in background thread without blocking server startup."""

    def __init__(self, model_runtime: PolicyModelRuntime):
        self.model = model_runtime
        self.ready = threading.Event()
        self.error: Exception | None = None

        thread = threading.Thread(target=self._load, daemon=True)
        thread.start()

    def _load(self) -> None:
        try:
            logger.info("Background: Loading policy model...")
            self.model.get_move_probabilities(chess.Board(), [])
            self.ready.set()
            logger.info("Background: Policy model loaded successfully")
        except Exception as e:
            self.error = e
            logger.exception("Background: Policy model failed to load")


class AnalysisEngine:
    """Unified interface for analysis engines - pluggable for different models/bots."""

    def __init__(
        self,
        evaluator: PositionEvaluator | None = None,
        policy: PolicyModelRuntime | None = None,
        settings: AnalysisSettings | None = None,
    ):
        self.settings = settings or AnalysisSettings()
        self.evaluator = evaluator or PositionEvaluator()
        self.policy = policy
        self.searcher = AlphaBetaSearch(
            evaluator=self.evaluator,
            depth=self.settings.analysis_depth,
            policy_runtime=self.policy,
            use_policy_ordering=self.settings.use_policy_ordering,
            time_limit_seconds=self.settings.analysis_timeout_seconds,
        )
        self.lock = threading.RLock()

    def _ensure_policy_ready(self) -> None:
        """Wait for policy model to load if present. No-op since policy is pre-loaded."""
        pass

    def analyze(self, board: chess.Board) -> dict[str, Any]:
        """Analyze a position."""
        with self.lock:
            self._ensure_policy_ready()
            return self.searcher.analyze_root(
                board,
                max_depth=self.settings.analysis_depth,
                breadth=self.settings.analysis_breadth,
                time_limit_seconds=self.settings.analysis_timeout_seconds,
            )


class AnalysisSession:
    """Session managing board state and analysis requests."""

    def __init__(self, engine: AnalysisEngine):
        self.engine = engine
        self.board = chess.Board()
        self.lock = threading.RLock()

    def set_position(self, fen: str) -> dict[str, Any]:
        """Load position from FEN."""
        with self.lock:
            try:
                self.board = chess.Board(fen)
            except ValueError as e:
                raise ValueError(f"Invalid FEN: {fen}") from e
            return self.get_state()

    def make_move(self, move_uci: str) -> dict[str, Any]:
        """Make a move in UCI notation."""
        with self.lock:
            try:
                move = chess.Move.from_uci(move_uci)
                if move not in self.board.legal_moves:
                    raise ValueError(f"Illegal move: {move_uci}")
                self.board.push(move)
            except (ValueError, chess.InvalidMoveError) as e:
                raise ValueError(f"Invalid move: {move_uci}") from e
            return self.get_state()

    def get_state(self) -> dict[str, Any]:
        """Get current board state and analysis."""
        with self.lock:
            analysis = self.engine.analyze(self.board)
            
            # Build valid_moves map: { from_square: [to_squares...] }
            valid_moves_map = {}
            for move in self.board.legal_moves:
                from_sq = move.uci()[:2]
                to_sq = move.uci()[2:4]
                if from_sq not in valid_moves_map:
                    valid_moves_map[from_sq] = []
                valid_moves_map[from_sq].append(to_sq)
            
            return {
                "fen": self.board.fen(),
                "turn": "white" if self.board.turn == chess.WHITE else "black",
                "legal_moves": [move.uci() for move in self.board.legal_moves],
                "valid_moves": valid_moves_map,
                "analysis": analysis,
            }


class AnalysisHandler(BaseHTTPRequestHandler):
    """HTTP handler for analysis API and UI."""

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route == "/":
            self._write_html(HTML_PAGE)
        elif route == "/api/state":
            self._write_json(self.server.session.get_state())
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        
        if route == "/api/position":
            fen = str(payload.get("fen", "")).strip()
            if not fen:
                self._write_json({"error": "Missing FEN"}, 400)
                return
            try:
                state = self.server.session.set_position(fen)
                self._write_json(state)
            except ValueError as e:
                self._write_json({"error": str(e)}, 400)
                
        elif route == "/api/move":
            move_uci = str(payload.get("move", "")).strip()
            if not move_uci:
                self._write_json({"error": "Missing move"}, 400)
                return
            try:
                state = self.server.session.make_move(move_uci)
                self._write_json(state)
            except ValueError as e:
                self._write_json({"error": str(e)}, 400)
        else:
            self.send_error(404)

    def _write_json(self, data: dict[str, Any], status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _write_html(self, html: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        logger.info(format, *args)


HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chess Analysis</title>
    <style>
        :root {
            --light: #f0d9b5;
            --dark: #b58863;
            --white: #ffffff;
            --black: #000000;
            --panel: #1a1a1a;
            --text: #e0e0e0;
            --muted: #888;
            --good: #50d18d;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--panel);
            color: var(--text);
            padding: 20px;
            max-width: 1000px;
            margin: 0 auto;
        }

        h1 {
            margin-bottom: 20px;
            font-size: 28px;
        }

        .container {
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 40px;
            margin-bottom: 40px;
        }

        .board {
            display: grid;
            grid-template-columns: repeat(8, 1fr);
            gap: 0;
            background: var(--dark);
            padding: 4px;
            border-radius: 4px;
            aspect-ratio: 1;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }

        .square {
            aspect-ratio: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 36px;
            font-weight: bold;
            font-family: serif;
            user-select: none;
            color: var(--white);
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            position: relative;
            cursor: pointer;
            transition: background-color 0.1s;
        }

        .square.light {
            background: var(--light);
            color: var(--black);
            text-shadow: none;
        }

        .square.dark {
            background: var(--dark);
        }

        .square.selected.light {
            background: #baca44;
        }

        .square.selected.dark {
            background: #9ca84e;
        }

        .square.valid::after {
            content: '';
            position: absolute;
            width: 14px;
            height: 14px;
            background: rgba(80, 209, 141, 0.6);
            border-radius: 50%;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            pointer-events: none;
        }

        .fen-input {
            display: flex;
            gap: 8px;
            margin-top: 16px;
        }

        .fen-input input {
            flex: 1;
            background: #333;
            border: 1px solid #555;
            color: var(--text);
            padding: 8px 12px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 11px;
        }

        .fen-input button {
            background: var(--good);
            border: none;
            color: var(--black);
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 600;
            font-size: 13px;
        }

        .fen-input button:hover {
            background: #45c976;
        }

        .analysis {
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        .eval-box {
            font-size: 56px;
            font-weight: bold;
            color: var(--good);
            text-align: center;
            padding: 24px;
            background: rgba(80, 209, 141, 0.08);
            border: 1px solid rgba(80, 209, 141, 0.2);
            border-radius: 6px;
            font-variant-numeric: tabular-nums;
            font-family: 'Courier New', monospace;
        }

        .eval-box.negative {
            color: #ff6b6b;
        }

        .moves-title {
            font-size: 12px;
            text-transform: uppercase;
            color: var(--muted);
            letter-spacing: 1px;
            font-weight: 600;
        }

        .moves {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .move {
            padding: 10px 12px;
            background: rgba(100, 100, 100, 0.15);
            border-left: 3px solid var(--good);
            border-radius: 3px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13px;
            transition: background 0.15s;
        }

        .move:hover {
            background: rgba(100, 100, 100, 0.25);
        }

        .move-san {
            font-weight: 600;
            letter-spacing: 0.5px;
        }

        .move-eval {
            font-family: 'Courier New', monospace;
            font-size: 12px;
            color: var(--muted);
        }

        .lines {
            margin-top: 8px;
        }

        .line {
            font-family: 'Courier New', monospace;
            font-size: 12px;
            color: var(--muted);
            line-height: 1.5;
            word-break: break-all;
            padding: 10px 12px;
            background: rgba(100, 100, 100, 0.1);
            border-radius: 3px;
        }

        .status {
            color: var(--muted);
            font-size: 12px;
            text-align: center;
            padding: 12px;
        }
    </style>
</head>
<body>
    <h1>Chess Analysis</h1>

    <div class="container">
        <div>
            <div class="board" id="board"></div>
            <div class="fen-input">
                <input type="text" id="fen" placeholder="FEN...">
                <button onclick="analyze()">Load</button>
            </div>
            <div class="status" id="status">Loading...</div>
        </div>

        <div class="analysis">
            <div class="eval-box" id="eval">0.0</div>

            <div>
                <div class="moves-title">Top Moves</div>
                <div class="moves" id="moves"></div>
            </div>

            <div class="lines">
                <div class="moves-title">Main Line</div>
                <div class="line" id="line"></div>
            </div>
        </div>
    </div>

    <script>
        // SVG pieces map (base64 encoded)
        const PIECES_MAP = window.PIECES_MAP || {};

        let state = null;
        let selectedSquare = null;
        let validMoves = [];

        function squareToCoords(rank, file) {
            return String.fromCharCode(97 + file) + (8 - rank);
        }

        function coordsToSquare(square) {
            const file = square.charCodeAt(0) - 97;
            const rank = 8 - parseInt(square[1]);
            return { rank, file };
        }

        function parseBoardFen(fen) {
            const board = [];
            const boardPart = fen.split(' ')[0];
            for (const row of boardPart.split('/')) {
                const boardRow = [];
                for (const char of row) {
                    if (char >= '1' && char <= '8') {
                        for (let i = 0; i < parseInt(char); i++) boardRow.push(null);
                    } else {
                        boardRow.push(char);
                    }
                }
                board.push(boardRow);
            }
            return board;
        }

        function getPieceSvg(piece) {
            if (!piece) return null;
            const isWhite = piece === piece.toUpperCase();
            const pMap = { 'K': 'k', 'Q': 'q', 'R': 'r', 'B': 'b', 'N': 'n', 'P': 'p' };
            const pLower = pMap[piece.toUpperCase()] || piece.toLowerCase();
            const key = pLower + (isWhite ? 'l' : 'd');
            return PIECES_MAP[key] || null;
        }

        function renderBoard() {
            if (!state || !state.fen) return;

            const boardEl = document.getElementById('board');
            boardEl.innerHTML = '';

            const board = parseBoardFen(state.fen);
            for (let rank = 0; rank < 8; rank++) {
                for (let file = 0; file < 8; file++) {
                    const square = document.createElement('div');
                    // Fixed: h1 should be white/light (rank=7, file=7) => (7+7)%2=0 => dark
                    // So negate: isLight = (rank + file) % 2 === 0
                    const isLight = (rank + file) % 2 === 0;
                    const squareId = squareToCoords(rank, file);
                    const isSelected = selectedSquare === squareId;
                    const isValid = validMoves.includes(squareId);

                    square.className = `square ${isLight ? 'light' : 'dark'} ${isSelected ? 'selected' : ''} ${isValid ? 'valid' : ''}`;
                    square.id = `sq-${squareId}`;
                    square.dataset.rank = rank;
                    square.dataset.file = file;
                    square.dataset.square = squareId;

                    const piece = board[rank][file];
                    if (piece) {
                        const svg = getPieceSvg(piece);
                        if (svg) {
                            const img = document.createElement('img');
                            img.src = svg;
                            img.alt = piece;
                            img.style.width = '100%';
                            img.style.height = '100%';
                            img.style.objectFit = 'contain';
                            square.appendChild(img);
                        }
                    }

                    square.addEventListener('click', () => handleSquareClick(squareId, piece));
                    boardEl.appendChild(square);
                }
            }
        }

        async function handleSquareClick(square, piece) {
            if (!selectedSquare) {
                // Select piece
                if (piece) {
                    selectedSquare = square;
                    validMoves = (state.valid_moves || {})[square] || [];
                    renderBoard();
                }
            } else {
                // Try to move
                if (square === selectedSquare) {
                    // Deselect
                    selectedSquare = null;
                    validMoves = [];
                    renderBoard();
                } else if (validMoves.includes(square)) {
                    // Make move
                    await makeMove(selectedSquare + square);
                } else if (piece) {
                    // Select different piece
                    selectedSquare = square;
                    validMoves = (state.valid_moves || {})[square] || [];
                    renderBoard();
                }
            }
        }

        async function makeMove(move) {
            try {
                const res = await fetch('/api/move', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ move, fen: state.fen })
                });

                if (!res.ok) {
                    const err = await res.json();
                    alert('Invalid move: ' + (err.error || 'Unknown'));
                    return;
                }

                state = await res.json();
                selectedSquare = null;
                validMoves = [];
                renderBoard();
                renderAnalysis();
                document.getElementById('status').textContent = state.turn === 'white' ? 'White to move' : 'Black to move';
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }

        function formatEval(value) {
            if (value === undefined || value === null || isNaN(value)) return '0.0';
            if (value > 10000) return '#+' + Math.round(value / 1000);
            if (value < -10000) return '#-' + Math.round(-value / 1000);
            return (value >= 0 ? '+' : '') + value.toFixed(1);
        }

        function formatLine(node, depth = 0, moveNum = 1, isWhite = true) {
            if (!node || depth > 12) return '';

            let line = '';
            let current = node;
            let num = moveNum;

            for (let i = 0; i < 15 && current; i++) {
                const children = current.children || [];
                if (children.length === 0) break;

                const best = children[0];
                const move = best.san || best.uci;

                if (isWhite) {
                    line += `${num}. ${move}`;
                } else {
                    line += move;
                }

                line += ' ';
                if (isWhite) num++;

                current = best.child;
                isWhite = !isWhite;
            }

            return line.trim();
        }

        function renderAnalysis() {
            if (!state || !state.analysis) return;

            const analysis = state.analysis;

            // Eval
            const evalEl = document.getElementById('eval');
            const eval_ = formatEval(analysis.white_eval);
            evalEl.textContent = eval_;
            evalEl.classList.toggle('negative', analysis.white_eval < 0);

            // Moves
            const movesEl = document.getElementById('moves');
            movesEl.innerHTML = '';

            const children = analysis.children || [];
            const topMoves = children.slice(0, 5);

            if (topMoves.length === 0) {
                movesEl.innerHTML = '<div class="status">No legal moves</div>';
                return;
            }

            // Opacity gradient based on eval difference
            const bestEval = topMoves[0]?.white_eval ?? 0;
            const worstEval = topMoves[topMoves.length - 1]?.white_eval ?? bestEval;
            const spread = Math.max(Math.abs(bestEval - worstEval), 0.1);

            topMoves.forEach((moveData, idx) => {
                const move = document.createElement('div');
                move.className = 'move';

                // Calculate opacity based on how much worse this move is
                const evalDiff = bestEval - moveData.white_eval;
                const opacity = Math.max(0.3, 1 - (evalDiff / spread) * 0.6);
                move.style.opacity = opacity.toString();

                const san = moveData.san || moveData.uci;
                const eval_ = formatEval(moveData.white_eval);

                move.innerHTML = `
                    <span class="move-san">${san}</span>
                    <span class="move-eval">${eval_}</span>
                `;

                movesEl.appendChild(move);
            });

            // Main line
            const lineEl = document.getElementById('line');
            if (topMoves[0] && topMoves[0].child) {
                lineEl.textContent = formatLine(topMoves[0].child);
            } else {
                lineEl.textContent = '';
            }
        }

        async function analyze() {
            const fen = document.getElementById('fen').value.trim();
            if (!fen) return;

            document.getElementById('status').textContent = 'Analyzing...';

            try {
                const res = await fetch('/api/position', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ fen })
                });

                if (!res.ok) {
                    const err = await res.json();
                    document.getElementById('status').textContent = 'Error: ' + (err.error || 'Unknown');
                    return;
                }

                state = await res.json();
                renderBoard();
                renderAnalysis();
                document.getElementById('status').textContent = state.turn === 'white' ? 'White to move' : 'Black to move';
            } catch (e) {
                document.getElementById('status').textContent = 'Error: ' + e.message;
            }
        }

        async function init() {
            try {
                const res = await fetch('/api/state');
                state = await res.json();
                document.getElementById('fen').value = state.fen;
                renderBoard();
                renderAnalysis();
                document.getElementById('status').textContent = state.turn === 'white' ? 'White to move' : 'Black to move';
            } catch (e) {
                document.getElementById('status').textContent = 'Error: ' + e.message;
            }
        }

        window.addEventListener('load', init);
        document.getElementById('fen').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') analyze();
        });
    </script>
</body>
</html>
""".strip()


def main():
    """Run the analysis server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s: %(message)s",
    )

    # Create pluggable engine - easy to swap models/evaluators
    engine = AnalysisEngine(
        evaluator=PositionEvaluator(),
        policy=PolicyModelRuntime(model_path="engines/train_eval/model_artifacts/policy_xgb.joblib"),
        settings=AnalysisSettings(
            analysis_depth=3,
            analysis_breadth=5,
            analysis_timeout_seconds=2.75,
            use_policy_ordering=True,
        ),
    )

    session = AnalysisSession(engine)

    server = ThreadingHTTPServer(("127.0.0.1", 8000), AnalysisHandler)
    server.session = session

    logger.info("Chess analysis running on http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
