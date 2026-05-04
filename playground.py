"""Local chess UI for playing against the homemade bot.

Run with:
  .\\venv\\Scripts\\python.exe playground.py
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from textwrap import dedent
from typing import Any
from urllib.parse import urlparse

import chess
import yaml

from engines.evaluator import PositionEvaluator
from engines.policy_model import PolicyModelRuntime
from engines.search import AlphaBetaSearch


logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger(__name__)


class BackgroundModelLoader:
    """Loads the policy model asynchronously in the background."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.policy = None
        self.ready = threading.Event()
        self.load_thread = None

    def start_loading(self) -> None:
        """Start loading the model in a background thread."""
        if self.load_thread is not None:
            return  # Already loading
        
        def load():
            logger.info("Background: Loading policy model from %s...", self.model_path)
            try:
                self.policy = PolicyModelRuntime(model_path=self.model_path, threshold=0.5)
                logger.info("Background: Policy model loaded successfully")
            except Exception as exc:
                logger.warning("Background: Failed to load policy model: %s", exc)
                self.policy = None
            finally:
                self.ready.set()
        
        self.load_thread = threading.Thread(target=load, daemon=True)
        self.load_thread.start()

    def get_model(self, timeout: float | None = None) -> PolicyModelRuntime | None:
        """Get the loaded model, waiting up to timeout seconds if needed."""
        self.ready.wait(timeout=timeout)
        return self.policy if self.ready.is_set() else None


# Global model loader instance
_model_loader: BackgroundModelLoader | None = None


HTML_PAGE = dedent(
    """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Chess Arena</title>
      <style>
        :root {
          color-scheme: dark;
          --bg: #0d1117;
          --panel: #141a22;
          --panel-2: #0f151c;
          --line: rgba(255,255,255,0.08);
          --text: #e5ecf4;
          --muted: #8b96a8;
          --accent: #50d18d;
          --accent-2: #7cc7ff;
          --danger: #ef6c73;
          --board-light: #e9d8b8;
          --board-dark: #769656;
          --shadow: 0 22px 55px rgba(0,0,0,0.42);
          --radius: 18px;
          --square: min(7.5vw, 56px);
          font-family: "Bahnschrift", "Trebuchet MS", "Segoe UI", sans-serif;
        }

        * { box-sizing: border-box; }

        body {
          margin: 0;
          min-height: 100vh;
          color: var(--text);
          background:
            radial-gradient(circle at top left, rgba(80, 209, 141, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(124, 199, 255, 0.14), transparent 26%),
            linear-gradient(180deg, #111722 0%, #0b0f14 100%);
        }

        .shell {
          max-width: 1520px;
          margin: 0 auto;
          padding: 20px;
        }

        .topbar {
          display: flex;
          align-items: end;
          justify-content: space-between;
          gap: 16px;
          margin-bottom: 18px;
        }

        .titleblock h1 {
          margin: 0;
          font-size: 30px;
          letter-spacing: 0.02em;
        }

        .titleblock p {
          margin: 6px 0 0;
          color: var(--muted);
          font-size: 14px;
        }

        .badge {
          padding: 10px 14px;
          border: 1px solid var(--line);
          border-radius: 999px;
          background: rgba(255,255,255,0.04);
          color: var(--muted);
          font-size: 13px;
        }

        .layout {
          display: grid;
          grid-template-columns: minmax(0, 1.25fr) minmax(330px, 0.75fr);
          gap: 18px;
          align-items: start;
        }

        .board-panel,
        .side-panel > section {
          background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
          border: 1px solid var(--line);
          border-radius: var(--radius);
          box-shadow: var(--shadow);
        }

        .board-panel {
          padding: 16px;
        }

        .board-wrap {
          position: relative;
          width: fit-content;
          margin: 0 auto;
          border-radius: 16px;
          overflow: hidden;
          border: 1px solid rgba(255,255,255,0.08);
          box-shadow: inset 0 0 0 1px rgba(0,0,0,0.2), 0 24px 60px rgba(0,0,0,0.35);
        }

        #board {
          display: grid;
          grid-template-columns: repeat(8, var(--square));
          grid-template-rows: repeat(8, var(--square));
          position: relative;
          z-index: 1;
        }

        .square {
          position: relative;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: calc(var(--square) * 0.66);
          user-select: none;
          cursor: pointer;
        }

        .light { background: linear-gradient(180deg, #f0f0f0, #e0e0e0); }
        .dark { background: linear-gradient(180deg, #7a8c56, #5f7d44); }

        .square.selected::after,
        .square.target::after,
        .square.hover-target::after {
          content: "";
          position: absolute;
          inset: 0;
          pointer-events: none;
        }

        .square.selected::after {
          outline: 3px solid rgba(124, 199, 255, 0.95);
          outline-offset: -3px;
        }

        .square.target::after {
          background: radial-gradient(circle, rgba(80, 209, 141, 0.40) 0 18%, transparent 20%);
        }

        .square.hover-target::after {
          background: radial-gradient(circle, rgba(255, 204, 92, 0.45) 0 20%, transparent 22%);
        }

        .piece {
          position: relative;
          z-index: 2;
          text-shadow: 0 2px 10px rgba(0,0,0,0.5);
          transform: translateY(-1px);
          color: #1a1a1a;
          font-weight: bold;
          opacity: 1;
        }

        .coords {
          position: absolute;
          inset: auto 5px 4px auto;
          font-size: 10px;
          font-weight: 700;
          color: rgba(0,0,0,0.46);
          mix-blend-mode: multiply;
          pointer-events: none;
        }

        .arrow-layer {
          position: absolute;
          inset: 0;
          width: 100%;
          height: 100%;
          overflow: visible;
          z-index: 3;
          pointer-events: none;
        }

        .arrow {
          stroke-width: 6;
          stroke-linecap: round;
          opacity: 0.55;
          fill: none;
          pointer-events: auto;
        }

        .arrow.best { stroke: var(--accent); }
        .arrow.second { stroke: #70d6ff; }
        .arrow.third { stroke: #ffd166; }
        .arrow.active { opacity: 1; filter: drop-shadow(0 0 10px rgba(255,255,255,0.26)); }

        .controls {
          display: flex;
          flex-wrap: wrap;
          gap: 12px;
          align-items: center;
          justify-content: space-between;
          margin-top: 14px;
        }

        .controls button {
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(180deg, rgba(80,209,141,0.25), rgba(80,209,141,0.14));
          color: var(--text);
          border-radius: 12px;
          padding: 11px 14px;
          font: inherit;
          cursor: pointer;
        }

        .controls button:hover { filter: brightness(1.08); }

        .status {
          color: var(--muted);
          font-size: 14px;
        }

        .side-panel {
          display: grid;
          gap: 14px;
        }

        .side-panel > section {
          padding: 16px;
        }

        .section-title {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          margin-bottom: 12px;
        }

        .section-title h2,
        .section-title h3 {
          margin: 0;
          font-size: 15px;
          letter-spacing: 0.03em;
          text-transform: uppercase;
          color: #dbe5f2;
        }

        .section-title small {
          color: var(--muted);
        }

        .eval-meter {
          position: relative;
          height: 340px;
          border-radius: 18px;
          overflow: hidden;
          background:
            linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02)),
            linear-gradient(180deg, #0f151d 0%, #0a0f14 100%);
          border: 1px solid rgba(255,255,255,0.08);
        }

        .eval-center {
          position: absolute;
          left: 0;
          right: 0;
          top: 50%;
          height: 2px;
          background: rgba(255,255,255,0.18);
          transform: translateY(-1px);
        }

        .eval-fill {
          position: absolute;
          left: 0;
          right: 0;
          top: 50%;
          height: 0;
          transition: height 220ms ease, top 220ms ease, bottom 220ms ease;
          border-radius: 14px 14px 0 0;
          box-shadow: 0 0 30px rgba(80,209,141,0.24);
        }

        .eval-fill.white {
          bottom: 50%;
          top: auto;
          background: linear-gradient(180deg, rgba(80,209,141,0.95), rgba(80,209,141,0.25));
        }

        .eval-fill.black {
          top: 50%;
          bottom: auto;
          border-radius: 0 0 14px 14px;
          background: linear-gradient(180deg, rgba(239,108,115,0.25), rgba(239,108,115,0.95));
          box-shadow: 0 0 30px rgba(239,108,115,0.24);
        }

        .eval-value {
          position: absolute;
          inset: 50% 10px auto;
          transform: translateY(-50%);
          text-align: center;
          font-size: 22px;
          font-weight: 800;
          letter-spacing: 0.02em;
          color: white;
          text-shadow: 0 2px 14px rgba(0,0,0,0.55);
          pointer-events: none;
        }

        .eval-caption {
          display: flex;
          justify-content: space-between;
          margin-top: 10px;
          color: var(--muted);
          font-size: 12px;
        }

        .list {
          display: grid;
          gap: 8px;
        }

        .move-card {
          padding: 10px 11px;
          border-radius: 12px;
          border: 1px solid rgba(255,255,255,0.08);
          background: rgba(255,255,255,0.035);
          cursor: pointer;
          transition: transform 120ms ease, background 120ms ease, border-color 120ms ease;
        }

        .move-card:hover,
        .move-card.active {
          transform: translateY(-1px);
          border-color: rgba(124,199,255,0.36);
          background: rgba(124,199,255,0.08);
        }

        .move-card .line1 {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 12px;
          font-size: 14px;
          font-weight: 700;
        }

        .move-card .line2 {
          margin-top: 4px;
          color: var(--muted);
          font-size: 12px;
          display: flex;
          justify-content: space-between;
          gap: 10px;
        }

        .tree {
          display: grid;
          gap: 8px;
          max-height: 44vh;
          overflow: auto;
          padding-right: 4px;
        }

        .tree-node {
          border-left: 1px solid rgba(255,255,255,0.09);
          padding-left: 12px;
          margin-left: 8px;
        }

        .tree-node.root {
          border-left: 0;
          margin-left: 0;
          padding-left: 0;
        }

        .tree-row {
          display: flex;
          flex-wrap: wrap;
          justify-content: space-between;
          gap: 8px;
          align-items: baseline;
          padding: 6px 0;
        }

        .tree-move {
          font-weight: 800;
        }

        .tree-score {
          color: var(--muted);
          font-size: 12px;
        }

        .tree-children {
          display: grid;
          gap: 6px;
        }

        .empty {
          color: var(--muted);
          font-size: 14px;
          padding: 10px 0 2px;
        }

        @media (max-width: 1100px) {
          .layout { grid-template-columns: 1fr; }
          :root { --square: min(10.2vw, 76px); }
        }

        @media (max-width: 760px) {
          .shell { padding: 12px; }
          .topbar { flex-direction: column; align-items: start; }
          :root { --square: 11.2vw; }
          .board-panel { padding: 12px; }
          .side-panel > section { padding: 14px; }
        }
      </style>
    </head>
    <body>
      <div class="shell">
        <div class="topbar">
          <div class="titleblock">
            <h1>Chess Arena</h1>
            <p>Click a piece, click a target, and watch the engine's candidate tree update live.</p>
          </div>
          <div class="badge" id="turnBadge">Loading...</div>
        </div>

        <main class="layout">
          <section class="board-panel">
            <div class="board-wrap" id="boardWrap">
              <svg class="arrow-layer" id="arrowLayer" viewBox="0 0 800 800" preserveAspectRatio="none"></svg>
              <div id="board"></div>
            </div>
            <div class="controls">
              <button id="newGameBtn">New game</button>
              <div class="status" id="statusText">Loading board...</div>
            </div>
          </section>

          <aside class="side-panel">
            <section>
              <div class="section-title">
                <h2>Evaluation</h2>
                <small id="evalLabel">White positive</small>
              </div>
              <div class="eval-meter">
                <div class="eval-fill white" id="evalFillWhite"></div>
                <div class="eval-fill black" id="evalFillBlack"></div>
                <div class="eval-center"></div>
                <div class="eval-value" id="evalValue">0.0</div>
              </div>
              <div class="eval-caption">
                <span>Black winning</span>
                <span>White winning</span>
              </div>
            </section>

            <section>
              <div class="section-title">
                <h3>Candidate moves</h3>
                <small>Top 3</small>
              </div>
              <div class="list" id="candidateList"></div>
            </section>

            <section>
              <div class="section-title">
                <h3 id="treeHeader">Tree</h3>
                <small id="treeHint">Hover a move</small>
              </div>
              <div class="tree" id="treeView"></div>
            </section>
          </aside>
        </main>
      </div>

      <div id="treeTooltip" style="position:fixed;display:none;z-index:1000;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px;max-width:180px;"></div>

      <script>
        const PIECES = {
          'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
          'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
        };

        const STATE = {
          data: null,
          selected: null,
          hoveredCandidate: null,
          squareSize: 80,
        };

        const boardEl = document.getElementById('board');
        const arrowLayer = document.getElementById('arrowLayer');
        const candidateList = document.getElementById('candidateList');
        const treeView = document.getElementById('treeView');
        const treeHeader = document.getElementById('treeHeader');
        const treeHint = document.getElementById('treeHint');
        const turnBadge = document.getElementById('turnBadge');
        const statusText = document.getElementById('statusText');
        const evalValue = document.getElementById('evalValue');
        const evalFillWhite = document.getElementById('evalFillWhite');
        const evalFillBlack = document.getElementById('evalFillBlack');
        const newGameBtn = document.getElementById('newGameBtn');
        const boardWrap = document.getElementById('boardWrap');

        function fmtEval(value) {
          const signed = Number(value).toFixed(1);
          return value > 0 ? `+${signed}` : signed;
        }

        function clamp(value, min, max) {
          return Math.max(min, Math.min(max, value));
        }

        function squareToCoords(square) {
          const file = square.charCodeAt(0) - 97;
          const rank = parseInt(square[1], 10) - 1;
          const x = (file + 0.5) * STATE.squareSize;
          const y = (7 - rank + 0.5) * STATE.squareSize;
          return { x, y };
        }

        function squareToIndex(square) {
          return (parseInt(square[1], 10) - 1) * 8 + (square.charCodeAt(0) - 97);
        }

        function parseBoardFen(fen) {
          const placement = fen.split(' ')[0];
          const rows = placement.split('/');
          const squares = new Array(64).fill(null);
          for (let fenRank = 0; fenRank < 8; fenRank += 1) {
            const row = rows[fenRank];
            let file = 0;
            for (const char of row) {
              if (char >= '1' && char <= '8') {
                file += parseInt(char, 10);
              } else {
                const boardRank = 7 - fenRank;
                squares[boardRank * 8 + file] = char;
                file += 1;
              }
            }
          }
          return squares;
        }

        function pieceColor(piece) {
          if (!piece) return null;
          return piece === piece.toUpperCase() ? 'white' : 'black';
        }

        function currentBoardSquares() {
          if (!STATE.data) return new Array(64).fill(null);
          return parseBoardFen(STATE.data.fen);
        }

        function getPieceAt(square) {
          return currentBoardSquares()[squareToIndex(square)];
        }

        function getPieceSVG(piece) {
          const isWhite = piece === piece.toUpperCase();
          const color = isWhite ? '#fff' : '#000';
          const stroke = isWhite ? '#000' : '#fff';
          const strokeWidth = 1.5;
          
          const svgs = {
            'P': `<svg viewBox="0 0 100 100" style="width:100%;height:100%;"><circle cx="50" cy="30" r="12" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><rect x="45" y="40" width="10" height="35" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><polygon points="40,75 60,75 65,85 35,85" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/></svg>`,
            'N': `<svg viewBox="0 0 100 100" style="width:100%;height:100%;"><path d="M30,60 Q20,40 40,20 Q60,10 70,30 Q75,50 60,70 L40,85 Q30,80 30,60" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><circle cx="50" cy="35" r="4" fill="${stroke}"/></svg>`,
            'B': `<svg viewBox="0 0 100 100" style="width:100%;height:100%;"><circle cx="50" cy="25" r="8" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><polygon points="40,35 50,50 60,35 55,40 50,45 45,40" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><polygon points="35,55 65,55 70,85 30,85" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/></svg>`,
            'R': `<svg viewBox="0 0 100 100" style="width:100%;height:100%;"><rect x="35" y="15" width="30" height="20" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><polygon points="30,35 70,35 68,80 32,80" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><rect x="38" y="20" width="6" height="12" fill="${stroke}"/><rect x="56" y="20" width="6" height="12" fill="${stroke}"/></svg>`,
            'Q': `<svg viewBox="0 0 100 100" style="width:100%;height:100%;"><circle cx="50" cy="20" r="8" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><circle cx="35" cy="28" r="6" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><circle cx="65" cy="28" r="6" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><polygon points="30,40 70,40 68,85 32,85" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/></svg>`,
            'K': `<svg viewBox="0 0 100 100" style="width:100%;height:100%;"><circle cx="50" cy="25" r="8" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><rect x="48" y="18" width="4" height="15" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><polygon points="35,40 65,40 68,85 32,85" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><line x1="50" y1="38" x2="50" y2="50" stroke="${stroke}" stroke-width="2"/></svg>`,
            'p': `<svg viewBox="0 0 100 100" style="width:100%;height:100%;"><circle cx="50" cy="30" r="12" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><rect x="45" y="40" width="10" height="35" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><polygon points="40,75 60,75 65,85 35,85" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/></svg>`,
            'n': `<svg viewBox="0 0 100 100" style="width:100%;height:100%;"><path d="M30,60 Q20,40 40,20 Q60,10 70,30 Q75,50 60,70 L40,85 Q30,80 30,60" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><circle cx="50" cy="35" r="4" fill="${stroke}"/></svg>`,
            'b': `<svg viewBox="0 0 100 100" style="width:100%;height:100%;"><circle cx="50" cy="25" r="8" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><polygon points="40,35 50,50 60,35 55,40 50,45 45,40" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><polygon points="35,55 65,55 70,85 30,85" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/></svg>`,
            'r': `<svg viewBox="0 0 100 100" style="width:100%;height:100%;"><rect x="35" y="15" width="30" height="20" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><polygon points="30,35 70,35 68,80 32,80" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><rect x="38" y="20" width="6" height="12" fill="${stroke}"/><rect x="56" y="20" width="6" height="12" fill="${stroke}"/></svg>`,
            'q': `<svg viewBox="0 0 100 100" style="width:100%;height:100%;"><circle cx="50" cy="20" r="8" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><circle cx="35" cy="28" r="6" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><circle cx="65" cy="28" r="6" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><polygon points="30,40 70,40 68,85 32,85" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/></svg>`,
            'k': `<svg viewBox="0 0 100 100" style="width:100%;height:100%;"><circle cx="50" cy="25" r="8" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><rect x="48" y="18" width="4" height="15" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><polygon points="35,40 65,40 68,85 32,85" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}"/><line x1="50" y1="38" x2="50" y2="50" stroke="${stroke}" stroke-width="2"/></svg>`,
          };
          
          return svgs[piece] || '?';
        }

        function legalMoveForSquares(from, to) {
          if (!STATE.data) return null;
          const prefix = `${from}${to}`;
          const legal = STATE.data.legal_moves || [];
          const exact = legal.find((uci) => uci === prefix) || legal.find((uci) => uci.startsWith(prefix));
          return exact || null;
        }

        function createSquare(rank, file, piece, isLight) {
          const squareName = String.fromCharCode(97 + file) + (rank + 1);
          const square = document.createElement('div');
          square.className = `square ${isLight ? 'light' : 'dark'}`;
          square.dataset.square = squareName;

          if ((rank === 0 || rank === 7) && file === 0) {
            const coord = document.createElement('div');
            coord.className = 'coords';
            coord.textContent = squareName;
            square.appendChild(coord);
          }

          if (piece) {
            const glyph = document.createElement('div');
            glyph.className = 'piece';
            glyph.innerHTML = getPieceSVG(piece);
            square.appendChild(glyph);
          }

          const selected = STATE.selected === squareName;
          if (selected) square.classList.add('selected');

          const legalMoves = STATE.data ? (STATE.data.legal_moves || []) : [];
          const targets = legalMoves.filter((uci) => uci.startsWith(`${STATE.selected || ''}${squareName}`));
          if (STATE.selected && targets.length && squareName !== STATE.selected) {
            square.classList.add('target');
          }

          square.addEventListener('click', () => handleSquareClick(squareName));
          square.addEventListener('mouseenter', () => {
            if (STATE.selected && squareName !== STATE.selected && legalMoves.some((uci) => uci.startsWith(`${STATE.selected}${squareName}`))) {
              square.classList.add('hover-target');
            }
          });
          square.addEventListener('mouseleave', () => square.classList.remove('hover-target'));

          return square;
        }

        function renderBoard() {
          if (!STATE.data) return;

          const fenPieces = currentBoardSquares();
          const boardSize = Math.min(boardWrap.clientWidth, window.innerWidth - 40, 860);
          STATE.squareSize = boardSize / 8;
          boardEl.style.gridTemplateColumns = `repeat(8, ${STATE.squareSize}px)`;
          boardEl.style.gridTemplateRows = `repeat(8, ${STATE.squareSize}px)`;
          arrowLayer.setAttribute('viewBox', `0 0 ${boardSize} ${boardSize}`);

          boardEl.innerHTML = '';
          for (let rank = 7; rank >= 0; rank -= 1) {
            for (let file = 0; file < 8; file += 1) {
              const index = rank * 8 + file;
              const piece = fenPieces[index];
              const isLight = (rank + file) % 2 === 1;
              boardEl.appendChild(createSquare(rank, file, piece, isLight));
            }
          }

          drawArrows();
        }

        function drawArrows() {
          if (!STATE.data) return;

          const boardSize = STATE.squareSize * 8;
          arrowLayer.setAttribute('viewBox', `0 0 ${boardSize} ${boardSize}`);
          arrowLayer.innerHTML = '';

          const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
          defs.innerHTML = `
            <marker id="arrowHead" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 z" fill="currentColor"></path>
            </marker>
          `;
          arrowLayer.appendChild(defs);

          const candidates = (STATE.data.analysis && STATE.data.analysis.children) ? STATE.data.analysis.children.slice(0, 3) : [];
          candidates.forEach((candidate, index) => {
            const from = candidate.uci.slice(0, 2);
            const to = candidate.uci.slice(2, 4);
            const start = squareToCoords(from);
            const end = squareToCoords(to);
            const dx = end.x - start.x;
            const dy = end.y - start.y;
            const distance = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
            const shorten = Math.min(12, distance * 0.15);
            const ux = dx / distance;
            const uy = dy / distance;
            const x1 = start.x + ux * 12;
            const y1 = start.y + uy * 12;
            const x2 = end.x - ux * shorten;
            const y2 = end.y - uy * shorten;

            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', x1);
            line.setAttribute('y1', y1);
            line.setAttribute('x2', x2);
            line.setAttribute('y2', y2);
            line.setAttribute('stroke', 'currentColor');
            line.setAttribute('marker-end', 'url(#arrowHead)');
            line.classList.add('arrow');
            line.classList.add(index === 0 ? 'best' : index === 1 ? 'second' : 'third');
            if (STATE.hoveredCandidate && STATE.hoveredCandidate.uci === candidate.uci) {
              line.classList.add('active');
            }
            line.addEventListener('mouseenter', () => setHoveredCandidate(candidate));
            line.addEventListener('mouseleave', () => clearHoveredCandidate());
            arrowLayer.appendChild(line);
          });
        }

        function formatBranchScore(node) {
          if (!node) return 'n/a';
          return `W ${fmtEval(node.white_eval)} | B ${fmtEval(node.black_eval)}`;
        }

        function renderTreeNode(node, parentMove = '', depth = 0, parentEl = null) {
          if (depth > 3 || !node) return;
          
          if (depth === 0) {
            window.treeDebug = {
              node,
              children: node.children,
              childCount: node.children ? node.children.length : 0
            };
            if (node.children && node.children.length > 0) {
              window.firstChild = node.children[0];
              console.log('First child:', node.children[0]);
            }
          }
          
          const container = document.createElement('div');
          container.style.marginLeft = depth > 0 ? '12px' : '0';
          container.style.fontSize = depth === 0 ? '13px' : '11px';
          container.style.marginBottom = '2px';
          container.style.lineHeight = '1.4';
          
          const nodeEl = document.createElement('div');
          nodeEl.style.padding = '4px 6px';
          nodeEl.style.borderRadius = '4px';
          nodeEl.style.cursor = 'pointer';
          nodeEl.style.userSelect = 'none';
          nodeEl.style.transition = 'background 0.15s';
          
          // Store FEN for later SVG generation
          nodeEl.dataset.fen = node.fen || '';
          
          const moveText = parentMove || 'Root';
          const wEval = node.white_eval !== undefined ? node.white_eval.toFixed(1) : '0.0';
          const bEval = node.black_eval !== undefined ? node.black_eval.toFixed(1) : '0.0';
          
          nodeEl.innerHTML = `<span style="color:var(--accent);font-weight:600">${moveText}</span> <span style="color:var(--muted);font-size:10px">W:${wEval} B:${bEval}</span>`;
          
          nodeEl.addEventListener('mouseenter', (e) => {
            nodeEl.style.background = 'rgba(80,209,141,0.15)';
            showTreeTooltip(node, e.clientX, e.clientY);
          });
          nodeEl.addEventListener('mouseleave', () => {
            nodeEl.style.background = 'transparent';
            hideTreeTooltip();
          });
          
          container.appendChild(nodeEl);
          
          if (node.children && node.children.length > 0) {
            const childrenContainer = document.createElement('div');
            node.children.slice(0, 3).forEach((childWrapper, idx) => {
              if (!childWrapper.child) return;
              const moveLabel = childWrapper.san || childWrapper.uci || `${idx + 1}`;
              const childLabel = `${idx + 1}. ${moveLabel}`;
              const childNode = renderTreeNode(childWrapper.child, childLabel, depth + 1, childrenContainer);
              if (childNode) childrenContainer.appendChild(childNode);
            });
            if (childrenContainer.children.length > 0) {
              container.appendChild(childrenContainer);
            }
          }
          
          return container;
        }

        let tooltipTimeout = null;
        
        function showTreeTooltip(node, x, y) {
          clearTimeout(tooltipTimeout);
          
          const tooltip = document.getElementById('treeTooltip');
          if (!tooltip) return;
          
          tooltipTimeout = setTimeout(() => {
            if (!node.fen) return;
            
            // Generate SVG board from FEN using canvas and ASCII rendering
            const svg = generateBoardSVG(node.fen);
            tooltip.innerHTML = `<div style="background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:8px;font-size:11px;max-width:200px;">
              ${svg}
              <div style="margin-top:6px;color:var(--muted);text-align:center;">
                W: ${(node.white_eval || 0).toFixed(1)} | B: ${(node.black_eval || 0).toFixed(1)}
              </div>
            </div>`;
            tooltip.style.display = 'block';
            tooltip.style.left = (x + 10) + 'px';
            tooltip.style.top = (y - 120) + 'px';
          }, 200);
        }
        
        function hideTreeTooltip() {
          clearTimeout(tooltipTimeout);
          const tooltip = document.getElementById('treeTooltip');
          if (tooltip) tooltip.style.display = 'none';
        }

        function generateBoardSVG(fen) {
          // Simple SVG board generator from FEN
          const board = parseBoardFen(fen);
          let svg = '<svg viewBox="0 0 320 320" style="width:160px;height:160px;border:1px solid var(--line);border-radius:4px;">';
          
          for (let rank = 7; rank >= 0; rank--) {
            for (let file = 0; file < 8; file++) {
              const isLight = (rank + file) % 2 === 1;
              const x = file * 40;
              const y = (7 - rank) * 40;
              const color = isLight ? '#f0f0f0' : '#7a8c56';
              const index = rank * 8 + file;
              const piece = board[index];
              
              svg += `<rect x="${x}" y="${y}" width="40" height="40" fill="${color}"/>`;
              
              if (piece) {
                const fill = piece === piece.toUpperCase() ? '#fff' : '#000';
                const pieceChar = ['K', 'Q', 'R', 'B', 'N', 'P'].includes(piece.toUpperCase()) ? piece : '?';
                svg += `<text x="${x + 20}" y="${y + 28}" font-size="20" font-weight="bold" fill="${fill}" text-anchor="middle" font-family="serif">${pieceChar}</text>`;
              }
            }
          }
          
          svg += '</svg>';
          return svg;
        }

        function renderTree() {
          window.renderTreeCalled = true;
          try {
            if (!STATE.data || !STATE.data.analysis) {
              treeHeader.textContent = 'Tree';
              treeHint.textContent = 'Waiting for analysis';
              treeView.innerHTML = '<div class="empty">No analysis yet.</div>';
              return;
            }

            treeHeader.textContent = 'Alpha-Beta Tree';
            treeHint.textContent = 'Hover over moves to see positions';
            treeView.innerHTML = '';
            
            window.beforeRenderTreeNode = true;
            const treeNode = renderTreeNode(STATE.data.analysis, 'Root', 0);
            window.afterRenderTreeNode = true;
            window.treeNodeResult = treeNode;
            if (treeNode) {
              treeView.appendChild(treeNode);
            } else {
              window.treeNodeWasNull = true;
            }
          } catch (e) {
            window.renderTreeError = e.message;
            console.error('renderTree error:', e);
            treeView.innerHTML = '<div class="empty">Error rendering tree.</div>';
          }
        }

        function renderCandidates() {
          if (!STATE.data || !STATE.data.analysis) {
            candidateList.innerHTML = '<div class="empty">No candidates yet.</div>';
            return;
          }

          const candidates = (STATE.data.analysis.children || []).slice(0, 3);
          if (!candidates.length) {
            candidateList.innerHTML = '<div class="empty">No legal moves.</div>';
            return;
          }

          candidateList.innerHTML = '';
          candidates.forEach((candidate, index) => {
            const card = document.createElement('div');
            card.className = 'move-card';
            if (STATE.hoveredCandidate && STATE.hoveredCandidate.uci === candidate.uci) card.classList.add('active');
            card.innerHTML = `
              <div class="line1"><span>${index + 1}. ${candidate.san}</span><span>${candidate.uci}</span></div>
              <div class="line2"><span>White ${fmtEval(candidate.white_eval)}</span><span>Black ${fmtEval(candidate.black_eval)}</span></div>
            `;
            card.addEventListener('mouseenter', () => setHoveredCandidate(candidate));
            card.addEventListener('mouseleave', () => clearHoveredCandidate());
            candidateList.appendChild(card);
          });
        }

        function renderEvalBar() {
          if (!STATE.data || !STATE.data.analysis) {
            evalValue.textContent = '0.0';
            evalFillWhite.style.height = '0%';
            evalFillBlack.style.height = '0%';
            return;
          }

          const whiteEval = Number(STATE.data.analysis.white_eval || 0);
          const clamped = clamp(whiteEval, -2000, 2000);
          const magnitude = Math.min(Math.abs(clamped) / 2000, 1) * 50;
          evalValue.textContent = fmtEval(whiteEval);

          if (clamped >= 0) {
            evalFillWhite.style.height = `${magnitude}%`;
            evalFillWhite.style.bottom = '50%';
            evalFillWhite.style.top = 'auto';
            evalFillWhite.style.display = 'block';
            evalFillBlack.style.height = '0%';
          } else {
            evalFillBlack.style.height = `${magnitude}%`;
            evalFillBlack.style.top = '50%';
            evalFillBlack.style.bottom = 'auto';
            evalFillBlack.style.display = 'block';
            evalFillWhite.style.height = '0%';
          }

          document.getElementById('evalLabel').textContent = `White ${fmtEval(whiteEval)} / Black ${fmtEval(-whiteEval)}`;
        }

        function updateStatus() {
          if (!STATE.data) {
            turnBadge.textContent = 'Loading...';
            statusText.textContent = 'Loading board...';
            return;
          }

          turnBadge.textContent = STATE.data.game_over ? `Game over · ${STATE.data.result}` : `${STATE.data.turn === 'white' ? 'White' : 'Black'} to move`;
          const history = (STATE.data.history_san || []).join(' ');
          statusText.textContent = STATE.data.game_over
            ? `Result: ${STATE.data.result} · ${STATE.data.termination || 'finished'}`
            : (history ? `Moves: ${history}` : 'Your move as White.');
        }

        function setHoveredCandidate(candidate) {
          STATE.hoveredCandidate = candidate;
          renderCandidates();
          renderTree();
          drawArrows();
        }

        function clearHoveredCandidate() {
          STATE.hoveredCandidate = null;
          renderCandidates();
          renderTree();
          drawArrows();
        }

        function handleSquareClick(square) {
          if (!STATE.data || STATE.data.game_over) return;
          const piece = getPieceAt(square);
          const turn = STATE.data.turn;

          if (!STATE.selected) {
            if (piece && pieceColor(piece) === turn) {
              STATE.selected = square;
              renderBoard();
            }
            return;
          }

          if (STATE.selected === square) {
            STATE.selected = null;
            renderBoard();
            return;
          }

          const move = legalMoveForSquares(STATE.selected, square);
          if (move) {
            submitMove(move);
            return;
          }

          if (piece && pieceColor(piece) === turn) {
            STATE.selected = square;
            renderBoard();
          } else {
            STATE.selected = null;
            renderBoard();
          }
        }

        async function loadState() {
          const response = await fetch('/api/state');
          STATE.data = await response.json();
          STATE.selected = null;
          STATE.hoveredCandidate = null;
          renderBoard();
          renderCandidates();
          renderTree();
          renderEvalBar();
          updateStatus();
        }

        async function submitMove(moveUci) {
          const response = await fetch('/api/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ move: moveUci }),
          });

          if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Move rejected' }));
            statusText.textContent = error.error || 'Move rejected';
            return;
          }

          STATE.data = await response.json();
          STATE.selected = null;
          STATE.hoveredCandidate = null;
          renderBoard();
          renderCandidates();
          renderTree();
          renderEvalBar();
          updateStatus();
        }

        async function newGame() {
          await fetch('/api/reset', { method: 'POST' });
          await loadState();
        }

        newGameBtn.addEventListener('click', newGame);
        window.addEventListener('resize', () => renderBoard());
        document.addEventListener('mouseleave', () => {
          if (STATE.hoveredCandidate) clearHoveredCandidate();
        });

        loadState();
      </script>
    </body>
    </html>
    """
).strip()


def load_config() -> dict[str, Any]:
    config_path = Path("config.yml")
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            return loaded
    return {}


@dataclass(slots=True)
class ArenaSettings:
    model_path: Path
    search_depth: int = 3
    search_timeout_seconds: float = 2.75
    analysis_depth: int = 3
    analysis_breadth: int = 3
    analysis_timeout_seconds: float = 1.25


def build_settings() -> ArenaSettings:
    config = load_config()
    engine_config_raw = config.get("engine", {})
    engine_config = engine_config_raw if isinstance(engine_config_raw, dict) else {}
    homemade_options_raw = engine_config.get("homemade_options", {})
    homemade_options = homemade_options_raw if isinstance(homemade_options_raw, dict) else {}

    def coerce_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def coerce_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    model_path = Path(str(homemade_options.get("model_path", "engines/train_eval/model_artifacts/policy_xgb.joblib")))
    search_depth = coerce_int(homemade_options.get("search_depth", 3), 3)
    search_timeout_seconds = coerce_float(homemade_options.get("search_timeout_seconds", 2.75), 2.75)

    analysis_depth = max(2, min(search_depth, 3))
    analysis_timeout_seconds = min(search_timeout_seconds, 1.25)
    return ArenaSettings(
        model_path=model_path,
        search_depth=search_depth,
        search_timeout_seconds=search_timeout_seconds,
        analysis_depth=analysis_depth,
        analysis_breadth=3,
        analysis_timeout_seconds=analysis_timeout_seconds,
    )


class ChessArenaSession:
    def __init__(self, settings: ArenaSettings) -> None:
        global _model_loader
        
        self.settings = settings
        self.lock = threading.RLock()
        self.board = chess.Board()

        if not self.settings.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.settings.model_path}")

        evaluator = PositionEvaluator()
        
        # Get the global model loader (starts loading if not already started)
        if _model_loader is None:
            _model_loader = BackgroundModelLoader(self.settings.model_path)
            _model_loader.start_loading()
        
        # Don't wait for model to load; use it once available
        self.model_loader = _model_loader
        self.searcher = AlphaBetaSearch(
            evaluator=evaluator,
            depth=self.settings.search_depth,
            policy_runtime=None,  # Will be set lazily
            use_policy_ordering=False,  # Will be updated when model loads
            time_limit_seconds=self.settings.search_timeout_seconds,
        )
        self._policy_checked = False

    def _ensure_policy_loaded(self) -> None:
        """Check if policy model has loaded and update searcher if so."""
        if self._policy_checked:
            return  # Already checked and doesn't need to check again
        
        if self.model_loader.ready.is_set():
            policy = self.model_loader.policy
            if policy:
                self.searcher.policy = policy
                self.searcher.use_policy_ordering = True
                logger.info("Session: Policy model now available; enabling policy guidance")
            self._policy_checked = True

    def reset(self) -> None:
        with self.lock:
            self.board = chess.Board()

    def _history_san(self, board: chess.Board) -> list[str]:
        history_board = chess.Board()
        history: list[str] = []
        for move in board.move_stack:
            history.append(history_board.san(move))
            history_board.push(move)
        return history

    def _board_state(self, board: chess.Board) -> dict[str, Any]:
        self._ensure_policy_loaded()
        analysis = self.searcher.analyze_root(
            board,
            max_depth=self.settings.analysis_depth,
            breadth=self.settings.analysis_breadth,
            time_limit_seconds=self.settings.analysis_timeout_seconds,
        )
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            result = "*"
            termination = None
        else:
            result = outcome.result()
            termination = outcome.termination.name.replace("_", " ")

        return {
            "fen": board.fen(),
            "turn": "white" if board.turn == chess.WHITE else "black",
            "game_over": board.is_game_over(claim_draw=True),
            "result": result,
            "termination": termination,
            "legal_moves": [move.uci() for move in board.legal_moves],
            "analysis": analysis,
            "history_san": self._history_san(board),
        }

    def state(self) -> dict[str, Any]:
        with self.lock:
            return self._board_state(self.board.copy(stack=True))

    def play_move(self, move_uci: str) -> dict[str, Any]:
        with self.lock:
            if self.board.is_game_over(claim_draw=True):
                return self._board_state(self.board.copy(stack=True))

            move = chess.Move.from_uci(move_uci)
            if move not in self.board.legal_moves:
                raise ValueError(f"Illegal move: {move_uci}")

            self.board.push(move)
            if not self.board.is_game_over(claim_draw=True):
                self._ensure_policy_loaded()
                bot_move = self.searcher.choose(self.board, list(self.board.legal_moves))
                self.board.push(bot_move)

            return self._board_state(self.board.copy(stack=True))


class ArenaRequestHandler(BaseHTTPRequestHandler):
    server_version = "ChessArena/1.0"

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route == "/":
            self._write_html(HTML_PAGE)
            return
        if route == "/api/state":
            self._write_json(self.server.session.state())
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route == "/api/reset":
            self.server.session.reset()
            self._write_json(self.server.session.state())
            return

        if route == "/api/move":
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length) or b"{}")
            move_uci = str(payload.get("move", "")).strip()
            if not move_uci:
                self._write_json({"error": "Missing move"}, status=400)
                return

            try:
                state = self.server.session.play_move(move_uci)
            except ValueError as exc:
                self._write_json({"error": str(exc)}, status=400)
                return
            except Exception as exc:  # noqa: BLE001
                logger.exception("Move handling failed")
                self._write_json({"error": str(exc)}, status=500)
                return

            self._write_json(state)
            return

        self.send_error(404, "Not found")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        logger.info(format, *args)

    def _write_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _write_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def make_server(port: int) -> ThreadingHTTPServer:
    global _model_loader
    
    settings = build_settings()
    
    # Initialize global model loader if not already done
    if _model_loader is None:
        _model_loader = BackgroundModelLoader(settings.model_path)
        _model_loader.start_loading()
    
    session = ChessArenaSession(settings)

    class ArenaServer(ThreadingHTTPServer):
        def __init__(self, server_address: tuple[str, int], RequestHandlerClass: type[BaseHTTPRequestHandler]):
            super().__init__(server_address, RequestHandlerClass)
            self.session = session

    return ArenaServer(("127.0.0.1", port), ArenaRequestHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local chess arena UI.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the local server to.")
    args = parser.parse_args()

    server = make_server(args.port)
    logger.info("Chess UI running on http://127.0.0.1:%d", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()