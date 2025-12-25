import os
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# ---- Optional visualization helpers (won't crash if missing) ----
try:
    import pygame
except Exception:
    pygame = None

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


# =========================
# Checkers Environment
# American checkers:
# - 8x8 board, only dark squares used
# - men move forward diagonally
# - kings move both directions diagonally
# - captures are mandatory
# - multi-jump captures allowed and must be continued
# - win by capturing all or opponent has no legal moves
# =========================

EMPTY = 0
P1_MAN = 1
P1_KING = 2
P2_MAN = -1
P2_KING = -2

BOARD_SIZE = 8

# Directions: (dr, dc)
P1_DIRS = [(-1, -1), (-1, 1)]  # P1 moves "up" (row decreasing)
P2_DIRS = [(1, -1), (1, 1)]    # P2 moves "down" (row increasing)
KING_DIRS = P1_DIRS + P2_DIRS


@dataclass(frozen=True)
class Move:
    # path is sequence of squares visited (including start and final), e.g. [(r0,c0),(r1,c1),...]
    # captures is list of captured squares (r,c) in order
    path: Tuple[Tuple[int, int], ...]
    captures: Tuple[Tuple[int, int], ...]


def is_dark_square(r: int, c: int) -> bool:
    return (r + c) % 2 == 1


def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def piece_owner(p: int) -> int:
    if p > 0:
        return 1
    if p < 0:
        return -1
    return 0


def is_king(p: int) -> bool:
    return abs(p) == 2


class CheckersEnv:
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=np.int8)
        self.player = 1  # 1 = P1 to move, -1 = P2 to move
        self.done = False
        self.winner: Optional[int] = None  # 1, -1, or None

    def reset(self):
        self.board[:] = 0
        # Standard initial setup: 12 pieces each on dark squares
        for r in range(3):  # top rows for P2
            for c in range(8):
                if is_dark_square(r, c):
                    self.board[r, c] = P2_MAN
        for r in range(5, 8):  # bottom rows for P1
            for c in range(8):
                if is_dark_square(r, c):
                    self.board[r, c] = P1_MAN

        self.player = 1
        self.done = False
        self.winner = None
        return self._obs()

    def _obs(self) -> np.ndarray:
        # 4 channels: current player's men, current player's kings, opponent men, opponent kings
        # plus a side-to-move channel baked in by perspective flipping
        b = self.board.copy()

        # Perspective: always encode "side to move" as +1 pieces on our side
        if self.player == -1:
            b = -b

        ch = np.zeros((5, 8, 8), dtype=np.float32)
        ch[0] = (b == P1_MAN).astype(np.float32)
        ch[1] = (b == P1_KING).astype(np.float32)
        ch[2] = (b == P2_MAN).astype(np.float32)
        ch[3] = (b == P2_KING).astype(np.float32)
        ch[4] = np.ones((8, 8), dtype=np.float32)  # side-to-move indicator (always 1 after perspective)
        return ch

    def step(self, move: Move):
        if self.done:
            raise RuntimeError("Game is over. Call reset().")

        legal = self.legal_moves()
        if move not in legal:
            raise ValueError("Illegal move.")

        self._apply_move(move)

        # Check terminal
        opp_moves = self.legal_moves()  # legal_moves uses current self.player
        if len(opp_moves) == 0:
            self.done = True
            self.winner = -self.player  # if current player has no moves, previous player won
        else:
            self.done = False
            self.winner = None

        # Reward from the perspective of the player who just moved:
        # If game ended, previous player gets +1 for win / -1 for loss
        reward = 0.0
        if self.done:
            reward = 1.0  # the side who made the move won
        return self._obs(), reward, self.done, {"winner": self.winner}

    def legal_moves(self) -> List[Move]:
        # Generate all legal moves for current self.player
        captures = self._all_captures()
        if captures:
            return captures  # forced capture
        return self._all_simple_moves()

    def _all_simple_moves(self) -> List[Move]:
        moves: List[Move] = []
        for r in range(8):
            for c in range(8):
                p = self.board[r, c]
                if piece_owner(p) != self.player:
                    continue
                dirs = KING_DIRS if is_king(p) else (P1_DIRS if self.player == 1 else P2_DIRS)
                for dr, dc in dirs:
                    r2, c2 = r + dr, c + dc
                    if in_bounds(r2, c2) and self.board[r2, c2] == EMPTY:
                        moves.append(Move(path=((r, c), (r2, c2)), captures=()))
        return moves

    def _all_captures(self) -> List[Move]:
        # Captures may be multi-jump; generate all sequences.
        moves: List[Move] = []
        for r in range(8):
            for c in range(8):
                p = self.board[r, c]
                if piece_owner(p) != self.player:
                    continue
                self._dfs_captures_from(r, c, moves)
        return moves

    def _dfs_captures_from(self, r0: int, c0: int, out_moves: List[Move]):
        p0 = self.board[r0, c0]
        dirs = KING_DIRS if is_king(p0) else (P1_DIRS if self.player == 1 else P2_DIRS)

        # DFS stack carries board mutations; easiest is recursion with undo
        def recurse(path: List[Tuple[int, int]], caps: List[Tuple[int, int]]):
            r, c = path[-1]
            p = self.board[r, c]
            found = False

            dirs_local = KING_DIRS if is_king(p) else dirs
            for dr, dc in dirs_local:
                rm, cm = r + dr, c + dc
                r2, c2 = r + 2 * dr, c + 2 * dc
                if not (in_bounds(r2, c2) and in_bounds(rm, cm)):
                    continue
                mid = self.board[rm, cm]
                if mid == EMPTY or piece_owner(mid) == self.player:
                    continue
                if self.board[r2, c2] != EMPTY:
                    continue

                # do capture
                captured_piece = self.board[rm, cm]
                moving_piece = self.board[r, c]
                self.board[rm, cm] = EMPTY
                self.board[r, c] = EMPTY
                self.board[r2, c2] = moving_piece

                # kinging can happen at end of move; in American checkers,
                # multi-jump continues even if you land on king row? Many rulesets stop;
                # we’ll apply kinging only after the sequence ends for simplicity.
                found = True
                path.append((r2, c2))
                caps.append((rm, cm))
                recurse(path, caps)
                caps.pop()
                path.pop()

                # undo
                self.board[r, c] = moving_piece
                self.board[rm, cm] = captured_piece
                self.board[r2, c2] = EMPTY

            if not found and caps:
                out_moves.append(Move(path=tuple(path), captures=tuple(caps)))

        recurse([(r0, c0)], [])

    def _apply_move(self, move: Move):
        path = list(move.path)
        start = path[0]
        end = path[-1]

        p = self.board[start]
        self.board[start] = EMPTY

        # remove captures
        for (cr, cc) in move.captures:
            self.board[cr, cc] = EMPTY

        # place piece
        self.board[end] = p

        # kinging
        if p == P1_MAN and end[0] == 0:
            self.board[end] = P1_KING
        elif p == P2_MAN and end[0] == 7:
            self.board[end] = P2_KING

        # switch player
        self.player *= -1

    def clone(self) -> "CheckersEnv":
        e = CheckersEnv()
        e.board = self.board.copy()
        e.player = self.player
        e.done = self.done
        e.winner = self.winner
        return e


# =========================
# Neural Net: Value Network
# Input: (5,8,8)
# Output: scalar in [-1,1] predicting outcome for side to move
# =========================

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.head(x)


# =========================
# Training (self-play TD learning)
# =========================

@dataclass
class TrainConfig:
    episodes: int = 2000
    max_moves: int = 200
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 0.40
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 1500
    device: str = "cpu"
    save_path: str = "checkers_value.pt"


def epsilon_by_episode(ep: int, cfg: TrainConfig) -> float:
    t = min(1.0, ep / max(1, cfg.epsilon_decay_episodes))
    return cfg.epsilon_start + t * (cfg.epsilon_end - cfg.epsilon_start)


@torch.no_grad()
def pick_move_value_greedy(env: CheckersEnv, net: ValueNet, eps: float, device: str) -> Move:
    moves = env.legal_moves()
    if not moves:
        raise RuntimeError("No legal moves.")
    if random.random() < eps:
        return random.choice(moves)

    # Evaluate each candidate by value of resulting position (for the next player).
    # Since env.step flips player, the value output is always for side-to-move.
    # After we apply a move, it's opponent's turn; good for us means BAD for them => minimize their value.
    best = None
    best_score = float("inf")
    for mv in moves:
        e2 = env.clone()
        e2._apply_move(mv)
        obs = torch.tensor(e2._obs(), dtype=torch.float32, device=device).unsqueeze(0)
        v = net(obs).item()
        if v < best_score:
            best_score = v
            best = mv
    return best if best is not None else random.choice(moves)


def train(cfg: TrainConfig):
    device = torch.device(cfg.device)
    env = CheckersEnv()
    net = ValueNet().to(device)
    opt = optim.Adam(net.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    net.train()

    win_counts = {1: 0, -1: 0, 0: 0}

    for ep in range(1, cfg.episodes + 1):
        env.reset()
        eps = epsilon_by_episode(ep, cfg)

        # store transitions as (obs, target)
        transitions: List[Tuple[np.ndarray, float]] = []

        move_count = 0
        while not env.done and move_count < cfg.max_moves:
            obs = env._obs()
            mv = pick_move_value_greedy(env, net, eps, cfg.device)
            next_obs, reward, done, info = env.step(mv)

            # We will compute TD targets later using bootstrap
            transitions.append((obs, 0.0))
            move_count += 1

            if done:
                # If the mover won, reward=+1 for mover.
                # Our obs is "side to move" at that time, so reward pertains to that mover.
                # We need to assign final outcome (+1/-1) alternating backward.
                pass

        # Determine outcome from the perspective of "side to move" at each stored obs:
        # If winner is 1, then positions where player=1 to move should trend +1, and player=-1 to move trend -1.
        winner = env.winner if env.done else 0
        win_counts[winner] = win_counts.get(winner, 0) + 1

        # Re-run game to align players for each obs:
        # easiest: simulate again from fresh using the stored moves? we didn't store them.
        # Instead we compute targets with TD(0) online during play in a second loop:
        # We'll do a simpler method: while playing, compute TD target immediately.
        # So for now, re-play the episode quickly by sampling random? Not correct.

        # Fix: switch to online TD updates: do another episode loop but with updates each step.
        # We'll implement that now for correctness.

        # ---- Online TD episode (redo properly) ----
        env.reset()
        move_count = 0
        while not env.done and move_count < cfg.max_moves:
            obs = env._obs()
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            mv = pick_move_value_greedy(env, net, eps, cfg.device)
            next_obs, reward, done, info = env.step(mv)

            with torch.no_grad():
                if done:
                    target = torch.tensor([[reward]], dtype=torch.float32, device=device)
                else:
                    next_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    target = torch.tensor([[0.0]], dtype=torch.float32, device=device) + cfg.gamma * net(next_t)

            pred = net(obs_t)
            loss = loss_fn(pred, target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            move_count += 1

        if ep % 50 == 0:
            total = win_counts[1] + win_counts[-1] + win_counts[0]
            print(
                f"ep {ep:5d} | eps={eps:.3f} | "
                f"wins P1={win_counts[1]} P2={win_counts[-1]} draws/limit={win_counts[0]} | total={total}"
            )

        if ep % 200 == 0:
            torch.save(net.state_dict(), cfg.save_path)
            print(f"saved: {cfg.save_path}")

    torch.save(net.state_dict(), cfg.save_path)
    print(f"done, saved: {cfg.save_path}")


# =========================
# Visualization + Video
# =========================

# =========================
# Pygame 3D-style rendering + Human vs AI Play
# =========================

def _shade(color, factor):
    """Darken/lighten a (r,g,b) tuple by factor (0..2)."""
    r, g, b = color
    return (
        max(0, min(255, int(r * factor))),
        max(0, min(255, int(g * factor))),
        max(0, min(255, int(b * factor))),
    )


def draw_board_3d(
    screen,
    board: np.ndarray,
    cell_size: int = 80,
    selected: Optional[Tuple[int, int]] = None,
    origin_squares: Optional[set] = None,
    capture_origins: Optional[set] = None,
    dest_squares: Optional[set] = None,
    status_text: str = "",
):
    """
    Draw an 8x8 board with a slightly 3D look, highlighting:
      - origin_squares: where you can move from
      - capture_origins: where you must move from (captures)
      - dest_squares: where the selected piece can land
    """
    if origin_squares is None:
        origin_squares = set()
    if capture_origins is None:
        capture_origins = set()
    if dest_squares is None:
        dest_squares = set()

    width = board.shape[1] * cell_size
    height = board.shape[0] * cell_size + 30  # extra status bar

    LIGHT_SQ = (230, 230, 230)
    DARK_SQ = (130, 120, 120)
    BG_COLOR = (240, 240, 240)
    HUMAN_COLOR = (200, 60, 60)
    HUMAN_KING = (255, 120, 120)
    AI_COLOR = (40, 40, 40)
    AI_KING = (120, 120, 120)

    screen.fill(BG_COLOR)

    font = pygame.font.SysFont("consolas", 18)

    # Board
    for r in range(8):
        for c in range(8):
            x = c * cell_size
            y = r * cell_size

            base = DARK_SQ if is_dark_square(r, c) else LIGHT_SQ
            # top surface
            pygame.draw.rect(screen, base, (x, y, cell_size, cell_size))

            # 3D bevel: darker bottom strip
            bevel_color = _shade(base, 0.8)
            pygame.draw.rect(
                screen,
                bevel_color,
                (x, y + int(cell_size * 0.65), cell_size, int(cell_size * 0.35)),
            )

            # highlights
            rect_inner = (x + 3, y + 3, cell_size - 6, cell_size - 6)
            if (r, c) in capture_origins:
                pygame.draw.rect(screen, (255, 140, 0), rect_inner, 3)  # orange border
            elif (r, c) in origin_squares:
                pygame.draw.rect(screen, (0, 180, 255), rect_inner, 2)  # blue border

            if (r, c) in dest_squares:
                pygame.draw.rect(screen, (50, 220, 50), rect_inner, 3)  # green border

            if selected is not None and (r, c) == selected:
                pygame.draw.rect(screen, (255, 255, 0), rect_inner, 3)  # yellow

            p = int(board[r, c])
            if p == EMPTY:
                continue

            owner = piece_owner(p)
            is_k = is_king(p)

            if owner == 1:
                base_col = HUMAN_KING if is_k else HUMAN_COLOR
            else:
                base_col = AI_KING if is_k else AI_COLOR

            cx = x + cell_size // 2
            cy = y + cell_size // 2
            radius = int(cell_size * 0.33)

            # shadow
            shadow_rect = pygame.Rect(
                cx - radius + 4, cy - radius + 6, radius * 2, radius * 2
            )
            pygame.draw.ellipse(screen, (40, 40, 40), shadow_rect)

            # main piece
            pygame.draw.circle(screen, base_col, (cx, cy - 1), radius)

            # top highlight
            highlight_col = _shade(base_col, 1.3)
            pygame.draw.arc(
                screen,
                highlight_col,
                (cx - radius, cy - radius - 4, radius * 2, radius * 2),
                3.5,
                6.0,
                2,
            )

            # king ring
            if is_k:
                pygame.draw.circle(screen, (255, 255, 255), (cx, cy - 1), radius // 2, 2)

    # Status bar
    pygame.draw.rect(screen, (210, 210, 210), (0, height - 30, width, 30))
    txt = font.render(status_text, True, (0, 0, 0))
    screen.blit(txt, (10, height - 24))

    pygame.display.flip()


def human_select_move_with_hints(env: "CheckersEnv", human_side: int, screen, cell_size: int = 80):
    """
    Human input with:
      - 3D board
      - highlighted legal origins / mandatory capture origins
      - highlighted destinations for selected piece
    Returns a Move or None if user quits.
    """
    clock = pygame.time.Clock()
    selected: Optional[Tuple[int, int]] = None

    while True:
        clock.tick(60)
        moves = env.legal_moves()
        if not moves:
            return None

        # Find all origins and capture origins
        origins = set()
        cap_origins = set()
        for mv in moves:
            o = mv.path[0]
            origins.add(o)
            if mv.captures:
                cap_origins.add(o)
        # If any captures exist, only those origins are actually legal in checkers.
        if cap_origins:
            origins = cap_origins

        # Destination squares for currently selected piece (if any)
        dests = set()
        if selected is not None:
            for mv in moves:
                if mv.path[0] == selected:
                    dests.add(mv.path[-1])

        status = f"Your turn (player {human_side}). Click a highlighted piece, then a green target."
        draw_board_3d(
            screen,
            env.board,
            cell_size=cell_size,
            selected=selected,
            origin_squares=origins,
            capture_origins=cap_origins,
            dest_squares=dests,
            status_text=status,
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return None
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = pygame.mouse.get_pos()
                c = mx // cell_size
                r = my // cell_size
                if not (0 <= r < 8 and 0 <= c < 8):
                    continue

                if selected is None:
                    # select a piece that has legal moves and belongs to human
                    p = int(env.board[r, c])
                    if p != EMPTY and piece_owner(p) == human_side and (r, c) in origins:
                        selected = (r, c)
                else:
                    # attempt to pick a destination
                    possible = [
                        mv for mv in moves
                        if mv.path[0] == selected and mv.path[-1] == (r, c)
                    ]
                    if possible:
                        return possible[0]
                    else:
                        # click same square to deselect
                        if (r, c) == selected:
                            selected = None
                        # or click another origin to switch selection
                        elif (r, c) in origins:
                            selected = (r, c)
                        else:
                            # invalid; keep selection
                            pass

def animate_move_pygame(
    screen,
    env_before: "CheckersEnv",
    env_after: "CheckersEnv",
    move: Move,
    human_side: int,
    step_delay_ms: int = 140,
    cell_size: int = 80,
    status_prefix: str = "",
):
    """
    Animate a move path step-by-step on a 3D board.
    We simulate on a copy of env_before.board so we don't touch the real env.
    """
    board_anim = env_before.board.copy()
    piece = board_anim[move.path[0]]
    board_anim[move.path[0]] = EMPTY

    # We expect len(captures) == number of jumps for a capture move
    captures = list(move.captures)

    for i in range(len(move.path) - 1):
        r1, c1 = move.path[i]
        r2, c2 = move.path[i + 1]

        # place piece at landing square
        board_anim[r2, c2] = piece

        # remove captured piece for this jump (if any)
        if i < len(captures):
            cr, cc = captures[i]
            board_anim[cr, cc] = EMPTY

        draw_board_3d(
            screen,
            board_anim,
            cell_size=cell_size,
            status_text=f"{status_prefix} Moving from {move.path[0]} to {move.path[-1]}",
        )
        pygame.time.delay(step_delay_ms)

        # clear current pos if we have another step to go
        if i < len(move.path) - 2:
            board_anim[r2, c2] = EMPTY

    # final board should match env_after.board (for sanity, but we don't assert)
    draw_board_3d(
        screen,
        env_after.board,
        cell_size=cell_size,
        status_text=status_prefix + " Move complete.",
    )

def play_vs_ai_pygame(model_path: str, human_side: int = 1, device: str = "cpu", cell_size: int = 80):
    if pygame is None:
        raise RuntimeError("pygame not installed. pip install pygame")

    pygame.init()
    width = 8 * cell_size
    height = 8 * cell_size + 30
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("NeuroCheckers - Play vs AI")

    env = CheckersEnv()
    env.reset()

    net = ValueNet().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    clock = pygame.time.Clock()
    running = True

    # If human is player -1, let AI move first
    if human_side == -1 and env.player == 1:
        env_before = env.clone()
        mv = pick_move_value_greedy(env, net, eps=0.0, device=device)
        env.step(mv)
        animate_move_pygame(
            screen,
            env_before,
            env,
            mv,
            human_side=human_side,
            status_prefix="AI (first move): ",
            cell_size=cell_size,
        )

    while running:
        clock.tick(60)

        if env.done:
            if env.winner == human_side:
                status = "Game over: YOU WIN! Press ESC or close window."
            elif env.winner == -human_side:
                status = "Game over: AI wins. Press ESC or close window."
            else:
                status = "Game over: no moves. Press ESC or close window."

            draw_board_3d(screen, env.board, cell_size=cell_size, status_text=status)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            continue

        if env.player == human_side:
            mv = human_select_move_with_hints(env, human_side, screen, cell_size=cell_size)
            if mv is None:
                # user quit
                running = False
                continue
            env_before = env.clone()
            env.step(mv)
            animate_move_pygame(
                screen,
                env_before,
                env,
                mv,
                human_side=human_side,
                status_prefix="Your move: ",
                cell_size=cell_size,
            )
        else:
            # AI turn
            draw_board_3d(
                screen,
                env.board,
                cell_size=cell_size,
                status_text=f"AI (player {env.player}) is thinking...",
            )
            pygame.display.flip()
            pygame.time.delay(250)

            env_before = env.clone()
            mv = pick_move_value_greedy(env, net, eps=0.0, device=device)
            env.step(mv)
            animate_move_pygame(
                screen,
                env_before,
                env,
                mv,
                human_side=human_side,
                status_prefix="AI move: ",
                cell_size=cell_size,
            )

    pygame.quit()


def render_board_rgb(board: np.ndarray, cell: int = 80) -> np.ndarray:
    """
    Headless renderer (no pygame required) -> returns RGB image.
    """
    h = w = 8 * cell
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # colors
    light = np.array([230, 230, 230], dtype=np.uint8)
    dark = np.array([120, 120, 120], dtype=np.uint8)

    red = np.array([200, 60, 60], dtype=np.uint8)
    red_k = np.array([255, 120, 120], dtype=np.uint8)
    blk = np.array([30, 30, 30], dtype=np.uint8)
    blk_k = np.array([90, 90, 90], dtype=np.uint8)

    # draw squares
    for r in range(8):
        for c in range(8):
            y0, y1 = r * cell, (r + 1) * cell
            x0, x1 = c * cell, (c + 1) * cell
            img[y0:y1, x0:x1] = dark if is_dark_square(r, c) else light

            p = board[r, c]
            if p == EMPTY:
                continue

            # simple filled circle raster (no cv2)
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            radius = int(cell * 0.33)
            yy, xx = np.ogrid[:cell, :cell]
            mask = (yy - cell / 2) ** 2 + (xx - cell / 2) ** 2 <= radius ** 2

            piece = red if p == P1_MAN else red_k if p == P1_KING else blk if p == P2_MAN else blk_k
            img[y0:y1, x0:x1][mask] = piece

    return img


def record_selfplay_video(
    model_path: str,
    out_mp4: str = "checkers_play.mp4",
    max_moves: int = 200,
    fps: int = 15,
    speed: float = 1.0,
    device: str = "cpu",
):
    """
    Record a self-play game to MP4.
      - fps: base frames per second
      - speed: multiplier; >1.0 = faster, <1.0 = slower
    """
    if imageio is None:
        raise RuntimeError("imageio not installed. pip install imageio")

    env = CheckersEnv()
    env.reset()

    net = ValueNet().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    frames = []
    moves = 0

    # one frame before any move
    frames.append(render_board_rgb(env.board, cell=80))

    while not env.done and moves < max_moves:
        mv = pick_move_value_greedy(env, net, eps=0.0, device=device)  # deterministic
        env.step(mv)
        frames.append(render_board_rgb(env.board, cell=80))
        moves += 1

    effective_fps = max(1, int(fps * speed))
    imageio.mimsave(out_mp4, frames, fps=effective_fps)
    print(f"wrote video: {out_mp4} | winner={env.winner} | moves={moves} | fps={effective_fps}")



def save_network_diagram_simple(path_png: str = "network_diagram.png"):
    """
    Simple diagram without graphviz: draws boxes and arrows.
    """
    import matplotlib.pyplot as plt

    layers = [
        "Input (5x8x8)",
        "Conv 3x3: 5->32 + ReLU",
        "Conv 3x3: 32->64 + ReLU",
        "Flatten (64*8*8)",
        "FC 256 + ReLU",
        "FC 128 + ReLU",
        "FC 1 + Tanh",
        "Output: Value in [-1,1]",
    ]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis("off")

    y = 0.95
    for i, text in enumerate(layers):
        ax.text(0.5, y, text, ha="center", va="center", fontsize=12,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black"))
        if i < len(layers) - 1:
            ax.annotate("", xy=(0.5, y - 0.08), xytext=(0.5, y - 0.02),
                        arrowprops=dict(arrowstyle="->"))
        y -= 0.12

    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    print(f"wrote: {path_png}")


def save_network_graph_torchviz(path_png: str = "network_graph.png", device: str = "cpu"):
    """
    Optional: requires torchviz + graphviz installed.
    """
    try:
        from torchviz import make_dot
    except Exception:
        raise RuntimeError("torchviz not installed. pip install torchviz graphviz")

    net = ValueNet().to(device)
    x = torch.zeros((1, 5, 8, 8), dtype=torch.float32, device=device)
    y = net(x)
    dot = make_dot(y, params=dict(net.named_parameters()))
    dot.format = "png"
    dot.render(path_png.replace(".png", ""), cleanup=True)
    print(f"wrote: {path_png}")


# =========================
# Main entry
# =========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--episodes", type=int, default=2000)
    p_train.add_argument("--device", type=str, default="cpu")
    p_train.add_argument("--save", type=str, default="checkers_value.pt")

    p_vid = sub.add_parser("video")
    p_vid.add_argument("--model", type=str, default="checkers_value.pt")
    p_vid.add_argument("--out", type=str, default="checkers_play.mp4")
    p_vid.add_argument("--device", type=str, default="cpu")
    p_vid.add_argument("--fps", type=int, default=15, help="Base frames per second.")
    p_vid.add_argument("--speed", type=float, default=1.0,
                       help="Speed multiplier for playback ( >1 faster, <1 slower ).")

    p_play = sub.add_parser("play")
    p_play.add_argument("--model", type=str, default="checkers_value.pt")
    p_play.add_argument("--device", type=str, default="cpu")
    p_play.add_argument("--human-side", type=int, default=1, choices=[1, -1],
                        help="1 = Human is Player 1 (moves first), -1 = Player 2.")
    p_play.add_argument("--cell-size", type=int, default=80, help="Board cell size in pixels.")


    p_diag = sub.add_parser("netdiag")
    p_diag.add_argument("--out", type=str, default="network_diagram.png")

    p_graph = sub.add_parser("netgraph")
    p_graph.add_argument("--out", type=str, default="network_graph.png")
    p_graph.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    if args.cmd == "train":
        cfg = TrainConfig(episodes=args.episodes, device=args.device, save_path=args.save)
        train(cfg)

    elif args.cmd == "video":
        record_selfplay_video(
            model_path=args.model,
            out_mp4=args.out,
            fps=args.fps,
            speed=args.speed,
            device=args.device,
        )

    elif args.cmd == "play":
        play_vs_ai_pygame(
            model_path=args.model,
            human_side=args.human_side,
            device=args.device,
            cell_size=args.cell_size,
        )


    elif args.cmd == "netdiag":
        save_network_diagram_simple(args.out)

    elif args.cmd == "netgraph":
        save_network_graph_torchviz(args.out, device=args.device)
