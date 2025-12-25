import argparse
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Optional deps
try:
    import pygame
except Exception:
    pygame = None

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


# =========================
# Connect 4 Environment
# Standard 7x6 board, 1 = player1, -1 = player2
# =========================

ROWS = 6
COLS = 7


class Connect4Env:
    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.player = 1  # 1 or -1, side to move
        self.done = False
        self.winner: Optional[int] = None  # 1, -1, or 0 for draw

    def reset(self):
        self.board[:] = 0
        self.player = 1
        self.done = False
        self.winner = None
        return self._obs()

    def clone(self) -> "Connect4Env":
        env = Connect4Env()
        env.board = self.board.copy()
        env.player = self.player
        env.done = self.done
        env.winner = self.winner
        return env

    def legal_moves(self) -> List[int]:
        if self.done:
            return []
        # A column is legal if its top cell is empty
        return [c for c in range(COLS) if self.board[0, c] == 0]

    def _drop_piece(self, board: np.ndarray, player: int, col: int) -> int:
        """
        Drop a piece for 'player' into 'col' on given board.
        Returns the row index where it landed.
        """
        for r in range(ROWS - 1, -1, -1):
            if board[r, col] == 0:
                board[r, col] = player
                return r
        raise ValueError(f"Column {col} is full")

    def _is_winning_move(self, board: np.ndarray, player: int, row: int, col: int) -> bool:
        """
        Check if placing 'player' at (row, col) resulted in 4 in a row.
        """
        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diag down-right
            (1, -1),  # diag down-left
        ]

        for dr, dc in directions:
            count = 1

            # forward
            r, c = row + dr, col + dc
            while 0 <= r < ROWS and 0 <= c < COLS and board[r, c] == player:
                count += 1
                r += dr
                c += dc

            # backward
            r, c = row - dr, col - dc
            while 0 <= r < ROWS and 0 <= c < COLS and board[r, c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= 4:
                return True

        return False

    def _obs(self) -> np.ndarray:
        """
        Encode board as (3, 6, 7):
          ch0: current player's pieces
          ch1: opponent pieces
          ch2: all 1's (side-to-move plane)
        Uses perspective trick: flip signs so current player is +1.
        """
        b = self.board.copy().astype(np.int8)
        if self.player == -1:
            b = -b

        ch = np.zeros((3, ROWS, COLS), dtype=np.float32)
        ch[0] = (b == 1).astype(np.float32)
        ch[1] = (b == -1).astype(np.float32)
        ch[2] = 1.0  # side-to-move plane
        return ch

    def step(self, col: int):
        """
        Apply a move (column index).
        Returns (obs, reward, done, info).
        reward is +1.0 for the player who just moved if they win, 0 otherwise.
        """
        if self.done:
            raise RuntimeError("Game is over, call reset().")

        legal = self.legal_moves()
        if col not in legal:
            raise ValueError(f"Illegal move: column {col}")

        current_player = self.player
        row = self._drop_piece(self.board, current_player, col)

        reward = 0.0
        if self._is_winning_move(self.board, current_player, row, col):
            self.done = True
            self.winner = current_player
            reward = 1.0
        else:
            if len(self.legal_moves()) == 0:
                self.done = True
                self.winner = 0  # draw
            else:
                self.done = False
                self.winner = None

        # switch player for next state
        self.player *= -1

        return self._obs(), reward, self.done, {"winner": self.winner}


# =========================
# Value Network for Connect 4
# =========================

class ValueNetConnect4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(128 * ROWS * COLS, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),  # value in [-1, 1]
        )

    def forward(self, x):
        x = self.conv(x)
        return self.head(x)


@dataclass
class TrainConfig:
    episodes: int = 4000
    max_moves: int = 42  # full board
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 0.4
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 3000
    device: str = "cpu"
    save_path: str = "connect4_value.pt"


def epsilon_by_episode(ep: int, cfg: TrainConfig) -> float:
    t = min(1.0, ep / max(1, cfg.epsilon_decay_episodes))
    return cfg.epsilon_start + t * (cfg.epsilon_end - cfg.epsilon_start)


@torch.no_grad()
def pick_move_value_greedy(env: Connect4Env, net: ValueNetConnect4, eps: float, device: str) -> int:
    moves = env.legal_moves()
    if not moves:
        raise RuntimeError("No legal moves.")

    # exploration
    if random.random() < eps:
        return random.choice(moves)

    # exploitation: choose move that minimizes opponent's value
    best_move = None
    best_score = float("inf")

    for col in moves:
        e2 = env.clone()
        # simulate move
        e2._drop_piece(e2.board, e2.player, col)
        e2.player *= -1
        obs = torch.tensor(e2._obs(), dtype=torch.float32, device=device).unsqueeze(0)
        v = net(obs).item()
        if v < best_score:
            best_score = v
            best_move = col

    return best_move if best_move is not None else random.choice(moves)


def train(cfg: TrainConfig):
    device = torch.device(cfg.device)
    env = Connect4Env()
    net = ValueNetConnect4().to(device)
    opt = optim.Adam(net.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    net.train()

    win_counts = {1: 0, -1: 0, 0: 0, "unfinished": 0}

    for ep in range(1, cfg.episodes + 1):
        env.reset()
        eps = epsilon_by_episode(ep, cfg)

        move_count = 0
        while not env.done and move_count < cfg.max_moves:
            obs = env._obs()
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            col = pick_move_value_greedy(env, net, eps, cfg.device)
            next_obs, reward, done, info = env.step(col)
            move_count += 1

            with torch.no_grad():
                if done:
                    target = torch.tensor([[reward]], dtype=torch.float32, device=device)
                else:
                    next_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    # reward is 0 except terminal; bootstrap only
                    target = cfg.gamma * net(next_t)

            pred = net(obs_t)
            loss = loss_fn(pred, target)

            opt.zero_grad()
            loss.backward()
            opt.step()

        if env.done:
            win_counts[env.winner] = win_counts.get(env.winner, 0) + 1
        else:
            win_counts["unfinished"] += 1

        if ep % 50 == 0:
            print(
                f"ep {ep:5d} eps={eps:.3f} | "
                f"P1 wins={win_counts[1]} P2 wins={win_counts[-1]} "
                f"draws={win_counts[0]} unfinished={win_counts['unfinished']}"
            )

        if ep % 500 == 0:
            torch.save(net.state_dict(), cfg.save_path)
            print(f"saved: {cfg.save_path}")

    torch.save(net.state_dict(), cfg.save_path)
    print(f"done, saved: {cfg.save_path}")


# =========================
# Headless video rendering
# =========================

def render_board_rgb_connect4(board: np.ndarray, cell: int = 80) -> np.ndarray:
    """
    Simple RGB rendering (no pygame) for video.
    board: (ROWS, COLS) with {0,1,-1}
    """
    h = ROWS * cell
    w = COLS * cell
    img = np.zeros((h, w, 3), dtype=np.uint8)

    bg = np.array([240, 240, 240], dtype=np.uint8)
    board_color = np.array([30, 80, 200], dtype=np.uint8)
    p1_color = np.array([220, 50, 50], dtype=np.uint8)
    p2_color = np.array([240, 220, 40], dtype=np.uint8)

    img[:] = bg

    # board rectangle
    y0, y1 = 0, ROWS * cell
    x0, x1 = 0, COLS * cell
    img[y0:y1, x0:x1] = board_color

    for r in range(ROWS):
        for c in range(COLS):
            cy = r * cell + cell // 2
            cx = c * cell + cell // 2
            radius = int(cell * 0.35)

            yy, xx = np.ogrid[:cell, :cell]
            mask = (yy - cell / 2) ** 2 + (xx - cell / 2) ** 2 <= radius ** 2

            y_start, y_end = r * cell, (r + 1) * cell
            x_start, x_end = c * cell, (c + 1) * cell

            val = board[r, c]
            if val == 0:
                color = bg
            elif val == 1:
                color = p1_color
            else:
                color = p2_color

            img[y_start:y_end, x_start:x_end][mask] = color

    return img


def record_selfplay_video(
    model_path: str,
    out_mp4: str = "connect4_play.mp4",
    max_moves: int = 42,
    fps: int = 15,
    speed: float = 1.0,
    device: str = "cpu",
):
    if imageio is None:
        raise RuntimeError("imageio not installed. pip install imageio")

    env = Connect4Env()
    env.reset()

    net = ValueNetConnect4().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    frames = []
    moves = 0

    frames.append(render_board_rgb_connect4(env.board, cell=80))

    while not env.done and moves < max_moves:
        col = pick_move_value_greedy(env, net, eps=0.0, device=device)
        env.step(col)
        frames.append(render_board_rgb_connect4(env.board, cell=80))
        moves += 1

    effective_fps = max(1, int(fps * speed))
    imageio.mimsave(out_mp4, frames, fps=effective_fps)
    print(f"wrote video: {out_mp4} | winner={env.winner} | moves={moves} | fps={effective_fps}")


# =========================
# Pygame 3D-ish board + play vs AI
# =========================

def _shade(color, factor: float):
    r, g, b = color
    return (
        max(0, min(255, int(r * factor))),
        max(0, min(255, int(g * factor))),
        max(0, min(255, int(b * factor))),
    )


def draw_board_3d_connect4(
    screen,
    board: np.ndarray,
    cell_size: int = 80,
    highlight_cols: Optional[Set[int]] = None,
    selected_col: Optional[int] = None,
    status_text: str = "",
):
    if highlight_cols is None:
        highlight_cols = set()

    width = COLS * cell_size
    height = ROWS * cell_size + 30

    BG = (240, 240, 240)
    BOARD = (30, 80, 200)
    BOARD_DARK = _shade(BOARD, 0.7)
    P1 = (220, 50, 50)
    P2 = (240, 220, 40)

    screen.fill(BG)
    font = pygame.font.SysFont("consolas", 20)

    # Board rectangle
    board_rect = pygame.Rect(0, 0, width, ROWS * cell_size)
    pygame.draw.rect(screen, BOARD, board_rect)

    for r in range(ROWS):
        for c in range(COLS):
            x = c * cell_size
            y = r * cell_size

            # bevel at bottom of each cell section
            pygame.draw.rect(
                screen,
                BOARD_DARK,
                (x, y + int(cell_size * 0.7), cell_size, int(cell_size * 0.3)),
            )

            cx = x + cell_size // 2
            cy = y + cell_size // 2
            radius = int(cell_size * 0.35)

            val = int(board[r, c])

            if val == 0:
                # empty "hole"
                pygame.draw.circle(screen, BG, (cx, cy), radius)
            else:
                col = P1 if val == 1 else P2
                # shadow
                shadow_rect = pygame.Rect(cx - radius + 4, cy - radius + 6, radius * 2, radius * 2)
                pygame.draw.ellipse(screen, (40, 40, 40), shadow_rect)
                pygame.draw.circle(screen, col, (cx, cy - 1), radius)
                highlight = _shade(col, 1.3)
                pygame.draw.arc(
                    screen,
                    highlight,
                    (cx - radius, cy - radius - 4, radius * 2, radius * 2),
                    3.5,
                    6.0,
                    2,
                )

    # highlight columns at top
    for c in range(COLS):
        x = c * cell_size
        y = 0
        rect = (x + 2, y + 2, cell_size - 4, ROWS * cell_size - 4)
        if c == selected_col:
            pygame.draw.rect(screen, (255, 255, 0), rect, 3)
        elif c in highlight_cols:
            pygame.draw.rect(screen, (0, 200, 255), rect, 2)

    # status bar
    pygame.draw.rect(screen, (210, 210, 210), (0, height - 30, width, 30))
    txt = font.render(status_text, True, (0, 0, 0))
    screen.blit(txt, (10, height - 24))

    pygame.display.flip()


def human_select_move_with_hints_connect4(env: Connect4Env, human_player: int, screen, cell_size: int = 80) -> Optional[int]:
    """
    Human picks a column by clicking; highlights legal columns.
    Returns col index or None to quit.
    """
    clock = pygame.time.Clock()
    selected_col: Optional[int] = None

    while True:
        clock.tick(60)
        legal = env.legal_moves()
        if not legal:
            return None

        legal_set = set(legal)
        status = "Your turn. Click a highlighted column."
        draw_board_3d_connect4(
            screen,
            env.board,
            cell_size=cell_size,
            highlight_cols=legal_set,
            selected_col=selected_col,
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
                if not (0 <= c < COLS):
                    continue
                if c in legal_set:
                    # second click same column to confirm, or single click to pick
                    if selected_col is None or selected_col != c:
                        selected_col = c
                    else:
                        return c


def animate_drop_connect4(
    screen,
    board_before: np.ndarray,
    board_after: np.ndarray,
    col: int,
    row_end: int,
    player: int,
    status_prefix: str = "",
    cell_size: int = 80,
    steps: int = 10,
    step_delay_ms: int = 50,
):
    """
    Animate a piece falling from top into (row_end, col).
    """
    # base colors for disc
    P1 = (220, 50, 50)
    P2 = (240, 220, 40)
    disc_color = P1 if player == 1 else P2

    width = COLS * cell_size
    height = ROWS * cell_size + 30

    BG = (240, 240, 240)
    BOARD = (30, 80, 200)
    BOARD_DARK = _shade(BOARD, 0.7)

    for step in range(steps):
        t = step / max(1, steps - 1)
        screen.fill(BG)

        font = pygame.font.SysFont("consolas", 20)

        # draw static board_before
        board_rect = pygame.Rect(0, 0, width, ROWS * cell_size)
        pygame.draw.rect(screen, BOARD, board_rect)

        for r in range(ROWS):
            for c in range(COLS):
                x = c * cell_size
                y = r * cell_size

                pygame.draw.rect(
                    screen,
                    BOARD_DARK,
                    (x, y + int(cell_size * 0.7), cell_size, int(cell_size * 0.3)),
                )

                cy = r * cell_size + cell_size // 2
                cx = c * cell_size + cell_size // 2
                radius = int(cell_size * 0.35)

                val = int(board_before[r, c])

                if val == 0:
                    pygame.draw.circle(screen, BG, (cx, cy), radius)
                else:
                    ccol = (220, 50, 50) if val == 1 else (240, 220, 40)
                    shadow_rect = pygame.Rect(cx - radius + 4, cy - radius + 6, radius * 2, radius * 2)
                    pygame.draw.ellipse(screen, (40, 40, 40), shadow_rect)
                    pygame.draw.circle(screen, ccol, (cx, cy - 1), radius)
                    highlight = _shade(ccol, 1.3)
                    pygame.draw.arc(
                        screen,
                        highlight,
                        (cx - radius, cy - radius - 4, radius * 2, radius * 2),
                        3.5,
                        6.0,
                        2,
                    )

        # animated disc
        cx = col * cell_size + cell_size // 2
        start_y = cell_size // 2
        end_y = row_end * cell_size + cell_size // 2
        cy = int(start_y + t * (end_y - start_y))
        radius = int(cell_size * 0.35)

        shadow_rect = pygame.Rect(cx - radius + 4, cy - radius + 6, radius * 2, radius * 2)
        pygame.draw.ellipse(screen, (40, 40, 40), shadow_rect)
        pygame.draw.circle(screen, disc_color, (cx, cy - 1), radius)
        highlight = _shade(disc_color, 1.3)
        pygame.draw.arc(
            screen,
            highlight,
            (cx - radius, cy - radius - 4, radius * 2, radius * 2),
            3.5,
            6.0,
            2,
        )

        # status bar
        pygame.draw.rect(screen, (210, 210, 210), (0, height - 30, width, 30))
        txt = font.render(status_prefix + " Dropping...", True, (0, 0, 0))
        screen.blit(txt, (10, height - 24))

        pygame.display.flip()
        pygame.time.delay(step_delay_ms)

    # final board
    draw_board_3d_connect4(
        screen,
        board_after,
        cell_size=cell_size,
        status_text=status_prefix + " Move complete.",
    )


def play_vs_ai_pygame(
    model_path: str,
    human_player: int = 1,
    device: str = "cpu",
    cell_size: int = 80,
):
    if pygame is None:
        raise RuntimeError("pygame not installed. pip install pygame")

    pygame.init()
    width = COLS * cell_size
    height = ROWS * cell_size + 30
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("NeuroConnect4 - Play vs AI")

    env = Connect4Env()
    env.reset()

    net = ValueNetConnect4().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    clock = pygame.time.Clock()
    running = True

    # If human is player -1 and env.player == 1, AI moves first
    if human_player == -1 and env.player == 1:
        col = pick_move_value_greedy(env, net, eps=0.0, device=device)
        # compute landing row on current board
        row = None
        for r in range(ROWS - 1, -1, -1):
            if env.board[r, col] == 0:
                row = r
                break
        env_before = env.clone()
        env.step(col)
        animate_drop_connect4(
            screen,
            env_before.board,
            env.board,
            col,
            row,
            player=1,
            status_prefix="AI (first move): ",
            cell_size=cell_size,
        )

    while running:
        clock.tick(60)

        if env.done:
            if env.winner == human_player:
                msg = "Game over: YOU WIN! Press ESC or close."
            elif env.winner == 0:
                msg = "Game over: Draw. Press ESC or close."
            else:
                msg = "Game over: AI wins. Press ESC or close."

            draw_board_3d_connect4(screen, env.board, cell_size=cell_size, status_text=msg)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            continue

        if env.player == human_player:
            col = human_select_move_with_hints_connect4(env, human_player, screen, cell_size=cell_size)
            if col is None:
                running = False
                continue
            # landing row
            row = None
            for r in range(ROWS - 1, -1, -1):
                if env.board[r, col] == 0:
                    row = r
                    break
            env_before = env.clone()
            env.step(col)
            animate_drop_connect4(
                screen,
                env_before.board,
                env.board,
                col,
                row,
                player=human_player,
                status_prefix="Your move:",
                cell_size=cell_size,
            )
        else:
            status = "AI thinking..."
            draw_board_3d_connect4(screen, env.board, cell_size=cell_size, status_text=status)
            pygame.display.flip()
            pygame.time.delay(200)

            col = pick_move_value_greedy(env, net, eps=0.0, device=device)
            row = None
            for r in range(ROWS - 1, -1, -1):
                if env.board[r, col] == 0:
                    row = r
                    break
            env_before = env.clone()
            env.step(col)
            animate_drop_connect4(
                screen,
                env_before.board,
                env.board,
                col,
                row,
                player=env_before.player,
                status_prefix="AI move:",
                cell_size=cell_size,
            )

    pygame.quit()


# =========================
# Network diagram helpers
# =========================

def save_network_diagram_simple(path_png: str = "connect4_network_diagram.png"):
    import matplotlib.pyplot as plt

    layers = [
        "Input (3 x 6 x 7)",
        "Conv 3x3: 3->64 + ReLU",
        "Conv 3x3: 64->128 + ReLU",
        "Flatten (128*6*7)",
        "FC 256 + ReLU",
        "FC 128 + ReLU",
        "FC 1 + Tanh",
        "Output: Value in [-1, 1]",
    ]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis("off")

    y = 0.95
    for i, text in enumerate(layers):
        ax.text(
            0.5,
            y,
            text,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black"),
        )
        if i < len(layers) - 1:
            ax.annotate(
                "",
                xy=(0.5, y - 0.08),
                xytext=(0.5, y - 0.02),
                arrowprops=dict(arrowstyle="->"),
            )
        y -= 0.12

    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    print(f"wrote: {path_png}")


def save_network_graph_torchviz(path_png: str = "connect4_network_graph.png", device: str = "cpu"):
    try:
        from torchviz import make_dot
    except Exception:
        raise RuntimeError("torchviz not installed. pip install torchviz graphviz")

    net = ValueNetConnect4().to(device)
    x = torch.zeros((1, 3, ROWS, COLS), dtype=torch.float32, device=device)
    y = net(x)
    dot = make_dot(y, params=dict(net.named_parameters()))
    dot.format = "png"
    dot.render(path_png.replace(".png", ""), cleanup=True)
    print(f"wrote: {path_png}")


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(description="NeuroConnect4: simple RL + visualization for Connect 4.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train value net via self-play TD(0).")
    p_train.add_argument("--episodes", type=int, default=4000)
    p_train.add_argument("--max-moves", type=int, default=42)
    p_train.add_argument("--device", type=str, default="cpu")
    p_train.add_argument("--save", type=str, default="connect4_value.pt")

    p_vid = sub.add_parser("video", help="Record self-play video.")
    p_vid.add_argument("--model", type=str, default="connect4_value.pt")
    p_vid.add_argument("--out", type=str, default="connect4_play.mp4")
    p_vid.add_argument("--device", type=str, default="cpu")
    p_vid.add_argument("--fps", type=int, default=15, help="Base frames per second.")
    p_vid.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed multiplier (>1 faster, <1 slower).",
    )

    p_play = sub.add_parser("play", help="Play vs trained AI in a 3D-ish pygame GUI.")
    p_play.add_argument("--model", type=str, default="connect4_value.pt")
    p_play.add_argument("--device", type=str, default="cpu")
    p_play.add_argument(
        "--human-player",
        type=int,
        default=1,
        choices=[1, -1],
        help="1 = you go first, -1 = AI goes first.",
    )
    p_play.add_argument("--cell-size", type=int, default=80, help="Cell size in pixels.")

    p_diag = sub.add_parser("netdiag", help="Save a simple drawn network diagram.")
    p_diag.add_argument("--out", type=str, default="connect4_network_diagram.png")

    p_graph = sub.add_parser("netgraph", help="Save a torchviz computation graph.")
    p_graph.add_argument("--out", type=str, default="connect4_network_graph.png")
    p_graph.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    if args.cmd == "train":
        cfg = TrainConfig(
            episodes=args.episodes,
            max_moves=args.max_moves,
            device=args.device,
            save_path=args.save,
        )
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
            human_player=args.human_player,
            device=args.device,
            cell_size=args.cell_size,
        )

    elif args.cmd == "netdiag":
        save_network_diagram_simple(args.out)

    elif args.cmd == "netgraph":
        save_network_graph_torchviz(args.out, device=args.device)


if __name__ == "__main__":
    main()
