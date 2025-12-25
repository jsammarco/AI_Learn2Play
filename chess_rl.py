import argparse
import math
import random
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import chess  # python-chess

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
# Chess environment wrapper
# =========================

class ChessEnv:
    """
    Simple wrapper around python-chess Board for RL.
    """

    def __init__(self):
        self.board = chess.Board()
        self.done = False
        self.result: Optional[str] = None  # "1-0", "0-1", "1/2-1/2"

    def reset(self):
        self.board.reset()
        self.done = False
        self.result = None
        return self._obs()

    def clone(self) -> "ChessEnv":
        env = ChessEnv()
        env.board = self.board.copy(stack=False)
        env.done = self.done
        env.result = self.result
        return env

    # ------------- Observation encoding -------------

    def _obs(self) -> np.ndarray:
        """
        Encode board to (13, 8, 8) float32:
          0..5   : current side's pieces (P,N,B,R,Q,K)
          6..11  : opponent's pieces (P,N,B,R,Q,K)
          12     : side-to-move plane: all 1 if white to move, 0 otherwise

        Board orientation: rank 8 at top (row 0), rank 1 at bottom (row 7).
        """
        board = self.board
        turn = board.turn  # True=white, False=black

        planes = np.zeros((13, 8, 8), dtype=np.float32)

        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None:
                continue
            rank = chess.square_rank(sq)  # 0..7 (0 is rank1)
            file = chess.square_file(sq)  # 0..7 (0 is file a)
            r = 7 - rank  # display rank8 (7) at row0
            c = file

            type_idx = piece.piece_type - 1  # P,N,B,R,Q,K -> 0..5
            if piece.color == turn:
                ch = type_idx
            else:
                ch = 6 + type_idx
            planes[ch, r, c] = 1.0

        # side-to-move plane
        if turn:
            planes[12, :, :] = 1.0  # white to move
        else:
            planes[12, :, :] = 0.0  # black to move

        return planes

    # ------------- Interface -------------

    def legal_moves(self) -> List[chess.Move]:
        if self.done:
            return []
        return list(self.board.legal_moves)

    def step(self, move: chess.Move):
        """
        Apply move, return (obs, reward, done, info).
        Reward is from the perspective of the player who moved.
        """
        if self.done:
            raise RuntimeError("Game already over, call reset()")

        prev_turn = self.board.turn  # color about to move
        self.board.push(move)

        reward = 0.0
        if self.board.is_game_over():
            self.done = True
            self.result = self.board.result()  # "1-0", "0-1", "1/2-1/2"
            if self.result == "1-0":
                winner = chess.WHITE
            elif self.result == "0-1":
                winner = chess.BLACK
            else:
                winner = None  # draw
            if winner is None:
                reward = 0.0
            else:
                reward = 1.0 if winner == prev_turn else -1.0

        return self._obs(), reward, self.done, {"result": self.result}


# =========================
# Value network
# =========================

class ValueNetChess(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(13, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),  # value in [-1, 1]
        )

    def forward(self, x):
        x = self.conv(x)
        return self.head(x)


@dataclass
class TrainConfig:
    episodes: int = 2000
    max_moves: int = 200
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 0.4
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 1500
    device: str = "cpu"
    save_path: str = "chess_value.pt"


def epsilon_by_episode(ep: int, cfg: TrainConfig) -> float:
    t = min(1.0, ep / max(1, cfg.epsilon_decay_episodes))
    return cfg.epsilon_start + t * (cfg.epsilon_end - cfg.epsilon_start)


@torch.no_grad()
def pick_move_value_greedy(env: ChessEnv, net: ValueNetChess, eps: float, device: str) -> chess.Move:
    moves = env.legal_moves()
    if not moves:
        raise RuntimeError("No legal moves")

    # exploration
    if random.random() < eps:
        return random.choice(moves)

    # exploitation: choose move that minimizes opponent's value
    best_move = None
    best_score = float("inf")

    for mv in moves:
        e2 = env.clone()
        e2.board.push(mv)
        obs = torch.tensor(e2._obs(), dtype=torch.float32, device=device).unsqueeze(0)
        v = net(obs).item()  # value for side-to-move (opponent)
        if v < best_score:
            best_score = v
            best_move = mv

    return best_move if best_move is not None else random.choice(moves)


def train(cfg: TrainConfig):
    device = torch.device(cfg.device)
    env = ChessEnv()
    net = ValueNetChess().to(device)
    opt = optim.Adam(net.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    net.train()

    win_counts = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "unfinished": 0}

    for ep in range(1, cfg.episodes + 1):
        env.reset()
        eps = epsilon_by_episode(ep, cfg)

        move_count = 0
        while not env.done and move_count < cfg.max_moves:
            obs = env._obs()
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            mv = pick_move_value_greedy(env, net, eps, cfg.device)
            next_obs, reward, done, info = env.step(mv)
            move_count += 1

            with torch.no_grad():
                if done:
                    target = torch.tensor([[reward]], dtype=torch.float32, device=device)
                else:
                    next_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    target = reward + cfg.gamma * net(next_t)

            pred = net(obs_t)
            loss = loss_fn(pred, target)

            opt.zero_grad()
            loss.backward()
            opt.step()

        if env.done:
            win_counts[env.result] = win_counts.get(env.result, 0) + 1
        else:
            win_counts["unfinished"] += 1

        if ep % 20 == 0:
            print(
                f"ep {ep:5d} eps={eps:.3f} | "
                f"1-0={win_counts['1-0']} 0-1={win_counts['0-1']} "
                f"draw={win_counts['1/2-1/2']} unfinished={win_counts['unfinished']}"
            )

        if ep % 200 == 0:
            torch.save(net.state_dict(), cfg.save_path)
            print(f"saved: {cfg.save_path}")

    torch.save(net.state_dict(), cfg.save_path)
    print(f"done, saved: {cfg.save_path}")


# =========================
# Headless video rendering
# =========================

def render_board_rgb_chess(board: chess.Board, cell: int = 80) -> np.ndarray:
    """
    Simple RGB rendering (no pygame) for video.
    """
    h = w = 8 * cell
    img = np.zeros((h, w, 3), dtype=np.uint8)

    light = np.array([240, 240, 240], dtype=np.uint8)
    dark = np.array([120, 120, 120], dtype=np.uint8)

    white_piece = np.array([230, 230, 230], dtype=np.uint8)
    black_piece = np.array([40, 40, 40], dtype=np.uint8)

    for r in range(8):
        for c in range(8):
            y0, y1 = r * cell, (r + 1) * cell
            x0, x1 = c * cell, (c + 1) * cell

            # board square
            if (r + c) % 2 == 0:
                img[y0:y1, x0:x1] = light
            else:
                img[y0:y1, x0:x1] = dark

            # piece
            rank = 7 - r
            file = c
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if piece is None:
                continue

            color = white_piece if piece.color == chess.WHITE else black_piece
            cy = (y0 + y1) // 2
            cx = (x0 + x1) // 2
            radius = int(cell * 0.32)

            yy, xx = np.ogrid[:cell, :cell]
            mask = (yy - cell / 2) ** 2 + (xx - cell / 2) ** 2 <= radius ** 2
            img[y0:y1, x0:x1][mask] = color

    return img


def record_selfplay_video(
    model_path: str,
    out_mp4: str = "chess_play.mp4",
    max_moves: int = 200,
    fps: int = 15,
    speed: float = 1.0,
    device: str = "cpu",
):
    if imageio is None:
        raise RuntimeError("imageio not installed. pip install imageio")

    env = ChessEnv()
    env.reset()

    net = ValueNetChess().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    frames = []
    moves = 0

    frames.append(render_board_rgb_chess(env.board, cell=80))

    while not env.done and moves < max_moves:
        mv = pick_move_value_greedy(env, net, eps=0.0, device=device)
        env.step(mv)
        frames.append(render_board_rgb_chess(env.board, cell=80))
        moves += 1

    effective_fps = max(1, int(fps * speed))
    imageio.mimsave(out_mp4, frames, fps=effective_fps)
    print(f"wrote video: {out_mp4} | result={env.result} | moves={moves} | fps={effective_fps}")


# =========================
# Pygame 3D-ish board + play vs AI
# =========================

def _shade(color, factor):
    r, g, b = color
    return (
        max(0, min(255, int(r * factor))),
        max(0, min(255, int(g * factor))),
        max(0, min(255, int(b * factor))),
    )


def rc_to_square(r: int, c: int) -> chess.Square:
    # r=0 top (rank8); r=7 bottom (rank1)
    rank = 7 - r
    file = c
    return chess.square(file, rank)


def square_to_rc(sq: chess.Square) -> Tuple[int, int]:
    rank = chess.square_rank(sq)
    file = chess.square_file(sq)
    r = 7 - rank
    c = file
    return r, c


def draw_board_3d_chess(
    screen,
    board: chess.Board,
    cell_size: int = 80,
    selected: Optional[Tuple[int, int]] = None,
    origin_squares: Optional[Set[Tuple[int, int]]] = None,
    dest_squares: Optional[Set[Tuple[int, int]]] = None,
    status_text: str = "",
    anim_overlay: Optional[Dict] = None,
):
    """
    3D-ish board with highlighted moves and optional animated overlay piece.
    anim_overlay: {
        "piece": chess.Piece,
        "from_sq": chess.Square,
        "to_sq": chess.Square,
        "alpha": float in [0,1],
    }
    """
    if origin_squares is None:
        origin_squares = set()
    if dest_squares is None:
        dest_squares = set()

    width = 8 * cell_size
    height = 8 * cell_size + 30

    BG = (240, 240, 240)
    LIGHT_SQ = (230, 230, 230)
    DARK_SQ = (130, 120, 120)

    screen.fill(BG)

    font = pygame.font.SysFont("consolas", 20)

    from_sq_anim = anim_overlay["from_sq"] if anim_overlay else None
    to_sq_anim = anim_overlay["to_sq"] if anim_overlay else None
    alpha = anim_overlay["alpha"] if anim_overlay else 1.0
    anim_piece = anim_overlay["piece"] if anim_overlay else None

    def draw_piece_at_center(cx, cy, piece_obj: chess.Piece):
        if piece_obj is None:
            return
        base_col = (230, 230, 230) if piece_obj.color == chess.WHITE else (40, 40, 40)
        cx = int(cx)
        cy = int(cy)
        radius = int(cell_size * 0.33)

        # shadow
        shadow_rect = pygame.Rect(cx - radius + 4, cy - radius + 6, radius * 2, radius * 2)
        pygame.draw.ellipse(screen, (40, 40, 40), shadow_rect)

        pygame.draw.circle(screen, base_col, (cx, cy - 1), radius)
        highlight = _shade(base_col, 1.3)
        pygame.draw.arc(
            screen,
            highlight,
            (cx - radius, cy - radius - 4, radius * 2, radius * 2),
            3.5,
            6.0,
            2,
        )

        # piece letter
        letter = piece_obj.symbol().upper() if piece_obj.color == chess.WHITE else piece_obj.symbol().lower()
        txt = font.render(letter, True, (0, 0, 0) if piece_obj.color == chess.WHITE else (255, 255, 255))
        rect = txt.get_rect(center=(cx, cy - 1))
        screen.blit(txt, rect)

    # Board + stationary pieces
    for r in range(8):
        for c in range(8):
            x = c * cell_size
            y = r * cell_size

            base = DARK_SQ if (r + c) % 2 else LIGHT_SQ
            pygame.draw.rect(screen, base, (x, y, cell_size, cell_size))

            bevel = _shade(base, 0.8)
            pygame.draw.rect(
                screen,
                bevel,
                (x, y + int(cell_size * 0.65), cell_size, int(cell_size * 0.35)),
            )

            inner = (x + 3, y + 3, cell_size - 6, cell_size - 6)
            if (r, c) in origin_squares:
                pygame.draw.rect(screen, (0, 180, 255), inner, 2)
            if (r, c) in dest_squares:
                pygame.draw.rect(screen, (50, 220, 50), inner, 3)
            if selected is not None and (r, c) == selected:
                pygame.draw.rect(screen, (255, 255, 0), inner, 3)

            sq = rc_to_square(r, c)
            piece = board.piece_at(sq)

            # hide from/to squares if animating
            if anim_overlay is not None and alpha < 1.0:
                if sq == from_sq_anim or sq == to_sq_anim:
                    piece = None

            if piece is None:
                continue

            cx = x + cell_size // 2
            cy = y + cell_size // 2
            draw_piece_at_center(cx, cy, piece)

    # Animated moving piece
    if anim_overlay is not None and anim_piece is not None and 0.0 <= alpha < 1.0:
        r_from, c_from = square_to_rc(from_sq_anim)
        r_to, c_to = square_to_rc(to_sq_anim)
        x_from = c_from * cell_size + cell_size / 2
        y_from = r_from * cell_size + cell_size / 2
        x_to = c_to * cell_size + cell_size / 2
        y_to = r_to * cell_size + cell_size / 2
        cx = x_from + alpha * (x_to - x_from)
        cy = y_from + alpha * (y_to - y_from)
        draw_piece_at_center(cx, cy, anim_piece)

    # Status bar
    pygame.draw.rect(screen, (210, 210, 210), (0, height - 30, width, 30))
    txt = font.render(status_text, True, (0, 0, 0))
    screen.blit(txt, (10, height - 24))

    pygame.display.flip()


def human_select_move_with_hints_chess(env: ChessEnv, human_color: bool, screen, cell_size: int = 80):
    """
    Human clicks: select origin (highlighted), then destination (green squares).
    Returns a python-chess Move or None if user quits.
    """
    clock = pygame.time.Clock()
    selected: Optional[Tuple[int, int]] = None

    while True:
        clock.tick(60)
        moves = env.legal_moves()
        if not moves:
            return None

        origins: Set[Tuple[int, int]] = set()
        for mv in moves:
            if env.board.piece_at(mv.from_square) and env.board.piece_at(mv.from_square).color == human_color:
                origins.add(square_to_rc(mv.from_square))

        dests: Set[Tuple[int, int]] = set()
        if selected is not None:
            from_sq = rc_to_square(*selected)
            for mv in moves:
                if mv.from_square == from_sq:
                    dests.add(square_to_rc(mv.to_square))

        status = "Your turn. Click a highlighted piece, then a green target."
        draw_board_3d_chess(
            screen,
            env.board,
            cell_size=cell_size,
            selected=selected,
            origin_squares=origins,
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
                    if (r, c) in origins:
                        selected = (r, c)
                else:
                    from_sq = rc_to_square(*selected)
                    to_sq = rc_to_square(r, c)
                    candidates = [
                        mv for mv in moves if mv.from_square == from_sq and mv.to_square == to_sq
                    ]
                    if candidates:
                        # Choose promotion to queen if multiple options
                        queen_moves = [m for m in candidates if m.promotion == chess.QUEEN]
                        return queen_moves[0] if queen_moves else candidates[0]
                    else:
                        if (r, c) == selected:
                            selected = None
                        elif (r, c) in origins:
                            selected = (r, c)


def animate_move_pygame_chess(
    screen,
    env_before: ChessEnv,
    env_after: ChessEnv,
    move: chess.Move,
    status_prefix: str = "",
    cell_size: int = 80,
    steps: int = 8,
    step_delay_ms: int = 60,
):
    """
    Slide the moved piece from from_sq to to_sq over 'steps' frames.
    """
    piece = env_before.board.piece_at(move.from_square)
    if piece is None:
        # fallback: just draw final position
        draw_board_3d_chess(screen, env_after.board, cell_size=cell_size, status_text=status_prefix)
        return

    for i in range(steps):
        alpha = i / max(1, steps - 1)
        draw_board_3d_chess(
            screen,
            env_before.board,
            cell_size=cell_size,
            status_text=status_prefix,
            anim_overlay={
                "piece": piece,
                "from_sq": move.from_square,
                "to_sq": move.to_square,
                "alpha": alpha,
            },
        )
        pygame.time.delay(step_delay_ms)

    draw_board_3d_chess(
        screen,
        env_after.board,
        cell_size=cell_size,
        status_text=status_prefix + " Move complete.",
    )


def play_vs_ai_pygame(
    model_path: str,
    human_color_str: str = "white",
    device: str = "cpu",
    cell_size: int = 80,
):
    if pygame is None:
        raise RuntimeError("pygame not installed. pip install pygame")

    human_color = chess.WHITE if human_color_str.lower().startswith("w") else chess.BLACK

    pygame.init()
    width = 8 * cell_size
    height = 8 * cell_size + 30
    screen = pygame.display.set_mode((width, height))
    pygame.display.setCaption = "NeuroChess - Play vs AI"

    env = ChessEnv()
    env.reset()

    net = ValueNetChess().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    clock = pygame.time.Clock()
    running = True

    # AI moves first if human is black and white to move
    if human_color == chess.BLACK and env.board.turn == chess.WHITE:
        env_before = env.clone()
        mv = pick_move_value_greedy(env, net, eps=0.0, device=device)
        env.step(mv)
        animate_move_pygame_chess(
            screen,
            env_before,
            env,
            mv,
            status_prefix="AI (first move): ",
            cell_size=cell_size,
        )

    while running:
        clock.tick(60)

        if env.done:
            if env.result == "1-0":
                winner_color = "White"
            elif env.result == "0-1":
                winner_color = "Black"
            else:
                winner_color = "Draw"

            if env.result is None:
                msg = "Game over: no result. Press ESC or close."
            else:
                msg = f"Game over: {winner_color} ({env.result}). Press ESC or close."

            draw_board_3d_chess(screen, env.board, cell_size=cell_size, status_text=msg)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            continue

        if env.board.turn == human_color:
            mv = human_select_move_with_hints_chess(env, human_color, screen, cell_size=cell_size)
            if mv is None:
                running = False
                continue
            env_before = env.clone()
            env.step(mv)
            animate_move_pygame_chess(
                screen,
                env_before,
                env,
                mv,
                status_prefix="Your move: ",
                cell_size=cell_size,
            )
        else:
            status = f"AI ({'White' if env.board.turn == chess.WHITE else 'Black'}) thinking..."
            draw_board_3d_chess(screen, env.board, cell_size=cell_size, status_text=status)
            pygame.display.flip()
            pygame.time.delay(200)

            env_before = env.clone()
            mv = pick_move_value_greedy(env, net, eps=0.0, device=device)
            env.step(mv)
            animate_move_pygame_chess(
                screen,
                env_before,
                env,
                mv,
                status_prefix="AI move: ",
                cell_size=cell_size,
            )

    pygame.quit()


# =========================
# Network diagrams
# =========================

def save_network_diagram_simple(path_png: str = "chess_network_diagram.png"):
    import matplotlib.pyplot as plt

    layers = [
        "Input (13 x 8 x 8)",
        "Conv 3x3: 13->64 + ReLU",
        "Conv 3x3: 64->128 + ReLU",
        "Flatten (128*8*8)",
        "FC 512 + ReLU",
        "FC 256 + ReLU",
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


def save_network_graph_torchviz(path_png: str = "chess_network_graph.png", device: str = "cpu"):
    try:
        from torchviz import make_dot
    except Exception:
        raise RuntimeError("torchviz not installed. pip install torchviz graphviz")

    net = ValueNetChess().to(device)
    x = torch.zeros((1, 13, 8, 8), dtype=torch.float32, device=device)
    y = net(x)
    dot = make_dot(y, params=dict(net.named_parameters()))
    dot.format = "png"
    dot.render(path_png.replace(".png", ""), cleanup=True)
    print(f"wrote: {path_png}")


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(description="NeuroChess: simple RL chess value net + visualization.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train value net via self-play TD(0).")
    p_train.add_argument("--episodes", type=int, default=2000)
    p_train.add_argument("--max-moves", type=int, default=200)
    p_train.add_argument("--device", type=str, default="cpu")
    p_train.add_argument("--save", type=str, default="chess_value.pt")

    p_vid = sub.add_parser("video", help="Record self-play video.")
    p_vid.add_argument("--model", type=str, default="chess_value.pt")
    p_vid.add_argument("--out", type=str, default="chess_play.mp4")
    p_vid.add_argument("--device", type=str, default="cpu")
    p_vid.add_argument("--fps", type=int, default=15, help="Base frames per second.")
    p_vid.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed multiplier for playback (>1 faster, <1 slower).",
    )

    p_play = sub.add_parser("play", help="Play vs trained AI in a 3D-ish pygame GUI.")
    p_play.add_argument("--model", type=str, default="chess_value.pt")
    p_play.add_argument("--device", type=str, default="cpu")
    p_play.add_argument(
        "--human-color",
        type=str,
        default="white",
        choices=["white", "black"],
        help="Which side you play.",
    )
    p_play.add_argument("--cell-size", type=int, default=80, help="Board cell size in pixels.")

    p_diag = sub.add_parser("netdiag", help="Save a simple drawn network diagram (no graphviz).")
    p_diag.add_argument("--out", type=str, default="chess_network_diagram.png")

    p_graph = sub.add_parser("netgraph", help="Save a torchviz computation graph (requires graphviz).")
    p_graph.add_argument("--out", type=str, default="chess_network_graph.png")
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
            human_color_str=args.human_color,
            device=args.device,
            cell_size=args.cell_size,
        )

    elif args.cmd == "netdiag":
        save_network_diagram_simple(args.out)

    elif args.cmd == "netgraph":
        save_network_graph_torchviz(args.out, device=args.device)


if __name__ == "__main__":
    main()
