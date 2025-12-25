# AI_Learn2Play

Neural networks that **learn to play classic board games** (Checkers, Chess, Connect 4) via self‑play, with:

- PyTorch value networks
- Simple TD(0) self‑play training loops
- 3D‑style pygame visualizations
- Human vs AI play with move hints and animations
- MP4 video recording of games with adjustable playback speed
- Network diagrams for docs / presentations

Repo: [AI_Learn2Play](https://github.com/jsammarco/AI_Learn2Play)  
Author / site: [ConsultingJoe](https://consultingjoe.com)

---

## 1. Installation

### 1.1. Python

Tested with **Python 3.10**.

### 1.2. Core dependencies

Install the common stack:

```bash
pip install torch numpy pygame imageio matplotlib
```

Game‑specific extra:

- **Chess**: `python-chess`
- **Optional network graph**: `torchviz` + Graphviz

```bash
# For Chess
pip install python-chess

# For network graphs (all games, optional)
pip install torchviz graphviz
```

On Windows, also install **Graphviz** and ensure `dot` is on your `PATH` to use the `netgraph` commands.

---

## 2. Project Layout

Suggested layout in the repo:

```text
AI_Learn2Play/
├── checkers_rl.py      # NeuroCheckers: Checkers RL + visualization
├── chess_rl.py         # NeuroChess: Chess RL + visualization
├── connect4_rl.py      # NeuroConnect4: Connect 4 RL + visualization
└── README.md           # This file
```

Each script is **standalone** – you can run them directly with `python <script>.py <subcommand> ...`.

---

## 3. Common Concepts

All three games share the same core ideas:

- A **value network** that takes a board state and predicts a scalar value in **[-1, 1]**  
  (`+1` = winning for side‑to‑move, `-1` = losing).
- Training via **self‑play TD(0)**:
  - The current network plays against itself with an epsilon‑greedy policy.
  - After each move, it updates towards the bootstrapped target `r + γ V(next_state)`.
- During inference (play / video):
  - For each legal move, simulate the next state.
  - Evaluate the value of the resulting position for the *next* player.
  - Pick the move that **minimizes** the opponent’s value.

Visualizations:

- **Pygame GUIs**:
  - 3D‑ish boards and pieces/discs.
  - Human vs AI play.
  - Legal moves highlighted.
  - Animated moves.
- **Headless MP4 recording**:
  - Uses `imageio` to write an MP4 from a sequence of rendered frames.
  - `--speed` lets you scale playback speed (e.g. 2× faster).

Network diagrams:

- `netdiag`: simple drawn diagram (no Graphviz).
- `netgraph`: full computation graph via `torchviz` (requires Graphviz).

---

## 4. NeuroCheckers – `checkers_rl.py`

A Checkers (American draughts) environment plus value network and visualizations.

### 4.1. Training

```bash
python checkers_rl.py train [--episodes N] [--device DEVICE] [--save PATH]
```

**Arguments:**

- `--episodes`  
  Number of self‑play games to train on.  
  Default: `2000`

- `--device`  
  PyTorch device: `"cpu"` or e.g. `"cuda"`.  
  Default: `"cpu"`

- `--save`  
  Path to save the trained value network `state_dict`.  
  Default: `checkers_value.pt`

### 4.2. Record self‑play video

```bash
python checkers_rl.py video     --model checkers_value.pt     --out checkers_play.mp4     [--device DEVICE]     [--fps FPS]     [--speed MULTIPLIER]
```

**Arguments:**

- `--model`  
  Path to saved model (`state_dict`). Default: `checkers_value.pt`

- `--out`  
  Output MP4 path. Default: `checkers_play.mp4`

- `--device`  
  Device for inference (`cpu` / `cuda`). Default: `cpu`

- `--fps`  
  Base frames per second. Default: `15`

- `--speed`  
  Playback speed multiplier. Effective FPS is `fps * speed`.  
  Default: `1.0`  
  Examples:
  - `--speed 2.0` → ~2× faster
  - `--speed 0.5` → ~0.5× (slow motion)

### 4.3. Play vs AI (3D board + hints + animation)

```bash
python checkers_rl.py play     --model checkers_value.pt     [--device DEVICE]     [--human-side {1,-1}]     [--cell-size PX]
```

**Arguments:**

- `--model`  
  Path to trained model. Default: `checkers_value.pt`

- `--device`  
  `cpu` or `cuda`. Default: `cpu`

- `--human-side`  
  Which player you control:
  - `1`  → Player 1 (starts at bottom, moves first)
  - `-1` → Player 2 (AI moves first)  
  Default: `1`

- `--cell-size`  
  Pixel size of each board square. Default: `80`

**Features:**

- 3D‑ish board and checkers pieces.
- Legal move origins highlighted; mandatory capture origins highlighted differently.
- Destinations for selected piece highlighted.
- Multi‑jump captures animated step‑by‑step.

### 4.4. Network diagrams

#### Simple diagram (no Graphviz)

```bash
python checkers_rl.py netdiag [--out FILE]
```

- `--out` – Output PNG path. Default: `network_diagram.png`

#### Torchviz computation graph

```bash
python checkers_rl.py netgraph [--out FILE] [--device DEVICE]
```

- `--out` – Output PNG path. Default: `network_graph.png`  
- `--device` – `cpu` or `cuda`. Default: `cpu`  

Requires: `torchviz`, `graphviz`, and `dot` on PATH.

---

## 5. NeuroChess – `chess_rl.py`

Chess environment via `python-chess`, plus value net and visualization.

### 5.1. Training

```bash
python chess_rl.py train     [--episodes N]     [--max-moves M]     [--device DEVICE]     [--save PATH]
```

**Arguments:**

- `--episodes` – Number of self‑play games. Default: `2000`  
- `--max-moves` – Max plies per game before stopping. Default: `200`  
- `--device` – `"cpu"` or `"cuda"`. Default: `"cpu"`  
- `--save` – Path for `state_dict`. Default: `chess_value.pt`

### 5.2. Record self‑play video

```bash
python chess_rl.py video     --model chess_value.pt     --out chess_play.mp4     [--device DEVICE]     [--fps FPS]     [--speed MULTIPLIER]
```

Same semantics as Checkers:

- `--model` – Model path (default: `chess_value.pt`)  
- `--out` – Output MP4 (default: `chess_play.mp4`)  
- `--device` – `cpu` / `cuda`  
- `--fps` – Base FPS (default: `15`)  
- `--speed` – Playback speed multiplier (default: `1.0`)

### 5.3. Play vs AI (3D board + hints + animation)

```bash
python chess_rl.py play     --model chess_value.pt     [--device DEVICE]     [--human-color {white,black}]     [--cell-size PX]
```

**Arguments:**

- `--model` – Model path. Default: `chess_value.pt`  
- `--device` – `cpu` / `cuda`. Default: `cpu`  
- `--human-color` – `"white"` or `"black"` (your side). Default: `"white"`  
- `--cell-size` – Board square size in pixels. Default: `80`

**Features:**

- 3D‑ish chessboard with bevels.
- Circular 3D‑style pieces with letter overlay (`P`, `N`, `B`, `R`, `Q`, `K`).
- Legal origin squares highlighted; destinations for selected piece highlighted.
- Single‑move animation: piece slides from source to destination.
- Promotions default to queen in ambiguous promotion situations.

### 5.4. Network diagrams

#### Simple diagram

```bash
python chess_rl.py netdiag [--out FILE]
```

- `--out` – Default: `chess_network_diagram.png`

#### Torchviz computation graph

```bash
python chess_rl.py netgraph [--out FILE] [--device DEVICE]
```

- `--out` – Default: `chess_network_graph.png`  
- `--device` – `cpu` / `cuda` (default: `cpu`)

Requires `torchviz` + Graphviz.

---

## 6. NeuroConnect4 – `connect4_rl.py`

Connect 4 on a 7×6 grid with value net, animated GUI, and videos.

### 6.1. Training

```bash
python connect4_rl.py train     [--episodes N]     [--max-moves M]     [--device DEVICE]     [--save PATH]
```

**Arguments:**

- `--episodes` – Number of self‑play games. Default: `4000`  
- `--max-moves` – Max moves per game. Default: `42` (full board)  
- `--device` – `cpu` / `cuda`. Default: `cpu`  
- `--save` – Model path. Default: `connect4_value.pt`

### 6.2. Record self‑play video

```bash
python connect4_rl.py video     --model connect4_value.pt     --out connect4_play.mp4     [--device DEVICE]     [--fps FPS]     [--speed MULTIPLIER]
```

**Arguments:**

- `--model` – Model path. Default: `connect4_value.pt`  
- `--out` – Output MP4. Default: `connect4_play.mp4`  
- `--device` – `cpu` / `cuda`  
- `--fps` – Base FPS (default: `15`)  
- `--speed` – Playback multiplier (default: `1.0`)

### 6.3. Play vs AI (3D‑ish board + falling discs)

```bash
python connect4_rl.py play     --model connect4_value.pt     [--device DEVICE]     [--human-player {1,-1}]     [--cell-size PX]
```

**Arguments:**

- `--model` – Model path. Default: `connect4_value.pt`  
- `--device` – `cpu` / `cuda`. Default: `cpu`  
- `--human-player`:
  - `1`  → You are Player 1 (red, go first)
  - `-1` → You are Player 2 (yellow, AI first)  
  Default: `1`

- `--cell-size` – Cell size in pixels. Default: `80`

**Features:**

- 3D‑style blue board with circular “holes”.
- Red and yellow discs with shadow + highlight.
- Legal columns highlighted; click to select and confirm.
- Animated falling discs into the correct row.

### 6.4. Network diagrams

#### Simple diagram

```bash
python connect4_rl.py netdiag [--out FILE]
```

- `--out` – Default: `connect4_network_diagram.png`

#### Torchviz computation graph

```bash
python connect4_rl.py netgraph [--out FILE] [--device DEVICE]
```

- `--out` – Default: `connect4_network_graph.png`  
- `--device` – `cpu` / `cuda`  

Requires `torchviz` + Graphviz.

---

## 7. Tips & Next Steps

Ideas to extend this project:

- Add **policy heads** (move distributions) alongside value heads for stronger play.
- Integrate a small **MCTS** for Chess/Checkers using the networks.
- Add **replay buffers** instead of pure online TD(0).
- Add UI toggles for:
  - Showing predicted value on screen.
  - Logging moves to PGN (for Chess) or simple text logs (Checkers/Connect4).
- Wrap all three into a small launcher GUI that lets you pick the game and mode.

If you use this in a video or blog, feel free to link back to:

- Repo: [AI_Learn2Play](https://github.com/jsammarco/AI_Learn2Play)  
- Site: [ConsultingJoe](https://consultingjoe.com)
