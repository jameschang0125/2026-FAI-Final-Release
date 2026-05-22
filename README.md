# 2026 FAI Final - Building a 6 Nimmt! Agent 🐮

## Setup

Use `python 3.13.11` and `pip install -r requirements.txt`.

## Running Simulations and Tournaments

The framework provides two main scripts for evaluating players, both driven by JSON configuration files (examples located in `configs/game/` and `configs/tournament/`).

### 1. Single Game Simulation
Use `run_single_game.py` to run a single match. It supports detailed logging and captures initial hands and history outputs.
```bash
python run_single_game.py --config configs/game/example.json
```

### 2. Tournaments
Use `run_tournament.py` to run large-scale evaluations.
```bash
python run_tournament.py --config configs/tournament/example.json
```
*Optional config overrides*: You can mix-and-match configurations via the command line using `--player-cfg`, `--engine-cfg`, or `--tournament-cfg`.

## Config File Structure

Configuration files are structured as JSON objects with three main sections:

### 1. `players`
A list of players participating in the game or tournament. 
*   **Format**: Defined as either a dictionary `{"path": "...", "class": "...", "args": {...}, "label": "optional"}` or a compact list `["path", "class", {"args": "here"}, "optional_label"]`.
*   *Note: The compact list can be just length-2 `["path", "class"]` or 3 if no additional arguments or labels are needed. The label helps distinguish players using the same class.*

### 1b. `baselines`
An optional list with the same format as `players`. In `random_partition`-style tournaments, baseline entrants are scheduled exactly like normal players, but can also be used as score anchors for the final standings.

### 2. `engine`
Settings that control the inner game mechanics:
*   `n_players`: Number of players per game (default: 4).
*   `n_rounds`: Number of rounds played per game (default: 10).
*   `timeout` & `timeout_buffer`: Time limit (in seconds) allowed for a player's `action()`. If they exceed this, their card defaults. If they intentionally catch and swallow the alarm exception, they are disqualified.
*   `verbose`: Boolean to toggle detailed turn-by-turn print logs.

### 3. `tournament`
Settings specific to the `run_tournament.py` runner (ignored by `run_single_game.py`):
*   `grading_mode`: Boolean (default: `false`). When `true`, pins `torch.set_num_threads(1)` (and related env vars like `OMP_NUM_THREADS`) so accidental thread-level parallelism is disabled, and suppresses per-game stdout so only the final standings are printed. This is the mode used during official evaluation.
*   `type`: Type of tournament to run (`combination`, `random_partition`, or `grouped_random_partition`). **The `random_partition` tournament format will be used in final evaluations.**
*   `duplication_mode`: String (`"permutations"`, `"cycle"`, or `"none"`). Determines how hands are duplicated to reduce RNG variance. `"permutations"` plays $N!$ games with all seat assignments, `"cycle"` plays $N$ games shifting hands, and `"none"` plays 1 game. (Legacy boolean `use_permutations` is also supported).
*   `num_games_per_player` & `num_workers`: Used by `random_partition` to control the number of games played and parallel processing threads.
*   `scoring`: (Optional, used by `random_partition`). Adds a calibrated `Score` column in final standings by mapping `avg_rank` to a linear score scale defined by baseline percentiles. Supported keys are `baseline_upper_pct`, `baseline_lower_pct`, `score_at_upper_pct`, and `score_at_lower_pct`.
*   `num_groups`: (Optional, used by `grouped_random_partition`). The number of groups to split the players into for Stage 2 of the tournament based on their Stage 1 rank.
*   `max_memory_mb_per_matchup`: (Optional) Maximum memory limit in MB for a single matchup process to prevent memory leaks or out-of-memory issues from crashing the whole tournament.
*   `matchup_timeout_multiplier`: (Optional) Multiplier applied to the total expected matchup time to determine the hard timeout limit for the matchup subprocess.

## Player Disqualifications and Penalties
During games and tournaments, players may experience errors or consume excessive resources. These are marked in the final standings as:
*   **DQ (Disqualified)**: Player swallowed a timeout exception. Their subsequent moves defaults to their smallest available card.
*   **TO (Timeout)**: Player exceeded the `timeout` limit for a single turn. Their card defaults to their smallest available card.
*   **EXC (Exception)**: Player code raised an exception during their turn. Their card defaults to their smallest available card.
*   **OOM (Out of Memory)**: Player's matchup subprocess was killed due to exceeding the `max_memory_mb_per_matchup` limit. The entire matchup is aborted and players in that matchup receive no points or ranking.
*   **ERR (Generic Error)**: Player's matchup subprocess crashed for reasons other than OOM (e.g. fatal segfault, total process timeout). The entire matchup is aborted and players in that matchup receive no points or ranking.

## ⚠️ Important Warnings for Students
- **Do not use `multiprocessing` or `threading`!** Doing so will cause severe server instability and you will be penalized. The tournament orchestrator already runs games in parallel processes. Your agent must remain single-threaded.
- **Do NOT use bare `except:` or catch `BaseException` in your code.** You may accidentally catch system signals, timeout interrupts (`TimeoutException`), or out-of-memory errors (`MemoryError`), masking critical engine behaviors. Only catch specific exceptions you anticipate (e.g. `except ValueError:`).
- **Penalties for Errors:** High error rates or causing errors intentionally will result in score deductions.

## How to Add New Players
1.  Create a **subdirectory** under `src/players/` (e.g., `src/players/student_id(lowercase)/`).
2.  Add your player Python file(s) inside that directory.
3.  Your player class must implement the `action` method.

**Example:**
File: `src/players/student_id(lowercase)/best_player1.py`
```python
class BestPlayer1:
    def __init__(self, player_idx):
        self.player_idx = player_idx

    def action(self, hand, history):
        # hand: list of integers (your cards)
        # history: dict containing board state and past moves
        return hand[0] # Your logic here
```

## Game Engine Rules
*   **6th Card Rule**: If a row has 5 cards, the 6th card placed takes the row.
*   **Low Card Rule**: If a played card is lower than the last card of all rows, the player takes the row with the **lowest score**.
    *   Tie-breaking: Lowest card count -> Smallest row index.
