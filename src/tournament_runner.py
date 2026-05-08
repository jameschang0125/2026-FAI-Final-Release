
import copy
import random
import itertools
import multiprocessing as mp
import queue
import math
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed
from src.engine import Engine, is_oom, silenced_if
from src.game_utils import load_players, _preprocess_player_config

class BaseTournamentRunner:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.config = _preprocess_player_config(self.config)
        
        self.engine_config = self.config.get("engine", {})
        self.n_players_per_game = self.engine_config.get("n_players", 4)
        self.grading_mode = bool(self.engine_config.get("grading_mode", False))
        
        self.tournament_config = self.config.get("tournament", {})
        if "use_permutations" in self.tournament_config:
            self.duplication_mode = "permutations" if self.tournament_config["use_permutations"] else "none"
        else:
            self.duplication_mode = self.tournament_config.get("duplication_mode", "permutations")
        self.num_games_per_player = self.tournament_config.get("num_games_per_player", 10)

        # Runner-owned RNG: isolates tournament shuffles from the global
        # `random` state, which player code can mutate via random.seed().
        # seed=None falls back to fresh OS entropy (same as global default).
        self.rng = random.Random(self.tournament_config.get("seed"))

        # Load all player classes
        self.player_classes = load_players(self.config, verbose=True)
        self.player_configs = self.config.get("players", [])
        
        if len(self.player_classes) < self.n_players_per_game:
            raise ValueError(f"Not enough players! Have {len(self.player_classes)}, need {self.n_players_per_game}")

        # Elo estimation structures
        self.pairwise_wins = defaultdict(lambda: defaultdict(float))

        # Diagnostics: number of caught perm-level crashes inside matchups
        self.perm_failures = 0

    def _player_label(self, config_idx):
        cfg = self.player_configs[config_idx]
        return cfg.get("label", "-")

    def _build_notes(self, p):
        notes = []
        original_n = getattr(self, "original_num_players", None)
        if original_n is not None and p["id"] >= original_n:
            notes.append("(PAD)")
        for key, label in (("dq_count", "DQ"),
                           ("timeout_count", "TO"),
                           ("exception_count", "EXC"),
                           ("err_oom_count", "OOM"),
                           ("err_generic_count", "ERR")):
            v = p.get(key, 0)
            if v:
                notes.append(f"{label}: {v}")
        return " ".join(notes)

    def compute_elo(self, player_stats):
        """
        Computes the Bradley-Terry MLE for Elo ratings based on accumulated pairwise wins.
        """
        num_players = len(self.player_classes)
        p = [1.0] * num_players

        # Read pairwise_wins via .get(); subscripting a defaultdict here would
        # silently materialize empty rows for every (i, j) we touch.
        wins = self.pairwise_wins

        # Max iterations for Minorization-Maximization
        max_iters = 100
        for _ in range(max_iters):
            new_p = [0.0] * num_players
            for i in range(num_players):
                row_i = wins.get(i, {})
                wins_i = sum(row_i.values())
                denom_sum = 0.0
                for j in range(num_players):
                    if i != j:
                        n_ij = row_i.get(j, 0.0) + wins.get(j, {}).get(i, 0.0)
                        if n_ij > 0:
                            denom_sum += n_ij / (p[i] + p[j])
                if denom_sum > 0:
                    new_p[i] = wins_i / denom_sum
                else:
                    new_p[i] = p[i]

            # Keep average p at 1.0 to prevent drift
            avg_p = sum(new_p) / num_players
            if avg_p > 0:
                p = [x / avg_p for x in new_p]
            else:
                p = new_p

        # Convert p values to Elo: p_i = 10^(R_i / 400) -> R_i = 400 * log10(p_i)
        # Shift ratings so avg is 1500
        raw_elos = []
        for val in p:
            if val <= 0:
                val = 1e-6 # smoothing
            raw_elos.append(400 * math.log10(val))
            
        avg_elo = sum(raw_elos) / len(raw_elos)
        shift = 1500 - avg_elo
        final_elos = [r + shift for r in raw_elos]
        
        # Write to stats
        for stat in player_stats:
            stat["est_elo"] = final_elos[stat["config_idx"]]


    def _play_matchup_permutations(self, matchup_players_data, n_cards, n_rounds_game, matchup_seed):
        """
        Runs the set of permuted games for a single matchup.
        matchup_players_data: List of tuples/dicts/objects identifying the players for the engine.
                              Needs to allow retrieving class/config.
                              Inputs here are indices in self.player_classes/configs basically.

        matchup_seed: Per-matchup seed used to construct a local RNG for the deal.
        Allocated by the parent (advancing self.rng) so that each matchup gets a
        fresh deck even when this method runs inside a forked subprocess
        (multiprocessing._bootstrap auto-reseeds the global `random`, but it
        does NOT reseed instance-owned random.Random objects like self.rng).

        To allow flexibility between Swiss (standings dicts) and Combination (global indices),
        we'll expect a list of 'global_indices'.
        """
        # Generate ONE deal of hands and one initial board, from the same deck,
        # so all permutations of this matchup share the same starting state.
        deal_rng = random.Random(matchup_seed)
        deck = list(range(1, n_cards + 1))
        deal_rng.shuffle(deck)

        base_hands = []
        for _ in range(self.n_players_per_game):
            h = []
            for _ in range(n_rounds_game):
                h.append(deck.pop())
            base_hands.append(sorted(h))

        board_size_y = self.engine_config.get("board_size_y", 4)
        base_board = [[deck.pop()] for _ in range(board_size_y)]

        hand_indices = list(range(self.n_players_per_game))

        if self.duplication_mode == "permutations":
            selected_perms = list(itertools.permutations(hand_indices))
        elif self.duplication_mode == "cycle":
            selected_perms = [tuple(hand_indices[i:] + hand_indices[:i]) for i in range(len(hand_indices))]
        else:
            selected_perms = [tuple(hand_indices)]

        matchup_scores = [0] * self.n_players_per_game
        matchup_ranks = [0] * self.n_players_per_game
        n_completed_perms = 0
        n_failed_perms = 0

        # Local pairwise wins (to avoid modifying self directly when in a subprocess)
        local_pairwise_wins = {p_id: {p_id_2: 0.0 for p_id_2 in matchup_players_data} for p_id in matchup_players_data}
        local_dq_counts = {p_id: 0 for p_id in matchup_players_data}
        local_timeout_counts = {p_id: 0 for p_id in matchup_players_data}
        local_exception_counts = {p_id: 0 for p_id in matchup_players_data}

        seat_resolutions = []
        for global_idx in matchup_players_data:
            p_conf = self.player_configs[global_idx]
            seat_resolutions.append((self.player_classes[global_idx], p_conf.get("args")))

        for perm in selected_perms:
            current_fixed_hands = [base_hands[hand_idx] for hand_idx in perm]

            # Instantiate
            game_players = []
            for seat, (p_cls, p_args) in enumerate(seat_resolutions):
                try:
                    with silenced_if(self.grading_mode):
                        if p_args is None:
                            inst = p_cls(player_idx=seat)
                        else:
                            inst = p_cls(player_idx=seat, **p_args)
                    # Reset global random state after each __init__ to prevent
                    # players from manipulating each other's random streams.
                    random.seed(None)
                    game_players.append(inst)
                except Exception as e:
                    print(f"Error creating player {matchup_players_data[seat]}: {e}")
                    raise e

            # Build per-perm engine config (shallow; engine doesn't mutate it).
            current_engine_config = dict(
                self.engine_config,
                fixed_hands=current_fixed_hands,
                fixed_board=base_board,
            )

            try:
                engine = Engine(current_engine_config, game_players)
                scores, full_history = engine.play_game()
                
                # Compute fractional ranks dealing with ties (e.g., tie for 1st and 2nd gives 1.5)
                ranks = [0.0] * len(scores)
                for i, score in enumerate(scores):
                    better_count = sum(1 for s in scores if s < score)
                    same_count = sum(1 for s in scores if s == score)
                    ranks[i] = (2 * better_count + same_count + 1) / 2.0
                    
                for seat, score in enumerate(scores):
                    matchup_scores[seat] += score
                    matchup_ranks[seat] += ranks[seat]
                    
                # Track Pairwise pairwise logic
                for i in range(len(scores)):
                    for j in range(len(scores)):
                        if i == j: continue
                        p1_idx = matchup_players_data[i]
                        p2_idx = matchup_players_data[j]
                        if ranks[i] < ranks[j]:
                            local_pairwise_wins[p1_idx][p2_idx] += 1.0
                        elif ranks[i] == ranks[j]:
                            local_pairwise_wins[p1_idx][p2_idx] += 0.5
                            
                for seat in full_history.get("disqualified_players", []):
                    local_dq_counts[matchup_players_data[seat]] += 1
                
                for seat, counts in full_history.get("timeout_counts", {}).items():
                    local_timeout_counts[matchup_players_data[int(seat)]] += counts

                for seat, counts in full_history.get("exception_counts", {}).items():
                    local_exception_counts[matchup_players_data[int(seat)]] += counts

                n_completed_perms += 1

            except Exception as e:
                if is_oom(e):
                    raise
                n_failed_perms += 1
                print(f"Error running game in matchup {matchup_players_data}: {e}")

        return matchup_scores, matchup_ranks, n_completed_perms, n_failed_perms, local_pairwise_wins, local_dq_counts, local_timeout_counts, local_exception_counts


class CombinationTournamentRunner(BaseTournamentRunner):
    def __init__(self, config):
        super().__init__(config)
        self.player_stats = []
        for i in range(len(self.player_classes)):
            self.player_stats.append({
                "id": i,
                "config_idx": i,
                "total_score": 0,
                "total_rank": 0,
                "games_played": 0,
                "matchups_played": 0,
                "dq_count": 0,
                "timeout_count": 0,
                "exception_count": 0
            })
            
    def run(self):
        print(f"--- Starting Combination Tournament (All C({len(self.player_classes)}, {self.n_players_per_game})) ---")
        print(f"Duplication Mode: {self.duplication_mode.upper()}")
        
        all_indices = range(len(self.player_classes))
        combinations = list(itertools.combinations(all_indices, self.n_players_per_game))
        
        print(f"Total Matchups: {len(combinations)}")
        
        n_cards = self.engine_config.get("n_cards", 104)
        n_rounds_game = self.engine_config.get("n_rounds", 10)
        
        matchup_history = []
        
        for idx, combo in enumerate(tqdm(combinations, desc="Running Matchups")):
            # combo is tuple of global indices
            matchup_seed = self.rng.getrandbits(64)
            scores, ranks, n_games, n_failed, local_wins, local_dqs, local_timeouts, local_exceptions = self._play_matchup_permutations(combo, n_cards, n_rounds_game, matchup_seed)
            self.perm_failures += n_failed
            
            # Aggregate local pairwise wins
            for p1, opp_wins in local_wins.items():
                for p2, w in opp_wins.items():
                    self.pairwise_wins[p1][p2] += w
                    
            for p1, dqs in local_dqs.items():
                self.player_stats[p1]["dq_count"] += dqs
                
            for p1, tos in local_timeouts.items():
                self.player_stats[p1]["timeout_count"] += tos

            for p1, exc in local_exceptions.items():
                self.player_stats[p1]["exception_count"] += exc
            
            matchup_res_list = []
            for seat, score in enumerate(scores):
                global_p_id = combo[seat]
                self.player_stats[global_p_id]["total_score"] += score
                self.player_stats[global_p_id]["total_rank"] += ranks[seat]
                self.player_stats[global_p_id]["games_played"] += n_games
                self.player_stats[global_p_id]["matchups_played"] += 1
                matchup_res_list.append({"id": global_p_id, "score": score, "rank": ranks[seat]})
            
            matchup_history.append({
                "matchup_id": idx,
                "players": list(combo),
                "results": matchup_res_list
            })
            
        self.compute_elo(self.player_stats)
        return self.player_stats, matchup_history

    def print_standings(self):
        for p in self.player_stats:
            p["avg_score"] = p["total_score"] / p["games_played"] if p["games_played"] > 0 else float('inf')
            p["avg_rank"] = p["total_rank"] / p["games_played"] if p["games_played"] > 0 else float('inf')
        
        self.player_stats.sort(key=lambda x: (x["avg_rank"], x["avg_score"]))
        
        print(f"\nFinal Standings (Sorted by Avg Rank):")
        print("-" * 110)
        print(f"{'Rank':<5} {'ID':<5} {'Class':<22} {'Label':<9} {'Avg Rank':<9} {'Est. Elo':<9} {'Avg Score':<9} {'Games':<6} {'Note':<9}")
        print("-" * 110)
        
        for i, p in enumerate(self.player_stats):
            p_cls_name = self.player_configs[p["config_idx"]]["class"]
            if len(p_cls_name) > 21: p_cls_name = p_cls_name[:18] + "..."
            label = self._player_label(p["config_idx"])
            if len(label) > 9: label = label[:9]

            note_str = self._build_notes(p)

            elo = p.get("est_elo", 1500)
            print(f"{i+1:<5} {p['id']:<5} {p_cls_name:<22} {label:<9} {p['avg_rank']:<9.2f} {elo:<9.0f} {p['avg_score']:<9.2f} {p['games_played']:<6} {note_str:<9}")
        print("-" * 110)


class RandomPartitionTournamentRunner(BaseTournamentRunner):
    def __init__(self, config):
        super().__init__(config)
        
        self.original_num_players = len(self.player_classes)
        remainder = self.original_num_players % self.n_players_per_game
        if remainder != 0:
            num_pads = self.n_players_per_game - remainder
            print(f"Padding with {num_pads} players to make player count a multiple of {self.n_players_per_game}.")
            # Use RandomPlayer as padding
            from src.players.TA.random_player import RandomPlayer
            pad_cls = RandomPlayer
            pad_conf = {"path": "src.players.TA.random_player", "class": "RandomPlayer", "label": "(PAD)"}
            
            for _ in range(num_pads):
                self.player_classes.append(pad_cls)
                self.player_configs.append(pad_conf)
                
        self.player_stats = []
        for i in range(len(self.player_classes)):
            self.player_stats.append({
                "id": i,
                "config_idx": i,
                "is_baseline": self.player_configs[i].get("is_baseline", False),
                "total_score": 0,
                "total_rank": 0,
                "games_played": 0,
                "matchups_played": 0,
                "dq_count": 0,
                "timeout_count": 0,
                "exception_count": 0,
                "err_count": 0,
                "err_oom_count": 0,
                "err_generic_count": 0
            })

        self.matchup_timeout_multiplier = self.tournament_config.get("matchup_timeout_multiplier", 1.5)
        self.max_memory_mb_per_matchup = self.tournament_config.get("max_memory_mb_per_matchup", None)
        self.scoring_config = self.tournament_config.get("scoring", None)
        self.matchup_timeout_killed = 0
        self.matchup_oom_killed = 0
        self.matchup_crash = 0

    @staticmethod
    def _normalize_pct(value):
        if value is None:
            return None
        value = float(value)
        if value > 1.0 or value < 0.0:
            raise ValueError(f"Invalid percentage value: {value}")
        return value

    @staticmethod
    def _interpolate_sorted(values, pct):
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        pos = pct * (len(values) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return values[lo]
        w = pos - lo
        return values[lo] * (1.0 - w) + values[hi] * w

    def _compute_baseline_scores(self):
        if not self.scoring_config:
            return False

        upper_pct = self._normalize_pct(
            self.scoring_config.get("baseline_upper_pct", self.scoring_config.get("upper_pct"))
        )
        lower_pct = self._normalize_pct(
            self.scoring_config.get("baseline_lower_pct", self.scoring_config.get("lower_pct"))
        )
        upper_score = self.scoring_config.get("score_at_upper_pct", self.scoring_config.get("upper_score"))
        lower_score = self.scoring_config.get("score_at_lower_pct", self.scoring_config.get("lower_score"))
        if None in (upper_pct, lower_pct, upper_score, lower_score):
            return False

        baseline_ranks = [
            p["avg_rank"]
            for p in self.player_stats
            if p.get("is_baseline") and p["id"] < self.original_num_players and math.isfinite(p["avg_rank"])
        ]
        baseline_ranks.sort(reverse=True)
        if len(baseline_ranks) < 2:
            return False

        upper_rank = self._interpolate_sorted(baseline_ranks, upper_pct)
        lower_rank = self._interpolate_sorted(baseline_ranks, lower_pct)
        if upper_rank is None or lower_rank is None:
            return False
        if abs(lower_rank - upper_rank) < 1e-12:
            for p in self.player_stats:
                if p["id"] < self.original_num_players:
                    p["calibrated_score"] = float(upper_score)
            return True

        slope = (float(lower_score) - float(upper_score)) / (lower_rank - upper_rank)
        intercept = float(upper_score) - slope * upper_rank
        def _compute_calibrated_score(p):
            if math.isfinite(p["avg_rank"]):
                score = intercept + slope * p["avg_rank"]
                return max(0.0, min(100.0, score))
            return float("nan")
        for p in self.player_stats:
            if p["id"] < self.original_num_players:
                p["calibrated_score"] = _compute_calibrated_score(p)
        return True

    def _duplication_games_count(self):
        if self.duplication_mode == "permutations":
            return math.factorial(self.n_players_per_game)
        if self.duplication_mode == "cycle":
            return self.n_players_per_game
        return 1

    def _compute_matchup_timeout_seconds(self, n_rounds_game):
        timeout = self.engine_config.get("timeout", None)
        if timeout is None:
            return None
        timeout_buffer = self.engine_config.get("timeout_buffer", 0.5)
        dup_games = self._duplication_games_count()
        return (timeout + timeout_buffer) * self.n_players_per_game * n_rounds_game * dup_games * self.matchup_timeout_multiplier

    @staticmethod
    def _run_matchup_worker(runner, combo, n_cards, n_rounds_game, memory_mb, out_queue, matchup_seed):
        import os
        os.setpgrp() # Create a new process group for resource management and clean teardown

        try:
            if memory_mb is not None:
                import resource
                limit_bytes = int(float(memory_mb) * 1024 * 1024)
                if limit_bytes > 0:
                    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

            result = runner._play_matchup_permutations(combo, n_cards, n_rounds_game, matchup_seed)
            out_queue.put({"status": "ok", "result": result})
        except BaseException as e:
            if is_oom(e):
                out_queue.put({"status": "oom_killed", "error": str(e)})
            else:
                out_queue.put({"status": "crash", "error": f"{type(e).__name__}: {e}"})

    def _run_matchup_isolated(self, combo, n_cards, n_rounds_game, matchup_seed):
        timeout_s = self._compute_matchup_timeout_seconds(n_rounds_game)
        memory_mb = self.max_memory_mb_per_matchup

        # Fast path: no hardening controls configured
        if timeout_s is None and memory_mb is None:
            return {"status": "ok", "result": self._play_matchup_permutations(combo, n_cards, n_rounds_game, matchup_seed)}

        ctx = mp.get_context("fork")
        out_queue = ctx.Queue(maxsize=1)
        proc = ctx.Process(
            target=RandomPartitionTournamentRunner._run_matchup_worker,
            args=(self, combo, n_cards, n_rounds_game, memory_mb, out_queue, matchup_seed),
            daemon=True
        )
        proc.start()
        proc.join(timeout_s)

        if proc.is_alive():
            import os
            import signal
            
            # Kill the entire process group to prevent orphaned children
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.terminate()
            
            proc.join(0.5)
            
            if proc.is_alive():
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()
                proc.join(0.5)
            return {"status": "timeout_killed", "result": None}

        msg = None
        try:
            msg = out_queue.get_nowait()
        except queue.Empty:
            msg = None

        try:
            out_queue.close()
            out_queue.join_thread()
        except Exception:
            pass

        if msg is not None:
            return msg

        if memory_mb is not None and proc.exitcode in (-9, -11, 137):
            return {"status": "oom_killed", "result": None}
        return {"status": "crash", "error": f"Child exited with code {proc.exitcode}", "result": None}
            
    def _run_partition_games(self, groups, num_games):
        num_workers = self.tournament_config.get("num_workers", 1)

        # Snapshot diagnostic counters so the per-stage ERR summary reports
        # only this stage's failures, not cumulative across prior stages.
        # (matchup_*_killed/crash counters have the same global-accumulator
        # shape but are reported cumulatively today; not fixing here.)
        perm_failures_at_start = self.perm_failures

        n_cards = self.engine_config.get("n_cards", 104)
        n_rounds_game = self.engine_config.get("n_rounds", 10)
        
        # 1. Generate all matchups across all partitions upfront. We allocate
        # the per-matchup seeds in the parent so self.rng advances per matchup
        # and forked workers each get a unique deal (see _play_matchup_permutations).
        all_matchups = []
        for group in groups:
            for r in range(num_games):
                indices = list(group)
                self.rng.shuffle(indices)
                matchups = [indices[i:i + self.n_players_per_game] for i in range(0, len(indices), self.n_players_per_game)]
                for combo in matchups:
                    if len(combo) == self.n_players_per_game:
                        all_matchups.append((r + 1, combo, self.rng.getrandbits(64)))

        # 2. Parallel runner execution
        if num_workers > 1:
            results = Parallel(n_jobs=num_workers)(
                delayed(self._run_matchup_isolated)(combo, n_cards, n_rounds_game, seed)
                for r, combo, seed in tqdm(all_matchups, desc="Tournament")
            )
        else:
            results = [self._run_matchup_isolated(combo, n_cards, n_rounds_game, seed)
                       for r, combo, seed in tqdm(all_matchups, desc="Tournament")]

        # 3. Process results to build history and compute aggregated stats
        matchup_history = []
        current_partition = 1
        current_matchups = []
        current_round_results = []

        for idx, (outcome, (r, combo, _)) in enumerate(zip(results, all_matchups)):
            status = outcome.get("status", "crash")
            
            if r != current_partition:
                matchup_history.append({
                    "partition": current_partition,
                    "matchups": current_matchups,
                    "results": current_round_results
                })
                current_partition = r
                current_matchups = []
                current_round_results = []
                
            current_matchups.append(combo)

            if status != "ok":
                if status == "timeout_killed":
                    self.matchup_timeout_killed += 1
                elif status == "oom_killed":
                    self.matchup_oom_killed += 1
                else:
                    self.matchup_crash += 1

                err_result = []
                for global_p_id in combo:
                    self.player_stats[global_p_id]["matchups_played"] += 1
                    self.player_stats[global_p_id]["err_count"] += 1
                    if status == "oom_killed":
                        self.player_stats[global_p_id]["err_oom_count"] += 1
                    else:
                        self.player_stats[global_p_id]["err_generic_count"] += 1
                    err_result.append({"id": global_p_id, "score": None, "rank": None, "status": "ERR"})
                current_round_results.append(err_result)
                continue

            res = outcome["result"]
            scores, ranks, n_games, n_failed, local_wins, local_dqs, local_timeouts, local_exceptions = res
            self.perm_failures += n_failed
            
            # Aggregate local pairwise wins
            for p1, opp_wins in local_wins.items():
                for p2, w in opp_wins.items():
                    self.pairwise_wins[p1][p2] += w
                    
            for p1, dqs in local_dqs.items():
                self.player_stats[p1]["dq_count"] += dqs
                
            for p1, tos in local_timeouts.items():
                self.player_stats[p1]["timeout_count"] += tos

            for p1, exc in local_exceptions.items():
                self.player_stats[p1]["exception_count"] += exc
                    
            matchup_res_list = []
            for seat, score in enumerate(scores):
                global_p_id = combo[seat]
                self.player_stats[global_p_id]["total_score"] += score
                self.player_stats[global_p_id]["total_rank"] += ranks[seat]
                self.player_stats[global_p_id]["games_played"] += n_games
                self.player_stats[global_p_id]["matchups_played"] += 1
                matchup_res_list.append({"id": global_p_id, "score": score, "rank": ranks[seat]})
            
            current_round_results.append(matchup_res_list)
            
        # Append final partition
        if current_matchups:
            matchup_history.append({
                "partition": current_partition,
                "matchups": current_matchups,
                "results": current_round_results
            })

        failed_matchups = self.matchup_timeout_killed + self.matchup_oom_killed + self.matchup_crash
        stage_perm_failures = self.perm_failures - perm_failures_at_start
        if failed_matchups > 0 or stage_perm_failures > 0:
            print(
                f"Matchup ERR summary: total={failed_matchups}, "
                f"timeouts={self.matchup_timeout_killed}, "
                f"oom={self.matchup_oom_killed}, crash={self.matchup_crash}, "
                f"perm_crashes={stage_perm_failures}"
            )
            
        return matchup_history

    def run(self):
        print(f"--- Starting Random Partition Tournament ({self.num_games_per_player} partitions per player) ---")
        print(f"Duplication Mode: {self.duplication_mode.upper()}")
        print(f"Workers: {self.tournament_config.get('num_workers', 1)}")
        
        all_indices = list(range(len(self.player_classes)))
        matchup_history = self._run_partition_games([all_indices], self.num_games_per_player)
        self.compute_elo(self.player_stats)
        return self.player_stats, matchup_history

    def print_standings(self):
        for p in self.player_stats:
            p["avg_score"] = p["total_score"] / p["games_played"] if p["games_played"] > 0 else float('inf')
            p["avg_rank"] = p["total_rank"] / p["games_played"] if p["games_played"] > 0 else float('inf')
            p["calibrated_score"] = None

        has_calibrated_score = self._compute_baseline_scores()
        
        # Sort only original players primarily, but we can sort all
        self.player_stats.sort(key=lambda x: (x["avg_rank"], x["avg_score"]))
        
        print(f"\nFinal Standings (Sorted by Avg Rank):")
        line_width = 123 if has_calibrated_score else 110
        print("-" * line_width)
        if has_calibrated_score:
            print(f"{'Rank':<5} {'ID':<5} {'Class':<22} {'Label':<9} {'Avg Rank':<9} {'Score':<8} {'Est. Elo':<9} {'Avg Score':<9} {'Games':<6} {'Note':<9}")
        else:
            print(f"{'Rank':<5} {'ID':<5} {'Class':<22} {'Label':<9} {'Avg Rank':<9} {'Est. Elo':<9} {'Avg Score':<9} {'Games':<6} {'Note':<9}")
        print("-" * line_width)
        
        for i, p in enumerate(self.player_stats):
            p_cls_name = self.player_configs[p["config_idx"]]["class"]
            if len(p_cls_name) > 21: p_cls_name = p_cls_name[:18] + "..."
            label = self._player_label(p["config_idx"])
            if len(label) > 9: label = label[:9]

            note_str = self._build_notes(p)

            elo = p.get("est_elo", 1500)
            if has_calibrated_score:
                score_str = f"{p['calibrated_score']:.2f}" if p.get("calibrated_score") is not None and math.isfinite(p["calibrated_score"]) else "-"
                print(f"{i+1:<5} {p['id']:<5} {p_cls_name:<22} {label:<9} {p['avg_rank']:<9.2f} {score_str:<8} {elo:<9.0f} {p['avg_score']:<9.2f} {p['games_played']:<6} {note_str:<9}")
            else:
                print(f"{i+1:<5} {p['id']:<5} {p_cls_name:<22} {label:<9} {p['avg_rank']:<9.2f} {elo:<9.0f} {p['avg_score']:<9.2f} {p['games_played']:<6} {note_str:<9}")
        print("-" * line_width)


class GroupedRandomPartitionTournamentRunner(RandomPartitionTournamentRunner):
    def __init__(self, config):
        super().__init__(config)
        self.num_groups = self.tournament_config.get("num_groups", 2)
        self.num_games_stage_2 = self.num_games_per_player
        
        for p in self.player_stats:
            p["total_score_1"] = 0
            p["total_rank_1"] = 0
            p["games_played_1"] = 0
            p["avg_score_1"] = 0.0
            p["avg_rank_1"] = 0.0
            p["total_score_2"] = 0
            p["total_rank_2"] = 0
            p["games_played_2"] = 0
            p["avg_score_2"] = 0.0
            p["avg_rank_2"] = 0.0
            p["group_id"] = -1

    def run(self):
        print(f"--- Starting Grouped Random Partition Tournament (Stage 1: {self.num_games_per_player} games, Stage 2: {self.num_games_stage_2} games) ---")
        print(f"Duplication Mode: {self.duplication_mode.upper()}")
        print(f"Workers: {self.tournament_config.get('num_workers', 1)}")
        
        all_indices = list(range(len(self.player_classes)))
        
        print("\n=== STAGE 1 (All Players) ===")
        history_1 = self._run_partition_games([all_indices], self.num_games_per_player)
        
        for p in self.player_stats:
            p["avg_score_1"] = p["total_score"] / p["games_played"] if p["games_played"] > 0 else float('inf')
            p["avg_rank_1"] = p["total_rank"] / p["games_played"] if p["games_played"] > 0 else float('inf')
            p["total_score_1"] = p["total_score"]
            p["total_rank_1"] = p["total_rank"]
            p["games_played_1"] = p["games_played"]
            
        sorted_indices = sorted(all_indices, key=lambda i: (self.player_stats[i]["avg_rank_1"], self.player_stats[i]["avg_score_1"]))
        
        num_players = len(sorted_indices)
        num_tables = num_players // self.n_players_per_game
        tables_per_group = [num_tables // self.num_groups] * self.num_groups
        for i in range(num_tables % self.num_groups):
            tables_per_group[i] += 1
            
        groups = []
        curr = 0
        for g_id, t_count in enumerate(tables_per_group):
            g_size = t_count * self.n_players_per_game
            group = sorted_indices[curr:curr+g_size]
            groups.append(group)
            for p_id in group:
                self.player_stats[p_id]["group_id"] = g_id
            curr += g_size
            
        print(f"\n=== STAGE 2 (Groups: {[len(g) for g in groups]}) ===")
        history_2 = self._run_partition_games(groups, self.num_games_stage_2)
        
        for p in self.player_stats:
            p["total_score_2"] = p["total_score"] - p["total_score_1"]
            p["total_rank_2"] = p["total_rank"] - p["total_rank_1"]
            p["games_played_2"] = p["games_played"] - p["games_played_1"]
            p["avg_score_2"] = p["total_score_2"] / p["games_played_2"] if p["games_played_2"] > 0 else float('inf')
            p["avg_rank_2"] = p["total_rank_2"] / p["games_played_2"] if p["games_played_2"] > 0 else float('inf')
            p["avg_score"] = p["total_score"] / p["games_played"] if p["games_played"] > 0 else float('inf')
            p["avg_rank"] = p["total_rank"] / p["games_played"] if p["games_played"] > 0 else float('inf')
            
        self.compute_elo(self.player_stats)
        return self.player_stats, {"stage1": history_1, "stage2": history_2}

    def print_standings(self):
        self.player_stats.sort(key=lambda x: (x["group_id"], x["avg_rank_2"], x["avg_score_2"]))
        
        print(f"\nFinal Standings (Sorted by Group, then Stage 2 Rank):")
        print("-" * 132)
        print(f"{'Grp':<3} {'Rank':<5} {'ID':<5} {'Class':<22} {'Label':<9} {'AvgRk 1':<8} {'AvgRk 2':<8} {'Est. Elo':<9} {'TotalG':<6} {'Note':<9}")
        print("-" * 132)
        
        for i, p in enumerate(self.player_stats):
            p_cls_name = self.player_configs[p["config_idx"]]["class"]
            if len(p_cls_name) > 21: p_cls_name = p_cls_name[:18] + "..."
            label = self._player_label(p["config_idx"])
            if len(label) > 9: label = label[:9]

            note_str = self._build_notes(p)

            elo = p.get("est_elo", 1500)
            print(f"{p['group_id']:<3} {i+1:<5} {p['id']:<5} {p_cls_name:<22} {label:<9} {p['avg_rank_1']:<8.2f} {p['avg_rank_2']:<8.2f} {elo:<9.0f} {p['games_played']:<6} {note_str:<9}")
        print("-" * 132)
