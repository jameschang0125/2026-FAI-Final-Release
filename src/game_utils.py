import importlib
from src.engine import Engine, silenced_if
from copy import deepcopy
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import defaultdict
import itertools


def _grading_mode(config):
    """Pull grading_mode out of either top-level or tournament sub-config."""
    return bool(config.get("grading_mode")
                or config.get("tournament", {}).get("grading_mode"))


def load_players(config, verbose=False):
    assert "players" in config, "Config must have a 'players' key"
    assert isinstance(config["players"], list), "Players must be a list"

    if verbose:
        print("--- Importing Players ---")
    imported_players = []
    grading = _grading_mode(config)

    for i, player_conf in enumerate(config["players"]):
        try:
            path = player_conf["path"]
            cls_name = player_conf["class"]

            # 1. Import module (silence student import-time prints in grading mode)
            with silenced_if(grading):
                module = importlib.import_module(path)

            # 2. Get Class
            cls = getattr(module, cls_name)

            # 3. Append class
            imported_players.append(cls)

            if verbose:
                print(f"Imported {cls.__name__} from {path}")

        except Exception as e:
            print(f"Failed to import {cls_name} from {path} for player {i}:  {e}")
            raise e

    return imported_players

def _normalize_player_entries(entries, is_baseline):
    normalized = []
    for p in entries:
        if isinstance(p, list):
            item = {
                "path": p[0],
                "class": p[1],
            }
            if len(p) > 2:
                item["args"] = p[2]
            if len(p) > 3:
                item["label"] = p[3]
        elif isinstance(p, dict):
            item = dict(p)
        else:
            raise ValueError(f"Invalid player config: {p}")
        item["is_baseline"] = is_baseline
        normalized.append(item)
    return normalized


def _preprocess_player_config(config):
    import copy
    config = copy.deepcopy(config)
    players = _normalize_player_entries(config.get("players", []), is_baseline=False)
    baselines = _normalize_player_entries(config.get("baselines", []), is_baseline=True)
    merged_players = players + baselines
    for i, p in enumerate(merged_players):
        p["player_id"] = i
    config["players"] = merged_players
    config["baselines"] = baselines
    # Mirror grading_mode into the engine sub-block so Engine sees it
    # without having to peek at the top-level config.
    if _grading_mode(config):
        config.setdefault("engine", {})["grading_mode"] = True
    return config


