#!/usr/bin/env python3
"""
Patch Wavelength game attempts for agents in a population: remove dirty attempts
(missing or null response for any header), then run new game attempts until each
agent has target_attempts clean attempts. Uses one header per inference (no batch).
"""
from __future__ import annotations

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

import simulation_engine.settings as settings


def get_expected_header_ids(game_headers: List[Dict[str, Any]]) -> Set[str]:
    """Return set of header IDs: header_0, header_1, ... """
    return {f"header_{i}" for i in range(len(game_headers))}


def is_attempt_dirty(attempt: Dict[str, Any], expected_header_ids: Set[str]) -> bool:
    """
    Return True if the attempt has any header missing or any header with
    response None.
    """
    headers = attempt.get("headers") or {}
    for hid in expected_header_ids:
        h = headers.get(hid)
        if not h:
            return True
        if h.get("response") is None:
            return True
    return False


def clean_and_count_attempts(
    agent_folder: str,
    responses_data: Dict[str, Any],
    expected_header_ids: Set[str],
    builder: WavelengthResponseBuilder,
) -> Tuple[int, int]:
    """
    Remove dirty attempts from responses_data, save, and return
    (n_clean_after, n_removed).
    """
    attempts = responses_data.get("attempts") or []
    clean = [a for a in attempts if not is_attempt_dirty(a, expected_header_ids)]
    removed = len(attempts) - len(clean)
    if removed > 0:
        responses_data["attempts"] = clean
        for i, a in enumerate(responses_data["attempts"], start=1):
            a["attempt_id"] = i
        builder._save_agent_responses(agent_folder, responses_data)
    return len(clean), removed


def get_agent_folders(base_path: str) -> List[str]:
    """Return list of agent folder paths under base_path that have scratch.json."""
    base = Path(base_path)
    if not base.exists():
        raise ValueError(f"Base path does not exist: {base_path}")
    folders = []
    for agent_folder in sorted(base.iterdir()):
        if agent_folder.is_dir() and (agent_folder / "scratch.json").exists():
            folders.append(str(agent_folder))
    return folders


def main():
    parser = argparse.ArgumentParser(
        description="Patch Wavelength attempts: remove dirty attempts, then run new ones until target per agent. One header per inference."
    )
    parser.add_argument("--base", type=str, required=True,
                        help="Base path to population (e.g. agent_bank/populations/Llama-3.1-8B_agents)")
    parser.add_argument("--target-attempts", type=int, default=3,
                        help="Target number of clean attempts per agent (default: 3)")
    parser.add_argument("--game-headers", type=str, default=None,
                        help="Path to game_headers.json (default: Wavelength_Game/game_headers.json)")
    parser.add_argument("--no-reasoning", action="store_true", help="Do not store reasoning text")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report what would be done, do not run games")
    parser.add_argument("--model-choice", type=str, default=None,
                        help="Override MODEL_CHOICE (e.g. mistral-nemo-2407, llama3.1-8b, 7b, 14b). Use with CUDA_VISIBLE_DEVICES for parallel runs.")
    args = parser.parse_args()

    if args.model_choice:
        if args.model_choice not in settings.MODEL_PATHS:
            print(f"Error: Unknown model choice '{args.model_choice}'. Valid: {list(settings.MODEL_PATHS.keys())}")
            sys.exit(1)
        settings.MODEL_CHOICE = args.model_choice
        sel = settings.MODEL_PATHS[args.model_choice]
        settings.LOCAL_MODEL_NAME = sel["path"]
        settings.LLM_VERS = sel["name"]
        settings.DEVICE = "cuda:0"

    from Wavelength_Game.wavelength_response_builder import WavelengthResponseBuilder

    script_dir = Path(__file__).parent
    game_headers_path = args.game_headers or str(script_dir / "game_headers.json")
    if not os.path.exists(game_headers_path):
        print(f"Error: Game headers not found at {game_headers_path}")
        sys.exit(1)

    config = {
        "include_reasoning": not args.no_reasoning,
        "require_all_headers": True,
        "batch_by_header": False,  # one header per inference
    }
    builder = WavelengthResponseBuilder(game_headers_path, config)
    expected_header_ids = get_expected_header_ids(builder.game_headers)

    agent_folders = get_agent_folders(args.base)
    print(f"Population base: {args.base}")
    print(f"Target attempts per agent: {args.target_attempts}")
    print(f"Agents: {len(agent_folders)}")
    print(f"Mode: one header per inference (no batch)")
    if args.dry_run:
        print("DRY RUN — no games will be run")
    print()

    for agent_folder in agent_folders:
        responses_data = builder._load_agent_responses(agent_folder)
        n_clean, n_removed = clean_and_count_attempts(
            agent_folder, responses_data, expected_header_ids, builder
        )
        n_needed = max(0, args.target_attempts - n_clean)
        agent_id = Path(agent_folder).name
        if n_removed > 0:
            print(f"  {agent_id}: removed {n_removed} dirty attempt(s), {n_clean} clean remaining")
        if n_needed == 0:
            continue
        if args.dry_run:
            print(f"  {agent_id}: would run {n_needed} new attempt(s)")
            continue
        print(f"  {agent_id}: running {n_needed} new attempt(s)...")
        for _ in range(n_needed):
            try:
                builder.administer_game(agent_folder)
            except Exception as e:
                print(f"  ERROR {agent_id}: {e}", flush=True)
                break

    print("Done.")


if __name__ == "__main__":
    main()
