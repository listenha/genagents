#!/usr/bin/env python3
"""
Patch survey attempts for agents in a population: remove dirty attempts (missing or
empty response entries), then run new survey attempts until each agent has
target_attempts clean attempts. Uses one question per inference (no batch).
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

# Import settings first so we can override before builder import
import simulation_engine.settings as settings


def get_expected_question_ids_by_section(survey_data: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Build section_id -> set of question_id from survey_questions.json."""
    out = {}
    for section in survey_data.get("sections", []):
        sid = section.get("section_id")
        qids = {q.get("question_id") for q in section.get("questions", []) if q.get("question_id")}
        if sid:
            out[sid] = qids
    return out


def is_attempt_dirty(
    attempt: Dict[str, Any],
    expected_by_section: Dict[str, Set[str]]
) -> bool:
    """
    Return True if the attempt has any section missing, or any expected
    question missing or with None/empty response.
    """
    sections = attempt.get("sections") or {}
    for section_id, expected_qids in expected_by_section.items():
        if not expected_qids:
            continue
        sec = sections.get(section_id)
        if not sec:
            return True  # missing section
        responses = sec.get("responses") or {}
        for qid in expected_qids:
            if qid not in responses:
                return True  # missing question
            val = responses[qid]
            if val is None:
                return True
            # treat empty string as dirty if we ever store strings
            if isinstance(val, str) and not val.strip():
                return True
    return False


def clean_and_count_attempts(
    agent_folder: str,
    responses_data: Dict[str, Any],
    expected_by_section: Dict[str, Set[str]],
    builder: SurveyResponseBuilder,
) -> Tuple[int, int]:
    """
    Remove dirty attempts from responses_data, save, and return
    (n_clean_after, n_removed).
    """
    attempts = responses_data.get("attempts") or []
    clean = [a for a in attempts if not is_attempt_dirty(a, expected_by_section)]
    removed = len(attempts) - len(clean)
    if removed > 0:
        responses_data["attempts"] = clean
        # Re-number attempt_id to 1..len(clean) for consistency (optional)
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
        description="Patch survey attempts: remove dirty attempts, then run new ones until target per agent. Single question per inference."
    )
    parser.add_argument("--base", type=str, required=True,
                        help="Base path to population (e.g. agent_bank/populations/Llama-3.1-8B_agents)")
    parser.add_argument("--target-attempts", type=int, default=3,
                        help="Target number of clean attempts per agent (default: 3)")
    parser.add_argument("--survey", type=str, default=None,
                        help="Path to survey_questions.json (default: Surveys/survey_questions.json)")
    parser.add_argument("--no-reasoning", action="store_true", help="Do not store reasoning text")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report what would be done, do not run surveys")
    parser.add_argument("--model-choice", type=str, default=None,
                        help="Override MODEL_CHOICE (e.g. mistral-nemo-2407, llama3.1-8b, 7b, 14b). Use with CUDA_VISIBLE_DEVICES for parallel runs.")
    args = parser.parse_args()

    # Override settings before any code loads the model (for parallel multi-GPU runs)
    if args.model_choice:
        if args.model_choice not in settings.MODEL_PATHS:
            print(f"Error: Unknown model choice '{args.model_choice}'. Valid: {list(settings.MODEL_PATHS.keys())}")
            sys.exit(1)
        settings.MODEL_CHOICE = args.model_choice
        sel = settings.MODEL_PATHS[args.model_choice]
        settings.LOCAL_MODEL_NAME = sel["path"]
        settings.LLM_VERS = sel["name"]
        # When CUDA_VISIBLE_DEVICES is set to one GPU, use cuda:0 (the only visible device)
        settings.DEVICE = "cuda:0"

    from Surveys.survey_response_builder import SurveyResponseBuilder

    script_dir = Path(__file__).parent
    survey_path = args.survey or str(script_dir / "survey_questions.json")
    if not os.path.exists(survey_path):
        print(f"Error: Survey questions not found at {survey_path}")
        sys.exit(1)

    config = {
        "include_reasoning": not args.no_reasoning,
        "require_all_sections": True,
        "batch_by_section": False,  # one question per inference
    }
    builder = SurveyResponseBuilder(survey_path, config)
    survey_data = builder.survey_data
    expected_by_section = get_expected_question_ids_by_section(survey_data)

    agent_folders = get_agent_folders(args.base)
    print(f"Population base: {args.base}")
    print(f"Target attempts per agent: {args.target_attempts}")
    print(f"Agents: {len(agent_folders)}")
    print(f"Mode: one question per inference (no batch)")
    if args.dry_run:
        print("DRY RUN — no surveys will be run")
    print()

    for agent_folder in agent_folders:
        responses_data = builder._load_agent_responses(agent_folder)
        n_clean, n_removed = clean_and_count_attempts(
            agent_folder, responses_data, expected_by_section, builder
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
                builder.administer_survey(agent_folder)
            except Exception as e:
                print(f"  ERROR {agent_id}: {e}", flush=True)
                break

    print("Done.")


if __name__ == "__main__":
    main()
