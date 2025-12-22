#!/usr/bin/env python3
"""
Translate raw survey responses to distilled scores.

This script applies reverse scoring and calculates trait/total scores according to
the scoring rules for BFI-10, BES-A, and REI instruments.

Output: survey_distilled_responses.json (only distilled scores, no raw responses)
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_range(range_str: str) -> List[str]:
    """Parse a range string like "0000-0049" into a list of agent IDs."""
    try:
        start_str, end_str = range_str.split('-')
        start = int(start_str)
        end = int(end_str)
        
        if start > end:
            raise ValueError(f"Start ({start}) must be <= end ({end})")
        
        padding = len(start_str)
        return [f"{i:0{padding}d}" for i in range(start, end + 1)]
    except ValueError as e:
        raise ValueError(f"Invalid range format '{range_str}'. Expected format: '0000-0049'. Error: {e}")


def reverse_score(value: int) -> int:
    """Reverse score a 1-5 Likert scale value."""
    if value is None or value < 1 or value > 5:
        return None
    return 6 - value  # 1→5, 2→4, 3→3, 4→2, 5→1


def translate_bfi10(responses: Dict[str, Optional[int]]) -> Dict[str, Optional[float]]:
    """
    Translate BFI-10 raw responses to 5 trait scores.
    
    Reverse score items: 1, 3, 4, 5, 7
    Calculate: E, A, C, N, O trait scores
    """
    # Reverse score items
    item_1_rev = reverse_score(responses.get("BFI-10_1"))
    item_3_rev = reverse_score(responses.get("BFI-10_3"))
    item_4_rev = reverse_score(responses.get("BFI-10_4"))
    item_5_rev = reverse_score(responses.get("BFI-10_5"))
    item_7_rev = reverse_score(responses.get("BFI-10_7"))
    
    # Normal items
    item_2 = responses.get("BFI-10_2")
    item_6 = responses.get("BFI-10_6")
    item_8 = responses.get("BFI-10_8")
    item_9 = responses.get("BFI-10_9")
    item_10 = responses.get("BFI-10_10")
    
    # Calculate trait scores (average of 2 items each)
    # Skip if either item is None
    def safe_avg(val1, val2):
        if val1 is None or val2 is None:
            return None
        return (val1 + val2) / 2.0
    
    return {
        "Extraversion": safe_avg(item_1_rev, item_6),
        "Agreeableness": safe_avg(item_2, item_7_rev),
        "Conscientiousness": safe_avg(item_3_rev, item_8),
        "Neuroticism": safe_avg(item_4_rev, item_9),
        "Openness": safe_avg(item_5_rev, item_10)
    }


def translate_bes_a(responses: Dict[str, Optional[int]]) -> Dict[str, Optional[int]]:
    """
    Translate BES-A raw responses to Cognitive Empathy total score.
    
    Reverse score items: 2, 8, 9
    Sum all 9 items (after reverse scoring)
    """
    # Reverse score items
    item_2_rev = reverse_score(responses.get("BES-A_2"))
    item_8_rev = reverse_score(responses.get("BES-A_8"))
    item_9_rev = reverse_score(responses.get("BES-A_9"))
    
    # Normal items
    item_1 = responses.get("BES-A_1")
    item_3 = responses.get("BES-A_3")
    item_4 = responses.get("BES-A_4")
    item_5 = responses.get("BES-A_5")
    item_6 = responses.get("BES-A_6")
    item_7 = responses.get("BES-A_7")
    
    # Sum all items (skip None values)
    items = [item_1, item_2_rev, item_3, item_4, item_5, item_6, item_7, item_8_rev, item_9_rev]
    valid_items = [v for v in items if v is not None]
    
    if len(valid_items) == 0:
        ce_total = None
    else:
        ce_total = sum(valid_items)
        # If not all items present, we still calculate but note it's partial
        # (user said to skip null items, so partial scores are allowed)
    
    return {
        "CE_total": ce_total,
        "n_items": len(valid_items),  # Track how many items were used
        "CE_normalized": ce_total / len(valid_items) if len(valid_items) > 0 else None
    }


def translate_rei(responses: Dict[str, Optional[int]]) -> Dict[str, Optional[int]]:
    """
    Translate REI raw responses to Rational Ability total score.
    
    Reverse score items: 1, 2, 3, 4, 5
    Sum all 10 items (after reverse scoring)
    """
    # Reverse score items
    item_1_rev = reverse_score(responses.get("REI_1"))
    item_2_rev = reverse_score(responses.get("REI_2"))
    item_3_rev = reverse_score(responses.get("REI_3"))
    item_4_rev = reverse_score(responses.get("REI_4"))
    item_5_rev = reverse_score(responses.get("REI_5"))
    
    # Normal items
    item_6 = responses.get("REI_6")
    item_7 = responses.get("REI_7")
    item_8 = responses.get("REI_8")
    item_9 = responses.get("REI_9")
    item_10 = responses.get("REI_10")
    
    # Sum all items (skip None values)
    items = [item_1_rev, item_2_rev, item_3_rev, item_4_rev, item_5_rev,
             item_6, item_7, item_8, item_9, item_10]
    valid_items = [v for v in items if v is not None]
    
    if len(valid_items) == 0:
        ra_total = None
    else:
        ra_total = sum(valid_items)
    
    return {
        "RA_total": ra_total,
        "n_items": len(valid_items),  # Track how many items were used
        "RA_normalized": ra_total / len(valid_items) if len(valid_items) > 0 else None
    }


def translate_attempt(attempt: Dict[str, Any]) -> Dict[str, Any]:
    """Translate one attempt's raw responses to distilled scores."""
    distilled_attempt = {
        "attempt_id": attempt["attempt_id"],
        "timestamp": attempt["timestamp"]
    }
    
    distilled_sections = {}
    
    for section_id, section_data in attempt.get("sections", {}).items():
        raw_responses = section_data.get("responses", {})
        
        if section_id == "BFI-10":
            distilled_scores = translate_bfi10(raw_responses)
            distilled_sections[section_id] = {
                "trait_scores": distilled_scores
            }
        elif section_id == "BES-A":
            distilled_scores = translate_bes_a(raw_responses)
            distilled_sections[section_id] = {
                "total_score": distilled_scores["CE_total"],
                "n_items": distilled_scores["n_items"],
                "normalized_score": distilled_scores["CE_normalized"]
            }
        elif section_id == "REI":
            distilled_scores = translate_rei(raw_responses)
            distilled_sections[section_id] = {
                "total_score": distilled_scores["RA_total"],
                "n_items": distilled_scores["n_items"],
                "normalized_score": distilled_scores["RA_normalized"]
            }
        else:
            # Unknown section - skip or preserve as-is
            print(f"    Warning: Unknown section {section_id}, skipping")
            continue
    
    distilled_attempt["sections"] = distilled_sections
    distilled_attempt["completion_status"] = attempt.get("completion_status", "completed")
    
    return distilled_attempt


def translate_agent_responses(agent_folder: str) -> Optional[Dict[str, Any]]:
    """Translate survey responses for a single agent."""
    responses_path = Path(agent_folder) / "survey_responses.json"
    
    if not responses_path.exists():
        return None
    
    try:
        with open(responses_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"    Error loading {responses_path}: {e}")
        return None
    
    # Create distilled structure
    distilled_data = {
        "survey_metadata": raw_data.get("survey_metadata", {}),
        "agent_metadata": raw_data.get("agent_metadata", {}),
        "translation_metadata": {
            "translation_date": datetime.now().isoformat(),
            "translation_version": "1.0"
        },
        "attempts": []
    }
    
    # Translate each attempt
    for attempt in raw_data.get("attempts", []):
        distilled_attempt = translate_attempt(attempt)
        distilled_data["attempts"].append(distilled_attempt)
    
    return distilled_data


def save_distilled_responses(agent_folder: str, distilled_data: Dict[str, Any]):
    """Save distilled responses to agent folder."""
    output_path = Path(agent_folder) / "survey_distilled_responses.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(distilled_data, f, indent=2, ensure_ascii=False)
    
    return output_path


def translate_batch(base_path: str, agent_ids: List[str]) -> Dict[str, Any]:
    """Translate responses for multiple agents."""
    results = {
        "successful": [],
        "failed": [],
        "skipped": []
    }
    
    for agent_id in agent_ids:
        agent_folder = Path(base_path) / agent_id
        
        if not (agent_folder / "scratch.json").exists():
            print(f"Warning: Agent {agent_id} not found, skipping")
            results["skipped"].append(agent_id)
            continue
        
        if not (agent_folder / "survey_responses.json").exists():
            print(f"Warning: No survey_responses.json for agent {agent_id}, skipping")
            results["skipped"].append(agent_id)
            continue
        
        try:
            print(f"Translating agent {agent_id}...", flush=True)
            distilled_data = translate_agent_responses(str(agent_folder))
            
            if distilled_data is None:
                results["failed"].append(agent_id)
                continue
            
            output_path = save_distilled_responses(str(agent_folder), distilled_data)
            print(f"  Saved to: {output_path}", flush=True)
            
            results["successful"].append(agent_id)
            
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            results["failed"].append(agent_id)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Translate raw survey responses to distilled scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Translate single agent
  python3 translate_responses.py --agent agent_bank/populations/gss_agents/0000

  # Translate range of agents
  python3 translate_responses.py --base agent_bank/populations/gss_agents --range 0000-0005

  # Translate specific agents
  python3 translate_responses.py --base agent_bank/populations/gss_agents --agents 0000 0001 0002
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--agent',
        type=str,
        help='Path to a single agent folder'
    )
    input_group.add_argument(
        '--base',
        type=str,
        help='Base path to agent bank directory'
    )
    
    parser.add_argument(
        '--agents',
        nargs='+',
        help='Specific agent IDs to process (requires --base)'
    )
    
    parser.add_argument(
        '--range',
        type=str,
        help='Range of agent IDs to process (requires --base). Example: --range 0000-0005'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run: show what would be processed without actually translating'
    )
    
    args = parser.parse_args()
    
    # Determine agent folders
    if args.agent:
        agent_folders = [args.agent]
        agent_ids = [os.path.basename(args.agent)]
    elif args.base:
        if args.range:
            agent_ids = parse_range(args.range)
        elif args.agents:
            agent_ids = args.agents
        else:
            print("Error: --base requires either --agents or --range")
            sys.exit(1)
        agent_folders = [str(Path(args.base) / agent_id) for agent_id in agent_ids]
    else:
        print("Error: Must specify either --agent or --base")
        sys.exit(1)
    
    # Dry run
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN - No translations will be performed")
        print("=" * 60)
        print(f"Agents to process: {len(agent_folders)}")
        for folder in agent_folders:
            exists = Path(folder).exists() and (Path(folder) / "survey_responses.json").exists()
            print(f"  {'✓' if exists else '✗'} {folder}")
        return
    
    # Process agents
    print("=" * 60)
    print("SURVEY RESPONSE TRANSLATION")
    print("=" * 60)
    print(f"Agents to process: {len(agent_folders)}")
    print("=" * 60)
    print()
    
    if len(agent_folders) == 1:
        # Single agent
        agent_folder = agent_folders[0]
        try:
            distilled_data = translate_agent_responses(agent_folder)
            if distilled_data:
                output_path = save_distilled_responses(agent_folder, distilled_data)
                print(f"\nTranslation complete. Saved to: {output_path}")
            else:
                print(f"\nError: Could not translate agent responses")
                sys.exit(1)
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Batch processing
        base_path = args.base
        results = translate_batch(base_path, agent_ids)
        
        # Summary
        print("\n" + "=" * 60)
        print("TRANSLATION SUMMARY")
        print("=" * 60)
        print(f"Total agents: {len(agent_ids)}")
        print(f"Successful: {len(results['successful'])}")
        print(f"Failed: {len(results['failed'])}")
        print(f"Skipped: {len(results['skipped'])}")
        
        if results['failed']:
            print(f"\nFailed agents: {', '.join(results['failed'])}")
        if results['skipped']:
            print(f"Skipped agents: {', '.join(results['skipped'])}")
        
        print("=" * 60)


if __name__ == "__main__":
    main()

