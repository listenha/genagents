#!/usr/bin/env python3
"""
Run survey administration to collect agent responses.

This script administers the PRE-TASK SURVEY to agents and stores responses
in structured JSON format under each agent's folder. Survey interactions
are NOT recorded to memory stream.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Optional

# Add project root to path (genagents directory)
# Path(__file__) = Surveys/run_survey.py
# Path(__file__).parent = Surveys/
# Path(__file__).parent.parent = genagents/
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from Surveys.survey_response_builder import SurveyResponseBuilder


def parse_range(range_str: str) -> List[str]:
    """
    Parse a range string like "0000-0049" into a list of agent IDs.
    
    Args:
        range_str: Range string in format "START-END" (e.g., "0000-0049")
    
    Returns:
        List of zero-padded agent IDs
    """
    try:
        start_str, end_str = range_str.split('-')
        start = int(start_str)
        end = int(end_str)
        
        if start > end:
            raise ValueError(f"Start ({start}) must be <= end ({end})")
        
        # Determine padding length from the input strings
        padding = len(start_str)
        
        return [f"{i:0{padding}d}" for i in range(start, end + 1)]
    except ValueError as e:
        raise ValueError(f"Invalid range format '{range_str}'. Expected format: '0000-0049'. Error: {e}")


def get_agent_folders(base_path: str, agent_ids: Optional[List[str]] = None) -> List[str]:
    """
    Get list of agent folders to process.
    
    Args:
        base_path: Base path to agent bank (e.g., gss_agents/)
        agent_ids: Optional list of specific agent IDs to process. If None, processes all.
    
    Returns:
        List of agent folder paths
    """
    base = Path(base_path)
    if not base.exists():
        raise ValueError(f"Base path does not exist: {base_path}")
    
    if agent_ids:
        # Process specific agents
        folders = []
        for agent_id in agent_ids:
            agent_folder = base / agent_id
            if (agent_folder / "scratch.json").exists():
                folders.append(str(agent_folder))
            else:
                print(f"Warning: Agent {agent_id} not found at {agent_folder}")
        return folders
    else:
        # Process all agents
        folders = []
        for agent_folder in sorted(base.iterdir()):
            if agent_folder.is_dir() and (agent_folder / "scratch.json").exists():
                folders.append(str(agent_folder))
        return folders


def main():
    parser = argparse.ArgumentParser(
        description="Run survey administration to collect agent responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run survey on a single agent
  python3 run_survey.py --agent agent_bank/populations/gss_agents/0000

  # Run on range of agents (0000 to 0049)
  python3 run_survey.py --base agent_bank/populations/gss_agents --range 0000-0049

  # Run on multiple specific agents
  python3 run_survey.py --base agent_bank/populations/gss_agents --agents 0000 0001 0002

  # Run on all agents in directory
  python3 run_survey.py --base agent_bank/populations/gss_agents --all

  # New attempt for existing agent
  python3 run_survey.py --agent agent_bank/populations/gss_agents/0000 --new-attempt

  # Run without storing reasoning (faster, smaller files)
  python3 run_survey.py --agent agent_bank/populations/gss_agents/0000 --no-reasoning

  # Run with batch-by-section mode (all questions in section at once, faster)
  python3 run_survey.py --base agent_bank/populations/gss_agents --range 0000-0049 --batch-by-section
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--agent',
        type=str,
        help='Path to a single agent folder (must contain scratch.json)'
    )
    input_group.add_argument(
        '--base',
        type=str,
        help='Base path to agent bank directory (e.g., agent_bank/populations/gss_agents)'
    )
    
    parser.add_argument(
        '--agents',
        nargs='+',
        help='Specific agent IDs to process (requires --base). Example: --agents 0000 0001 0002'
    )
    parser.add_argument(
        '--range',
        type=str,
        help='Range of agent IDs to process (requires --base). Example: --range 0000-0049'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all agents in base directory (requires --base)'
    )
    
    # Survey options
    parser.add_argument(
        '--survey',
        type=str,
        default=None,
        help='Path to survey questions JSON (default: Surveys/survey_questions.json)'
    )
    parser.add_argument(
        '--sections',
        nargs='+',
        help='Specific sections to administer (default: all sections). Example: --sections BFI-10 REI'
    )
    parser.add_argument(
        '--new-attempt',
        action='store_true',
        help='Create a new attempt for agent(s) (auto-increments attempt_id)'
    )
    parser.add_argument(
        '--attempt-id',
        type=int,
        help='Specific attempt ID to use (overrides auto-increment)'
    )
    parser.add_argument(
        '--no-reasoning',
        action='store_true',
        help='Do not store reasoning text (faster, smaller files)'
    )
    parser.add_argument(
        '--batch-by-section',
        action='store_true',
        help='Send all questions in a section at once (faster but may truncate). Default: one question per inference (more reliable)'
    )
    
    # Execution options
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run: show what would be processed without actually running surveys'
    )
    
    args = parser.parse_args()
    
    # Determine survey questions path
    if args.survey:
        survey_questions_path = args.survey
    else:
        script_dir = Path(__file__).parent
        survey_questions_path = script_dir / "survey_questions.json"
    
    if not os.path.exists(survey_questions_path):
        print(f"Error: Survey questions file not found at {survey_questions_path}")
        print("Please run extract_survey_questions.py first")
        sys.exit(1)
    
    # Create builder
    config = {
        "include_reasoning": not args.no_reasoning,
        "require_all_sections": True,
        "batch_by_section": args.batch_by_section
    }
    builder = SurveyResponseBuilder(str(survey_questions_path), config)
    
    # Determine agent folders to process
    if args.agent:
        # Single agent
        agent_folder = Path(args.agent)
        if not (agent_folder / "scratch.json").exists():
            print(f"Error: Agent not found at {agent_folder}")
            sys.exit(1)
        agent_folders = [str(agent_folder)]
    elif args.base:
        # Multiple agents
        if args.range:
            # Parse range (e.g., "0000-0049")
            try:
                agent_ids = parse_range(args.range)
                agent_folders = get_agent_folders(args.base, agent_ids)
            except ValueError as e:
                print(f"Error: {e}")
                sys.exit(1)
        elif args.agents:
            agent_folders = get_agent_folders(args.base, args.agents)
        elif args.all:
            agent_folders = get_agent_folders(args.base)
        else:
            print("Error: --base requires either --agents, --range, or --all")
            sys.exit(1)
    else:
        print("Error: Must specify either --agent or --base")
        sys.exit(1)
    
    if not agent_folders:
        print("Error: No valid agent folders found")
        sys.exit(1)
    
    # Dry run
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN - No surveys will be administered")
        print("=" * 60)
        print(f"Survey questions: {survey_questions_path}")
        print(f"Total sections: {len(builder.survey_data['sections'])}")
        print(f"Agents to process: {len(agent_folders)}")
        if args.sections:
            print(f"Sections to administer: {args.sections}")
        else:
            print(f"Sections to administer: ALL")
        print(f"Include reasoning: {config['include_reasoning']}")
        print(f"Batch by section: {config['batch_by_section']}")
        print("\nAgents:")
        for folder in agent_folders:
            print(f"  - {folder}")
        return
    
    # Process agents
    print("=" * 60)
    print("SURVEY ADMINISTRATION")
    print("=" * 60)
    print(f"Survey questions: {survey_questions_path}")
    print(f"Total sections: {len(builder.survey_data['sections'])}")
    print(f"Agents to process: {len(agent_folders)}")
    if args.sections:
        print(f"Sections to administer: {args.sections}")
    else:
        print(f"Sections to administer: ALL")
    print(f"Include reasoning: {config['include_reasoning']}")
    print("=" * 60)
    
    if len(agent_folders) == 1:
        # Single agent processing
        agent_folder = agent_folders[0]
        
        try:
            result = builder.administer_survey(
                agent_folder,
                attempt_id=args.attempt_id,
                sections=args.sections,
                include_reasoning=config["include_reasoning"]
            )
            
            print("\n" + "=" * 60)
            print("RESULTS:")
            print("=" * 60)
            print(f"Agent: {result['agent_folder']}")
            print(f"Attempt ID: {result['attempt_id']}")
            print(f"Sections completed: {', '.join(result['sections_completed'])}")
            print(f"Total questions: {result['total_questions']}")
            if 'total_time_seconds' in result:
                print(f"Total time: {result['total_time_seconds']:.2f}s ({result['total_time_seconds']/60:.2f} minutes)")
            print(f"Status: {result['status']}")
            print("=" * 60)
            
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Batch processing
        print(f"\nStarting batch processing of {len(agent_folders)} agents...\n")
        results = builder.administer_survey_batch(agent_folders, include_reasoning=config["include_reasoning"])
        
        # Summary
        print("\n" + "=" * 60)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 60)
        
        successful = [r for r in results if r.get('status') == 'success']
        failed = [r for r in results if r.get('status') == 'error']
        
        print(f"Total agents: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            avg_questions = sum(r.get('total_questions', 0) for r in successful) / len(successful)
            avg_time = sum(r.get('agent_time_seconds', 0) for r in successful) / len(successful)
            print(f"\nAverage per agent:")
            print(f"  Questions: {avg_questions:.1f}")
            print(f"  Time: {avg_time:.2f}s ({avg_time/60:.2f} minutes)")
        
        if failed:
            print(f"\nFailed agents:")
            for r in failed:
                print(f"  - {r['agent_folder']}: {r.get('error', 'Unknown error')}")
        
        print("=" * 60)


if __name__ == "__main__":
    main()

