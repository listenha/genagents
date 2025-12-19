#!/usr/bin/env python3
"""
Run complete interview process to populate agent memory streams.

This script conducts semi-structured interviews with agents starting from their
demographic profiles, following the American Voices Project interview protocol.
Each question-answer pair is saved as an observation memory, and reflections
are triggered based on configurable strategies.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Optional

# Enable unbuffered output for real-time updates (use python3 -u flag or set PYTHONUNBUFFERED=1)

# Add project root to path (genagents directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environment.interview.interview_memory_builder import InterviewMemoryBuilder


def load_config(config_path: Optional[str]) -> dict:
    """Load configuration from JSON file or return None for defaults."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


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
        description="Run interview process to populate agent memory streams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full interview on a single agent
  python3 run_interview.py --agent agent_bank/populations/gss_agents/0000

  # Run on range of agents (0000 to 0049)
  python3 run_interview.py --base agent_bank/populations/gss_agents --range 0000-0049

  # Run on multiple specific agents
  python3 run_interview.py --base agent_bank/populations/gss_agents --agents 0000 0001 0002

  # Run on all agents in directory
  python3 run_interview.py --base agent_bank/populations/gss_agents --all

  # Test with first 10 questions only
  python3 run_interview.py --agent agent_bank/populations/gss_agents/0000 --limit 10

  # Use custom configuration
  python3 run_interview.py --agent agent_bank/populations/gss_agents/0000 --config my_config.json
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
    
    # Configuration options
    parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON configuration file (optional, uses defaults if not provided)'
    )
    parser.add_argument(
        '--script',
        type=str,
        default=None,
        help='Path to interview script JSON (default: agent_bank/scripts/interview_script.json)'
    )
    
    # Execution options
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of questions (for testing). Example: --limit 10'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for processed agents (default: saves to agent folder)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run: show what would be processed without actually running interviews'
    )
    
    args = parser.parse_args()
    
    # Determine interview script path
    if args.script:
        interview_script_path = args.script
    else:
        script_dir = Path(__file__).parent
        interview_script_path = script_dir / "interview_script.json"
    
    if not os.path.exists(interview_script_path):
        print(f"Error: Interview script not found at {interview_script_path}")
        print("Please run extract_interview_script.py first")
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    
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
    
    # Create builder
    builder = InterviewMemoryBuilder(str(interview_script_path), config)
    
    # Apply question limit if specified
    if args.limit:
        print(f"⚠ Limiting interview to first {args.limit} questions (for testing)")
        builder.interview_script = builder.interview_script[:args.limit]
    
    # Dry run
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN - No interviews will be conducted")
        print("=" * 60)
        print(f"Interview script: {interview_script_path}")
        print(f"Total questions: {len(builder.interview_script)}")
        print(f"Agents to process: {len(agent_folders)}")
        print("\nAgents:")
        for folder in agent_folders:
            print(f"  - {folder}")
        print("\nConfiguration:")
        print(json.dumps(builder.config, indent=2))
        return
    
    # Process agents
    print("=" * 60)
    print("INTERVIEW MEMORY BUILDER")
    print("=" * 60)
    print(f"Interview script: {interview_script_path}")
    print(f"Total questions: {len(builder.interview_script)}")
    print(f"Agents to process: {len(agent_folders)}")
    print("=" * 60)
    
    if len(agent_folders) == 1:
        # Single agent processing
        agent_folder = agent_folders[0]
        output_folder = args.output if args.output else None
        
        try:
            result = builder.build_memory(agent_folder, output_folder)
            
            print("\n" + "=" * 60)
            print("RESULTS:")
            print("=" * 60)
            print(f"Agent: {result['agent_folder']}")
            print(f"Output: {result['output_folder']}")
            print(f"Questions answered: {result['questions_answered']}")
            print(f"Memories created: {result['memories_created']}")
            print(f"Reflections triggered: {result['reflections_triggered']}")
            print("=" * 60)
            
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Batch processing
        output_base = args.output if args.output else None
        
        print(f"\nStarting batch processing of {len(agent_folders)} agents...\n")
        results = builder.build_memory_batch(agent_folders, output_base)
        
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
            avg_questions = sum(r['questions_answered'] for r in successful) / len(successful)
            avg_memories = sum(r['memories_created'] for r in successful) / len(successful)
            avg_reflections = sum(r['reflections_triggered'] for r in successful) / len(successful)
            
            print(f"\nAverage per agent:")
            print(f"  Questions answered: {avg_questions:.1f}")
            print(f"  Memories created: {avg_memories:.1f}")
            print(f"  Reflections triggered: {avg_reflections:.1f}")
        
        if failed:
            print(f"\nFailed agents:")
            for r in failed:
                print(f"  - {r['agent_folder']}: {r.get('error', 'Unknown error')}")
        
        print("=" * 60)


if __name__ == "__main__":
    main()

