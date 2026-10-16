#!/usr/bin/env python3
"""
Run Wavelength game administration to collect agent responses.

This script administers the Wavelength Game to agents and stores responses
in structured JSON format under each agent's folder. Game interactions
are NOT recorded to memory stream.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Optional

# Add project root to path (genagents directory)
# Path(__file__) = Wavelength_Game/run_wavelength.py
# Path(__file__).parent = Wavelength_Game/
# Path(__file__).parent.parent = genagents/
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from simulation_engine.agent_selection import parse_range, get_agent_folders

from Wavelength_Game.wavelength_response_builder import WavelengthResponseBuilder


def main():
    parser = argparse.ArgumentParser(
        description="Run Wavelength game administration to collect agent responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run game on a single agent
  python3 ./Wavelength_Game/run_wavelength.py --agent agent_bank/populations/Qwen3-14B_agents/0000

  # Run on range of agents (0000 to 0049)
  python3 run_wavelength.py --base agent_bank/populations/Qwen3-14B_agents --range 0000-0049

  # Run on multiple specific agents
  python3 ./Wavelength_Game/run_wavelength.py --base agent_bank/populations/Qwen3-14B_agents --agents 0000 0001 0002

  # Run on all agents in directory
  python3 run_wavelength.py --base agent_bank/populations/Qwen3-14B_agents --all

  # New attempt for existing agent
  python3 run_wavelength.py --agent agent_bank/populations/Qwen3-14B_agents/0000 --new-attempt

  # Run without storing reasoning (faster, smaller files)
  python3 run_wavelength.py --agent agent_bank/populations/Qwen3-14B_agents/0000 --no-reasoning

  # Run with batch-by-header mode (all headers at once, faster)
  python3 run_wavelength.py --base agent_bank/populations/Qwen3-14B_agents --range 0000-0049 --batch-by-header
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
        help='Base path to agent bank directory (e.g., agent_bank/populations/Qwen3-14B_agents)'
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
    
    # Game options
    parser.add_argument(
        '--game-headers',
        type=str,
        default=None,
        help='Path to game headers JSON (default: Wavelength_Game/game_headers.json)'
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
        '--batch-by-header',
        action='store_true',
        help='Send all headers at once (faster but may truncate). Default: one header per inference (more reliable)'
    )
    
    # Execution options
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run: show what would be processed without actually running games'
    )
    
    args = parser.parse_args()
    
    # Determine game headers path
    if args.game_headers:
        game_headers_path = args.game_headers
    else:
        script_dir = Path(__file__).parent
        game_headers_path = script_dir / "game_headers.json"
    
    if not os.path.exists(game_headers_path):
        print(f"Error: Game headers file not found at {game_headers_path}")
        sys.exit(1)
    
    # Create builder
    config = {
        "include_reasoning": not args.no_reasoning,
        "require_all_headers": True,
        "batch_by_header": args.batch_by_header
    }
    builder = WavelengthResponseBuilder(str(game_headers_path), config)
    
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
        print("DRY RUN - No games will be administered")
        print("=" * 60)
        print(f"Game headers: {game_headers_path}")
        print(f"Total headers: {len(builder.game_headers)}")
        print(f"Agents to process: {len(agent_folders)}")
        print(f"Include reasoning: {config['include_reasoning']}")
        print(f"Batch by header: {config['batch_by_header']}")
        print("\nAgents:")
        for folder in agent_folders:
            print(f"  - {folder}")
        return
    
    # Process agents
    print("=" * 60)
    print("WAVELENGTH GAME ADMINISTRATION")
    print("=" * 60)
    print(f"Game headers: {game_headers_path}")
    print(f"Total headers: {len(builder.game_headers)}")
    print(f"Agents to process: {len(agent_folders)}")
    print(f"Include reasoning: {config['include_reasoning']}")
    print(f"Batch by header: {config['batch_by_header']}")
    print("=" * 60)
    
    if len(agent_folders) == 1:
        # Single agent processing
        agent_folder = agent_folders[0]
        
        try:
            result = builder.administer_game(
                agent_folder,
                attempt_id=args.attempt_id,
                include_reasoning=config["include_reasoning"]
            )
            
            print("\n" + "=" * 60)
            print("RESULTS:")
            print("=" * 60)
            print(f"Agent: {result['agent_folder']}")
            print(f"Attempt ID: {result['attempt_id']}")
            print(f"Headers completed: {result['headers_completed']}/{result['total_headers']}")
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
        results = builder.administer_game_batch(agent_folders, include_reasoning=config["include_reasoning"])
        
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
            avg_headers = sum(r.get('headers_completed', 0) for r in successful) / len(successful)
            avg_time = sum(r.get('agent_time_seconds', 0) for r in successful) / len(successful)
            print(f"\nAverage per agent:")
            print(f"  Headers completed: {avg_headers:.1f}")
            print(f"  Time: {avg_time:.2f}s ({avg_time/60:.2f} minutes)")
        
        if failed:
            print(f"\nFailed agents:")
            for r in failed:
                print(f"  - {r['agent_folder']}: {r.get('error', 'Unknown error')}")
        
        print("=" * 60)


if __name__ == "__main__":
    main()