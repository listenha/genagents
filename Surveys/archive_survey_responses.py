#!/usr/bin/env python3
"""
Archive existing survey response files before generating new ones.

This script creates an "archive" subfolder in each agent folder and moves
survey_responses.json and survey_distilled_responses.json into it.
"""

import sys
import os
import shutil
import argparse
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Add project root to path (genagents directory)
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))


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
        base_path: Base path to agent bank (e.g., agent_bank/populations/gss_agents)
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
            if agent_folder.exists() and agent_folder.is_dir():
                folders.append(str(agent_folder))
            else:
                print(f"Warning: Agent folder {agent_id} not found at {agent_folder}")
        return folders
    else:
        # Process all agents
        folders = []
        for agent_folder in sorted(base.iterdir()):
            if agent_folder.is_dir():
                folders.append(str(agent_folder))
        return folders


def archive_agent_responses(agent_folder: str, dry_run: bool = False) -> dict:
    """
    Archive survey response files for a single agent.
    
    Args:
        agent_folder: Path to agent folder
        dry_run: If True, only show what would be done without actually moving files
    
    Returns:
        Dictionary with archive results
    """
    agent_path = Path(agent_folder)
    agent_id = agent_path.name
    
    files_to_archive = [
        "survey_responses.json",
        "survey_distilled_responses.json"
    ]
    
    archive_folder = agent_path / "archive"
    results = {
        "agent_id": agent_id,
        "agent_folder": str(agent_folder),
        "files_found": [],
        "files_moved": [],
        "errors": []
    }
    
    # Check which files exist
    existing_files = []
    for filename in files_to_archive:
        file_path = agent_path / filename
        if file_path.exists():
            existing_files.append(filename)
            results["files_found"].append(filename)
    
    if not existing_files:
        return results
    
    if dry_run:
        print(f"  [DRY RUN] Would archive {len(existing_files)} file(s) for agent {agent_id}")
        for filename in existing_files:
            print(f"    - {filename}")
        results["files_moved"] = existing_files
        return results
    
    # Create archive folder
    try:
        archive_folder.mkdir(exist_ok=True)
    except Exception as e:
        results["errors"].append(f"Failed to create archive folder: {e}")
        return results
    
    # Move files to archive
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for filename in existing_files:
        source_path = agent_path / filename
        # Add timestamp to archived filename to avoid conflicts if archiving multiple times
        archived_filename = f"{timestamp}_{filename}"
        dest_path = archive_folder / archived_filename
        
        try:
            shutil.move(str(source_path), str(dest_path))
            results["files_moved"].append(filename)
            print(f"    Moved: {filename} -> archive/{archived_filename}")
        except Exception as e:
            error_msg = f"Failed to move {filename}: {e}"
            results["errors"].append(error_msg)
            print(f"    ERROR: {error_msg}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Archive existing survey response files before generating new ones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Archive files for a single agent
  python3 archive_survey_responses.py --agent agent_bank/populations/gss_agents/0000

  # Archive files for range of agents (0000 to 0049)
  python3 archive_survey_responses.py --base agent_bank/populations/gss_agents --range 0000-0049

  # Archive files for multiple specific agents
  python3 archive_survey_responses.py --base agent_bank/populations/gss_agents --agents 0000 0001 0002

  # Archive files for all agents in directory
  python3 archive_survey_responses.py --base agent_bank/populations/gss_agents --all

  # Dry run (show what would be archived without actually moving files)
  python3 archive_survey_responses.py --base agent_bank/populations/gss_agents --all --dry-run
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
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run: show what would be archived without actually moving files'
    )
    
    args = parser.parse_args()
    
    # Determine agent folders to process
    if args.agent:
        # Single agent
        agent_folder = Path(args.agent)
        if not agent_folder.exists():
            print(f"Error: Agent folder not found at {agent_folder}")
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
    
    # Print summary
    print("=" * 60)
    print("ARCHIVE SURVEY RESPONSES")
    print("=" * 60)
    print(f"Total agents: {len(agent_folders)}")
    if args.dry_run:
        print("Mode: DRY RUN (no files will be moved)")
    else:
        print("Mode: ARCHIVE (files will be moved to archive/)")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    
    # Process agents
    all_results = []
    total_files_found = 0
    total_files_moved = 0
    total_errors = 0
    
    for idx, agent_folder in enumerate(agent_folders):
        agent_id = Path(agent_folder).name
        print(f"[{idx + 1}/{len(agent_folders)}] Processing agent {agent_id}...")
        
        result = archive_agent_responses(agent_folder, dry_run=args.dry_run)
        all_results.append(result)
        
        total_files_found += len(result["files_found"])
        total_files_moved += len(result["files_moved"])
        total_errors += len(result["errors"])
        
        if not result["files_found"]:
            print(f"  No survey response files found")
        print()
    
    # Print summary
    print("=" * 60)
    print("ARCHIVE SUMMARY")
    print("=" * 60)
    print(f"Total agents processed: {len(agent_folders)}")
    print(f"Total files found: {total_files_found}")
    if not args.dry_run:
        print(f"Total files moved: {total_files_moved}")
        print(f"Total errors: {total_errors}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Show agents with errors
    agents_with_errors = [r for r in all_results if r["errors"]]
    if agents_with_errors:
        print("\nAgents with errors:")
        for result in agents_with_errors:
            print(f"  {result['agent_id']}:")
            for error in result["errors"]:
                print(f"    - {error}")
    
    # Show agents with no files
    agents_without_files = [r for r in all_results if not r["files_found"]]
    if agents_without_files and len(agents_without_files) <= 10:
        print(f"\nAgents with no survey response files ({len(agents_without_files)}):")
        for result in agents_without_files:
            print(f"  {result['agent_id']}")
    elif agents_without_files:
        print(f"\n{len(agents_without_files)} agents had no survey response files to archive")


if __name__ == "__main__":
    main()

