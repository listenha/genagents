#!/usr/bin/env python3
"""
Generate agent profile summaries from scratch.json and memory_stream/nodes.json.

This script reads agent data, formats it into a prompt template, calls an LLM
to generate a profile summary, and appends it to the agent's scratch.json file.
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Add project root to path (genagents directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from simulation_engine.gpt_structure import gpt_request
from simulation_engine.settings import LLM_VERS


def setup_logging(log_file: Optional[str] = None):
    """Setup logging to both file and console."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode='a'))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )


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
                logging.warning(f"Agent {agent_id} not found at {agent_folder}")
        return folders
    else:
        # Process all agents
        folders = []
        for agent_folder in sorted(base.iterdir()):
            if agent_folder.is_dir() and (agent_folder / "scratch.json").exists():
                folders.append(str(agent_folder))
        return folders


def load_prompt_template(template_path: str) -> str:
    """Load the prompt template from file."""
    template_file = Path(template_path)
    if not template_file.exists():
        raise FileNotFoundError(f"Prompt template not found at {template_path}")
    
    with open(template_file, 'r', encoding='utf-8') as f:
        return f.read()


def format_prompt(template: str, scratch_json: dict, nodes_json: list) -> str:
    """
    Format the prompt template by replacing placeholders with actual data.
    
    Args:
        template: The prompt template string
        scratch_json: The scratch.json content as a dict
        nodes_json: The nodes.json content as a list
    
    Returns:
        Formatted prompt string
    """
    # Convert JSON to formatted string
    scratch_str = json.dumps(scratch_json, indent=2, ensure_ascii=False)
    nodes_str = json.dumps(nodes_json, indent=2, ensure_ascii=False)
    
    # Replace placeholders
    prompt = template.replace("{{PASTE SCRATCH.JSON CONTENT HERE}}", scratch_str)
    prompt = prompt.replace("{{PASTE NODES.JSON CONTENT HERE}}", nodes_str)
    
    return prompt


def has_existing_summary(scratch_path: str) -> bool:
    """
    Check if agent already has a non-empty profile summary.
    
    Args:
        scratch_path: Path to scratch.json
    
    Returns:
        True if summary exists and is not empty/null, False otherwise
    """
    try:
        with open(scratch_path, 'r', encoding='utf-8') as f:
            scratch = json.load(f)
        
        summary = scratch.get("agent_profile_summary")
        if summary is None:
            return False
        
        # Check if summary is empty string or whitespace
        if isinstance(summary, str) and summary.strip():
            return True
        
        return False
    except (json.JSONDecodeError, FileNotFoundError):
        return False


def append_summary_to_scratch(scratch_path: str, summary: str):
    """
    Append the profile summary to scratch.json as the last field.
    
    Args:
        scratch_path: Path to scratch.json
        summary: The generated profile summary text
    """
    # Load existing scratch.json
    with open(scratch_path, 'r', encoding='utf-8') as f:
        scratch = json.load(f)
    
    # Add the summary field
    scratch["agent_profile_summary"] = summary
    
    # Write back with proper formatting (indent=2 for readability)
    with open(scratch_path, 'w', encoding='utf-8') as f:
        json.dump(scratch, f, indent=2, ensure_ascii=False)
        # Add newline at end of file
        f.write('\n')


def generate_profile_summary(agent_folder: str, prompt_template: str, max_tokens: int = 500, force: bool = False) -> dict:
    """
    Generate profile summary for a single agent.
    
    Args:
        agent_folder: Path to agent folder
        prompt_template: The prompt template string
        max_tokens: Maximum tokens for LLM response
        force: If True, overwrite existing summary; if False, skip if summary exists
    
    Returns:
        Dictionary with status, summary, and error (if any)
    """
    agent_path = Path(agent_folder)
    scratch_path = agent_path / "scratch.json"
    nodes_path = agent_path / "memory_stream" / "nodes.json"
    
    try:
        # Check if summary already exists (unless force is True)
        if not force and has_existing_summary(str(scratch_path)):
            logging.info(f"Skipping {agent_path.name}: summary already exists (use --force to overwrite)")
            return {
                'status': 'skipped',
                'agent_folder': agent_folder,
                'summary': None,
                'error': None
            }
        
        # If force is True and summary exists, log that we're overwriting
        if force and has_existing_summary(str(scratch_path)):
            logging.info(f"Overwriting existing summary for {agent_path.name} (--force flag set)")
        
        # Load scratch.json
        if not scratch_path.exists():
            raise FileNotFoundError(f"scratch.json not found at {scratch_path}")
        
        with open(scratch_path, 'r', encoding='utf-8') as f:
            scratch_data = json.load(f)
        
        # Load nodes.json
        if not nodes_path.exists():
            raise FileNotFoundError(f"nodes.json not found at {nodes_path}")
        
        with open(nodes_path, 'r', encoding='utf-8') as f:
            nodes_data = json.load(f)
        
        # Format prompt
        formatted_prompt = format_prompt(prompt_template, scratch_data, nodes_data)
        
        # Call LLM
        logging.info(f"Generating summary for {agent_path.name}...")
        response = gpt_request(formatted_prompt, model=LLM_VERS, max_tokens=max_tokens)
        
        # Check for errors
        if response.startswith("GENERATION ERROR"):
            raise Exception(f"LLM generation error: {response}")
        
        # Clean up response (remove any markdown formatting if present)
        summary = response.strip()
        if summary.startswith("```"):
            # Remove markdown code blocks if present
            lines = summary.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            summary = '\n'.join(lines).strip()
        
        # Append to scratch.json
        append_summary_to_scratch(str(scratch_path), summary)
        
        logging.info(f"✓ Successfully generated summary for {agent_path.name}")
        
        return {
            'status': 'success',
            'agent_folder': agent_folder,
            'summary': summary,
            'error': None
        }
        
    except Exception as e:
        error_msg = str(e)
        logging.error(f"✗ Error processing {agent_path.name}: {error_msg}")
        return {
            'status': 'error',
            'agent_folder': agent_folder,
            'summary': None,
            'error': error_msg
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate agent profile summaries from scratch.json and memory_stream/nodes.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate summary for a single agent
  python3 run_profile_summary.py --agent agent_bank/populations/gss_agents/0000

  # Generate summaries for range of agents (0000 to 0049)
  python3 run_profile_summary.py --base agent_bank/populations/gss_agents --range 0000-0049

  # Generate summaries for multiple specific agents
  python3 run_profile_summary.py --base agent_bank/populations/gss_agents --agents 0000 0001 0002

  # Generate summaries for all agents in directory
  python3 run_profile_summary.py --base agent_bank/populations/gss_agents --all

  # Use custom prompt template
  python3 run_profile_summary.py --base agent_bank/populations/gss_agents --range 0000-0049 --template custom_prompt.txt

  # Dry run (show what would be processed)
  python3 run_profile_summary.py --base agent_bank/populations/gss_agents --range 0000-0049 --dry-run

  # Overwrite existing summaries
  python3 run_profile_summary.py --base agent_bank/populations/gss_agents --range 0000-0049 --force
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
        '--template',
        type=str,
        default=None,
        help='Path to prompt template file (default: Agent_Profile_Summary/prompt.txt)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=500,
        help='Maximum tokens for LLM response (default: 500)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file (default: logs/profile_summary_YYYYMMDD_HHMMSS.log)'
    )
    
    # Execution options
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing summaries instead of skipping them'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run: show what would be processed without actually generating summaries'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.log_file:
        log_file = args.log_file
    else:
        # Default log file with timestamp
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = str(log_dir / f"profile_summary_{timestamp}.log")
    
    setup_logging(log_file)
    
    # Determine prompt template path
    if args.template:
        template_path = args.template
    else:
        script_dir = Path(__file__).parent
        template_path = str(script_dir / "prompt.txt")
    
    if not os.path.exists(template_path):
        logging.error(f"Error: Prompt template not found at {template_path}")
        sys.exit(1)
    
    # Load prompt template
    try:
        prompt_template = load_prompt_template(template_path)
    except Exception as e:
        logging.error(f"Error loading prompt template: {e}")
        sys.exit(1)
    
    # Determine agent folders to process
    if args.agent:
        # Single agent
        agent_folder = Path(args.agent)
        if not (agent_folder / "scratch.json").exists():
            logging.error(f"Error: Agent not found at {agent_folder}")
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
                logging.error(f"Error: {e}")
                sys.exit(1)
        elif args.agents:
            agent_folders = get_agent_folders(args.base, args.agents)
        elif args.all:
            agent_folders = get_agent_folders(args.base)
        else:
            logging.error("Error: --base requires either --agents, --range, or --all")
            sys.exit(1)
    else:
        logging.error("Error: Must specify either --agent or --base")
        sys.exit(1)
    
    if not agent_folders:
        logging.error("Error: No valid agent folders found")
        sys.exit(1)
    
    # Dry run
    if args.dry_run:
        logging.info("=" * 60)
        logging.info("DRY RUN - No summaries will be generated")
        logging.info("=" * 60)
        logging.info(f"Prompt template: {template_path}")
        logging.info(f"Agents to process: {len(agent_folders)}")
        logging.info(f"Max tokens: {args.max_tokens}")
        logging.info("\nAgents:")
        for folder in agent_folders:
            agent_path = Path(folder)
            scratch_path = agent_path / "scratch.json"
            has_summary = has_existing_summary(str(scratch_path))
            if has_summary and not args.force:
                status = "✓ (has summary, will skip)"
            elif has_summary and args.force:
                status = "⚠ (has summary, will overwrite)"
            else:
                status = "→ (will generate)"
            logging.info(f"  {status} {folder}")
        return
    
    # Process agents
    logging.info("=" * 60)
    logging.info("AGENT PROFILE SUMMARY GENERATOR")
    logging.info("=" * 60)
    logging.info(f"Prompt template: {template_path}")
    logging.info(f"Agents to process: {len(agent_folders)}")
    logging.info(f"Max tokens: {args.max_tokens}")
    if args.force:
        logging.info("Force mode: Will overwrite existing summaries")
    logging.info(f"Log file: {log_file}")
    logging.info("=" * 60)
    
    if len(agent_folders) == 1:
        # Single agent processing
        agent_folder = agent_folders[0]
        
        try:
            result = generate_profile_summary(agent_folder, prompt_template, args.max_tokens, args.force)
            
            logging.info("\n" + "=" * 60)
            logging.info("RESULTS:")
            logging.info("=" * 60)
            logging.info(f"Agent: {result['agent_folder']}")
            logging.info(f"Status: {result['status']}")
            if result['status'] == 'success':
                logging.info(f"Summary length: {len(result['summary'])} characters")
            elif result['status'] == 'error':
                logging.info(f"Error: {result['error']}")
            logging.info("=" * 60)
            
        except Exception as e:
            logging.error(f"\nERROR: {e}")
            import traceback
            logging.error(traceback.format_exc())
            sys.exit(1)
    else:
        # Batch processing
        logging.info(f"\nStarting batch processing of {len(agent_folders)} agents...\n")
        
        results = []
        for agent_folder in agent_folders:
            result = generate_profile_summary(agent_folder, prompt_template, args.max_tokens, args.force)
            results.append(result)
        
        # Summary
        logging.info("\n" + "=" * 60)
        logging.info("BATCH PROCESSING SUMMARY")
        logging.info("=" * 60)
        
        successful = [r for r in results if r.get('status') == 'success']
        skipped = [r for r in results if r.get('status') == 'skipped']
        failed = [r for r in results if r.get('status') == 'error']
        
        logging.info(f"Total agents: {len(results)}")
        logging.info(f"Successful: {len(successful)}")
        logging.info(f"Skipped (already have summary): {len(skipped)}")
        logging.info(f"Failed: {len(failed)}")
        
        if successful:
            avg_length = sum(len(r['summary']) for r in successful if r['summary']) / len(successful)
            logging.info(f"\nAverage summary length: {avg_length:.0f} characters")
        
        if failed:
            logging.info(f"\nFailed agents:")
            for r in failed:
                logging.info(f"  - {Path(r['agent_folder']).name}: {r.get('error', 'Unknown error')}")
        
        logging.info("=" * 60)
        logging.info(f"Full log saved to: {log_file}")


if __name__ == "__main__":
    main()

