"""
Helpers shared by the CLI runners for selecting agent folders.

Deliberately does not import simulation_engine.settings: the patch scripts set
GENAGENTS_MODEL after parsing arguments, so settings must not be loaded yet.
"""

from pathlib import Path
from typing import List, Optional


def parse_range(range_str: str) -> List[str]:
  """
  Parse a range string like "0000-0049" into a list of zero-padded agent IDs.
  The padding width is taken from the start of the range.
  """
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


def get_agent_folders(base_path: str, agent_ids: Optional[List[str]] = None) -> List[str]:
  """
  Agent folders (those containing scratch.json) under base_path, sorted.
  If agent_ids is given, only those agents are returned; missing ones are
  reported and skipped.
  """
  base = Path(base_path)
  if not base.exists():
    raise ValueError(f"Base path does not exist: {base_path}")

  if agent_ids:
    folders = []
    for agent_id in agent_ids:
      agent_folder = base / agent_id
      if (agent_folder / "scratch.json").exists():
        folders.append(str(agent_folder))
      else:
        print(f"Warning: Agent {agent_id} not found at {agent_folder}")
    return folders

  return [str(agent_folder) for agent_folder in sorted(base.iterdir())
          if agent_folder.is_dir() and (agent_folder / "scratch.json").exists()]
