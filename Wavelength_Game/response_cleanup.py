# Wavelength_Game/cleanup_failed_agents.py
#!/usr/bin/env python3
"""
Cleanup script to remove wavelength_responses.json files for agents that failed.
Use this before re-running with correct model configuration.
"""

import sys
from pathlib import Path
import json

def has_all_null_responses(agent_folder):
    """Check if agent has only null responses."""
    responses_path = Path(agent_folder) / "wavelength_responses.json"
    if not responses_path.exists():
        return False
    
    with open(responses_path, 'r') as f:
        data = json.load(f)
    
    if not data.get("attempts"):
        return False
    
    # Check all attempts
    for attempt in data["attempts"]:
        headers = attempt.get("headers", {})
        # If any attempt has valid responses, don't delete
        if any(h.get("response") is not None for h in headers.values()):
            return False
    
    return True

def main():
    base_path = Path("agent_bank/populations/gss_agents")
    
    # Agents 0100-0199
    agents_to_clean = []
    for agent_id in range(100, 200):
        agent_folder = base_path / f"{agent_id:04d}"
        if agent_folder.exists() and has_all_null_responses(agent_folder):
            agents_to_clean.append(agent_folder)
    
    if not agents_to_clean:
        print("No agents found with all null responses.")
        return
    
    print(f"Found {len(agents_to_clean)} agents with all null responses:")
    for folder in agents_to_clean:
        print(f"  - {folder}")
    
    response = input(f"\nDelete wavelength_responses.json for these {len(agents_to_clean)} agents? (yes/no): ")
    if response.lower() != "yes":
        print("Cancelled.")
        return
    
    for agent_folder in agents_to_clean:
        responses_path = agent_folder / "wavelength_responses.json"
        if responses_path.exists():
            responses_path.unlink()
            print(f"Deleted: {responses_path}")
    
    print(f"\nCleanup complete. {len(agents_to_clean)} files deleted.")

if __name__ == "__main__":
    main()