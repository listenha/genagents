#!/usr/bin/env python3
"""
Extract demographic information from all agent scratch.json files to CSV.

This script scans all agent folders in agent_bank/populations/gss_agents/
and extracts demographic information from each agent's scratch.json file,
converting it to a CSV file for easy filtering and statistical analysis.
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Any


def get_agent_folders(base_path: str) -> List[str]:
    """Get all agent folder paths."""
    agent_folders = []
    base = Path(base_path)
    
    if not base.exists():
        print(f"Error: Base path does not exist: {base_path}")
        return []
    
    # Get all subdirectories (agent folders)
    for item in base.iterdir():
        if item.is_dir():
            agent_folders.append(str(item))
    
    # Sort by folder name for consistent ordering
    agent_folders.sort()
    return agent_folders


def read_scratch_json(agent_folder: str) -> Dict[str, Any]:
    """Read scratch.json from an agent folder."""
    scratch_path = Path(agent_folder) / "scratch.json"
    
    if not scratch_path.exists():
        print(f"Warning: scratch.json not found in {agent_folder}")
        return {}
    
    try:
        with open(scratch_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON in {scratch_path}: {e}")
        return {}
    except Exception as e:
        print(f"Error: Failed to read {scratch_path}: {e}")
        return {}


def extract_all_demographics(agent_folders: List[str]) -> List[Dict[str, Any]]:
    """Extract demographic data from all agents."""
    all_data = []
    
    print(f"Processing {len(agent_folders)} agent folders...")
    
    for i, agent_folder in enumerate(agent_folders):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(agent_folders)} agents...")
        
        # Get agent ID from folder name
        agent_id = Path(agent_folder).name
        
        # Read scratch.json
        scratch_data = read_scratch_json(agent_folder)
        
        if scratch_data:
            # Add agent_id to the data
            scratch_data['agent_id'] = agent_id
            all_data.append(scratch_data)
    
    print(f"Successfully extracted data from {len(all_data)} agents.")
    return all_data


def get_all_field_names(data_list: List[Dict[str, Any]]) -> List[str]:
    """Get all unique field names from all agent data."""
    field_names = set()
    
    for data in data_list:
        field_names.update(data.keys())
    
    # Sort fields, but put agent_id first
    sorted_fields = ['agent_id'] + sorted([f for f in field_names if f != 'agent_id'])
    return sorted_fields


def write_to_csv(data_list: List[Dict[str, Any]], output_path: str):
    """Write demographic data to CSV file."""
    if not data_list:
        print("No data to write.")
        return
    
    # Get all field names
    fieldnames = get_all_field_names(data_list)
    
    print(f"Writing {len(data_list)} records to {output_path}...")
    print(f"Columns: {len(fieldnames)} fields")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        
        for data in data_list:
            # Ensure all fields are present (fill missing with empty string)
            row = {field: data.get(field, '') for field in fieldnames}
            writer.writerow(row)
    
    print(f"✓ CSV file created: {output_path}")


def main():
    """Main function."""
    # Get the base directory of this script
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # Define paths
    agents_base_path = project_root / "agent_bank" / "populations" / "gss_agents"
    output_csv = project_root / "agent_bank" / "gss_agents_demographics.csv"
    
    print("=" * 60)
    print("Agent Demographics Extractor")
    print("=" * 60)
    print(f"Agents directory: {agents_base_path}")
    print(f"Output CSV: {output_csv}")
    print()
    
    # Get all agent folders
    agent_folders = get_agent_folders(str(agents_base_path))
    
    if not agent_folders:
        print("No agent folders found. Exiting.")
        return
    
    # Extract demographic data
    all_data = extract_all_demographics(agent_folders)
    
    if not all_data:
        print("No data extracted. Exiting.")
        return
    
    # Write to CSV
    write_to_csv(all_data, str(output_csv))
    
    print()
    print("=" * 60)
    print("Extraction complete!")
    print(f"Total agents processed: {len(all_data)}")
    print(f"Output file: {output_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()

