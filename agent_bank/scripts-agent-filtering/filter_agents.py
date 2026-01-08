#!/usr/bin/env python3
"""
Filter and manage agent bank based on demographic criteria.

This script allows filtering agents by demographic criteria, with three modes:
1. statistics: Show statistics about matching agents
2. edit_csv: Create a filtered CSV file
3. filter_agent_bank: Move non-matching agents to backup folder
"""

import json
import csv
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from datetime import datetime
import re


class FilterTypes:
    """Filter type implementations."""
    
    @staticmethod
    def numeric_range(value: Any, config: Dict[str, Any]) -> bool:
        """Check if numeric value is within range."""
        try:
            num_value = float(value) if value else None
            if num_value is None:
                return False
            
            min_val = config.get('min')
            max_val = config.get('max')
            
            if min_val is not None and num_value < min_val:
                return False
            if max_val is not None and num_value > max_val:
                return False
            
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def categorical(value: Any, config: Dict[str, Any]) -> bool:
        """Check if value matches categorical criteria."""
        value_str = str(value).strip() if value else ""
        
        include = config.get('include', [])
        exclude = config.get('exclude', [])
        
        # If include list exists, value must be in it
        if include and value_str not in include:
            return False
        
        # If exclude list exists, value must not be in it
        if exclude and value_str in exclude:
            return False
        
        return True
    
    @staticmethod
    def regex(value: Any, config: Dict[str, Any]) -> bool:
        """Check if value matches regex pattern."""
        pattern = config.get('pattern', '')
        if not pattern:
            return True
        
        value_str = str(value) if value else ""
        try:
            return bool(re.match(pattern, value_str))
        except re.error:
            return False


class AgentFilter:
    """Main filter class for agent demographics."""
    
    def __init__(self, csv_path: str, config_path: str):
        self.csv_path = Path(csv_path)
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.agents_data = self._load_csv()
        self.filter_types = FilterTypes()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load filter configuration from JSON file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Support both 'filters' (AND logic) and 'filter_groups' (OR logic)
        if 'filters' not in config and 'filter_groups' not in config:
            raise ValueError("Config file must contain either 'filters' or 'filter_groups' key")
        
        return config
    
    def _load_csv(self) -> List[Dict[str, Any]]:
        """Load agent demographics CSV file."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        agents = []
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                agents.append(row)
        
        return agents
    
    def _apply_filter(self, agent: Dict[str, Any], column: str, filter_config: Dict[str, Any]) -> bool:
        """Apply a single filter to an agent."""
        filter_type = filter_config.get('type')
        value = agent.get(column, '')
        
        if filter_type == 'numeric_range':
            return self.filter_types.numeric_range(value, filter_config)
        elif filter_type == 'categorical':
            return self.filter_types.categorical(value, filter_config)
        elif filter_type == 'regex':
            return self.filter_types.regex(value, filter_config)
        else:
            print(f"Warning: Unknown filter type '{filter_type}' for column '{column}'")
            return True
    
    def _matches_filter_group(self, agent: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if an agent matches a filter group (AND logic within group)."""
        for column, filter_config in filters.items():
            if not self._apply_filter(agent, column, filter_config):
                return False
        return True
    
    def filter_agents(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Filter agents based on configuration. Returns (matching, non-matching).
        
        Supports two modes:
        1. 'filters' key: Single filter group with AND logic (backward compatible)
        2. 'filter_groups' key: Multiple filter groups with OR logic between groups
        """
        matching = []
        non_matching = []
        
        # Check if using filter_groups (OR logic) or filters (AND logic)
        filter_groups = self.config.get('filter_groups', None)
        
        if filter_groups:
            # OR logic: agent matches if it matches ANY filter group
            for agent in self.agents_data:
                matches_any = False
                
                # Check each filter group (AND logic within each group)
                for filter_group in filter_groups:
                    if self._matches_filter_group(agent, filter_group):
                        matches_any = True
                        break
                
                if matches_any:
                    matching.append(agent)
                else:
                    non_matching.append(agent)
        else:
            # AND logic: backward compatible single filter group
            filters = self.config.get('filters', {})
            
            for agent in self.agents_data:
                if self._matches_filter_group(agent, filters):
                    matching.append(agent)
                else:
                    non_matching.append(agent)
        
        return matching, non_matching
    
    def _generate_filter_label(self) -> str:
        """Generate a label for the filter condition."""
        filter_groups = self.config.get('filter_groups', None)
        
        if filter_groups:
            # OR logic: generate label for each group and combine
            group_labels = []
            for i, filter_group in enumerate(filter_groups):
                group_parts = []
                for column, filter_config in sorted(filter_group.items()):
                    filter_type = filter_config.get('type', '')
                    column_parts = []
                    
                    if filter_type == 'numeric_range':
                        min_val = filter_config.get('min', '')
                        max_val = filter_config.get('max', '')
                        if min_val and max_val:
                            column_parts.append(f"{min_val}-{max_val}")
                        elif min_val:
                            column_parts.append(f"min{min_val}")
                        elif max_val:
                            column_parts.append(f"max{max_val}")
                        if column_parts:
                            group_parts.append(f"{column}_{'_'.join(column_parts)}")
                    
                    elif filter_type == 'categorical':
                        include = filter_config.get('include', [])
                        exclude = filter_config.get('exclude', [])
                        if include:
                            column_parts.append(f"incl{len(include)}")
                        if exclude:
                            column_parts.append(f"excl{len(exclude)}")
                        if column_parts:
                            group_parts.append(f"{column}_{'_'.join(column_parts)}")
                    
                    elif filter_type == 'regex':
                        group_parts.append(f"{column}_regex")
                
                if group_parts:
                    group_labels.append(f"group{i+1}_{'_'.join(group_parts)}")
            
            if group_labels:
                return "_OR_".join(group_labels)
        else:
            # AND logic: single filter group (backward compatible)
            filters = self.config.get('filters', {})
            parts = []
            
            for column, filter_config in sorted(filters.items()):
                filter_type = filter_config.get('type', '')
                column_parts = []
                
                if filter_type == 'numeric_range':
                    min_val = filter_config.get('min', '')
                    max_val = filter_config.get('max', '')
                    if min_val and max_val:
                        column_parts.append(f"{min_val}-{max_val}")
                    elif min_val:
                        column_parts.append(f"min{min_val}")
                    elif max_val:
                        column_parts.append(f"max{max_val}")
                    if column_parts:
                        parts.append(f"{column}_{'_'.join(column_parts)}")
                
                elif filter_type == 'categorical':
                    include = filter_config.get('include', [])
                    exclude = filter_config.get('exclude', [])
                    if include:
                        column_parts.append(f"incl{len(include)}")
                    if exclude:
                        column_parts.append(f"excl{len(exclude)}")
                    if column_parts:
                        parts.append(f"{column}_{'_'.join(column_parts)}")
                
                elif filter_type == 'regex':
                    parts.append(f"{column}_regex")
            
            if parts:
                return "_".join(parts)
        
        # Fallback
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"filter_{timestamp}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about filtered agents."""
        matching, non_matching = self.filter_agents()
        
        total = len(self.agents_data)
        matching_count = len(matching)
        non_matching_count = len(non_matching)
        
        stats = {
            'total': total,
            'matching': matching_count,
            'non_matching': non_matching_count,
            'matching_percentage': (matching_count / total * 100) if total > 0 else 0,
            'non_matching_percentage': (non_matching_count / total * 100) if total > 0 else 0
        }
        
        # Breakdown by filter columns
        filter_groups = self.config.get('filter_groups', None)
        if filter_groups:
            # For OR logic, collect all columns from all groups
            all_columns = set()
            for filter_group in filter_groups:
                all_columns.update(filter_group.keys())
        else:
            # For AND logic, use single filter group
            all_columns = set(self.config.get('filters', {}).keys())
        
        breakdown = {}
        for column in all_columns:
            breakdown[column] = {}
            for agent in matching:
                value = agent.get(column, 'N/A')
                breakdown[column][value] = breakdown[column].get(value, 0) + 1
        
        stats['breakdown'] = breakdown
        
        return stats
    
    def create_filtered_csv(self, output_path: str = None) -> str:
        """Create a filtered CSV file."""
        matching, _ = self.filter_agents()
        
        if not matching:
            print("Warning: No matching agents found. CSV file will be empty.")
        
        # Generate output filename
        if output_path is None:
            filter_label = self._generate_filter_label()
            csv_stem = self.csv_path.stem
            csv_suffix = self.csv_path.suffix
            output_path = self.csv_path.parent / f"{csv_stem}_{filter_label}{csv_suffix}"
        
        output_path = Path(output_path)
        
        # Get fieldnames from first agent (all should have same keys)
        if matching:
            fieldnames = list(matching[0].keys())
        else:
            # If no matching, read from original CSV
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
        
        # Write filtered CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(matching)
        
        return str(output_path)
    
    def filter_agent_bank(self, agents_base_path: str, backup_base_path: str, 
                         dry_run: bool = True, confirm: bool = False) -> Dict[str, Any]:
        """Move non-matching agents to backup folder."""
        matching, non_matching = self.filter_agents()
        
        agents_base = Path(agents_base_path)
        backup_base = Path(backup_base_path)
        
        if not agents_base.exists():
            raise FileNotFoundError(f"Agents base path not found: {agents_base_path}")
        
        # Create backup folder with filter label
        filter_label = self._generate_filter_label()
        backup_folder = backup_base / filter_label
        backup_folder.mkdir(parents=True, exist_ok=True)
        
        # Get agent IDs to move
        agent_ids_to_move = {agent['agent_id'] for agent in non_matching}
        
        moved_agents = []
        failed_moves = []
        
        print(f"\n{'DRY RUN - ' if dry_run else ''}Filtering agent bank...")
        print(f"Total agents: {len(self.agents_data)}")
        print(f"Matching (keeping): {len(matching)}")
        print(f"Non-matching (moving): {len(non_matching)}")
        print(f"Backup folder: {backup_folder}")
        
        if dry_run:
            print("\nDRY RUN - No files will be moved.")
            print(f"Would move {len(agent_ids_to_move)} agent folders:")
            sample_ids = list(agent_ids_to_move)[:10]
            print(f"  Sample IDs: {', '.join(sample_ids)}")
            if len(agent_ids_to_move) > 10:
                print(f"  ... and {len(agent_ids_to_move) - 10} more")
            print("\nTo actually move files, run with --confirm flag")
            return {
                'dry_run': True,
                'would_move': len(agent_ids_to_move),
                'backup_folder': str(backup_folder)
            }
        
        if not confirm:
            print("\nError: --confirm flag required to actually move files.")
            print("This is a safety measure. Add --confirm to proceed.")
            return {
                'dry_run': True,
                'error': 'Confirmation required'
            }
        
        # Actually move the agents
        print(f"\nMoving {len(agent_ids_to_move)} agent folders to backup...")
        
        for agent_id in agent_ids_to_move:
            agent_folder = agents_base / agent_id
            
            if not agent_folder.exists():
                failed_moves.append(agent_id)
                continue
            
            try:
                dest_folder = backup_folder / agent_id
                shutil.move(str(agent_folder), str(dest_folder))
                moved_agents.append(agent_id)
            except Exception as e:
                print(f"Error moving {agent_id}: {e}")
                failed_moves.append(agent_id)
        
        # Create log file
        log_file = backup_folder / "move_log.json"
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'filter_config': self.config.get('filters', {}),
            'filter_groups': self.config.get('filter_groups', None),
            'total_agents': len(self.agents_data),
            'moved_count': len(moved_agents),
            'failed_count': len(failed_moves),
            'moved_agents': moved_agents,
            'failed_agents': failed_moves
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"\n✓ Moved {len(moved_agents)} agent folders")
        if failed_moves:
            print(f"✗ Failed to move {len(failed_moves)} agent folders")
        print(f"✓ Log file created: {log_file}")
        
        return {
            'dry_run': False,
            'moved': len(moved_agents),
            'failed': len(failed_moves),
            'backup_folder': str(backup_folder),
            'log_file': str(log_file)
        }


def print_statistics(stats: Dict[str, Any]):
    """Print statistics in a readable format."""
    print("\n" + "=" * 60)
    print("FILTER STATISTICS")
    print("=" * 60)
    print(f"Total agents: {stats['total']}")
    print(f"Matching agents: {stats['matching']} ({stats['matching_percentage']:.1f}%)")
    print(f"Non-matching agents: {stats['non_matching']} ({stats['non_matching_percentage']:.1f}%)")
    
    if stats.get('breakdown'):
        print("\nBreakdown by filter columns:")
        for column, values in stats['breakdown'].items():
            print(f"\n  {column}:")
            sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)
            for value, count in sorted_values[:10]:  # Show top 10
                percentage = (count / stats['matching'] * 100) if stats['matching'] > 0 else 0
                print(f"    {value}: {count} ({percentage:.1f}%)")
            if len(sorted_values) > 10:
                print(f"    ... and {len(sorted_values) - 10} more values")
    
    print("=" * 60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Filter and manage agent bank based on demographic criteria'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to filter configuration JSON file'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='agent_bank/gss_agents_demographics.csv',
        help='Path to demographics CSV file (default: agent_bank/gss_agents_demographics.csv)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['statistics', 'edit_csv', 'filter_agent_bank'],
        default='statistics',
        help='Operation mode (default: statistics)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV path (for edit_csv mode, optional)'
    )
    parser.add_argument(
        '--agents-path',
        type=str,
        default='agent_bank/populations/gss_agents',
        help='Path to agent folders (default: agent_bank/populations/gss_agents)'
    )
    parser.add_argument(
        '--backup-path',
        type=str,
        default='agent_bank/populations/backup_agents',
        help='Path to backup folder (default: agent_bank/populations/backup_agents)'
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Confirm destructive operations (required for filter_agent_bank mode)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location or as absolute paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    
    # Config path: try relative to script_dir, then absolute
    if Path(args.config).is_absolute():
        config_path = Path(args.config)
    else:
        # Try relative to script_dir first, then relative to current working directory
        config_path = script_dir / args.config
        if not config_path.exists():
            config_path = Path(args.config).resolve()
    
    # CSV path: try relative to project_root, then absolute
    if Path(args.csv).is_absolute():
        csv_path = Path(args.csv)
    else:
        csv_path = project_root / args.csv
    
    # Agents path: try relative to project_root, then absolute
    if Path(args.agents_path).is_absolute():
        agents_path = Path(args.agents_path)
    else:
        agents_path = project_root / args.agents_path
    
    # Backup path: try relative to project_root, then absolute
    if Path(args.backup_path).is_absolute():
        backup_path = Path(args.backup_path)
    else:
        backup_path = project_root / args.backup_path
    
    try:
        # Initialize filter
        agent_filter = AgentFilter(str(csv_path), str(config_path))
        
        if args.mode == 'statistics':
            stats = agent_filter.get_statistics()
            print_statistics(stats)
        
        elif args.mode == 'edit_csv':
            output_path = agent_filter.create_filtered_csv(args.output)
            print(f"\n✓ Filtered CSV created: {output_path}")
            stats = agent_filter.get_statistics()
            print(f"  Contains {stats['matching']} matching agents")
        
        elif args.mode == 'filter_agent_bank':
            result = agent_filter.filter_agent_bank(
                str(agents_path),
                str(backup_path),
                dry_run=not args.confirm,
                confirm=args.confirm
            )
            if result.get('error'):
                return 1
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

