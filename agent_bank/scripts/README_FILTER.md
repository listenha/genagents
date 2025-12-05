# Agent Filter Tool

A tool for filtering and managing the agent bank based on demographic criteria.

## Features

- **Statistics Mode**: View statistics about agents matching your filter criteria
- **CSV Edit Mode**: Create a filtered CSV file with matching agents
- **Agent Bank Filter Mode**: Move non-matching agents to backup folder

## Configuration

### AND Logic (Single Filter Group)

Edit `filter_config.json` to specify your filter criteria using the `filters` key:

```json
{
  "filters": {
    "age": {
      "type": "numeric_range",
      "min": 18,
      "max": 30
    },
    "highest_degree_received": {
      "type": "categorical",
      "include": ["High school", "Bachelor's", "Graduate"],
      "exclude": ["Less than high school"]
    }
  }
}
```

All filters use AND logic - an agent must match ALL filter conditions.

### OR Logic (Multiple Filter Groups)

Use the `filter_groups` key to specify multiple filter groups with OR logic:

```json
{
  "filter_groups": [
    {
      "age": {
        "type": "numeric_range",
        "min": 18,
        "max": 22
      },
      "highest_degree_received": {
        "type": "categorical",
        "include": ["High school", "Bachelor's", "Graduate"]
      }
    },
    {
      "age": {
        "type": "numeric_range",
        "min": 22,
        "max": 30
      },
      "highest_degree_received": {
        "type": "categorical",
        "include": ["Bachelor's", "Graduate"]
      }
    }
  ]
}
```

With `filter_groups`, an agent matches if it matches ANY filter group. Within each group, all conditions use AND logic.

**Example**: The above config matches:
- Group 1: Age 18-22 AND (High school OR Bachelor's OR Graduate)
- OR
- Group 2: Age 22-30 AND (Bachelor's OR Graduate)

### Filter Types

1. **numeric_range**: Filter numeric values by range
   - `min`: Minimum value (optional)
   - `max`: Maximum value (optional)

2. **categorical**: Filter by specific values
   - `include`: List of allowed values (optional)
   - `exclude`: List of disallowed values (optional)

3. **regex**: Filter by regular expression pattern
   - `pattern`: Regex pattern to match

## Usage

### Statistics Mode (Default)

View statistics about matching agents:

```bash
python3 agent_bank/scripts/filter_agents.py --config agent_bank/scripts/filter_config.json
```

or explicitly:

```bash
python3 agent_bank/scripts/filter_agents.py --config agent_bank/scripts/filter_config.json --mode statistics
```

### CSV Edit Mode

Create a filtered CSV file:

```bash
python3 agent_bank/scripts/filter_agents.py --config agent_bank/scripts/filter_config.json --mode edit_csv
```

The output CSV will be named based on your filter conditions, e.g.:
`gss_agents_demographics_age_18-30_highest_degree_received_incl4_excl1.csv`

You can also specify a custom output path:

```bash
python3 agent_bank/scripts/filter_agents.py --config agent_bank/scripts/filter_config.json --mode edit_csv --output my_filtered_agents.csv
```

### Agent Bank Filter Mode

Move non-matching agents to backup folder (dry-run by default):

```bash
python3 agent_bank/scripts/filter_agents.py --config agent_bank/scripts/filter_config.json --mode filter_agent_bank
```

To actually move the agents, add the `--confirm` flag:

```bash
python3 agent_bank/scripts/filter_agents.py --config agent_bank/scripts/filter_config.json --mode filter_agent_bank --confirm
```

Agents will be moved to:
`agent_bank/populations/backup_agents/{filter_label}/`

A log file will be created in the backup folder with details of the operation.

## Examples

### Filter by age only

```json
{
  "filters": {
    "age": {
      "type": "numeric_range",
      "min": 25,
      "max": 35
    }
  }
}
```

### Filter by education and exclude certain degrees

```json
{
  "filters": {
    "highest_degree_received": {
      "type": "categorical",
      "exclude": ["Less than high school"]
    }
  }
}
```

### Multiple filters (AND logic)

All filters must match for an agent to be included:

```json
{
  "filters": {
    "age": {
      "type": "numeric_range",
      "min": 18,
      "max": 30
    },
    "sex": {
      "type": "categorical",
      "include": ["Female"]
    },
    "political_views": {
      "type": "categorical",
      "include": ["Liberal", "Slightly liberal"]
    }
  }
}
```

### Complex OR conditions

Use `filter_groups` for complex OR logic:

```json
{
  "filter_groups": [
    {
      "age": {
        "type": "numeric_range",
        "min": 18,
        "max": 25
      },
      "sex": {
        "type": "categorical",
        "include": ["Female"]
      }
    },
    {
      "age": {
        "type": "numeric_range",
        "min": 25,
        "max": 35
      },
      "highest_degree_received": {
        "type": "categorical",
        "include": ["Graduate"]
      }
    }
  ]
}
```

This matches: (Age 18-25 AND Female) OR (Age 25-35 AND Graduate)

## Safety Features

- **Dry-run mode**: Default for `filter_agent_bank` mode - shows what would be moved without actually moving
- **Confirmation required**: Must use `--confirm` flag to actually move agents
- **Logging**: All moves are logged in the backup folder
- **Backup organization**: Each filter operation creates a separate backup folder

## Notes

- **AND logic**: When using `filters` key, all conditions must match
- **OR logic**: When using `filter_groups` key, agent matches if it matches ANY group (AND logic within each group)
- CSV files are never overwritten - new filtered files are created with descriptive names
- Original agent folders are moved (not copied) to backup when using `filter_agent_bank` mode
- The backup folder structure preserves the filter conditions for easy identification
- You can use either `filters` (AND) or `filter_groups` (OR), but not both in the same config file

