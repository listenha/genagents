#!/usr/bin/env python3
"""
Analyze consistency of Wavelength game responses across multiple attempts.

This script computes various consistency metrics (ICC, Cronbach's Alpha, CV, SD, Range)
for Wavelength game responses across multiple attempts and generates an interactive HTML report.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import re

import pandas as pd
import numpy as np
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_range(range_str: str) -> List[str]:
    """Parse a range string like "0000-0049" into a list of agent IDs."""
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


def parse_attempt_range(attempt_str: str) -> List[int]:
    """Parse attempt range like '1-10' into list of attempt IDs."""
    try:
        if '-' in attempt_str:
            start, end = attempt_str.split('-')
            return list(range(int(start), int(end) + 1))
        else:
            # Single attempt or comma-separated
            return [int(x.strip()) for x in attempt_str.split(',')]
    except ValueError as e:
        raise ValueError(f"Invalid attempt range format '{attempt_str}'. Error: {e}")


def load_wavelength_responses(agent_folder: str, attempt_ids: List[int]) -> Dict:
    """Load Wavelength game responses for an agent for specified attempts."""
    responses_path = Path(agent_folder) / "wavelength_responses.json"
    
    if not responses_path.exists():
        return None
    
    try:
        with open(responses_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Filter attempts
        filtered_attempts = [
            attempt for attempt in data.get("attempts", [])
            if attempt.get("attempt_id") in attempt_ids
        ]
        
        if not filtered_attempts:
            return None
        
        return {
            "agent_id": data.get("agent_metadata", {}).get("agent_id", os.path.basename(agent_folder)),
            "agent_name": data.get("agent_metadata", {}).get("agent_name", "Unknown"),
            "attempts": filtered_attempts
        }
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Error loading {responses_path}: {e}")
        return None


def compute_icc(data: np.ndarray) -> Tuple[float, float]:
    """
    Compute ICC(2,k) and ICC(3,k) for multiple attempts.
    
    For Wavelength game (0-100 scale):
    - Uses variance-based approach
    - ICC measures consistency of responses across attempts
    
    Args:
        data: 1D array of responses across attempts (for single header)
    
    Returns:
        Tuple of (ICC_2k, ICC_3k)
    """
    # Ensure 1D array
    if data.ndim > 1:
        data = data.flatten()
    
    n = len(data)
    if n < 2:
        return np.nan, np.nan
    
    # Remove NaN values
    data = data[~np.isnan(data)]
    n = len(data)
    if n < 2:
        return np.nan, np.nan
    
    mean_val = np.mean(data)
    var_val = np.var(data, ddof=1)
    
    if var_val == 0:
        # Perfect consistency
        return 1.0, 1.0
    
    # For Wavelength scale (0-100), use scale-relative ICC
    # Expected variance for uniform distribution on [0,100] = (100-0)^2/12 = 833.33
    expected_var_uniform = ((100 - 0) ** 2) / 12
    
    # ICC(2,k) - absolute agreement (stricter)
    icc_2k = 1 - (var_val / (var_val + expected_var_uniform))
    
    # ICC(3,k) - consistency (more lenient, allows systematic shifts)
    icc_3k = 1 - (var_val / (var_val + mean_val * 0.1))
    
    return max(0, min(1, icc_2k)), max(0, min(1, icc_3k))


def compute_cronbach_alpha(responses: List[float]) -> float:
    """
    Compute Cronbach's Alpha for single header across attempts.
    
    For Wavelength game (0-100 scale).
    """
    if len(responses) < 2:
        return np.nan
    
    responses_array = np.array(responses)
    responses_array = responses_array[~np.isnan(responses_array)]
    
    if len(responses_array) < 2:
        return np.nan
    
    var_val = np.var(responses_array, ddof=1)
    mean_val = np.mean(responses_array)
    
    if var_val == 0:
        return 1.0
    
    # Expected variance for random responses on [0,100] scale
    expected_var = ((100 - 0) ** 2) / 12
    
    # Simplified alpha: 1 - (observed_var / expected_var)
    alpha = 1 - (var_val / expected_var)
    
    return max(0, min(1, alpha))


def compute_consistency_metrics(responses: List[float]) -> Dict[str, float]:
    """Compute all consistency metrics for a list of responses across attempts."""
    if not responses or len(responses) < 2:
        return {
            "icc_2k": np.nan,
            "icc_3k": np.nan,
            "cronbach_alpha": np.nan,
            "cv": np.nan,
            "sd": np.nan,
            "range": np.nan,
            "mean": np.nan,
            "n_attempts": len(responses) if responses else 0
        }
    
    responses_array = np.array(responses)
    responses_array = responses_array[~np.isnan(responses_array)]
    
    if len(responses_array) < 2:
        return {
            "icc_2k": np.nan,
            "icc_3k": np.nan,
            "cronbach_alpha": np.nan,
            "cv": np.nan,
            "sd": np.nan,
            "range": np.nan,
            "mean": np.nan,
            "n_attempts": len(responses_array)
        }
    
    # Compute ICC
    icc_2k, icc_3k = compute_icc(responses_array)
    
    # Compute Cronbach's Alpha
    cronbach_alpha = compute_cronbach_alpha(responses)
    
    # Compute descriptive stats
    mean_val = np.mean(responses_array)
    sd_val = np.std(responses_array, ddof=1)
    cv_val = (sd_val / mean_val * 100) if mean_val > 0 else np.nan
    range_val = np.max(responses_array) - np.min(responses_array)
    
    return {
        "icc_2k": icc_2k,
        "icc_3k": icc_3k,
        "cronbach_alpha": cronbach_alpha,
        "cv": cv_val,
        "sd": sd_val,
        "range": range_val,
        "mean": mean_val,
        "n_attempts": len(responses_array)
    }


def extract_responses_data(agent_data: Dict, attempt_ids: List[int]) -> pd.DataFrame:
    """Extract responses into a DataFrame."""
    rows = []
    
    for attempt in agent_data["attempts"]:
        attempt_id = attempt["attempt_id"]
        if attempt_id not in attempt_ids:
            continue
        
        headers = attempt.get("headers", {})
        for header_id, header_data in headers.items():
            response_value = header_data.get("response")
            spectrum = header_data.get("spectrum", {})
            cue = header_data.get("cue", "")
            raw_response = header_data.get("raw_response", "")
            
            rows.append({
                "agent_id": agent_data["agent_id"],
                "agent_name": agent_data["agent_name"],
                "attempt_id": attempt_id,
                "header_id": header_id,
                "cue": cue,
                "spectrum_left": spectrum.get("left", ""),
                "spectrum_right": spectrum.get("right", ""),
                "response": response_value if response_value is not None else np.nan,
                "raw_response": raw_response
            })
    
    return pd.DataFrame(rows)


def extract_reasoning_data(agent_data: Dict, attempt_ids: List[int]) -> pd.DataFrame:
    """Extract reasoning (raw_response) data into a DataFrame."""
    rows = []
    
    for attempt in agent_data["attempts"]:
        attempt_id = attempt["attempt_id"]
        if attempt_id not in attempt_ids:
            continue
        
        headers = attempt.get("headers", {})
        for header_id, header_data in headers.items():
            raw_response = header_data.get("raw_response", "")
            cue = header_data.get("cue", "")
            spectrum = header_data.get("spectrum", {})
            
            rows.append({
                "agent_id": agent_data["agent_id"],
                "agent_name": agent_data["agent_name"],
                "attempt_id": attempt_id,
                "header_id": header_id,
                "cue": cue,
                "spectrum_left": spectrum.get("left", ""),
                "spectrum_right": spectrum.get("right", ""),
                "reasoning": raw_response
            })
    
    return pd.DataFrame(rows)


def analyze_consistency(base_path: str, agent_ids: List[str], attempt_ids: List[int]) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """
    Main analysis function.
    
    Returns:
        Tuple of (results, df_all, df_reasoning)
    """
    print(f"Loading Wavelength game responses for {len(agent_ids)} agents, attempts {min(attempt_ids)}-{max(attempt_ids)}...")
    
    all_data = []
    all_reasoning_data = []
    agent_folders = []
    
    for agent_id in agent_ids:
        agent_folder = Path(base_path) / agent_id
        if not (agent_folder / "scratch.json").exists():
            print(f"Warning: Agent {agent_id} not found, skipping")
            continue
        
        # Load responses
        agent_data = load_wavelength_responses(str(agent_folder), attempt_ids)
        if agent_data is None:
            print(f"Warning: No Wavelength data found for agent {agent_id}, skipping")
            continue
        
        df = extract_responses_data(agent_data, attempt_ids)
        df_reasoning = extract_reasoning_data(agent_data, attempt_ids)
        
        if not df.empty:
            all_data.append(df)
            all_reasoning_data.append(df_reasoning)
            agent_folders.append(agent_folder)
    
    if not all_data:
        raise ValueError("No Wavelength game data found for the specified agents and attempts")
    
    # Combine all data
    df_all = pd.concat(all_data, ignore_index=True)
    df_reasoning_all = pd.concat(all_reasoning_data, ignore_index=True) if all_reasoning_data else pd.DataFrame()
    
    print(f"Loaded {len(df_all)} response records")
    if not df_reasoning_all.empty:
        print(f"Loaded {len(df_reasoning_all)} reasoning records")
    
    # Analyze responses
    print(f"Analyzing response consistency metrics...")
    results = analyze_header_consistency(df_all, attempt_ids)
    
    return results, df_all, df_reasoning_all


def analyze_header_consistency(df_all: pd.DataFrame, attempt_ids: List[int]) -> Dict:
    """Analyze consistency for header responses."""
    results = {}
    
    # 1. Per-header consistency
    header_metrics = []
    for (agent_id, header_id), group in df_all.groupby(['agent_id', 'header_id']):
        responses = group['response'].dropna().tolist()
        metrics = compute_consistency_metrics(responses)
        metrics.update({
            'agent_id': agent_id,
            'header_id': header_id,
            'cue': group['cue'].iloc[0],
            'spectrum_left': group['spectrum_left'].iloc[0],
            'spectrum_right': group['spectrum_right'].iloc[0]
        })
        header_metrics.append(metrics)
    
    results['header_level'] = pd.DataFrame(header_metrics)
    
    # 2. Per-agent consistency (average ICC across headers for each agent)
    agent_metrics = []
    for agent_id in df_all['agent_id'].unique():
        # Get all header-level ICCs for this agent
        agent_headers = results['header_level'][results['header_level']['agent_id'] == agent_id]
        
        if len(agent_headers) == 0:
            continue
        
        # Get agent name from the data
        agent_data = df_all[df_all['agent_id'] == agent_id]
        agent_name = agent_data['agent_name'].iloc[0] if 'agent_name' in agent_data.columns else 'Unknown'
        
        metrics = {
            'agent_id': agent_id,
            'agent_name': agent_name,
            'icc_2k': agent_headers['icc_2k'].mean(),
            'icc_3k': agent_headers['icc_3k'].mean(),
            'cronbach_alpha': agent_headers['cronbach_alpha'].mean(),
            'cv': agent_headers['cv'].mean(),
            'sd': agent_headers['sd'].mean(),
            'range': agent_headers['range'].mean(),
            'mean': agent_headers['mean'].mean(),
            'n_headers': len(agent_headers),
            'n_attempts': agent_headers['n_attempts'].iloc[0] if len(agent_headers) > 0 else 0
        }
        agent_metrics.append(metrics)
    
    results['agent_level'] = pd.DataFrame(agent_metrics)
    
    # 3. Overall summary by header (aggregated across agents)
    header_aggregated = []
    for header_id, group in results['header_level'].groupby('header_id'):
        header_aggregated.append({
            'header_id': header_id,
            'cue': group['cue'].iloc[0],
            'spectrum_left': group['spectrum_left'].iloc[0],
            'spectrum_right': group['spectrum_right'].iloc[0],
            'icc_2k': group['icc_2k'].mean(),
            'icc_3k': group['icc_3k'].mean(),
            'cronbach_alpha': group['cronbach_alpha'].mean(),
            'cv': group['cv'].mean(),
            'sd': group['sd'].mean(),
            'range': group['range'].mean(),
            'mean': group['mean'].mean(),
            'n_agents': len(group),
            'n_attempts': group['n_attempts'].iloc[0] if len(group) > 0 else 0
        })
    
    results['header_aggregated'] = pd.DataFrame(header_aggregated)
    
    # 4. Aggregated by agent (average ICC across headers for each agent)
    agent_aggregated = []
    for agent_id in df_all['agent_id'].unique():
        agent_headers = results['header_level'][results['header_level']['agent_id'] == agent_id]
        if len(agent_headers) == 0:
            continue
        
        agent_data = df_all[df_all['agent_id'] == agent_id]
        agent_name = agent_data['agent_name'].iloc[0] if 'agent_name' in agent_data.columns else 'Unknown'
        
        agent_aggregated.append({
            'agent_id': agent_id,
            'agent_name': agent_name,
            'icc_2k': agent_headers['icc_2k'].mean(),
            'icc_3k': agent_headers['icc_3k'].mean(),
            'cronbach_alpha': agent_headers['cronbach_alpha'].mean(),
            'cv': agent_headers['cv'].mean(),
            'sd': agent_headers['sd'].mean(),
            'range': agent_headers['range'].mean(),
            'mean': agent_headers['mean'].mean(),
            'n_headers': len(agent_headers),
            'n_attempts': agent_headers['n_attempts'].iloc[0] if len(agent_headers) > 0 else 0
        })
    
    results['agent_aggregated'] = pd.DataFrame(agent_aggregated)
    
    return results


def generate_html_report(results: Dict, df_all: pd.DataFrame, output_path: str, 
                        agent_ids: List[str], attempt_ids: List[int], df_reasoning: pd.DataFrame = None):
    """Generate interactive HTML report with Plotly visualizations and tabbed interface."""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        print("Warning: Plotly not available. Generating basic HTML report without interactive charts.")
        generate_basic_html_report(results, df_all, output_path, agent_ids, attempt_ids, df_reasoning)
        return
    
    import json
    import re
    
    # Prepare data structures
    header_agg = results['header_aggregated']
    header_level = results['header_level']
    agent_level = results['agent_level']
    agent_agg = results.get('agent_aggregated', results['agent_level'])
    
    # Get unique headers and their info
    # Sort headers by numeric part (e.g., header_2 before header_10)
    def extract_header_number(header_id):
        """Extract numeric part from header ID for sorting (e.g., 'header_10' -> 10)."""
        try:
            # Extract number after underscore (most common format: header_0, header_1, etc.)
            parts = str(header_id).split('_')
            if len(parts) > 1:
                return int(parts[-1])
            # If no underscore, try to extract number from the end
            match = re.search(r'\d+$', str(header_id))
            if match:
                return int(match.group())
            return 0
        except (ValueError, IndexError):
            return 0
    
    unique_headers = sorted(df_all['header_id'].unique(), key=extract_header_number)
    header_info = {}
    for header_id in unique_headers:
        header_row = df_all[df_all['header_id'] == header_id].iloc[0]
        header_info[header_id] = {
            'cue': header_row['cue'],
            'spectrum_left': header_row['spectrum_left'],
            'spectrum_right': header_row['spectrum_right']
        }
    
    # Prepare reasoning data for JavaScript
    reasoning_data_dict = {}
    if df_reasoning is not None and not df_reasoning.empty:
        for _, row in df_reasoning.iterrows():
            key = f"{row['agent_id']}_{row['header_id']}_{int(row['attempt_id'])}"
            reasoning_data_dict[key] = {
                'agent_id': str(row['agent_id']),
                'agent_name': str(row['agent_name']),
                'header_id': str(row['header_id']),
                'cue': str(row['cue']),
                'attempt_id': int(row['attempt_id']),
                'reasoning': str(row.get('reasoning', ''))
            }
    
    # Prepare trajectory data for JavaScript
    trajectory_data_dict = {}
    for agent_id in agent_ids:
        agent_data = df_all[df_all['agent_id'] == agent_id]
        for header_id in unique_headers:
            header_data = agent_data[agent_data['header_id'] == header_id].sort_values('attempt_id')
            header_data = header_data[~header_data['response'].isna()]
            if len(header_data) > 0:
                key = f"{agent_id}_{header_id}"
                trajectory_data_dict[key] = {
                    'agent_id': str(agent_id),
                    'header_id': str(header_id),
                    'cue': str(header_data['cue'].iloc[0]),
                    'attempts': [int(x) for x in header_data['attempt_id'].tolist()],
                    'responses': [float(x) for x in header_data['response'].tolist()]
                }
    
    # Serialize data for JavaScript (escape properly)
    reasoning_data_json = json.dumps(reasoning_data_dict).replace('</script>', '<\\/script>')
    trajectory_data_json = json.dumps(trajectory_data_dict).replace('</script>', '<\\/script>')
    header_info_json = json.dumps(header_info).replace('</script>', '<\\/script>')
    agent_ids_json = json.dumps(agent_ids).replace('</script>', '<\\/script>')
    attempt_ids_json = json.dumps(attempt_ids).replace('</script>', '<\\/script>')
    unique_headers_json = json.dumps(unique_headers).replace('</script>', '<\\/script>')
    
    # Summary metrics
    mean_icc_2k = header_agg['icc_2k'].mean() if not header_agg['icc_2k'].isna().all() else 0
    mean_icc_3k = header_agg['icc_3k'].mean() if not header_agg['icc_3k'].isna().all() else 0
    mean_alpha = header_agg['cronbach_alpha'].mean() if not header_agg['cronbach_alpha'].isna().all() else 0
    
    # Summary metrics
    header_agg = results['header_aggregated']
    mean_icc_2k = header_agg['icc_2k'].mean() if not header_agg['icc_2k'].isna().all() else 0
    mean_icc_3k = header_agg['icc_3k'].mean() if not header_agg['icc_3k'].isna().all() else 0
    mean_alpha = header_agg['cronbach_alpha'].mean() if not header_agg['cronbach_alpha'].isna().all() else 0
    
    # Generate visualizations
    # Summary chart
    fig_summary = go.Figure()
    fig_summary.add_trace(go.Bar(
        x=header_agg['cue'],
        y=header_agg['icc_2k'].fillna(0),
        name='ICC(2,k)',
        marker_color='#4CAF50',
        hovertemplate='Cue: %{x}<br>ICC(2,k): %{y:.3f}<extra></extra>'
    ))
    fig_summary.update_layout(
        title='Consistency by Header (Average Across Agents)',
        xaxis_title='Cue',
        yaxis_title='ICC(2,k)',
        yaxis=dict(range=[0, 1]),
        height=500
    )
    
    # Question level - All view (heatmap)
    # Split into multiple heatmaps if more than 50 agents
    heatmap_figs = []
    try:
        pivot_all = header_level.pivot_table(
            index='header_id',
            columns='agent_id',
            values='icc_2k',
            aggfunc='mean'
        ).fillna(0)
        
        # Sort columns (agent IDs) for consistent ordering
        pivot_all = pivot_all.sort_index(axis=1)
        
        agent_ids_list = pivot_all.columns.tolist()
        header_ids_list = pivot_all.index.tolist()
        header_cues_list = [header_info[h]['cue'] for h in header_ids_list]
        
        # Split agents into chunks of 50
        chunk_size = 50
        num_chunks = (len(agent_ids_list) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(agent_ids_list))
            
            chunk_agent_ids = agent_ids_list[start_idx:end_idx]
            chunk_pivot = pivot_all[chunk_agent_ids]
            
            # Create customdata for hover - array of [agent_id, cue] pairs
            customdata = []
            for header_id in header_ids_list:
                row_data = []
                for agent_id in chunk_agent_ids:
                    row_data.append([agent_id, header_info[header_id]['cue']])
                customdata.append(row_data)
            
            # Create hovertemplate with proper Plotly syntax (single braces, not double)
            hovertemplate_str = '<b>Agent ID:</b> %{x}<br><b>Header:</b> %{y}<br><b>ICC(2,k):</b> %{z:.3f}<extra></extra>'
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=chunk_pivot.values.tolist(),
                x=chunk_agent_ids,
                y=header_cues_list,
                colorscale='RdYlGn',
                colorbar=dict(title="ICC(2,k)"),
                hovertemplate=hovertemplate_str,
                showscale=True
            ))
            
            title_suffix = f" (Agents {chunk_agent_ids[0]} to {chunk_agent_ids[-1]})" if num_chunks > 1 else ""
            fig_heatmap.update_layout(
                title=f'All Agent-Header Pairs{title_suffix}',
                xaxis=dict(
                    title='Agent ID',
                    type='category',
                    tickangle=-45
                ),
                yaxis=dict(
                    title='Header (Cue)',
                    type='category'
                ),
                height=600
            )
            
            heatmap_figs.append(fig_heatmap)
        
        # Add overall heatmap with all agents at the end
        if num_chunks > 1:
            hovertemplate_str_all = '<b>Agent ID:</b> %{x}<br><b>Header:</b> %{y}<br><b>ICC(2,k):</b> %{z:.3f}<extra></extra>'
            fig_heatmap_all = go.Figure(data=go.Heatmap(
                z=pivot_all.values.tolist(),
                x=agent_ids_list,
                y=header_cues_list,
                colorscale='RdYlGn',
                colorbar=dict(title="ICC(2,k)"),
                hovertemplate=hovertemplate_str_all,
                showscale=True
            ))
            fig_heatmap_all.update_layout(
                title=f'All Agent-Header Pairs (All {len(agent_ids_list)} Agents)',
                xaxis=dict(
                    title='Agent ID',
                    type='category',
                    tickangle=-45
                ),
                yaxis=dict(
                    title='Header (Cue)',
                    type='category'
                ),
                height=600
            )
            heatmap_figs.append(fig_heatmap_all)
        
    except Exception as e:
        fig_all = go.Figure()
        fig_all.add_annotation(text="Data not available", showarrow=False)
        print(f"Warning: Could not create all-view heatmap: {e}")
        heatmap_figs = [fig_all]
    
    # Question level - Aggregated by header
    # Sort by header_id to ensure consistent ordering
    header_agg_sorted = header_agg.sort_values('header_id').copy()
    header_agg_sorted['icc_2k'] = header_agg_sorted['icc_2k'].fillna(0)
    
    fig_header_agg = go.Figure()
    fig_header_agg.add_trace(go.Bar(
        x=header_agg_sorted['cue'].tolist(),
        y=header_agg_sorted['icc_2k'].tolist(),
        marker_color='#4CAF50',
        hovertemplate='Header: %{x}<br>Mean ICC(2,k): %{y:.3f}<extra></extra>',
        text=header_agg_sorted['icc_2k'].tolist(),
        texttemplate='%{{text:.3f}}',
        textposition='outside',
        width=0.8
    ))
    fig_header_agg.update_layout(
        title='Aggregated by Header (Average Across Agents)',
        xaxis_title='Header (Cue)',
        yaxis_title='Mean ICC(2,k)',
        yaxis=dict(range=[0, 1.1]),
        height=max(500, len(header_agg_sorted) * 25),
        xaxis=dict(tickangle=-45),
        bargap=0.1
    )
    
    # Question level - Aggregated by agent
    # Sort by agent_id to ensure consistent ordering
    agent_agg_sorted = agent_agg.sort_values('agent_id').copy()
    agent_agg_sorted['icc_2k'] = agent_agg_sorted['icc_2k'].fillna(0)
    
    # Split into multiple charts if more than 50 agents
    agent_agg_figs = []
    chunk_size = 50
    num_chunks = (len(agent_agg_sorted) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(agent_agg_sorted))
        
        chunk_data = agent_agg_sorted.iloc[start_idx:end_idx]
        chunk_agent_ids = chunk_data['agent_id'].tolist()
        chunk_icc_values = chunk_data['icc_2k'].tolist()
        
        fig_agent_agg = go.Figure()
        fig_agent_agg.add_trace(go.Bar(
            x=chunk_agent_ids,
            y=chunk_icc_values,
            marker_color='#4CAF50',
            hovertemplate='Agent: %{x}<br>Mean ICC(2,k): %{y:.3f}<extra></extra>',
            text=chunk_icc_values,
            texttemplate='%{{text:.3f}}',
            textposition='outside',
            width=0.8
        ))
        
        title_suffix = f" (Agents {chunk_agent_ids[0]} to {chunk_agent_ids[-1]})" if num_chunks > 1 else ""
        fig_agent_agg.update_layout(
            title=f'Aggregated by Agent (Average Across Headers){title_suffix}',
            xaxis_title='Agent ID',
            yaxis_title='Mean ICC(2,k)',
            yaxis=dict(range=[0, 1.1]),
            height=max(500, len(chunk_data) * 15),
            xaxis=dict(tickangle=-45),
            bargap=0.1
        )
        
        agent_agg_figs.append(fig_agent_agg)
    
    # Add overall agent aggregated chart with all agents at the end
    if num_chunks > 1:
        fig_agent_agg_all = go.Figure()
        fig_agent_agg_all.add_trace(go.Bar(
            x=agent_agg_sorted['agent_id'].tolist(),
            y=agent_agg_sorted['icc_2k'].tolist(),
            marker_color='#4CAF50',
            hovertemplate='Agent: %{x}<br>Mean ICC(2,k): %{y:.3f}<extra></extra>',
            text=agent_agg_sorted['icc_2k'].tolist(),
            texttemplate='%{{text:.3f}}',
            textposition='outside',
            width=0.8
        ))
        fig_agent_agg_all.update_layout(
            title=f'Aggregated by Agent (Average Across Headers) - All {len(agent_agg_sorted)} Agents',
            xaxis_title='Agent ID',
            yaxis_title='Mean ICC(2,k)',
            yaxis=dict(range=[0, 1.1]),
            height=max(500, len(agent_agg_sorted) * 15),
            xaxis=dict(tickangle=-45),
            bargap=0.1
        )
        agent_agg_figs.append(fig_agent_agg_all)
    
    # Convert to HTML
    summary_chart_html = fig_summary.to_html(full_html=False, include_plotlyjs='cdn', div_id='summary-chart')
    
    # Convert multiple heatmaps to HTML
    heatmap_htmls = []
    for idx, fig_heatmap in enumerate(heatmap_figs):
        div_id = f'question-all-view-{idx}'
        heatmap_html = fig_heatmap.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)
        heatmap_htmls.append(heatmap_html)
    
    header_agg_html = fig_header_agg.to_html(full_html=False, include_plotlyjs=False, div_id='question-header-agg')
    
    # Convert multiple agent aggregated charts to HTML
    agent_agg_htmls = []
    for idx, fig_agent_agg in enumerate(agent_agg_figs):
        div_id = f'question-agent-agg-{idx}'
        agent_agg_html = fig_agent_agg.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)
        agent_agg_htmls.append(agent_agg_html)
    
    # Extract divs and scripts
    def extract_div(html_str):
        div_match = re.search(r'<div[^>]*id="[^"]*"[^>]*>.*?</div>', html_str, re.DOTALL)
        return div_match.group(0) if div_match else html_str
    
    def extract_scripts(html_str):
        return re.findall(r'<script[^>]*>.*?</script>', html_str, re.DOTALL)
    
    # Extract all heatmap divs and arrange them
    all_heatmap_divs = [extract_div(html) for html in heatmap_htmls]
    header_agg_div = extract_div(header_agg_html)
    all_agent_agg_divs = [extract_div(html) for html in agent_agg_htmls]
    
    # Collect all scripts
    all_scripts = []
    for html in heatmap_htmls:
        all_scripts.extend(extract_scripts(html))
    all_scripts.extend(extract_scripts(header_agg_html))
    for html in agent_agg_htmls:
        all_scripts.extend(extract_scripts(html))
    
    # Create HTML structure for multiple heatmaps (max 2 per row, overall chart full width)
    heatmaps_container_html = '<div id="heatmaps-container" style="display: flex; flex-wrap: wrap; gap: 20px;">'
    for idx, heatmap_div in enumerate(all_heatmap_divs):
        # Last chart (overall) should be full width, others max 2 per row
        is_overall = (idx == len(all_heatmap_divs) - 1 and len(all_heatmap_divs) > 1)
        if is_overall:
            width_style = 'width: 100%;'
        else:
            # For split charts, max 2 per row
            width_style = 'width: calc(50% - 10px);' if len(all_heatmap_divs) > 1 else 'width: 100%;'
        heatmaps_container_html += f'<div style="{width_style} margin-bottom: 20px;">{heatmap_div}</div>'
    heatmaps_container_html += '</div>'
    
    # Create HTML structure for multiple agent aggregated charts (max 2 per row, overall chart full width)
    agent_agg_container_html = '<div id="agent-agg-container" style="display: flex; flex-wrap: wrap; gap: 20px;">'
    for idx, agent_agg_div in enumerate(all_agent_agg_divs):
        # Last chart (overall) should be full width, others max 2 per row
        is_overall = (idx == len(all_agent_agg_divs) - 1 and len(all_agent_agg_divs) > 1)
        if is_overall:
            width_style = 'width: 100%;'
        else:
            # For split charts, max 2 per row
            width_style = 'width: calc(50% - 10px);' if len(all_agent_agg_divs) > 1 else 'width: 100%;'
        agent_agg_container_html += f'<div style="{width_style} margin-bottom: 20px;">{agent_agg_div}</div>'
    agent_agg_container_html += '</div>'
    
    # Escape HTML strings for JavaScript embedding (replace backticks and script tags)
    summary_chart_html_escaped = json.dumps(summary_chart_html).replace('</script>', '<\\/script>')
    all_view_div_escaped = json.dumps(heatmaps_container_html).replace('</script>', '<\\/script>')
    header_agg_div_escaped = json.dumps(header_agg_div).replace('</script>', '<\\/script>')
    agent_agg_div_escaped = json.dumps(agent_agg_container_html).replace('</script>', '<\\/script>')
    
    # Generate headers description table
    headers_desc_rows = []
    for header_id in unique_headers:
        info = header_info[header_id]
        headers_desc_rows.append({
            'Header ID': header_id,
            'Cue': info['cue'],
            'Spectrum Left': info['spectrum_left'],
            'Spectrum Right': info['spectrum_right']
        })
    headers_desc_df = pd.DataFrame(headers_desc_rows)
    headers_desc_table = headers_desc_df.to_html(classes='data-table', index=False, escape=False)
    headers_desc_table_escaped = json.dumps(headers_desc_table).replace('</script>', '<\\/script>')
    
    # Build complete HTML with tabs
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Wavelength Game Consistency Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .tabs {{
            display: flex;
            border-bottom: 2px solid #ddd;
            margin: 20px 0;
        }}
        .tab {{
            padding: 10px 20px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 16px;
            color: #666;
            border-bottom: 3px solid transparent;
        }}
        .tab.active {{
            color: #4CAF50;
            border-bottom: 3px solid #4CAF50;
            font-weight: bold;
        }}
        .tab-content {{
            display: none;
            padding: 20px 0;
        }}
        .tab-content.active {{
            display: block;
        }}
        .metric-box {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }}
        .metric-label {{
            font-size: 14px;
            color: #666;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .filter-controls {{
            margin: 20px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }}
        .filter-controls label {{
            font-weight: bold;
            margin-right: 10px;
        }}
        .filter-controls select, .filter-controls input {{
            padding: 5px 10px;
            margin-right: 15px;
            font-size: 14px;
        }}
        .reasoning-cell {{
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            cursor: pointer;
            color: #0066cc;
        }}
        .reasoning-cell:hover {{
            text-decoration: underline;
        }}
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.5);
        }}
        .modal-content {{
            background-color: #fefefe;
            margin: 5% auto;
            padding: 20px;
            border: 1px solid #888;
            border-radius: 8px;
            width: 80%;
            max-width: 800px;
            max-height: 80vh;
            overflow-y: auto;
        }}
        .close {{
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }}
        .close:hover {{
            color: black;
        }}
        .checkbox-group {{
            display: inline-block;
            margin-right: 15px;
        }}
        .checkbox-group label {{
            margin-left: 5px;
            font-weight: normal;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Wavelength Game Consistency Analysis Report</h1>
        <p><strong>Agents:</strong> {', '.join(agent_ids[:10])}{'...' if len(agent_ids) > 10 else ''}</p>
        <p><strong>Attempts:</strong> {min(attempt_ids)}-{max(attempt_ids)}</p>
        <p><strong>Total Responses:</strong> {len(df_all)}</p>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('summary', event)">Summary</button>
            <button class="tab" onclick="showTab('question-level', event)">Question Level Consistency</button>
            <button class="tab" onclick="showTab('trajectories', event)">Trajectories</button>
            <button class="tab" onclick="showTab('reasoning', event)">Reasoning</button>
            <button class="tab" onclick="showTab('headers-desc', event)">Headers Description</button>
    </div>
        
        <div id="summary" class="tab-content active">
            <h2>Summary</h2>
            <div class="metric-box">
                <div class="metric-value">{mean_icc_2k:.3f}</div>
                <div class="metric-label">Mean ICC(2,k)</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{mean_icc_3k:.3f}</div>
                <div class="metric-label">Mean ICC(3,k)</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{mean_alpha:.3f}</div>
                <div class="metric-label">Mean Cronbach's α</div>
            </div>
            <div id="summary-chart-container"></div>
        </div>
        
        <div id="question-level" class="tab-content">
            <h2>Question Level Consistency</h2>
            <div class="filter-controls">
                <label for="question-view">View:</label>
                <select id="question-view" onchange="updateQuestionView()">
                    <option value="all">All (All Agent-Header Pairs)</option>
                    <option value="header-agg">Aggregated by Header</option>
                    <option value="agent-agg">Aggregated by Agent</option>
                </select>
            </div>
            <div id="question-all-view-container" style="display: block;"></div>
            <div id="question-header-agg-container" style="display: none;"></div>
            <div id="question-agent-agg-container" style="display: none;"></div>
        </div>
        
        <div id="trajectories" class="tab-content">
            <h2>Trajectories</h2>
            <div class="filter-controls">
                <label for="trajectory-view">View:</label>
                <select id="trajectory-view" onchange="updateTrajectoryView()">
                    <option value="all">All Agents</option>
                    <option value="single">Single Agent</option>
                </select>
                <select id="trajectory-agent1" style="display: none;" onchange="updateTrajectoryView()">
                    <option value="">Select Agent</option>
                </select>
                <select id="trajectory-header" onchange="updateTrajectoryView()">
                    <option value="all">All Headers</option>
                </select>
            </div>
            <div id="trajectory-charts-container"></div>
        </div>
        
        <div id="reasoning" class="tab-content">
            <h2>Reasoning</h2>
            <div class="filter-controls">
                <label for="reasoning-filter">Filter by:</label>
                <select id="reasoning-filter" onchange="updateReasoningFilter()">
                    <option value="header">By Header</option>
                    <option value="agent">By Agent</option>
                </select>
                <select id="reasoning-header-select" onchange="updateReasoningTable()">
                    <option value="">Select Header</option>
                </select>
                <select id="reasoning-agent-select" style="display: none;" onchange="updateReasoningTable()">
                    <option value="">Select Agent</option>
                </select>
                <div class="checkbox-group">
                    <label>Attempts:</label>
                    <input type="checkbox" id="attempt-1" checked onchange="updateReasoningTable()">
                    <label for="attempt-1">1</label>
                    <input type="checkbox" id="attempt-2" checked onchange="updateReasoningTable()">
                    <label for="attempt-2">2</label>
                    <input type="checkbox" id="attempt-3" checked onchange="updateReasoningTable()">
                    <label for="attempt-3">3</label>
                </div>
            </div>
            <div id="reasoning-table-container"></div>
        </div>
        
        <div id="headers-desc" class="tab-content">
            <h2>Headers Description</h2>
            <div id="headers-desc-table-container"></div>
        </div>
    </div>
    
    <div id="reasoning-modal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal()">&times;</span>
            <h3 id="modal-title"></h3>
            <p id="modal-text" style="white-space: pre-wrap;"></p>
        </div>
    </div>
    
    <script>
        // Data
        const reasoningData = {reasoning_data_json};
        const trajectoryData = {trajectory_data_json};
        const headerInfo = {header_info_json};
        const agentIds = {agent_ids_json};
        const attemptIds = {attempt_ids_json};
        const uniqueHeaders = {unique_headers_json};
        
        // Tab switching
        function showTab(tabName, event) {{
            const tabs = document.querySelectorAll('.tab');
            const contents = document.querySelectorAll('.tab-content');
            
            tabs.forEach(tab => tab.classList.remove('active'));
            contents.forEach(content => content.classList.remove('active'));
            
            if (event && event.target) {{
                event.target.classList.add('active');
            }}
            document.getElementById(tabName).classList.add('active');
            
            // Initialize views when tabs are shown
            if (tabName === 'trajectories') {{
                updateTrajectoryView();
            }} else if (tabName === 'reasoning') {{
                updateReasoningTable();
            }}
        }}
        
        // Question level view switching
        function updateQuestionView() {{
            const view = document.getElementById('question-view').value;
            document.getElementById('question-all-view-container').style.display = 
                view === 'all' ? 'block' : 'none';
            document.getElementById('question-header-agg-container').style.display = 
                view === 'header-agg' ? 'block' : 'none';
            document.getElementById('question-agent-agg-container').style.display = 
                view === 'agent-agg' ? 'block' : 'none';
        }}
        
        // Trajectory view switching
        function updateTrajectoryView() {{
            const view = document.getElementById('trajectory-view').value;
            const agent1Select = document.getElementById('trajectory-agent1');
            
            agent1Select.style.display = view === 'single' ? 'inline-block' : 'none';
            
            // Populate agent select
            if (agent1Select && agent1Select.options.length === 1) {{
                agentIds.forEach(agentId => {{
                    const option1 = document.createElement('option');
                    option1.value = agentId;
                    option1.textContent = agentId;
                    agent1Select.appendChild(option1);
                }});
            }}
            
            // Populate header select
            const headerSelect = document.getElementById('trajectory-header');
            if (headerSelect && headerSelect.options.length === 1) {{
                uniqueHeaders.forEach(headerId => {{
                    const option = document.createElement('option');
                    option.value = headerId;
                    option.textContent = headerInfo[headerId].cue;
                    headerSelect.appendChild(option);
                }});
            }}
            
            // Generate charts
            const container = document.getElementById('trajectory-charts-container');
            container.innerHTML = '';
            
            const selectedHeader = headerSelect.value;
            const headersToShow = selectedHeader === 'all' ? uniqueHeaders : [selectedHeader];
            
            if (view === 'all') {{
                // Show all agents for selected headers
                headersToShow.forEach(headerId => {{
                    const div = createTrajectoryChart(headerId, agentIds, 'All Agents');
                    if (div) container.appendChild(div);
                }});
            }} else if (view === 'single') {{
                const agent1 = agent1Select.value;
                if (agent1) {{
                    headersToShow.forEach(headerId => {{
                        const div = createTrajectoryChart(headerId, [agent1], `Agent ${{agent1}}`);
                        if (div) container.appendChild(div);
                    }});
                }}
            }}
        }}
        
        function createTrajectoryChart(headerId, agents, title) {{
            const traces = [];
            const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'];
            
            agents.forEach((agentId, idx) => {{
                const key = `${{agentId}}_${{headerId}}`;
                if (trajectoryData[key]) {{
                    const data = trajectoryData[key];
                    traces.push({{
                        x: data.attempts,
                        y: data.responses,
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: `Agent ${{agentId}}`,
                        line: {{color: colors[idx % colors.length], width: 2}},
                        marker: {{size: 8}}
                    }});
                }}
            }});
            
            if (traces.length === 0) return null;
            
            const layout = {{
                title: `${{title}} - ${{headerInfo[headerId].cue}}`,
                xaxis: {{title: 'Attempt ID'}},
                yaxis: {{title: 'Response (0-100)', range: [0, 100]}},
                height: 400,
                margin: {{l: 50, r: 50, t: 50, b: 50}}
            }};
            
            const div = document.createElement('div');
            div.style.marginBottom = '30px';
            Plotly.newPlot(div, traces, layout);
            return div;
        }}
        
        // Reasoning filter switching
        function updateReasoningFilter() {{
            const filter = document.getElementById('reasoning-filter').value;
            const headerSelect = document.getElementById('reasoning-header-select');
            const agentSelect = document.getElementById('reasoning-agent-select');
            
            headerSelect.style.display = filter === 'header' ? 'inline-block' : 'none';
            agentSelect.style.display = filter === 'agent' ? 'inline-block' : 'none';
            
            // Populate selects
            if (headerSelect.options.length === 1) {{
                uniqueHeaders.forEach(headerId => {{
                    const option = document.createElement('option');
                    option.value = headerId;
                    option.textContent = headerInfo[headerId].cue;
                    headerSelect.appendChild(option);
                }});
            }}
            
            if (agentSelect.options.length === 1) {{
                agentIds.forEach(agentId => {{
                    const option = document.createElement('option');
                    option.value = agentId;
                    option.textContent = agentId;
                    agentSelect.appendChild(option);
                }});
            }}
            
            updateReasoningTable();
        }}
        
        function updateReasoningTable() {{
            const filter = document.getElementById('reasoning-filter').value;
            const container = document.getElementById('reasoning-table-container');
            
            // Get selected attempts
            const selectedAttempts = [];
            if (document.getElementById('attempt-1').checked) selectedAttempts.push(1);
            if (document.getElementById('attempt-2').checked) selectedAttempts.push(2);
            if (document.getElementById('attempt-3').checked) selectedAttempts.push(3);
            
            if (selectedAttempts.length === 0) {{
                container.innerHTML = '<p>Please select at least one attempt.</p>';
                return;
            }}
            
            let table = '<table class="data-table"><thead><tr>';
            
            if (filter === 'header') {{
                const headerId = document.getElementById('reasoning-header-select').value;
                if (!headerId) {{
                    container.innerHTML = '<p>Please select a header.</p>';
                    return;
                }}
                
                table += '<th>Agent ID</th><th>Agent Name</th>';
                selectedAttempts.forEach(att => {{
                    table += `<th>Reasoning (Attempt ${{att}})</th>`;
                }});
                table += '</tr></thead><tbody>';
                
                agentIds.forEach(agentId => {{
                    table += `<tr><td>${{agentId}}</td>`;
                    const firstKey = `${{agentId}}_${{headerId}}_${{selectedAttempts[0]}}`;
                    const agentName = reasoningData[firstKey]?.agent_name || 'Unknown';
                    table += `<td>${{agentName}}</td>`;
                    
                    selectedAttempts.forEach(att => {{
                        const key = `${{agentId}}_${{headerId}}_${{att}}`;
                        const reasoning = reasoningData[key]?.reasoning || '';
                        const truncated = reasoning.length > 100 ? reasoning.substring(0, 100) + '...' : reasoning;
                        const escaped = truncated.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
                        table += `<td class="reasoning-cell" onclick="showReasoningModal('${{key}}', '${{agentId}}', '${{headerInfo[headerId].cue}}', ${{att}})">${{escaped}}</td>`;
                    }});
                    table += '</tr>';
                }});
            }} else {{
                const agentId = document.getElementById('reasoning-agent-select').value;
                if (!agentId) {{
                    container.innerHTML = '<p>Please select an agent.</p>';
                    return;
                }}
                
                table += '<th>Header ID</th><th>Cue</th>';
                selectedAttempts.forEach(att => {{
                    table += `<th>Reasoning (Attempt ${{att}})</th>`;
                }});
                table += '</tr></thead><tbody>';
                
                uniqueHeaders.forEach(headerId => {{
                    table += `<tr><td>${{headerId}}</td><td>${{headerInfo[headerId].cue}}</td>`;
                    
                    selectedAttempts.forEach(att => {{
                        const key = `${{agentId}}_${{headerId}}_${{att}}`;
                        const reasoning = reasoningData[key]?.reasoning || '';
                        const truncated = reasoning.length > 100 ? reasoning.substring(0, 100) + '...' : reasoning;
                        const escaped = truncated.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
                        table += `<td class="reasoning-cell" onclick="showReasoningModal('${{key}}', '${{agentId}}', '${{headerInfo[headerId].cue}}', ${{att}})">${{escaped}}</td>`;
                    }});
                    table += '</tr>';
                }});
            }}
            
            table += '</tbody></table>';
            container.innerHTML = table;
        }}
        
        function showReasoningModal(key, agentId, cue, attempt) {{
            const reasoning = reasoningData[key]?.reasoning || 'No reasoning available';
            document.getElementById('modal-title').textContent = 
                `Agent ${{agentId}} - ${{cue}} (Attempt ${{attempt}})`;
            document.getElementById('modal-text').textContent = reasoning;
            document.getElementById('reasoning-modal').style.display = 'block';
        }}
        
        function closeModal() {{
            document.getElementById('reasoning-modal').style.display = 'none';
        }}
        
        // Close modal when clicking outside
        window.onclick = function(event) {{
            const modal = document.getElementById('reasoning-modal');
            if (event.target === modal) {{
                closeModal();
            }}
        }}
        
        // Initialize
        document.getElementById('summary-chart-container').innerHTML = {summary_chart_html_escaped};
        document.getElementById('question-all-view-container').innerHTML = {all_view_div_escaped};
        document.getElementById('question-header-agg-container').innerHTML = {header_agg_div_escaped};
        document.getElementById('question-agent-agg-container').innerHTML = {agent_agg_div_escaped};
        document.getElementById('headers-desc-table-container').innerHTML = {headers_desc_table_escaped};
    </script>
    {''.join(all_scripts)}
</body>
</html>"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report saved to: {output_path}")


def generate_basic_html_report(results: Dict, df_all: pd.DataFrame, output_path: str,
                               agent_ids: List[str], attempt_ids: List[int], df_reasoning: pd.DataFrame = None):
    """Generate basic HTML report without Plotly."""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Wavelength Game Consistency Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <h1>Wavelength Game Consistency Analysis Report</h1>
    <p><strong>Agents:</strong> {', '.join(agent_ids)}</p>
    <p><strong>Attempts:</strong> {min(attempt_ids)}-{max(attempt_ids)}</p>
    
    <h2>Header-Level Results</h2>
    {results['header_level'].to_html(index=False, na_rep='N/A')}
    
    <h2>Agent-Level Results</h2>
    {results['agent_level'].to_html(index=False, na_rep='N/A')}
    
    <h2>Header Aggregated Results</h2>
    {results['header_aggregated'].to_html(index=False, na_rep='N/A')}
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Basic HTML report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze consistency of Wavelength game responses across multiple attempts"
    )
    
    parser.add_argument(
        '--base',
        type=str,
        required=True,
        help='Base path to agent bank directory'
    )
    
    parser.add_argument(
        '--agents',
        nargs='+',
        help='Specific agent IDs to analyze'
    )
    
    parser.add_argument(
        '--range',
        type=str,
        help='Range of agent IDs (e.g., 0000-0005)'
    )
    
    parser.add_argument(
        '--attempts',
        type=str,
        default='1-10',
        help='Attempt range (e.g., 1-10) or comma-separated list (default: 1-10)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='wavelength_consistency_report.html',
        help='Output HTML file path (default: wavelength_consistency_report.html)'
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='Also save results as CSV files (specify directory)'
    )
    
    args = parser.parse_args()
    
    # Parse agent IDs
    if args.range:
        agent_ids = parse_range(args.range)
    elif args.agents:
        agent_ids = args.agents
    else:
        print("Error: Must specify either --agents or --range")
        sys.exit(1)
    
    # Parse attempt IDs
    attempt_ids = parse_attempt_range(args.attempts)
    
    # Run analysis
    try:
        results, df_all, df_reasoning = analyze_consistency(
            args.base, agent_ids, attempt_ids
        )
        
        # Generate HTML report
        generate_html_report(results, df_all, args.output, agent_ids, attempt_ids, df_reasoning)
        
        # Save CSV if requested
        if args.csv:
            os.makedirs(args.csv, exist_ok=True)
            results['header_level'].to_csv(f"{args.csv}/header_level.csv", index=False)
            results['agent_level'].to_csv(f"{args.csv}/agent_level.csv", index=False)
            results['header_aggregated'].to_csv(f"{args.csv}/header_aggregated.csv", index=False)
            print(f"CSV files saved to: {args.csv}")
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()