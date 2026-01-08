#!/usr/bin/env python3
"""
Analyze distribution of Wavelength game responses across agents.

This script computes distribution metrics (mean, SD, min, max, range, CV) for each
agent-header pair, then analyzes the distribution of these metrics across agents.
It generates visualizations (box plots, violin plots, scatter plots, histograms)
and detects edge cases (lack of diversity, outliers).
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

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. HTML report will be basic.")


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


def compute_agent_header_metrics(responses: List[float]) -> Dict[str, float]:
    """Compute distribution metrics for an agent's responses to a header across attempts."""
    if not responses:
        return {
            "mean": np.nan,
            "sd": np.nan,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
            "cv": np.nan,
            "n_attempts": 0
        }
    
    responses_array = np.array(responses)
    responses_array = responses_array[~np.isnan(responses_array)]  # Remove NaN
    
    if len(responses_array) == 0:
        return {
            "mean": np.nan,
            "sd": np.nan,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
            "cv": np.nan,
            "n_attempts": 0
        }
    
    mean_val = np.mean(responses_array)
    sd_val = np.std(responses_array, ddof=1) if len(responses_array) > 1 else 0.0
    min_val = np.min(responses_array)
    max_val = np.max(responses_array)
    range_val = max_val - min_val
    cv_val = (sd_val / mean_val * 100) if mean_val > 0 else np.nan
    
    return {
        "mean": mean_val,
        "sd": sd_val,
        "min": min_val,
        "max": max_val,
        "range": range_val,
        "cv": cv_val,
        "n_attempts": len(responses_array)
    }


def extract_agent_header_metrics(agent_data: Dict, attempt_ids: List[int]) -> pd.DataFrame:
    """Extract per-agent, per-header metrics from responses."""
    rows = []
    
    # Group responses by agent and header
    agent_id = agent_data["agent_id"]
    agent_name = agent_data["agent_name"]
    
    header_responses = {}  # {header_id: [responses]}
    
    for attempt in agent_data["attempts"]:
        attempt_id = attempt["attempt_id"]
        if attempt_id not in attempt_ids:
            continue
        
        headers = attempt.get("headers", {})
        for header_id, header_data in headers.items():
            response_value = header_data.get("response")
            if header_id not in header_responses:
                header_responses[header_id] = []
            if response_value is not None:
                header_responses[header_id].append(float(response_value))
    
    # Compute metrics for each header
    for header_id, responses in header_responses.items():
        metrics = compute_agent_header_metrics(responses)
        rows.append({
            "agent_id": agent_id,
            "agent_name": agent_name,
            "header_id": header_id,
            **metrics
        })
    
    return pd.DataFrame(rows)


def compute_distribution_stats(df: pd.DataFrame, group_col: str, value_col: str = "mean") -> pd.DataFrame:
    """Compute distribution statistics across agents for each header."""
    stats_list = []
    
    for group_val in df[group_col].unique():
        group_data = df[df[group_col] == group_val][value_col]
        group_data = group_data.dropna()
        
        if len(group_data) == 0:
            continue
        
        stats_list.append({
            group_col: group_val,
            "n_agents": len(group_data),
            "mean": np.mean(group_data),
            "median": np.median(group_data),
            "std": np.std(group_data, ddof=1),
            "cv": (np.std(group_data, ddof=1) / np.mean(group_data) * 100) if np.mean(group_data) > 0 else np.nan,
            "min": np.min(group_data),
            "max": np.max(group_data),
        })
    
    return pd.DataFrame(stats_list)


def detect_edge_cases(df_metrics: pd.DataFrame, df_stats: pd.DataFrame, 
                     group_col: str = "header_id") -> pd.DataFrame:
    """Detect edge cases: lack of diversity, outliers (adapted for 0-100 scale)."""
    edge_cases = []
    
    for group_val in df_stats[group_col].unique():
        stats_row = df_stats[df_stats[group_col] == group_val].iloc[0]
        metrics_data = df_metrics[df_metrics[group_col] == group_val]
        
        issues = []
        severity = "low"
        
        # Check 1: Lack of diversity (low SD across agents)
        # For 0-100 scale, SD < 10 indicates low diversity
        if stats_row["std"] < 10:
            issues.append(f"Low diversity: SD={stats_row['std']:.2f} (agents cluster tightly)")
            severity = "medium"
        
        # Check 2: Extreme clustering (mean near 0 or 100 with low SD)
        # For 0-100 scale, mean < 10 or > 90 with SD < 10 indicates extreme clustering
        if stats_row["mean"] < 10 and stats_row["std"] < 10:
            issues.append(f"Extreme clustering at low end: mean={stats_row['mean']:.2f}, SD={stats_row['std']:.2f}")
            severity = "high"
        elif stats_row["mean"] > 90 and stats_row["std"] < 10:
            issues.append(f"Extreme clustering at high end: mean={stats_row['mean']:.2f}, SD={stats_row['std']:.2f}")
            severity = "high"
        
        # Check 3: Outliers (agents with very different patterns)
        if len(metrics_data) > 2:
            mean_vals = metrics_data["mean"].dropna()
            if len(mean_vals) > 0:
                q1 = np.percentile(mean_vals, 25)
                q3 = np.percentile(mean_vals, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = metrics_data[
                    (metrics_data["mean"] < lower_bound) | (metrics_data["mean"] > upper_bound)
                ]
                
                if len(outliers) > 0:
                    outlier_agents = outliers["agent_id"].tolist()
                    issues.append(f"Outliers detected: {len(outliers)} agent(s) ({', '.join(outlier_agents[:5])})")
                    if len(outliers) > len(metrics_data) * 0.1:  # More than 10% are outliers
                        severity = "high"
        
        if issues:
            edge_cases.append({
                group_col: group_val,
                "severity": severity,
                "issues": "; ".join(issues),
                "n_issues": len(issues)
            })
    
    return pd.DataFrame(edge_cases)


def extract_header_index(header_id: str) -> int:
    """Extract numeric index from header_id for sorting (e.g., 'header_10' -> 10, 'header_2' -> 2)."""
    try:
        # Extract number after underscore or at the end
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


def analyze_header_distribution(agent_folders: List[Path], attempt_ids: Optional[List[int]], 
                               base_path: Path) -> Dict[str, Any]:
    """Analyze distribution of header responses."""
    print("Loading Wavelength game responses...")
    all_metrics = []
    header_cue_map = {}  # {header_id: cue_name}
    
    for agent_folder in agent_folders:
        # If attempt_ids is None, detect from this agent's data
        if attempt_ids is None:
            responses_path = agent_folder / "wavelength_responses.json"
            if responses_path.exists():
                try:
                    with open(responses_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    attempt_ids = sorted(set([a.get("attempt_id") for a in data.get("attempts", [])]))
                except:
                    continue
        
        agent_data = load_wavelength_responses(str(agent_folder), attempt_ids)
        if agent_data is None:
            continue
        
        # Extract metrics
        metrics_df = extract_agent_header_metrics(agent_data, attempt_ids)
        all_metrics.append(metrics_df)
        
        # Extract cue names (same for all attempts of a header)
        for attempt in agent_data["attempts"]:
            if attempt.get("attempt_id") not in attempt_ids:
                continue
            headers = attempt.get("headers", {})
            for header_id, header_data in headers.items():
                if header_id not in header_cue_map:
                    cue = header_data.get("cue", "")
                    header_cue_map[header_id] = cue
    
    if not all_metrics:
        return None
    
    df_metrics = pd.concat(all_metrics, ignore_index=True)
    
    print(f"Analyzing distribution for {len(df_metrics['agent_id'].unique())} agents and {len(df_metrics['header_id'].unique())} headers...")
    
    # Compute distribution statistics per header
    header_stats = compute_distribution_stats(df_metrics, "header_id", "mean")
    
    # Detect edge cases
    edge_cases = detect_edge_cases(df_metrics, header_stats, "header_id")
    
    return {
        "metrics": df_metrics,
        "header_stats": header_stats,
        "edge_cases": edge_cases,
        "header_cue_map": header_cue_map
    }


def generate_html_report(results: Optional[Dict], output_path: Path, attempt_ids: Optional[List[int]]) -> None:
    """Generate interactive HTML report."""
    print("Generating HTML report...")
    
    # Format timestamp
    timestamp_str = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    attempts_str = ', '.join(map(str, sorted(attempt_ids))) if attempt_ids else 'All available'
    
    # Start building HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wavelength Game Distribution Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        h3 {{
            color: #555;
            margin-top: 20px;
        }}
        .tab-container {{
            margin-top: 20px;
        }}
        .tab-buttons {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #ddd;
        }}
        .tab-button {{
            padding: 10px 20px;
            background-color: #ecf0f1;
            border: none;
            cursor: pointer;
            border-radius: 5px 5px 0 0;
            font-size: 14px;
            font-weight: 500;
        }}
        .tab-button.active {{
            background-color: #3498db;
            color: white;
        }}
        .tab-content {{
            display: none;
            padding: 20px 0;
        }}
        .tab-content.active {{
            display: block;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 12px;
        }}
        .data-table th, .data-table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        .data-table th {{
            background-color: #3498db;
            color: white;
            font-weight: 600;
        }}
        .data-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .edge-case-high {{
            background-color: #ffebee;
            border-left: 4px solid #f44336;
        }}
        .edge-case-medium {{
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
        }}
        .edge-case-low {{
            background-color: #f1f8e9;
            border-left: 4px solid #8bc34a;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }}
        .stat-label {{
            font-size: 12px;
            color: #7f8c8d;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Wavelength Game Distribution Analysis Report</h1>
        <p><strong>Analysis Date:</strong> {timestamp_str}</p>
        <p><strong>Attempts Analyzed:</strong> {attempts_str}</p>
        
        <div class="tab-container">
            <div class="tab-buttons">
                <button class="tab-button active" onclick="showTab('summary', event)">Summary</button>
                <button class="tab-button" onclick="showTab('header-level', event)">Header Level Distribution</button>
            </div>
            
            <div id="summary" class="tab-content active">
                <h2>Summary Statistics</h2>
                <div id="summary-stats"></div>
                <div id="summary-edge-cases"></div>
            </div>
            
            <div id="header-level" class="tab-content">
                <h2>Header Level Distribution</h2>
                <div id="header-content"></div>
            </div>
        </div>
    </div>
    
    <script>
        function showTab(tabName, event) {{
            // Hide all tabs
            const tabs = document.querySelectorAll('.tab-content');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            const buttons = document.querySelectorAll('.tab-button');
            buttons.forEach(btn => btn.classList.remove('active'));
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            if (event && event.target) {{
                event.target.classList.add('active');
            }}
        }}
    </script>
</body>
</html>"""
    
    # Generate visualizations and populate content
    if PLOTLY_AVAILABLE:
        html_content = add_visualizations(html_content, results)
    else:
        html_content = add_basic_tables(html_content, results)
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Report saved to: {output_path}")


def add_visualizations(html_content: str, results: Optional[Dict]) -> str:
    """Add Plotly visualizations to HTML."""
    import re
    
    # Build summary statistics
    summary_html = '<div class="summary-stats">'
    
    if results:
        n_agents = len(results['metrics']['agent_id'].unique())
        n_headers = len(results['header_stats'])
        n_edge_cases = len(results['edge_cases'])
        summary_html += f'''
            <div class="stat-card">
                <div class="stat-value">{n_agents}</div>
                <div class="stat-label">Agents</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{n_headers}</div>
                <div class="stat-label">Headers</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{n_edge_cases}</div>
                <div class="stat-label">Edge Cases Detected</div>
            </div>
        '''
    
    summary_html += '</div>'
    
    # Add edge cases table
    edge_cases_html = ""
    if results and not results['edge_cases'].empty:
        edge_cases_html = "<h3>Edge Cases</h3>"
        edge_cases_html += results['edge_cases'].to_html(classes='data-table', index=False, escape=False)
    
    html_content = html_content.replace('<div id="summary-stats"></div>', f'<div id="summary-stats">{summary_html}</div>')
    html_content = html_content.replace('<div id="summary-edge-cases"></div>', f'<div id="summary-edge-cases">{edge_cases_html}</div>')
    
    # Generate header visualizations
    if results:
        header_html = generate_header_visualizations(results)
        html_content = html_content.replace('<div id="header-content"></div>', f'<div id="header-content">{header_html}</div>')
    
    return html_content


def generate_header_visualizations(results: Dict) -> str:
    """Generate all visualizations for headers."""
    df_metrics = results['metrics']
    header_stats = results['header_stats']
    header_cue_map = results.get('header_cue_map', {})
    
    html_parts = []
    
    # 1. Box plots per header
    # Sort headers by numeric index
    headers = sorted(df_metrics['header_id'].unique(), key=extract_header_index)
    
    # Prepare data for box plot - group by header
    box_data = []
    box_labels = []
    
    for header_id in headers:
        header_data = df_metrics[df_metrics['header_id'] == header_id]['mean'].dropna()
        if len(header_data) > 0:
            box_data.append(header_data.tolist())
            box_labels.append(header_id)
    
    if box_data:
        fig_box = go.Figure()
        for i, (data, label) in enumerate(zip(box_data, box_labels)):
            fig_box.add_trace(go.Box(
                y=data,
                name=label,
                boxpoints='outliers'
            ))
        
        fig_box.update_layout(
            title='Box Plot: Distribution of Agents\' Mean Scores per Header',
            xaxis_title='Header',
            yaxis_title='Mean Score (across attempts)',
            yaxis=dict(range=[0, 100]),
            height=600,
            showlegend=False
        )
    else:
        fig_box = go.Figure()
        fig_box.add_annotation(text="No data available", showarrow=False)
    box_html = fig_box.to_html(full_html=False, include_plotlyjs=False, div_id='header-box-plot')
    html_parts.append(f'<h3>Box Plots</h3>{box_html}')
    
    # 2. Violin plots per header
    fig_violin = go.Figure()
    
    for idx, header_id in enumerate(headers):
        header_data = df_metrics[df_metrics['header_id'] == header_id]['mean'].dropna()
        if len(header_data) > 0:
            fig_violin.add_trace(go.Violin(
                x=[header_id] * len(header_data),
                y=header_data.tolist(),
                name=header_id,
                box_visible=True,
                meanline_visible=True,
                side='positive',
                width=0.6,
                showlegend=False
            ))
    
    if len(fig_violin.data) > 0:
        fig_violin.update_layout(
            title='Violin Plot: Density Distribution of Agents\' Mean Scores per Header',
            xaxis_title='Header',
            yaxis_title='Mean Score (across attempts)',
            yaxis=dict(range=[0, 100]),
            height=600,
            showlegend=False
        )
    else:
        fig_violin.add_annotation(text="No data available", showarrow=False)
        fig_violin.update_layout(height=600)
    violin_html = fig_violin.to_html(full_html=False, include_plotlyjs=False, div_id='header-violin-plot')
    html_parts.append(f'<h3>Violin Plots</h3>{violin_html}')
    
    # 3. Scatter plot: Mean vs SD
    fig_scatter = go.Figure()
    
    for header_id in headers:
        header_data = df_metrics[df_metrics['header_id'] == header_id]
        header_data = header_data[header_data['mean'].notna() & header_data['sd'].notna()]
        if len(header_data) > 0:
            trace = go.Scatter(
                x=header_data['mean'].tolist(),
                y=header_data['sd'].tolist(),
                mode='markers',
                name=header_id,
                text=header_data['agent_id'].tolist(),
                hovertemplate='Agent: %{text}<br>Header: ' + header_id + '<br>Mean: %{x:.2f}<br>SD: %{y:.2f}<extra></extra>'
            )
            fig_scatter.add_trace(trace)
    
    if len(fig_scatter.data) > 0:
        fig_scatter.update_layout(
            title='Scatter Plot: Mean Score vs Standard Deviation (per Agent-Header)',
            xaxis_title='Mean Score',
            yaxis_title='Standard Deviation',
            xaxis=dict(range=[0, 100]),
            height=500,
            hovermode='closest'
        )
    else:
        fig_scatter.add_annotation(text="No data available", showarrow=False)
        fig_scatter.update_layout(height=500)
    scatter_html = fig_scatter.to_html(full_html=False, include_plotlyjs=False, div_id='header-scatter-plot')
    html_parts.append(f'<h3>Mean vs Standard Deviation</h3>{scatter_html}')
    
    # 4. Histograms for all headers
    # Create subplots for headers (3 columns, calculate rows needed)
    # Sort headers by numeric index (already sorted above)
    n_headers = len(headers)
    n_cols = 3
    n_rows = (n_headers + n_cols - 1) // n_cols  # Ceiling division
    
    # Create subplot titles with header_id and cue name
    subplot_titles = []
    for header_id in headers:
        title = header_id
        # cue = header_cue_map.get(header_id, "")
        # if cue:
        #     title = f"{header_id}: {cue}"
        # else:
        #     title = header_id
        subplot_titles.append(title)
    
    fig_hist = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    for idx, header_id in enumerate(headers):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1
        header_data = df_metrics[df_metrics['header_id'] == header_id]['mean'].dropna()
        if len(header_data) > 0:
            fig_hist.add_trace(
                go.Histogram(x=header_data.tolist(), nbinsx=20, name=header_id, showlegend=False),
                row=row, col=col
            )
    
    if len(fig_hist.data) > 0:
        fig_hist.update_layout(
            title='Histogram: Distribution of Agents\' Mean Scores (All Headers)',
            height=max(400, n_rows * 250)
        )
        fig_hist.update_xaxes(title_text="Mean Score", range=[0, 100])
        fig_hist.update_yaxes(title_text="Frequency")
    else:
        fig_hist.add_annotation(text="No data available", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
        fig_hist.update_layout(height=max(400, n_rows * 250))
    
    hist_html = fig_hist.to_html(full_html=False, include_plotlyjs=False, div_id='header-histogram')
    html_parts.append(f'<h3>Histograms (All Headers)</h3>{hist_html}')
    
    # 5. Summary statistics table
    stats_table = header_stats.to_html(classes='data-table', index=False, escape=False)
    html_parts.append(f'<h3>Summary Statistics per Header</h3>{stats_table}')
    
    # Combine all HTML
    combined_html = '\n'.join(html_parts)
    
    return combined_html


def add_basic_tables(html_content: str, results: Optional[Dict]) -> str:
    """Add basic HTML tables when Plotly is not available."""
    summary_html = "<h3>Distribution Analysis Complete</h3>"
    
    if results:
        summary_html += f"<p>Headers: {len(results['header_stats'])} headers analyzed</p>"
        summary_html += f"<h4>Header Statistics</h4>"
        summary_html += results['header_stats'].to_html(classes='data-table', index=False, escape=False)
    
    html_content = html_content.replace('<div id="summary-stats"></div>', f'<div id="summary-stats">{summary_html}</div>')
    
    return html_content


def main():
    parser = argparse.ArgumentParser(description="Analyze distribution of Wavelength game responses across agents")
    parser.add_argument("--base", type=str, default="agent_bank/populations/gss_agents",
                       help="Base path to agent folders")
    parser.add_argument("--agent", type=str, help="Single agent ID (e.g., '0000')")
    parser.add_argument("--range", type=str, help="Agent range (e.g., '0000-0049')")
    parser.add_argument("--attempts", type=str, help="Attempt range (e.g., '1-10') or comma-separated. Default: all available")
    parser.add_argument("--output", type=str, default="wavelength_distribution_report.html",
                       help="Output HTML file path")
    
    args = parser.parse_args()
    
    # Set base path first (needed for attempt detection)
    base_path = Path(args.base)
    
    # Determine agent IDs
    if args.agent:
        agent_ids = [args.agent]
    elif args.range:
        agent_ids = parse_range(args.range)
    else:
        parser.error("Must specify either --agent or --range")
    
    # Determine attempt IDs
    if args.attempts:
        attempt_ids = parse_attempt_range(args.attempts)
    else:
        # Default: detect all available attempts from first agent
        attempt_ids = None
        for agent_id in agent_ids:
            agent_folder = base_path / agent_id
            responses_path = agent_folder / "wavelength_responses.json"
            if responses_path.exists():
                try:
                    with open(responses_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    available_attempts = [a.get("attempt_id") for a in data.get("attempts", [])]
                    if available_attempts:
                        attempt_ids = sorted(set(available_attempts))
                        print(f"Detected {len(attempt_ids)} available attempts: {min(attempt_ids)}-{max(attempt_ids)}")
                        break
                except:
                    continue
        
        if attempt_ids is None:
            print("Warning: Could not detect available attempts. Using attempts 1-3 as default.")
            attempt_ids = list(range(1, 4))
    
    # Get agent folders
    agent_folders = []
    for agent_id in agent_ids:
        agent_folder = base_path / agent_id
        if (agent_folder / "scratch.json").exists():
            agent_folders.append(agent_folder)
        else:
            print(f"Warning: Agent {agent_id} not found, skipping")
    
    if not agent_folders:
        print("Error: No valid agent folders found")
        return
    
    # Analyze header distribution
    results = analyze_header_distribution(agent_folders, attempt_ids, base_path)
    
    if results is None:
        print("Error: No data found for analysis")
        return
    
    # Generate report
    output_path = Path(args.output)
    generate_html_report(results, output_path, attempt_ids)
    
    print("Distribution analysis complete!")


if __name__ == "__main__":
    main()

