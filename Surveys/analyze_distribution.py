#!/usr/bin/env python3
"""
Analyze distribution of survey responses across agents.

This script computes distribution metrics (mean, SD, min, max, range, CV) for each
agent-question/score pair, then analyzes the distribution of these metrics across agents.
It generates visualizations (box plots, violin plots, scatter plots, histograms, coverage heatmap)
and detects edge cases (lack of diversity, unused values, outliers).
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


def load_survey_responses(agent_folder: str, attempt_ids: List[int]) -> Dict:
    """Load survey responses for an agent for specified attempts."""
    responses_path = Path(agent_folder) / "survey_responses.json"
    
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


def load_distilled_responses(agent_folder: str, attempt_ids: List[int]) -> Dict:
    """Load distilled survey responses for an agent for specified attempts."""
    distilled_path = Path(agent_folder) / "survey_distilled_responses.json"
    
    if not distilled_path.exists():
        return None
    
    try:
        with open(distilled_path, 'r', encoding='utf-8') as f:
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
        print(f"Warning: Error loading {distilled_path}: {e}")
        return None


def load_survey_questions() -> Dict:
    """Load survey questions structure."""
    questions_path = Path(__file__).parent / "survey_questions.json"
    if not questions_path.exists():
        return {}
    
    with open(questions_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_agent_question_metrics(responses: List[float]) -> Dict[str, float]:
    """Compute distribution metrics for an agent's responses to a question across attempts."""
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


def extract_agent_question_metrics(agent_data: Dict, attempt_ids: List[int]) -> pd.DataFrame:
    """Extract per-agent, per-question metrics from raw responses."""
    rows = []
    
    # Group responses by agent and question
    agent_id = agent_data["agent_id"]
    agent_name = agent_data["agent_name"]
    
    question_responses = {}  # {(section_id, question_id): [responses]}
    
    for attempt in agent_data["attempts"]:
        attempt_id = attempt["attempt_id"]
        if attempt_id not in attempt_ids:
            continue
        
        for section_id, section_data in attempt.get("sections", {}).items():
            for question_id, response_value in section_data.get("responses", {}).items():
                key = (section_id, question_id)
                if key not in question_responses:
                    question_responses[key] = []
                if response_value is not None:
                    question_responses[key].append(float(response_value))
    
    # Compute metrics for each question
    for (section_id, question_id), responses in question_responses.items():
        metrics = compute_agent_question_metrics(responses)
        rows.append({
            "agent_id": agent_id,
            "agent_name": agent_name,
            "section_id": section_id,
            "question_id": question_id,
            **metrics
        })
    
    return pd.DataFrame(rows)


def extract_agent_score_metrics(agent_data: Dict, attempt_ids: List[int]) -> pd.DataFrame:
    """Extract per-agent, per-score metrics from distilled responses."""
    rows = []
    
    agent_id = agent_data["agent_id"]
    agent_name = agent_data["agent_name"]
    
    score_responses = {}  # {(section_id, score_name): [responses]}
    
    for attempt in agent_data["attempts"]:
        attempt_id = attempt["attempt_id"]
        if attempt_id not in attempt_ids:
            continue
        
        for section_id, section_data in attempt.get("sections", {}).items():
            if section_id == "BFI-10":
                trait_scores = section_data.get("trait_scores", {})
                for trait_name, trait_value in trait_scores.items():
                    if trait_value is not None:
                        key = (section_id, trait_name)
                        if key not in score_responses:
                            score_responses[key] = []
                        score_responses[key].append(float(trait_value))
            elif section_id == "BES-A":
                normalized_score = section_data.get("normalized_score")
                if normalized_score is not None:
                    key = (section_id, "CE_normalized")
                    if key not in score_responses:
                        score_responses[key] = []
                    score_responses[key].append(float(normalized_score))
            elif section_id == "REI":
                normalized_score = section_data.get("normalized_score")
                if normalized_score is not None:
                    key = (section_id, "RA_normalized")
                    if key not in score_responses:
                        score_responses[key] = []
                    score_responses[key].append(float(normalized_score))
    
    # Compute metrics for each score
    for (section_id, score_name), responses in score_responses.items():
        metrics = compute_agent_question_metrics(responses)
        rows.append({
            "agent_id": agent_id,
            "agent_name": agent_name,
            "section_id": section_id,
            "score_name": score_name,
            **metrics
        })
    
    return pd.DataFrame(rows)


def compute_distribution_stats(df: pd.DataFrame, group_col: str, value_col: str = "mean") -> pd.DataFrame:
    """Compute distribution statistics across agents for each question/score."""
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
            # "q25": np.percentile(group_data, 25),
            # "q75": np.percentile(group_data, 75),
            # "iqr": np.percentile(group_data, 75) - np.percentile(group_data, 25),
            # "skewness": stats.skew(group_data) if len(group_data) > 2 else np.nan,
            # "kurtosis": stats.kurtosis(group_data) if len(group_data) > 2 else np.nan
        })
    
    return pd.DataFrame(stats_list)


def compute_coverage(df: pd.DataFrame, question_col: str, response_col: str = "response") -> pd.DataFrame:
    """Compute coverage: percentage of agents who used each Likert value (1-5) per question."""
    coverage_rows = []
    
    for question_id in df[question_col].unique():
        question_data = df[df[question_col] == question_id]
        
        # Get all responses for this question across all agents and attempts
        all_responses = question_data[response_col].dropna().astype(int)
        
        total_count = len(all_responses)
        if total_count == 0:
            continue
        
        # Count occurrences of each value (1-5)
        value_counts = {}
        for val in range(1, 6):
            count = len(all_responses[all_responses == val])
            percentage = (count / total_count * 100) if total_count > 0 else 0
            value_counts[val] = {
                "count": count,
                "percentage": percentage
            }
        
        coverage_rows.append({
            question_col: question_id,
            "value_1_count": value_counts[1]["count"],
            "value_1_pct": value_counts[1]["percentage"],
            "value_2_count": value_counts[2]["count"],
            "value_2_pct": value_counts[2]["percentage"],
            "value_3_count": value_counts[3]["count"],
            "value_3_pct": value_counts[3]["percentage"],
            "value_4_count": value_counts[4]["count"],
            "value_4_pct": value_counts[4]["percentage"],
            "value_5_count": value_counts[5]["count"],
            "value_5_pct": value_counts[5]["percentage"],
            "total_responses": total_count
        })
    
    return pd.DataFrame(coverage_rows)


def detect_edge_cases(df_metrics: pd.DataFrame, df_stats: pd.DataFrame, 
                     df_coverage: Optional[pd.DataFrame] = None,
                     group_col: str = "question_id") -> pd.DataFrame:
    """Detect edge cases: lack of diversity, unused values, outliers."""
    edge_cases = []
    
    for group_val in df_stats[group_col].unique():
        stats_row = df_stats[df_stats[group_col] == group_val].iloc[0]
        metrics_data = df_metrics[df_metrics[group_col] == group_val]
        
        issues = []
        severity = "low"
        
        # Check 1: Lack of diversity (low SD across agents)
        if stats_row["std"] < 0.5:
            issues.append(f"Low diversity: SD={stats_row['std']:.2f} (agents cluster tightly)")
            severity = "medium"
        
        # Check 2: Extreme clustering (mean near 1 or 5 with low SD)
        if stats_row["mean"] < 1.5 and stats_row["std"] < 0.5:
            issues.append(f"Extreme clustering at low end: mean={stats_row['mean']:.2f}, SD={stats_row['std']:.2f}")
            severity = "high"
        elif stats_row["mean"] > 4.5 and stats_row["std"] < 0.5:
            issues.append(f"Extreme clustering at high end: mean={stats_row['mean']:.2f}, SD={stats_row['std']:.2f}")
            severity = "high"
        
        # Check 3: Unused values (from coverage data)
        if df_coverage is not None:
            coverage_row = df_coverage[df_coverage[group_col] == group_val]
            if not coverage_row.empty:
                row = coverage_row.iloc[0]
                unused_values = []
                if row["value_1_pct"] < 1.0:
                    unused_values.append("1")
                if row["value_2_pct"] < 1.0:
                    unused_values.append("2")
                if row["value_3_pct"] < 1.0:
                    unused_values.append("3")
                if row["value_4_pct"] < 1.0:
                    unused_values.append("4")
                if row["value_5_pct"] < 1.0:
                    unused_values.append("5")
                
                if len(unused_values) >= 3:
                    issues.append(f"Limited scale usage: {len(unused_values)} values rarely used ({', '.join(unused_values)})")
                    severity = "medium" if severity == "low" else severity
        
        # Check 4: Outliers (agents with very different patterns)
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


def analyze_raw_distribution(agent_folders: List[Path], attempt_ids: Optional[List[int]], 
                            base_path: Path) -> Dict[str, Any]:
    """Analyze distribution of raw question responses."""
    print("Loading raw survey responses...")
    all_metrics = []
    all_responses = []  # For coverage analysis
    
    for agent_folder in agent_folders:
        # If attempt_ids is None, detect from this agent's data
        if attempt_ids is None:
            responses_path = agent_folder / "survey_responses.json"
            if responses_path.exists():
                try:
                    with open(responses_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    attempt_ids = sorted(set([a.get("attempt_id") for a in data.get("attempts", [])]))
                except:
                    continue
        
        agent_data = load_survey_responses(str(agent_folder), attempt_ids)
        if agent_data is None:
            continue
        
        # Extract metrics
        metrics_df = extract_agent_question_metrics(agent_data, attempt_ids)
        all_metrics.append(metrics_df)
        
        # Extract raw responses for coverage
        for attempt in agent_data["attempts"]:
            if attempt["attempt_id"] not in attempt_ids:
                continue
            for section_id, section_data in attempt.get("sections", {}).items():
                for question_id, response_value in section_data.get("responses", {}).items():
                    if response_value is not None:
                        all_responses.append({
                            "agent_id": agent_data["agent_id"],
                            "question_id": question_id,
                            "section_id": section_id,
                            "response": response_value
                        })
    
    if not all_metrics:
        return None
    
    df_metrics = pd.concat(all_metrics, ignore_index=True)
    df_responses = pd.DataFrame(all_responses)
    
    print(f"Analyzing distribution for {len(df_metrics['agent_id'].unique())} agents and {len(df_metrics['question_id'].unique())} questions...")
    
    # Compute distribution statistics per question
    question_stats = compute_distribution_stats(df_metrics, "question_id", "mean")
    
    # Compute distribution statistics per section
    section_stats = compute_distribution_stats(df_metrics, "section_id", "mean")
    
    # Compute coverage
    df_coverage = compute_coverage(df_responses, "question_id", "response") if not df_responses.empty else None
    
    # Detect edge cases
    edge_cases = detect_edge_cases(df_metrics, question_stats, df_coverage, "question_id")
    
    return {
        "metrics": df_metrics,
        "question_stats": question_stats,
        "section_stats": section_stats,
        "coverage": df_coverage,
        "edge_cases": edge_cases,
        "responses": df_responses
    }


def analyze_distilled_distribution(agent_folders: List[Path], attempt_ids: Optional[List[int]],
                                  base_path: Path) -> Dict[str, Any]:
    """Analyze distribution of distilled scores."""
    print("Loading distilled survey responses...")
    all_metrics = []
    
    for agent_folder in agent_folders:
        # If attempt_ids is None, detect from this agent's data
        if attempt_ids is None:
            distilled_path = agent_folder / "survey_distilled_responses.json"
            if distilled_path.exists():
                try:
                    with open(distilled_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    attempt_ids = sorted(set([a.get("attempt_id") for a in data.get("attempts", [])]))
                except:
                    continue
        
        agent_data = load_distilled_responses(str(agent_folder), attempt_ids)
        if agent_data is None:
            continue
        
        metrics_df = extract_agent_score_metrics(agent_data, attempt_ids)
        all_metrics.append(metrics_df)
    
    if not all_metrics:
        return None
    
    df_metrics = pd.concat(all_metrics, ignore_index=True)
    
    print(f"Analyzing distribution for {len(df_metrics['agent_id'].unique())} agents and {len(df_metrics['score_name'].unique())} scores...")
    
    # Compute distribution statistics per score
    score_stats = compute_distribution_stats(df_metrics, "score_name", "mean")
    
    # Compute distribution statistics per section
    section_stats = compute_distribution_stats(df_metrics, "section_id", "mean")
    
    # Detect edge cases
    edge_cases = detect_edge_cases(df_metrics, score_stats, None, "score_name")
    
    return {
        "metrics": df_metrics,
        "score_stats": score_stats,
        "section_stats": section_stats,
        "edge_cases": edge_cases
    }


def generate_html_report(raw_results: Optional[Dict], distilled_results: Optional[Dict],
                        output_path: Path, attempt_ids: Optional[List[int]]) -> None:
    """Generate interactive HTML report."""
    print("Generating HTML report...")
    
    # Load survey questions for labels
    survey_data = load_survey_questions()
    question_labels = {}
    if survey_data:
        for section in survey_data.get("sections", []):
            for q in section.get("questions", []):
                question_labels[q["question_id"]] = q["question_text"]
    
    # Format timestamp
    timestamp_str = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    attempts_str = ', '.join(map(str, sorted(attempt_ids))) if attempt_ids else 'All available'
    
    # Build conditional HTML parts - merged tab structure
    question_level_tab_button = '<button class="tab-button" onclick="showTab(\'question-level\', event)">Question Level Distribution</button>' if (raw_results or distilled_results) else ''
    
    # Start building HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Survey Distribution Analysis Report</title>
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
        <h1>📊 Survey Distribution Analysis Report</h1>
        <p><strong>Analysis Date:</strong> {timestamp_str}</p>
        <p><strong>Attempts Analyzed:</strong> {attempts_str}</p>
        
        <div class="tab-container">
            <div class="tab-buttons">
                <button class="tab-button active" onclick="showTab('summary', event)">Summary</button>
                {question_level_tab_button}
            </div>
            
            <div id="summary" class="tab-content active">
                <h2>Summary Statistics</h2>
                <div id="summary-stats"></div>
                <div id="summary-edge-cases"></div>
            </div>
            
            <div id="question-level" class="tab-content">
                <h2>Question Level Distribution</h2>
                <div class="filter-controls" style="margin: 20px 0; padding: 15px; background-color: #f9f9f9; border-radius: 5px;">
                    <label for="data-type" style="font-weight: bold; margin-right: 10px;">Data Type:</label>
                    <select id="data-type" onchange="updateQuestionLevelView()" style="padding: 5px 10px; font-size: 14px;">
                        <option value="raw">Raw Responses</option>""" + ("""
                        <option value="distilled">Distilled Scores</option>""" if distilled_results is not None else "") + """
                    </select>
                </div>
                
                <div id="question-view-raw" class="question-view" style="display: block;">
                    <div id="raw-content"></div>
                </div>
                """ + ("""
                <div id="question-view-distilled" class="question-view" style="display: none;">
                    <div id="distilled-content"></div>
                </div>
                """ if distilled_results is not None else "") + """
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
        
        function updateQuestionLevelView() {{
            const dataType = document.getElementById('data-type').value;
            const rawView = document.getElementById('question-view-raw');
            const distilledView = document.getElementById('question-view-distilled');
            
            if (rawView) rawView.style.display = (dataType === 'raw') ? 'block' : 'none';
            if (distilledView) distilledView.style.display = (dataType === 'distilled') ? 'block' : 'none';
        }}
    </script>
</body>
</html>"""
    
    # Generate visualizations and populate content
    if PLOTLY_AVAILABLE:
        html_content = add_visualizations(html_content, raw_results, distilled_results, question_labels)
    else:
        html_content = add_basic_tables(html_content, raw_results, distilled_results)
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Report saved to: {output_path}")


def add_visualizations(html_content: str, raw_results: Optional[Dict], 
                      distilled_results: Optional[Dict], question_labels: Dict) -> str:
    """Add Plotly visualizations to HTML."""
    import re
    
    # Use the extract_div_and_script function defined below
    
    # Build summary statistics
    summary_html = '<div class="summary-stats">'
    
    if raw_results:
        n_agents = len(raw_results['metrics']['agent_id'].unique())
        n_questions = len(raw_results['question_stats'])
        n_edge_cases = len(raw_results['edge_cases'])
        summary_html += f'''
            <div class="stat-card">
                <div class="stat-value">{n_agents}</div>
                <div class="stat-label">Agents (Raw)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{n_questions}</div>
                <div class="stat-label">Questions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{n_edge_cases}</div>
                <div class="stat-label">Edge Cases Detected</div>
            </div>
        '''
    
    if distilled_results:
        n_agents_dist = len(distilled_results['metrics']['agent_id'].unique())
        n_scores = len(distilled_results['score_stats'])
        summary_html += f'''
            <div class="stat-card">
                <div class="stat-value">{n_agents_dist}</div>
                <div class="stat-label">Agents (Distilled)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{n_scores}</div>
                <div class="stat-label">Scores</div>
            </div>
        '''
    
    summary_html += '</div>'
    
    # Add edge cases table
    edge_cases_html = ""
    if raw_results and not raw_results['edge_cases'].empty:
        edge_cases_html = "<h3>Edge Cases - Raw Questions</h3>"
        edge_cases_html += raw_results['edge_cases'].to_html(classes='data-table', index=False, escape=False)
    
    if distilled_results and not distilled_results['edge_cases'].empty:
        edge_cases_html += "<h3>Edge Cases - Distilled Scores</h3>"
        edge_cases_html += distilled_results['edge_cases'].to_html(classes='data-table', index=False, escape=False)
    
    html_content = html_content.replace('<div id="summary-stats"></div>', f'<div id="summary-stats">{summary_html}</div>')
    html_content = html_content.replace('<div id="summary-edge-cases"></div>', f'<div id="summary-edge-cases">{edge_cases_html}</div>')
    
    # Generate raw question visualizations
    if raw_results:
        raw_html = generate_raw_visualizations(raw_results, question_labels)
        html_content = html_content.replace('<div id="raw-content"></div>', f'<div id="raw-content">{raw_html}</div>')
    
    # Generate distilled score visualizations
    if distilled_results:
        distilled_html = generate_distilled_visualizations(distilled_results)
        html_content = html_content.replace('<div id="distilled-content"></div>', f'<div id="distilled-content">{distilled_html}</div>')
    
    return html_content


def generate_raw_visualizations(raw_results: Dict, question_labels: Dict) -> str:
    """Generate all visualizations for raw questions."""
    df_metrics = raw_results['metrics']
    question_stats = raw_results['question_stats']
    df_coverage = raw_results.get('coverage')
    
    html_parts = []
    
    # 1. Box plots per question
    questions = sorted(df_metrics['question_id'].unique())
    
    # Prepare data for box plot - group by question
    box_data = []
    box_labels = []
    
    for question_id in questions:
        question_data = df_metrics[df_metrics['question_id'] == question_id]['mean'].dropna()
        if len(question_data) > 0:
            box_data.append(question_data.tolist())
            box_labels.append(question_id)  # Use question_id instead of full text for cleaner display
    
    if box_data:
        fig_box = go.Figure()
        for i, (data, label) in enumerate(zip(box_data, box_labels)):
            fig_box.add_trace(go.Box(
                y=data,
                name=label,
                boxpoints='outliers'
            ))
        
        fig_box.update_layout(
            title='Box Plot: Distribution of Agents\' Mean Scores per Question',
            xaxis_title='Question',
            yaxis_title='Mean Score (across attempts)',
            yaxis=dict(range=[0.5, 5.5]),
            height=600,
            showlegend=False
        )
    else:
        fig_box = go.Figure()
        fig_box.add_annotation(text="No data available", showarrow=False)
    # Embed the full HTML output directly (includes both div and script)
    box_html = fig_box.to_html(full_html=False, include_plotlyjs=False, div_id='raw-box-plot')
    html_parts.append(f'<h3>Box Plots</h3>{box_html}')
    
    # 2. Violin plots per question
    # For violin plots with multiple groups, we need to use x-axis positioning
    fig_violin = go.Figure()
    
    for idx, question_id in enumerate(questions):
        question_data = df_metrics[df_metrics['question_id'] == question_id]['mean'].dropna()
        if len(question_data) > 0:
            # Use x positions to separate violins
            fig_violin.add_trace(go.Violin(
                x=[question_id] * len(question_data),  # x-axis grouping
                y=question_data.tolist(),
                name=question_id,
                box_visible=True,
                meanline_visible=True,
                side='positive',
                width=0.6,
                showlegend=False
            ))
    
    if len(fig_violin.data) > 0:
        fig_violin.update_layout(
            title='Violin Plot: Density Distribution of Agents\' Mean Scores per Question',
            xaxis_title='Question',
            yaxis_title='Mean Score (across attempts)',
            yaxis=dict(range=[0.5, 5.5]),
            height=600,
            showlegend=False
        )
    else:
        fig_violin.add_annotation(text="No data available", showarrow=False)
        fig_violin.update_layout(height=600)
    violin_html = fig_violin.to_html(full_html=False, include_plotlyjs=False, div_id='raw-violin-plot')
    html_parts.append(f'<h3>Violin Plots</h3>{violin_html}')
    
    # 3. Scatter plot: Mean vs SD with section filtering
    # Get question to section mapping
    question_to_section = {}
    for question_id in questions:
        question_rows = df_metrics[df_metrics['question_id'] == question_id]
        if len(question_rows) > 0:
            question_to_section[question_id] = question_rows['section_id'].iloc[0]
    
    # Organize questions by section
    bfi10_questions = [q for q in questions if question_to_section.get(q) == 'BFI-10']
    besa_questions = [q for q in questions if question_to_section.get(q) == 'BES-A']
    rei_questions = [q for q in questions if question_to_section.get(q) == 'REI']
    
    fig_scatter = go.Figure()
    all_traces = []
    bfi10_traces = []
    besa_traces = []
    rei_traces = []
    
    for question_id in questions:
        question_data = df_metrics[df_metrics['question_id'] == question_id]
        question_data = question_data[question_data['mean'].notna() & question_data['sd'].notna()]
        if len(question_data) > 0:
            # Convert to lists for Plotly
            trace = go.Scatter(
                x=question_data['mean'].tolist(),
                y=question_data['sd'].tolist(),
                mode='markers',
                name=question_id,
                text=question_data['agent_id'].tolist(),
                hovertemplate='Agent: %{text}<br>Question: ' + question_id + '<br>Mean: %{x:.2f}<br>SD: %{y:.2f}<extra></extra>'
            )
            fig_scatter.add_trace(trace)
            trace_idx = len(fig_scatter.data) - 1
            all_traces.append(trace_idx)
            if question_id in bfi10_questions:
                bfi10_traces.append(trace_idx)
            elif question_id in besa_questions:
                besa_traces.append(trace_idx)
            elif question_id in rei_questions:
                rei_traces.append(trace_idx)
    
    # Create visibility lists
    total_traces = len(fig_scatter.data)
    all_visible = [True] * total_traces
    bfi10_visible = [False] * total_traces
    besa_visible = [False] * total_traces
    rei_visible = [False] * total_traces
    for idx in bfi10_traces:
        bfi10_visible[idx] = True
    for idx in besa_traces:
        besa_visible[idx] = True
    for idx in rei_traces:
        rei_visible[idx] = True
    
    # Create buttons for filtering
    buttons = [
        dict(
            label="All Questions",
            method="update",
            args=[
                {"visible": all_visible}
            ]
        ),
        dict(
            label="BFI-10 Only",
            method="update",
            args=[
                {"visible": bfi10_visible}
            ]
        ),
        dict(
            label="BES-A Only",
            method="update",
            args=[
                {"visible": besa_visible}
            ]
        ),
        dict(
            label="REI Only",
            method="update",
            args=[
                {"visible": rei_visible}
            ]
        )
    ]
    
    if len(fig_scatter.data) > 0:
        fig_scatter.update_layout(
            title='Scatter Plot: Mean Score vs Standard Deviation (per Agent-Question)',
            xaxis_title='Mean Score',
            yaxis_title='Standard Deviation',
            xaxis=dict(range=[0.5, 5.5]),
            height=500,
            hovermode='closest',
            updatemenus=[dict(
                type="dropdown",
                direction="down",
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top",
                buttons=buttons
            )]
        )
    else:
        fig_scatter.add_annotation(text="No data available", showarrow=False)
        fig_scatter.update_layout(height=500)
    scatter_html = fig_scatter.to_html(full_html=False, include_plotlyjs=False, div_id='raw-scatter-plot')
    html_parts.append(f'<h3>Mean vs Standard Deviation</h3><p>Use dropdown to filter by instrument: All Questions, BFI-10, BES-A, or REI</p>{scatter_html}')
    
    # 4. Histograms removed from raw questions - moved to distilled scores section
    
    # 5. Coverage heatmap with section filtering
    if df_coverage is not None and not df_coverage.empty:
        # Get question to section mapping from metrics
        question_to_section = {}
        for question_id in df_metrics['question_id'].unique():
            question_rows = df_metrics[df_metrics['question_id'] == question_id]
            if len(question_rows) > 0:
                question_to_section[question_id] = question_rows['section_id'].iloc[0]
        
        # Organize questions by section
        all_question_ids = sorted(df_coverage['question_id'].tolist())
        bfi10_questions = [q for q in all_question_ids if question_to_section.get(q) == 'BFI-10']
        besa_questions = [q for q in all_question_ids if question_to_section.get(q) == 'BES-A']
        rei_questions = [q for q in all_question_ids if question_to_section.get(q) == 'REI']
        
        # Build coverage matrices for each view
        def build_matrix(question_list):
            matrix = []
            for question_id in question_list:
                row = df_coverage[df_coverage['question_id'] == question_id].iloc[0]
                matrix.append([
                    row['value_1_pct'],
                    row['value_2_pct'],
                    row['value_3_pct'],
                    row['value_4_pct'],
                    row['value_5_pct']
                ])
            return matrix
        
        all_matrix = build_matrix(all_question_ids)
        bfi10_matrix = build_matrix(bfi10_questions)
        besa_matrix = build_matrix(besa_questions)
        rei_matrix = build_matrix(rei_questions)
        
        # Create figure with all data
        fig_coverage = go.Figure()
        
        # Add all traces (one per view)
        fig_coverage.add_trace(go.Heatmap(
            z=all_matrix,
            x=['1', '2', '3', '4', '5'],
            y=all_question_ids,
            colorscale='Blues',
            colorbar=dict(title="Percentage of Responses"),
            hovertemplate='Question: %{y}<br>Value: %{x}<br>Percentage: %{z:.1f}%<extra></extra>',
            visible=True
        ))
        
        fig_coverage.add_trace(go.Heatmap(
            z=bfi10_matrix,
            x=['1', '2', '3', '4', '5'],
            y=bfi10_questions,
            colorscale='Blues',
            colorbar=dict(title="Percentage of Responses"),
            hovertemplate='Question: %{y}<br>Value: %{x}<br>Percentage: %{z:.1f}%<extra></extra>',
            visible=False
        ))
        
        fig_coverage.add_trace(go.Heatmap(
            z=besa_matrix,
            x=['1', '2', '3', '4', '5'],
            y=besa_questions,
            colorscale='Blues',
            colorbar=dict(title="Percentage of Responses"),
            hovertemplate='Question: %{y}<br>Value: %{x}<br>Percentage: %{z:.1f}%<extra></extra>',
            visible=False
        ))
        
        fig_coverage.add_trace(go.Heatmap(
            z=rei_matrix,
            x=['1', '2', '3', '4', '5'],
            y=rei_questions,
            colorscale='Blues',
            colorbar=dict(title="Percentage of Responses"),
            hovertemplate='Question: %{y}<br>Value: %{x}<br>Percentage: %{z:.1f}%<extra></extra>',
            visible=False
        ))
        
        # Create buttons for filtering
        buttons = [
            dict(
                label="All Questions",
                method="update",
                args=[
                    {"visible": [True, False, False, False]},
                    {"yaxis": {"title": "Question"}}
                ]
            ),
            dict(
                label="BFI-10 Only",
                method="update",
                args=[
                    {"visible": [False, True, False, False]},
                    {"yaxis": {"title": "Question"}}
                ]
            ),
            dict(
                label="BES-A Only",
                method="update",
                args=[
                    {"visible": [False, False, True, False]},
                    {"yaxis": {"title": "Question"}}
                ]
            ),
            dict(
                label="REI Only",
                method="update",
                args=[
                    {"visible": [False, False, False, True]},
                    {"yaxis": {"title": "Question"}}
                ]
            )
        ]
        
        fig_coverage.update_layout(
            title='Coverage Heatmap: Percentage of Agents Using Each Likert Value (1-5)',
            xaxis_title='Likert Value',
            yaxis_title='Question',
            height=max(400, len(all_question_ids) * 15),
            updatemenus=[dict(
                type="dropdown",
                direction="down",
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top",
                buttons=buttons
            )]
        )
        
        coverage_html = fig_coverage.to_html(full_html=False, include_plotlyjs=False, div_id='raw-coverage-heatmap')
        html_parts.append(f'<h3>Coverage Heatmap</h3><p>Use dropdown to filter by instrument: All Questions, BFI-10, BES-A, or REI</p>{coverage_html}')
    
    # 6. Summary statistics table
    stats_table = question_stats.to_html(classes='data-table', index=False, escape=False)
    html_parts.append(f'<h3>Summary Statistics per Question</h3>{stats_table}')
    
    # Section-level comparison removed
    
    # Combine all HTML (scripts are already embedded in each plot's HTML)
    combined_html = '\n'.join(html_parts)
    
    return combined_html


def order_distilled_scores(scores: List[str]) -> List[str]:
    """Order scores: BFI traits first, then REI and BES-A at the end."""
    bfi_traits = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
    ordered = []
    remaining = list(scores)
    
    # Add BFI traits in order
    for trait in bfi_traits:
        if trait in remaining:
            ordered.append(trait)
            remaining.remove(trait)
    
    # Add remaining scores (REI and BES-A) at the end
    # Sort them so REI comes before BES-A (alphabetically)
    remaining_sorted = sorted(remaining)
    ordered.extend(remaining_sorted)
    
    return ordered


def generate_distilled_visualizations(distilled_results: Dict) -> str:
    """Generate all visualizations for distilled scores."""
    df_metrics = distilled_results['metrics']
    score_stats = distilled_results['score_stats']
    
    html_parts = []
    
    # Order scores: BFI traits first, then REI and BES-A
    all_scores = list(df_metrics['score_name'].unique())
    scores = order_distilled_scores(all_scores)
    
    # 1. Box plots per score with filtering dropdown
    # Organize scores by section (same logic as violin plots)
    score_to_section = {}
    for score_name in scores:
        score_rows = df_metrics[df_metrics['score_name'] == score_name]
        if len(score_rows) > 0:
            score_to_section[score_name] = score_rows['section_id'].iloc[0]
    
    bfi10_scores = [s for s in scores if score_to_section.get(s) == 'BFI-10']
    other_scores = [s for s in scores if score_to_section.get(s) in ['BES-A', 'REI']]
    
    fig_box = go.Figure()
    all_traces = []
    bfi10_traces = []
    other_traces = []
    
    for score_name in scores:
        score_data = df_metrics[df_metrics['score_name'] == score_name]['mean'].dropna()
        if len(score_data) > 0:
            trace = go.Box(
                x=[score_name] * len(score_data),
                y=score_data.tolist(),
                name=score_name,
                boxpoints='outliers',
                showlegend=False
            )
            fig_box.add_trace(trace)
            trace_idx = len(fig_box.data) - 1
            all_traces.append(trace_idx)
            if score_name in bfi10_scores:
                bfi10_traces.append(trace_idx)
            elif score_name in other_scores:
                other_traces.append(trace_idx)
    
    # Create visibility lists
    total_traces = len(fig_box.data)
    all_visible = [True] * total_traces
    bfi10_visible = [False] * total_traces
    other_visible = [False] * total_traces
    for idx in bfi10_traces:
        bfi10_visible[idx] = True
    for idx in other_traces:
        other_visible[idx] = True
    
    # Create buttons for filtering
    buttons = [
        dict(
            label="All Scores",
            method="update",
            args=[
                {"visible": all_visible},
                {"yaxis": {"fixedrange": False}}
            ]
        ),
        dict(
            label="BFI-10 Only",
            method="update",
            args=[
                {"visible": bfi10_visible},
                {"yaxis": {"fixedrange": False}}
            ]
        ),
        dict(
            label="REI & BES-A Only",
            method="update",
            args=[
                {"visible": other_visible},
                {"yaxis": {"fixedrange": False}}
            ]
        )
    ]
    
    if len(fig_box.data) > 0:
        fig_box.update_layout(
            title='Box Plot: Distribution of Agents\' Mean Scores per Distilled Score',
            xaxis_title='Score',
            yaxis_title='Mean Score (across attempts)',
            height=500,
            showlegend=False,
            updatemenus=[dict(
                type="dropdown",
                direction="down",
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top",
                buttons=buttons
            )],
            yaxis=dict(fixedrange=False)  # Make y-axis interactive
        )
    else:
        fig_box.add_annotation(text="No data available", showarrow=False)
        fig_box.update_layout(height=500)
    box_html = fig_box.to_html(full_html=False, include_plotlyjs=False, div_id='distilled-box-plot')
    html_parts.append(f'<h3>Box Plots</h3><p>Use dropdown to filter: All Scores, BFI-10 Only, or REI & BES-A Only</p>{box_html}')
    
    # 2. Violin plots per score with filtering dropdown
    # Organize scores by section
    score_to_section = {}
    for score_name in scores:
        score_rows = df_metrics[df_metrics['score_name'] == score_name]
        if len(score_rows) > 0:
            score_to_section[score_name] = score_rows['section_id'].iloc[0]
    
    bfi10_scores = [s for s in scores if score_to_section.get(s) == 'BFI-10']
    other_scores = [s for s in scores if score_to_section.get(s) in ['BES-A', 'REI']]
    
    # Create violin plots - all traces in one figure
    fig_violin = go.Figure()
    all_traces = []
    bfi10_traces = []
    other_traces = []
    
    # Add all scores
    for score_name in scores:
        score_data = df_metrics[df_metrics['score_name'] == score_name]['mean'].dropna()
        if len(score_data) > 0:
            trace = go.Violin(
                x=[score_name] * len(score_data),
                y=score_data.tolist(),
                name=score_name,
                box_visible=True,
                meanline_visible=True,
                side='positive',
                width=0.6,
                showlegend=False
            )
            fig_violin.add_trace(trace)
            trace_idx = len(fig_violin.data) - 1
            all_traces.append(trace_idx)
            if score_name in bfi10_scores:
                bfi10_traces.append(trace_idx)
            elif score_name in other_scores:
                other_traces.append(trace_idx)
    
    # Create visibility lists
    total_traces = len(fig_violin.data)
    all_visible = [True] * total_traces
    bfi10_visible = [False] * total_traces
    other_visible = [False] * total_traces
    for idx in bfi10_traces:
        bfi10_visible[idx] = True
    for idx in other_traces:
        other_visible[idx] = True
    
    # Create buttons for filtering
    buttons = [
        dict(
            label="All Scores",
            method="update",
            args=[
                {"visible": all_visible},
                {"yaxis": {"fixedrange": False}}
            ]
        ),
        dict(
            label="BFI-10 Only",
            method="update",
            args=[
                {"visible": bfi10_visible},
                {"yaxis": {"fixedrange": False}}
            ]
        ),
        dict(
            label="REI & BES-A Only",
            method="update",
            args=[
                {"visible": other_visible},
                {"yaxis": {"fixedrange": False}}
            ]
        )
    ]
    
    fig_violin.update_layout(
        title='Violin Plot: Density Distribution of Agents\' Mean Scores per Distilled Score',
        xaxis_title='Score',
        yaxis_title='Mean Score (across attempts)',
        height=500,
        showlegend=False,
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.15,
            yanchor="top",
            buttons=buttons
        )],
        yaxis=dict(fixedrange=False)  # Make y-axis interactive (zoom/pan enabled)
    )
    
    violin_html = fig_violin.to_html(full_html=False, include_plotlyjs=False, div_id='distilled-violin-plot')
    html_parts.append(f'<h3>Violin Plots</h3><p>Use dropdown to filter: All Scores, BFI-10 Only, or REI & BES-A Only</p>{violin_html}')
    
    # 3. Scatter plot: Mean vs SD with section filtering
    # Organize scores by section (reuse score_to_section from violin plots section)
    score_to_section = {}
    for score_name in scores:
        score_rows = df_metrics[df_metrics['score_name'] == score_name]
        if len(score_rows) > 0:
            score_to_section[score_name] = score_rows['section_id'].iloc[0]
    
    bfi10_scores = [s for s in scores if score_to_section.get(s) == 'BFI-10']
    besa_scores = [s for s in scores if score_to_section.get(s) == 'BES-A']
    rei_scores = [s for s in scores if score_to_section.get(s) == 'REI']
    
    fig_scatter = go.Figure()
    all_traces = []
    bfi10_traces = []
    besa_traces = []
    rei_traces = []
    
    for score_name in scores:
        score_data = df_metrics[df_metrics['score_name'] == score_name]
        score_data = score_data[score_data['mean'].notna() & score_data['sd'].notna()]
        if len(score_data) > 0:
            # Convert to lists for Plotly
            trace = go.Scatter(
                x=score_data['mean'].tolist(),
                y=score_data['sd'].tolist(),
                mode='markers',
                name=score_name,
                text=score_data['agent_id'].tolist(),
                hovertemplate=f'Agent: %{{text}}<br>Score: {score_name}<br>Mean: %{{x:.2f}}<br>SD: %{{y:.2f}}<extra></extra>'
            )
            fig_scatter.add_trace(trace)
            trace_idx = len(fig_scatter.data) - 1
            all_traces.append(trace_idx)
            if score_name in bfi10_scores:
                bfi10_traces.append(trace_idx)
            elif score_name in besa_scores:
                besa_traces.append(trace_idx)
            elif score_name in rei_scores:
                rei_traces.append(trace_idx)
    
    # Create visibility lists
    total_traces = len(fig_scatter.data)
    all_visible = [True] * total_traces
    bfi10_visible = [False] * total_traces
    besa_visible = [False] * total_traces
    rei_visible = [False] * total_traces
    for idx in bfi10_traces:
        bfi10_visible[idx] = True
    for idx in besa_traces:
        besa_visible[idx] = True
    for idx in rei_traces:
        rei_visible[idx] = True
    
    # Create buttons for filtering
    buttons = [
        dict(
            label="All Scores",
            method="update",
            args=[
                {"visible": all_visible}
            ]
        ),
        dict(
            label="BFI-10 Only",
            method="update",
            args=[
                {"visible": bfi10_visible}
            ]
        ),
        dict(
            label="BES-A Only",
            method="update",
            args=[
                {"visible": besa_visible}
            ]
        ),
        dict(
            label="REI Only",
            method="update",
            args=[
                {"visible": rei_visible}
            ]
        )
    ]
    
    if len(fig_scatter.data) > 0:
        fig_scatter.update_layout(
            title='Scatter Plot: Mean Score vs Standard Deviation (per Agent-Score)',
            xaxis_title='Mean Score',
            yaxis_title='Standard Deviation',
            height=500,
            hovermode='closest',
            updatemenus=[dict(
                type="dropdown",
                direction="down",
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top",
                buttons=buttons
            )]
        )
    else:
        fig_scatter.add_annotation(text="No data available", showarrow=False)
        fig_scatter.update_layout(height=500)
    scatter_html = fig_scatter.to_html(full_html=False, include_plotlyjs=False, div_id='distilled-scatter-plot')
    html_parts.append(f'<h3>Mean vs Standard Deviation</h3><p>Use dropdown to filter by instrument: All Scores, BFI-10, BES-A, or REI</p>{scatter_html}')
    
    # 4. Histograms for all 7 distilled scores (ordered: BFI traits first, then REI and BES-A)
    # Create subplots for all 7 scores (3 columns, 3 rows, last row has 1)
    fig_hist = make_subplots(
        rows=3, cols=3,
        subplot_titles=scores,  # All 7 scores in ordered format
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Use ordered scores for histogram
    for idx, score_name in enumerate(scores):
        row = (idx // 3) + 1
        col = (idx % 3) + 1
        score_data = df_metrics[df_metrics['score_name'] == score_name]['mean'].dropna()
        if len(score_data) > 0:
            fig_hist.add_trace(
                go.Histogram(x=score_data.tolist(), nbinsx=15, name=score_name, showlegend=False),
                row=row, col=col
            )
    
    if len(fig_hist.data) > 0:
        fig_hist.update_layout(
            title='Histogram: Distribution of Agents\' Mean Scores (All Distilled Scores)',
            height=800
        )
        fig_hist.update_xaxes(title_text="Mean Score", range=[0, 5])
        fig_hist.update_yaxes(title_text="Frequency")
    else:
        fig_hist.add_annotation(text="No data available", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
        fig_hist.update_layout(height=800)
    
    hist_html = fig_hist.to_html(full_html=False, include_plotlyjs=False, div_id='distilled-histogram')
    html_parts.append(f'<h3>Histograms (All Distilled Scores)</h3>{hist_html}')
    
    # 5. Summary statistics table
    stats_table = score_stats.to_html(classes='data-table', index=False, escape=False)
    html_parts.append(f'<h3>Summary Statistics per Score</h3>{stats_table}')
    
    # Section-level comparison removed
    
    # Combine all HTML (scripts are already embedded in each plot's HTML)
    combined_html = '\n'.join(html_parts)
    
    return combined_html


def extract_div_and_script(html_str: str) -> Tuple[str, str]:
    """Extract div and script from Plotly HTML output."""
    import re
    div_match = re.search(r'<div[^>]*id="[^"]*"[^>]*>.*?</div>', html_str, re.DOTALL)
    script_match = re.search(r'<script[^>]*>.*?</script>', html_str, re.DOTALL)
    div_content = div_match.group(0) if div_match else ""
    script_content = script_match.group(0) if script_match else ""
    return div_content, script_content


def add_basic_tables(html_content: str, raw_results: Optional[Dict],
                    distilled_results: Optional[Dict]) -> str:
    """Add basic HTML tables when Plotly is not available."""
    summary_html = "<h3>Distribution Analysis Complete</h3>"
    
    if raw_results:
        summary_html += f"<p>Raw Questions: {len(raw_results['question_stats'])} questions analyzed</p>"
        summary_html += f"<h4>Question Statistics</h4>"
        summary_html += raw_results['question_stats'].to_html(classes='data-table', index=False, escape=False)
    
    if distilled_results:
        summary_html += f"<p>Distilled Scores: {len(distilled_results['score_stats'])} scores analyzed</p>"
        summary_html += f"<h4>Score Statistics</h4>"
        summary_html += distilled_results['score_stats'].to_html(classes='data-table', index=False, escape=False)
    
    html_content = html_content.replace('<div id="summary-stats"></div>', f'<div id="summary-stats">{summary_html}</div>')
    
    return html_content


def add_basic_tables(html_content: str, raw_results: Optional[Dict],
                    distilled_results: Optional[Dict]) -> str:
    """Add basic HTML tables when Plotly is not available."""
    # Placeholder implementation
    return html_content


def main():
    parser = argparse.ArgumentParser(description="Analyze distribution of survey responses across agents")
    parser.add_argument("--base", type=str, default="agent_bank/populations/gss_agents",
                       help="Base path to agent folders")
    parser.add_argument("--agent", type=str, help="Single agent ID (e.g., '0000')")
    parser.add_argument("--range", type=str, help="Agent range (e.g., '0000-0049')")
    parser.add_argument("--attempts", type=str, help="Attempt range (e.g., '1-10') or comma-separated. Default: all available")
    parser.add_argument("--output", type=str, default="distribution_report.html",
                       help="Output HTML file path")
    parser.add_argument("--no-distilled", action="store_true",
                       help="Skip distilled score analysis")
    
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
            responses_path = agent_folder / "survey_responses.json"
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
            print("Warning: Could not detect available attempts. Using attempts 1-10 as default.")
            attempt_ids = list(range(1, 11))
    
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
    
    # Analyze raw distribution
    raw_results = analyze_raw_distribution(agent_folders, attempt_ids, base_path)
    
    # Analyze distilled distribution
    distilled_results = None
    if not args.no_distilled:
        distilled_results = analyze_distilled_distribution(agent_folders, attempt_ids, base_path)
    
    # Generate report
    output_path = Path(args.output)
    generate_html_report(raw_results, distilled_results, output_path, attempt_ids)
    
    print("Distribution analysis complete!")


if __name__ == "__main__":
    main()

