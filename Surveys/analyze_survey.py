#!/usr/bin/env python3
"""
Analyze consistency of survey responses across multiple attempts.

This script computes various consistency metrics (ICC, Cronbach's Alpha, CV, SD, Range)
for survey responses across multiple attempts and generates an interactive HTML report.
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
from scipy.stats import chi2

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


def compute_stability_index(data: np.ndarray) -> float:
    """
    Compute Stability Index for single item across attempts (test-retest reliability).
    
    This is a variance-based approximation (not standard ICC) that measures consistency
    of responses across attempts for a single item.
    
    Args:
        data: 1D array of responses across attempts (for single item)
    
    Returns:
        Stability Index (0-1, where 1 = perfect consistency)
    """
    # Ensure 1D array
    if data.ndim > 1:
        data = data.flatten()
    
    n = len(data)
    if n < 2:
        return np.nan
    
    # Remove NaN values
    data = data[~np.isnan(data)]
    n = len(data)
    if n < 2:
        return np.nan
    
    mean_val = np.mean(data)
    var_val = np.var(data, ddof=1)
    
    if var_val == 0:
        # Perfect consistency
        return 1.0
    
    # For Likert scale (1-5), use scale-relative stability index
    # Stability Index = 1 - (variance / expected_variance_if_random)
    # Expected variance for uniform distribution on [1,5] = (5-1)^2/12 = 1.33
    expected_var_uniform = ((5 - 1) ** 2) / 12
    
    # Stability Index - absolute agreement (considers both systematic and random error)
    stability_index = 1 - (var_val / (var_val + expected_var_uniform))
    
    return max(0, min(1, stability_index))


def compute_standard_icc(data_matrix: np.ndarray) -> Tuple[float, float]:
    """
    Compute standard ICC(2,k) and ICC(3,k) using ANOVA-based formulas.
    
    This implements the standard ICC formulas for two-way random effects model:
    - ICC(2,k) = (MS_R - MS_E) / (MS_R + (MS_C - MS_E) / n)
    - ICC(3,k) = (MS_R - MS_E) / MS_R
    
    Where:
    - MS_R = Mean Square Rows (Agents)
    - MS_C = Mean Square Columns (Attempts)
    - MS_E = Mean Square Error
    - n = number of agents (rows)
    - k = number of attempts (columns)
    
    Args:
        data_matrix: 2D array where rows are agents and columns are attempts
                    Shape: (n_agents, n_attempts)
                    Missing values should be NaN
    
    Returns:
        Tuple of (ICC_2k, ICC_3k)
    """
    # Remove rows/columns that are all NaN
    valid_rows = ~np.isnan(data_matrix).all(axis=1)
    valid_cols = ~np.isnan(data_matrix).all(axis=0)
    
    if not valid_rows.any() or not valid_cols.any():
        return np.nan, np.nan
    
    data_matrix = data_matrix[valid_rows][:, valid_cols]
    
    n_agents, n_attempts = data_matrix.shape
    
    if n_agents < 2 or n_attempts < 2:
        return np.nan, np.nan
    
    # Compute grand mean
    grand_mean = np.nanmean(data_matrix)
    
    # Compute row means (agent means) and column means (attempt means)
    row_means = np.nanmean(data_matrix, axis=1)  # Mean across attempts for each agent
    col_means = np.nanmean(data_matrix, axis=0)  # Mean across agents for each attempt
    
    # Compute Sum of Squares
    # SS_Total = sum of (X_ij - grand_mean)^2
    ss_total = np.nansum((data_matrix - grand_mean) ** 2)
    
    # SS_Rows (Agents) = n_attempts * sum of (row_mean - grand_mean)^2
    # But need to account for missing values - use actual number of non-NaN values per row
    n_per_row = np.sum(~np.isnan(data_matrix), axis=1)
    ss_rows = np.sum(n_per_row * (row_means - grand_mean) ** 2)
    
    # SS_Columns (Attempts) = n_agents * sum of (col_mean - grand_mean)^2
    # But need to account for missing values - use actual number of non-NaN values per column
    n_per_col = np.sum(~np.isnan(data_matrix), axis=0)
    ss_cols = np.sum(n_per_col * (col_means - grand_mean) ** 2)
    
    # SS_Error = SS_Total - SS_Rows - SS_Columns
    ss_error = ss_total - ss_rows - ss_cols
    
    # Degrees of Freedom
    df_rows = n_agents - 1
    df_cols = n_attempts - 1
    # For error df, we need to account for missing values
    # In balanced case: df_error = (n_agents - 1) * (n_attempts - 1)
    # For unbalanced, we approximate using total valid cells
    n_valid = np.sum(~np.isnan(data_matrix))
    df_error = n_valid - n_agents - n_attempts + 1
    
    if df_error <= 0:
        df_error = 1  # Minimum df to avoid division by zero
    
    # Mean Squares
    ms_rows = ss_rows / df_rows if df_rows > 0 else 0
    ms_cols = ss_cols / df_cols if df_cols > 0 else 0
    ms_error = ss_error / df_error if df_error > 0 else 0
    
    # Compute ICC(2,k) and ICC(3,k)
    # ICC(2,k) = (MS_R - MS_E) / (MS_R + (MS_C - MS_E) / n)
    numerator_2k = ms_rows - ms_error
    denominator_2k = ms_rows + (ms_cols - ms_error) / n_agents
    
    if denominator_2k <= 0 or numerator_2k < 0:
        icc_2k = 0.0
    else:
        icc_2k = numerator_2k / denominator_2k
    
    # ICC(3,k) = (MS_R - MS_E) / MS_R
    if ms_rows <= 0 or numerator_2k < 0:
        icc_3k = 0.0
    else:
        icc_3k = numerator_2k / ms_rows
    
    # Ensure values are in [0, 1] range
    icc_2k = max(0, min(1, icc_2k))
    icc_3k = max(0, min(1, icc_3k))
    
    return icc_2k, icc_3k


def compute_cronbach_alpha(responses: List[float]) -> float:
    """
    Compute Cronbach's Alpha for single item across attempts (for individual agent-question pairs).
    
    For single item, Cronbach's Alpha is not directly applicable.
    We use a simplified measure: 1 - (variance / expected_variance)
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
    
    # For single item, use variance-based reliability
    # Higher variance = lower reliability
    # Expected variance for random responses on [1,5] scale
    expected_var = ((5 - 1) ** 2) / 12
    
    # Simplified alpha: 1 - (observed_var / expected_var)
    alpha = 1 - (var_val / expected_var)
    
    return max(0, min(1, alpha))


def compute_cronbach_alpha_question_level(data_matrix: np.ndarray) -> float:
    """
    Compute standard Cronbach's Alpha for a question across attempts (treating attempts as items).
    
    Standard formula: α = (k / (k-1)) × (1 - Σσⱼ² / σₜ²)
    Where:
    - k = number of attempts (items)
    - σⱼ² = variance of attempt j across all agents
    - σₜ² = variance of total scores (sum of all attempts for each agent)
    
    Args:
        data_matrix: 2D array where rows are agents and columns are attempts
                    Shape: (n_agents, n_attempts)
                    Missing values should be NaN
    
    Returns:
        Cronbach's Alpha (0-1, where 1 = perfect consistency)
    """
    # Remove rows/columns that are all NaN
    valid_rows = ~np.isnan(data_matrix).all(axis=1)
    valid_cols = ~np.isnan(data_matrix).all(axis=0)
    
    if not valid_rows.any() or not valid_cols.any():
        return np.nan
    
    data_matrix = data_matrix[valid_rows][:, valid_cols]
    n_agents, n_attempts = data_matrix.shape
    
    if n_agents < 2 or n_attempts < 2:
        return np.nan
    
    # Compute variance of each attempt (column) across agents
    attempt_vars = []
    for j in range(n_attempts):
        attempt_data = data_matrix[:, j]
        attempt_data_valid = attempt_data[~np.isnan(attempt_data)]
        if len(attempt_data_valid) > 1:
            attempt_vars.append(np.var(attempt_data_valid, ddof=1))
    
    if len(attempt_vars) == 0:
        return np.nan
    
    sum_attempt_vars = np.sum(attempt_vars)
    
    # Compute total scores (sum across attempts for each agent)
    total_scores = np.nansum(data_matrix, axis=1)
    total_scores_valid = total_scores[~np.isnan(total_scores)]
    
    if len(total_scores_valid) < 2:
        return np.nan
    
    var_total = np.var(total_scores_valid, ddof=1)
    
    if var_total == 0:
        return 1.0
    
    # Standard Cronbach's Alpha formula
    alpha = (n_attempts / (n_attempts - 1)) * (1 - sum_attempt_vars / var_total)
    
    return max(0, min(1, alpha))


def compute_consistency_metrics(responses: List[float]) -> Dict[str, float]:
    """Compute all consistency metrics for a list of responses across attempts."""
    if not responses or len(responses) < 2:
        return {
            "stability_index": np.nan,
            "cronbach_alpha": np.nan,
            "cv": np.nan,
            "sd": np.nan,
            "range": np.nan,
            "mean": np.nan,
            "n_attempts": len(responses) if responses else 0
        }
    
    responses_array = np.array(responses)
    responses_array = responses_array[~np.isnan(responses_array)]  # Remove NaN
    
    if len(responses_array) < 2:
        return {
            "stability_index": np.nan,
            "cronbach_alpha": np.nan,
            "cv": np.nan,
            "sd": np.nan,
            "range": np.nan,
            "mean": np.nan,
            "n_attempts": len(responses_array)
        }
    
    # Compute Stability Index (variance-based approximation for single item)
    stability_index = compute_stability_index(responses_array)
    
    # Compute Cronbach's Alpha
    cronbach_alpha = compute_cronbach_alpha(responses)
    
    # Compute descriptive stats
    mean_val = np.mean(responses_array)
    sd_val = np.std(responses_array, ddof=1)
    cv_val = (sd_val / mean_val * 100) if mean_val > 0 else np.nan
    range_val = np.max(responses_array) - np.min(responses_array)
    
    return {
        "stability_index": stability_index,
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
        
        for section_id, section_data in attempt.get("sections", {}).items():
            for question_id, response_value in section_data.get("responses", {}).items():
                rows.append({
                    "agent_id": agent_data["agent_id"],
                    "agent_name": agent_data["agent_name"],
                    "attempt_id": attempt_id,
                    "section_id": section_id,
                    "question_id": question_id,
                    "response": response_value if response_value is not None else np.nan
                })
    
    return pd.DataFrame(rows)


def extract_distilled_data(agent_data: Dict, attempt_ids: List[int]) -> pd.DataFrame:
    """Extract distilled scores into a DataFrame."""
    rows = []
    
    for attempt in agent_data["attempts"]:
        attempt_id = attempt["attempt_id"]
        if attempt_id not in attempt_ids:
            continue
        
        for section_id, section_data in attempt.get("sections", {}).items():
            if section_id == "BFI-10":
                # Extract trait scores
                trait_scores = section_data.get("trait_scores", {})
                for trait_name, trait_value in trait_scores.items():
                    rows.append({
                        "agent_id": agent_data["agent_id"],
                        "agent_name": agent_data["agent_name"],
                        "attempt_id": attempt_id,
                        "section_id": section_id,
                        "score_type": "trait",
                        "score_name": trait_name,
                        "score_value": trait_value if trait_value is not None else np.nan
                    })
            elif section_id == "BES-A":
                # Extract total score
                # total_score = section_data.get("total_score")
                normalized_score = section_data.get("normalized_score")
                rows.append({
                    "agent_id": agent_data["agent_id"],
                    "agent_name": agent_data["agent_name"],
                    "attempt_id": attempt_id,
                    "section_id": section_id,
                    "score_type": "total",
                    "score_name": "CE_normalized",  # Cognitive Empathy total
                    "score_value": normalized_score if normalized_score is not None else np.nan
                })
            elif section_id == "REI":
                # Extract total score
                # total_score = section_data.get("total_score")
                normalized_score = section_data.get("normalized_score")
                rows.append({
                    "agent_id": agent_data["agent_id"],
                    "agent_name": agent_data["agent_name"],
                    "attempt_id": attempt_id,
                    "section_id": section_id,
                    "score_type": "total",
                    "score_name": "RA_normalized",  # Rational Ability total
                    "score_value": normalized_score if normalized_score is not None else np.nan
                })
    
    return pd.DataFrame(rows)


# ============================================================================
# Distribution Analysis Functions (from analyze_distribution.py)
# ============================================================================

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
    print("Loading raw survey responses for distribution analysis...")
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
    print("Loading distilled survey responses for distribution analysis...")
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


# ============================================================================
# Distribution Visualization Functions
# ============================================================================

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


def generate_distribution_summary_html(raw_results: Optional[Dict], distilled_results: Optional[Dict]) -> str:
    """Generate summary HTML for distribution analysis."""
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
    
    return summary_html, edge_cases_html


def generate_raw_distribution_visualizations(raw_results: Dict, question_labels: Dict) -> str:
    """Generate all visualizations for raw questions (distribution analysis)."""
    # Import plotly here since it's needed
    try:
        import plotly.graph_objects as go
    except ImportError:
        return "<p>Plotly not available for visualizations.</p>"
    
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
            box_labels.append(question_id)
    
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
    box_html = fig_box.to_html(full_html=False, include_plotlyjs=False, div_id='raw-box-plot')
    html_parts.append(f'<h3>Box Plots</h3>{box_html}')
    
    # 2. Violin plots per question
    fig_violin = go.Figure()
    
    for idx, question_id in enumerate(questions):
        question_data = df_metrics[df_metrics['question_id'] == question_id]['mean'].dropna()
        if len(question_data) > 0:
            fig_violin.add_trace(go.Violin(
                x=[question_id] * len(question_data),
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
    question_to_section = {}
    for question_id in questions:
        question_rows = df_metrics[df_metrics['question_id'] == question_id]
        if len(question_rows) > 0:
            question_to_section[question_id] = question_rows['section_id'].iloc[0]
    
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
        dict(label="All Questions", method="update", args=[{"visible": all_visible}]),
        dict(label="BFI-10 Only", method="update", args=[{"visible": bfi10_visible}]),
        dict(label="BES-A Only", method="update", args=[{"visible": besa_visible}]),
        dict(label="REI Only", method="update", args=[{"visible": rei_visible}])
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
    
    # 4. Coverage heatmap with section filtering
    if df_coverage is not None and not df_coverage.empty:
        question_to_section = {}
        for question_id in df_metrics['question_id'].unique():
            question_rows = df_metrics[df_metrics['question_id'] == question_id]
            if len(question_rows) > 0:
                question_to_section[question_id] = question_rows['section_id'].iloc[0]
        
        all_question_ids = sorted(df_coverage['question_id'].tolist())
        bfi10_questions = [q for q in all_question_ids if question_to_section.get(q) == 'BFI-10']
        besa_questions = [q for q in all_question_ids if question_to_section.get(q) == 'BES-A']
        rei_questions = [q for q in all_question_ids if question_to_section.get(q) == 'REI']
        
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
        
        fig_coverage = go.Figure()
        
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
        
        buttons = [
            dict(label="All Questions", method="update", args=[{"visible": [True, False, False, False]}, {"yaxis": {"title": "Question"}}]),
            dict(label="BFI-10 Only", method="update", args=[{"visible": [False, True, False, False]}, {"yaxis": {"title": "Question"}}]),
            dict(label="BES-A Only", method="update", args=[{"visible": [False, False, True, False]}, {"yaxis": {"title": "Question"}}]),
            dict(label="REI Only", method="update", args=[{"visible": [False, False, False, True]}, {"yaxis": {"title": "Question"}}])
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
    
    # 5. Summary statistics table
    stats_table = question_stats.to_html(classes='data-table', index=False, escape=False)
    html_parts.append(f'<h3>Summary Statistics per Question</h3>{stats_table}')
    
    return '\n'.join(html_parts)


def generate_distilled_distribution_visualizations(distilled_results: Dict) -> str:
    """Generate all visualizations for distilled scores (distribution analysis)."""
    # Import plotly here since it's needed
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return "<p>Plotly not available for visualizations.</p>"
    
    df_metrics = distilled_results['metrics']
    score_stats = distilled_results['score_stats']
    
    html_parts = []
    
    # Order scores: BFI traits first, then REI and BES-A
    all_scores = list(df_metrics['score_name'].unique())
    scores = order_distilled_scores(all_scores)
    
    # 1. Box plots per score with filtering dropdown
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
    
    total_traces = len(fig_box.data)
    all_visible = [True] * total_traces
    bfi10_visible = [False] * total_traces
    other_visible = [False] * total_traces
    for idx in bfi10_traces:
        bfi10_visible[idx] = True
    for idx in other_traces:
        other_visible[idx] = True
    
    buttons = [
        dict(label="All Scores", method="update", args=[{"visible": all_visible}, {"yaxis": {"fixedrange": False}}]),
        dict(label="BFI-10 Only", method="update", args=[{"visible": bfi10_visible}, {"yaxis": {"fixedrange": False}}]),
        dict(label="REI & BES-A Only", method="update", args=[{"visible": other_visible}, {"yaxis": {"fixedrange": False}}])
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
            yaxis=dict(fixedrange=False)
        )
    else:
        fig_box.add_annotation(text="No data available", showarrow=False)
        fig_box.update_layout(height=500)
    box_html = fig_box.to_html(full_html=False, include_plotlyjs=False, div_id='distilled-box-plot')
    html_parts.append(f'<h3>Box Plots</h3><p>Use dropdown to filter: All Scores, BFI-10 Only, or REI & BES-A Only</p>{box_html}')
    
    # 2. Violin plots per score with filtering dropdown
    score_to_section = {}
    for score_name in scores:
        score_rows = df_metrics[df_metrics['score_name'] == score_name]
        if len(score_rows) > 0:
            score_to_section[score_name] = score_rows['section_id'].iloc[0]
    
    bfi10_scores = [s for s in scores if score_to_section.get(s) == 'BFI-10']
    other_scores = [s for s in scores if score_to_section.get(s) in ['BES-A', 'REI']]
    
    fig_violin = go.Figure()
    all_traces = []
    bfi10_traces = []
    other_traces = []
    
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
    
    total_traces = len(fig_violin.data)
    all_visible = [True] * total_traces
    bfi10_visible = [False] * total_traces
    other_visible = [False] * total_traces
    for idx in bfi10_traces:
        bfi10_visible[idx] = True
    for idx in other_traces:
        other_visible[idx] = True
    
    buttons = [
        dict(label="All Scores", method="update", args=[{"visible": all_visible}, {"yaxis": {"fixedrange": False}}]),
        dict(label="BFI-10 Only", method="update", args=[{"visible": bfi10_visible}, {"yaxis": {"fixedrange": False}}]),
        dict(label="REI & BES-A Only", method="update", args=[{"visible": other_visible}, {"yaxis": {"fixedrange": False}}])
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
        yaxis=dict(fixedrange=False)
    )
    
    violin_html = fig_violin.to_html(full_html=False, include_plotlyjs=False, div_id='distilled-violin-plot')
    html_parts.append(f'<h3>Violin Plots</h3><p>Use dropdown to filter: All Scores, BFI-10 Only, or REI & BES-A Only</p>{violin_html}')
    
    # 3. Scatter plot: Mean vs SD with section filtering
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
    
    buttons = [
        dict(label="All Scores", method="update", args=[{"visible": all_visible}]),
        dict(label="BFI-10 Only", method="update", args=[{"visible": bfi10_visible}]),
        dict(label="BES-A Only", method="update", args=[{"visible": besa_visible}]),
        dict(label="REI Only", method="update", args=[{"visible": rei_visible}])
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
    
    # 4. Histograms for all 7 distilled scores
    fig_hist = make_subplots(
        rows=3, cols=3,
        subplot_titles=scores,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
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
    
    return '\n'.join(html_parts)


def analyze_consistency(base_path: str, agent_ids: List[str], attempt_ids: List[int], 
                        include_distilled: bool = True) -> Tuple[Dict, pd.DataFrame, Optional[Dict], Optional[pd.DataFrame], Optional[Dict], Optional[Dict]]:
    """
    Main analysis function - runs both consistency and distribution analysis.
    
    Returns:
        Tuple of (raw_results, raw_df, distilled_results, distilled_df, raw_distribution_results, distilled_distribution_results)
        If distilled data is not available, distilled_results and distilled_df will be None
        Distribution results are also optional
    """
    print(f"Loading survey responses for {len(agent_ids)} agents, attempts {min(attempt_ids)}-{max(attempt_ids)}...")
    
    all_data = []
    all_distilled_data = []
    agent_folders = []
    has_distilled = False
    
    for agent_id in agent_ids:
        agent_folder = Path(base_path) / agent_id
        if not (agent_folder / "scratch.json").exists():
            print(f"Warning: Agent {agent_id} not found, skipping")
            continue
        
        # Load raw responses
        agent_data = load_survey_responses(str(agent_folder), attempt_ids)
        if agent_data is None:
            print(f"Warning: No survey data found for agent {agent_id}, skipping")
            continue
        
        df = extract_responses_data(agent_data, attempt_ids)
        if not df.empty:
            all_data.append(df)
            agent_folders.append(agent_folder)
        
        # Load distilled responses if requested
        if include_distilled:
            distilled_data = load_distilled_responses(str(agent_folder), attempt_ids)
            if distilled_data is not None:
                df_distilled = extract_distilled_data(distilled_data, attempt_ids)
                if not df_distilled.empty:
                    all_distilled_data.append(df_distilled)
                    has_distilled = True
    
    if not all_data:
        raise ValueError("No survey data found for the specified agents and attempts")
    
    # Combine all raw data
    df_all = pd.concat(all_data, ignore_index=True)
    
    print(f"Loaded {len(df_all)} raw response records")
    
    # Analyze raw responses (consistency)
    print(f"Analyzing raw response consistency metrics...")
    raw_results = analyze_raw_consistency(df_all, attempt_ids)
    
    # Analyze distilled responses if available (consistency)
    distilled_results = None
    df_distilled_all = None
    if has_distilled and all_distilled_data:
        df_distilled_all = pd.concat(all_distilled_data, ignore_index=True)
        print(f"Loaded {len(df_distilled_all)} distilled score records")
        print(f"Analyzing distilled score consistency metrics...")
        distilled_results = analyze_distilled_consistency(df_distilled_all, attempt_ids)
    
    # Analyze distribution (raw and distilled)
    print(f"Analyzing distribution metrics...")
    raw_distribution_results = analyze_raw_distribution(agent_folders, attempt_ids, Path(base_path))
    distilled_distribution_results = None
    if has_distilled:
        distilled_distribution_results = analyze_distilled_distribution(agent_folders, attempt_ids, Path(base_path))
    
    return raw_results, df_all, distilled_results, df_distilled_all, raw_distribution_results, distilled_distribution_results


def analyze_raw_consistency(df_all: pd.DataFrame, attempt_ids: List[int]) -> Dict:
    """Analyze consistency for raw responses."""
    results = {}
    
    # 1. Per-question consistency
    question_metrics = []
    for (agent_id, question_id), group in df_all.groupby(['agent_id', 'question_id']):
        responses = group['response'].dropna().tolist()
        metrics = compute_consistency_metrics(responses)
        # Cache first row to avoid multiple column accesses
        first_row = group.iloc[0]
        metrics.update({
            'agent_id': agent_id,
            'question_id': question_id,
            'section_id': first_row['section_id']
        })
        question_metrics.append(metrics)
    
    results['question_level'] = pd.DataFrame(question_metrics)
    
    # 2. Per-section consistency (aggregate across questions)
    section_metrics = []
    for (agent_id, section_id), group in df_all.groupby(['agent_id', 'section_id']):
        # Get all responses for this section across all attempts
        # More efficient: group by attempt_id first, then compute mean
        section_responses = []
        attempt_groups = group.groupby('attempt_id')['response']
        for attempt_id in attempt_ids:
            if attempt_id in attempt_groups.groups:
                attempt_responses = attempt_groups.get_group(attempt_id).dropna().tolist()
            if attempt_responses:
                section_responses.append(np.mean(attempt_responses))  # Mean across questions per attempt
        
        metrics = compute_consistency_metrics(section_responses)
        metrics.update({
            'agent_id': agent_id,
            'section_id': section_id
        })
        section_metrics.append(metrics)
    
    results['section_level'] = pd.DataFrame(section_metrics)
    
    # 3. Per-agent consistency (average Stability Index across questions for each agent)
    # Create agent name lookup once (more efficient)
    agent_name_map = df_all[['agent_id', 'agent_name']].drop_duplicates().set_index('agent_id')['agent_name'].to_dict() if 'agent_name' in df_all.columns else {}
    
    agent_metrics = []
    # Use groupby instead of filtering in a loop
    for agent_id, agent_questions in results['question_level'].groupby('agent_id'):
        if len(agent_questions) == 0:
            continue
        
        agent_name = agent_name_map.get(agent_id, 'Unknown')
        
        metrics = {
            'agent_id': agent_id,
            'agent_name': agent_name,
            'stability_index': agent_questions['stability_index'].mean(),
            'cronbach_alpha': agent_questions['cronbach_alpha'].mean(),
            'cv': agent_questions['cv'].mean(),
            'sd': agent_questions['sd'].mean(),
            'range': agent_questions['range'].mean(),
            'mean': agent_questions['mean'].mean(),
            'n_questions': len(agent_questions),
            'n_attempts': agent_questions['n_attempts'].iloc[0] if len(agent_questions) > 0 else 0
        }
        agent_metrics.append(metrics)
    
    results['agent_level'] = pd.DataFrame(agent_metrics)
    
    # 4. Overall summary by section
    # Use groupby instead of filtering in a loop
    overall_section = []
    for section_id, section_questions in results['question_level'].groupby('section_id'):
        overall_section.append({
            'section_id': section_id,
            'mean_stability_index': section_questions['stability_index'].mean(),
            'mean_cronbach_alpha': section_questions['cronbach_alpha'].mean(),
            'mean_cv': section_questions['cv'].mean(),
            'mean_sd': section_questions['sd'].mean(),
            'n_questions': len(section_questions)
        })
    
    results['overall_section'] = pd.DataFrame(overall_section)
    
    # 5. Aggregated question-level metrics (standard ICC across agents for each question)
    question_aggregated = []
    for question_id, question_data in df_all.groupby('question_id'):
        # Get all agents' responses for this question across all attempts
        # Create matrix: agents (rows) × attempts (columns)
        agent_ids = question_data['agent_id'].unique()
        attempt_ids_sorted = sorted(question_data['attempt_id'].unique())
        
        # Build data matrix
        data_matrix = []
        for agent_id in agent_ids:
            agent_responses = []
            for attempt_id in attempt_ids_sorted:
                response = question_data[(question_data['agent_id'] == agent_id) & 
                                        (question_data['attempt_id'] == attempt_id)]['response']
                if len(response) > 0:
                    val = response.iloc[0]
                    agent_responses.append(val if not pd.isna(val) else np.nan)
                else:
                    agent_responses.append(np.nan)
            data_matrix.append(agent_responses)
        
        data_matrix = np.array(data_matrix)
        
        # Compute standard ICC using ANOVA
        icc_2k, icc_3k = compute_standard_icc(data_matrix)
        
        # Compute standard Cronbach's Alpha at question level
        cronbach_alpha_question = compute_cronbach_alpha_question_level(data_matrix)
        
        # Compute MS_R, MS_C, MS_E for diagnostics
        # Remove rows/columns that are all NaN
        valid_rows = ~np.isnan(data_matrix).all(axis=1)
        valid_cols = ~np.isnan(data_matrix).all(axis=0)
        
        if valid_rows.any() and valid_cols.any():
            data_matrix_valid = data_matrix[valid_rows][:, valid_cols]
            n_agents_valid, n_attempts_valid = data_matrix_valid.shape
            
            if n_agents_valid >= 2 and n_attempts_valid >= 2:
                # Compute means
                grand_mean = np.nanmean(data_matrix_valid)
                row_means = np.nanmean(data_matrix_valid, axis=1)  # Agent means
                col_means = np.nanmean(data_matrix_valid, axis=0)  # Attempt means
                
                # Compute SS and MS for diagnostic
                n_per_row = np.sum(~np.isnan(data_matrix_valid), axis=1)
                ss_rows = np.sum(n_per_row * (row_means - grand_mean) ** 2)
                n_per_col = np.sum(~np.isnan(data_matrix_valid), axis=0)
                ss_cols = np.sum(n_per_col * (col_means - grand_mean) ** 2)
                ss_total = np.nansum((data_matrix_valid - grand_mean) ** 2)
                ss_error = ss_total - ss_rows - ss_cols
                
                df_rows = n_agents_valid - 1
                df_cols = n_attempts_valid - 1
                n_valid = np.sum(~np.isnan(data_matrix_valid))
                df_error = n_valid - n_agents_valid - n_attempts_valid + 1
                if df_error <= 0:
                    df_error = 1
                
                ms_rows = ss_rows / df_rows if df_rows > 0 else 0
                ms_cols = ss_cols / df_cols if df_cols > 0 else 0
                ms_error = ss_error / df_error if df_error > 0 else 0
                
                # Min and max values for this question
                min_value = np.nanmin(data_matrix_valid)
                max_value = np.nanmax(data_matrix_valid)
            else:
                ms_rows = np.nan
                ms_cols = np.nan
                ms_error = np.nan
                min_value = np.nan
                max_value = np.nan
        else:
            ms_rows = np.nan
            ms_cols = np.nan
            ms_error = np.nan
            min_value = np.nan
            max_value = np.nan
        
        # Also compute mean Stability Index and other metrics for comparison
        question_level_data = results['question_level'][results['question_level']['question_id'] == question_id]
        first_row = question_level_data.iloc[0] if len(question_level_data) > 0 else None
        
        question_aggregated.append({
            'question_id': question_id,
            'section_id': first_row['section_id'] if first_row is not None else None,
            'icc_2k': icc_2k,  # Standard ICC(2,k)
            'icc_3k': icc_3k,  # Standard ICC(3,k)
            'mean_stability_index': question_level_data['stability_index'].mean() if len(question_level_data) > 0 else np.nan,
            'cronbach_alpha': cronbach_alpha_question,  # Standard Cronbach's Alpha at question level
            'cv': question_level_data['cv'].mean() if len(question_level_data) > 0 else np.nan,
            'sd': question_level_data['sd'].mean() if len(question_level_data) > 0 else np.nan,
            'range': question_level_data['range'].mean() if len(question_level_data) > 0 else np.nan,
            'mean': question_level_data['mean'].mean() if len(question_level_data) > 0 else np.nan,
            'n_agents': len(agent_ids),
            'n_attempts': first_row['n_attempts'] if first_row is not None else 0,
            # Diagnostic variance components (MS values for ICC calculation)
            'ms_rows': ms_rows,  # MS_R: Mean Square for Rows (Agents)
            'ms_cols': ms_cols,  # MS_C: Mean Square for Columns (Attempts)
            'ms_error': ms_error,  # MS_E: Mean Square Error
            'min_value': min_value,
            'max_value': max_value
        })
    
    results['question_aggregated'] = pd.DataFrame(question_aggregated)
    
    return results


def analyze_distilled_consistency(df_distilled: pd.DataFrame, attempt_ids: List[int]) -> Dict:
    """Analyze consistency for distilled scores."""
    results = {}
    
    # 1. Per-score consistency (trait scores for BFI-10, total scores for BES-A/REI)
    score_metrics = []
    for (agent_id, score_name), group in df_distilled.groupby(['agent_id', 'score_name']):
        scores = group['score_value'].dropna().tolist()
        metrics = compute_consistency_metrics(scores)
        # Cache first row to avoid multiple column accesses
        first_row = group.iloc[0]
        metrics.update({
            'agent_id': agent_id,
            'score_name': score_name,
            'section_id': first_row['section_id'],
            'score_type': first_row['score_type']
        })
        score_metrics.append(metrics)
    
    results['score_level'] = pd.DataFrame(score_metrics)
    
    # 2. Per-section consistency (aggregate across scores)
    section_metrics = []
    for (agent_id, section_id), group in df_distilled.groupby(['agent_id', 'section_id']):
        # Get all scores for this section across all attempts
        # More efficient: group by attempt_id first, then compute mean
        section_scores = []
        attempt_groups = group.groupby('attempt_id')['score_value']
        for attempt_id in attempt_ids:
            if attempt_id in attempt_groups.groups:
                attempt_scores = attempt_groups.get_group(attempt_id).dropna().tolist()
            if attempt_scores:
                section_scores.append(np.mean(attempt_scores))  # Mean across scores per attempt
        
        metrics = compute_consistency_metrics(section_scores)
        metrics.update({
            'agent_id': agent_id,
            'section_id': section_id
        })
        section_metrics.append(metrics)
    
    results['section_level'] = pd.DataFrame(section_metrics)
    
    # 3. Per-agent consistency (average Stability Index across scores for each agent)
    # Create agent name lookup once (more efficient)
    agent_name_map = df_distilled[['agent_id', 'agent_name']].drop_duplicates().set_index('agent_id')['agent_name'].to_dict() if 'agent_name' in df_distilled.columns else {}
    
    agent_metrics = []
    # Use groupby instead of filtering in a loop
    for agent_id, agent_scores in results['score_level'].groupby('agent_id'):
        if len(agent_scores) == 0:
            continue
        
        agent_name = agent_name_map.get(agent_id, 'Unknown')
        
        metrics = {
            'agent_id': agent_id,
            'agent_name': agent_name,
            'stability_index': agent_scores['stability_index'].mean(),
            'cronbach_alpha': agent_scores['cronbach_alpha'].mean(),
            'cv': agent_scores['cv'].mean(),
            'sd': agent_scores['sd'].mean(),
            'range': agent_scores['range'].mean(),
            'mean': agent_scores['mean'].mean(),
            'n_scores': len(agent_scores),
            'n_attempts': agent_scores['n_attempts'].iloc[0] if len(agent_scores) > 0 else 0
        }
        agent_metrics.append(metrics)
    
    results['agent_level'] = pd.DataFrame(agent_metrics)
    
    # 4. Overall summary by section
    # Use groupby instead of filtering in a loop
    overall_section = []
    for section_id, section_scores in results['score_level'].groupby('section_id'):
        overall_section.append({
            'section_id': section_id,
            'mean_stability_index': section_scores['stability_index'].mean(),
            'mean_cronbach_alpha': section_scores['cronbach_alpha'].mean(),
            'mean_cv': section_scores['cv'].mean(),
            'mean_sd': section_scores['sd'].mean(),
            'n_scores': len(section_scores)
        })
    
    results['overall_section'] = pd.DataFrame(overall_section)
    
    # 5. Aggregated score-level metrics (standard ICC across agents for each score)
    score_aggregated = []
    for score_name, score_data in df_distilled.groupby('score_name'):
        # Get all agents' scores for this score_name across all attempts
        # Create matrix: agents (rows) × attempts (columns)
        agent_ids = score_data['agent_id'].unique()
        attempt_ids_sorted = sorted(score_data['attempt_id'].unique())
        
        # Build data matrix
        data_matrix = []
        for agent_id in agent_ids:
            agent_scores = []
            for attempt_id in attempt_ids_sorted:
                score_value = score_data[(score_data['agent_id'] == agent_id) & 
                                        (score_data['attempt_id'] == attempt_id)]['score_value']
                if len(score_value) > 0:
                    val = score_value.iloc[0]
                    agent_scores.append(val if not pd.isna(val) else np.nan)
                else:
                    agent_scores.append(np.nan)
            data_matrix.append(agent_scores)
        
        data_matrix = np.array(data_matrix)
        
        # Compute standard ICC using ANOVA
        icc_2k, icc_3k = compute_standard_icc(data_matrix)
        
        # Compute standard Cronbach's Alpha at score level
        cronbach_alpha_score = compute_cronbach_alpha_question_level(data_matrix)
        
        # Compute MS_R, MS_C, MS_E for diagnostics
        # Remove rows/columns that are all NaN
        valid_rows = ~np.isnan(data_matrix).all(axis=1)
        valid_cols = ~np.isnan(data_matrix).all(axis=0)
        
        if valid_rows.any() and valid_cols.any():
            data_matrix_valid = data_matrix[valid_rows][:, valid_cols]
            n_agents_valid, n_attempts_valid = data_matrix_valid.shape
            
            if n_agents_valid >= 2 and n_attempts_valid >= 2:
                # Compute means
                grand_mean = np.nanmean(data_matrix_valid)
                row_means = np.nanmean(data_matrix_valid, axis=1)  # Agent means
                col_means = np.nanmean(data_matrix_valid, axis=0)  # Attempt means
                
                # Compute SS and MS for diagnostic
                n_per_row = np.sum(~np.isnan(data_matrix_valid), axis=1)
                ss_rows = np.sum(n_per_row * (row_means - grand_mean) ** 2)
                n_per_col = np.sum(~np.isnan(data_matrix_valid), axis=0)
                ss_cols = np.sum(n_per_col * (col_means - grand_mean) ** 2)
                ss_total = np.nansum((data_matrix_valid - grand_mean) ** 2)
                ss_error = ss_total - ss_rows - ss_cols
                
                df_rows = n_agents_valid - 1
                df_cols = n_attempts_valid - 1
                n_valid = np.sum(~np.isnan(data_matrix_valid))
                df_error = n_valid - n_agents_valid - n_attempts_valid + 1
                if df_error <= 0:
                    df_error = 1
                
                ms_rows = ss_rows / df_rows if df_rows > 0 else 0
                ms_cols = ss_cols / df_cols if df_cols > 0 else 0
                ms_error = ss_error / df_error if df_error > 0 else 0
                
                # Min and max values for this score
                min_value = np.nanmin(data_matrix_valid)
                max_value = np.nanmax(data_matrix_valid)
            else:
                ms_rows = np.nan
                ms_cols = np.nan
                ms_error = np.nan
                min_value = np.nan
                max_value = np.nan
        else:
            ms_rows = np.nan
            ms_cols = np.nan
            ms_error = np.nan
            min_value = np.nan
            max_value = np.nan
        
        # Also compute mean Stability Index and other metrics for comparison
        score_level_data = results['score_level'][results['score_level']['score_name'] == score_name]
        first_row = score_level_data.iloc[0] if len(score_level_data) > 0 else None
        
        score_aggregated.append({
            'score_name': score_name,
            'section_id': first_row['section_id'] if first_row is not None else None,
            'score_type': first_row['score_type'] if first_row is not None else None,
            'icc_2k': icc_2k,  # Standard ICC(2,k)
            'icc_3k': icc_3k,  # Standard ICC(3,k)
            'mean_stability_index': score_level_data['stability_index'].mean() if len(score_level_data) > 0 else np.nan,
            'cronbach_alpha': cronbach_alpha_score,  # Standard Cronbach's Alpha at score level
            'cv': score_level_data['cv'].mean() if len(score_level_data) > 0 else np.nan,
            'sd': score_level_data['sd'].mean() if len(score_level_data) > 0 else np.nan,
            'range': score_level_data['range'].mean() if len(score_level_data) > 0 else np.nan,
            'mean': score_level_data['mean'].mean() if len(score_level_data) > 0 else np.nan,
            'n_agents': len(agent_ids),
            'n_attempts': first_row['n_attempts'] if first_row is not None else 0,
            # Diagnostic variance components (MS values for ICC calculation)
            'ms_rows': ms_rows,  # MS_R: Mean Square for Rows (Agents)
            'ms_cols': ms_cols,  # MS_C: Mean Square for Columns (Attempts)
            'ms_error': ms_error,  # MS_E: Mean Square Error
            'min_value': min_value,
            'max_value': max_value
        })
    
    results['score_aggregated'] = pd.DataFrame(score_aggregated)
    
    return results


def extract_question_number(question_id):
    """Extract numeric part from question ID for sorting (e.g., 'BFI-10_1' -> 1, 'BES-A_5' -> 5)."""
    try:
        # Extract number after underscore (format: SECTION_NUMBER)
        parts = str(question_id).split('_')
        if len(parts) > 1:
            return int(parts[-1])
        # If no underscore, try to extract number from the end
        match = re.search(r'\d+$', str(question_id))
        if match:
            return int(match.group())
        return 0
    except (ValueError, IndexError):
        return 0


def generate_html_report(raw_results: Dict, df_all: pd.DataFrame, output_path: str, 
                        agent_ids: List[str], attempt_ids: List[int],
                        distilled_results: Optional[Dict] = None, df_distilled: Optional[pd.DataFrame] = None,
                        raw_distribution_results: Optional[Dict] = None, distilled_distribution_results: Optional[Dict] = None):
    """Generate interactive HTML report with Plotly visualizations."""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
    except ImportError:
        print("Warning: Plotly not available. Generating basic HTML report without interactive charts.")
        generate_basic_html_report(raw_results, df_all, output_path, agent_ids, attempt_ids,
                                  distilled_results, df_distilled)
        return
    
    # Construct JavaScript code separately to avoid f-string escaping issues
    js_code = """
        function showTab(tabName, event) {
            // Hide all tabs
            var tabs = document.getElementsByClassName('tab-content');
            for (var i = 0; i < tabs.length; i++) {
                tabs[i].classList.remove('active');
            }
            var tabButtons = document.getElementsByClassName('tab');
            for (var i = 0; i < tabButtons.length; i++) {
                tabButtons[i].classList.remove('active');
            }
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            if (event && event.target) {
                event.target.classList.add('active');
            }
        }
        
        function updateTrajectoryView() {
            var viewType = document.getElementById('trajectory-view').value;
            var agent1Select = document.getElementById('trajectory-agent1');
            var container = document.getElementById('trajectory-charts-container');
            
            // Show/hide agent selector
            agent1Select.style.display = viewType === 'single' ? 'inline-block' : 'none';
            
            // Populate agent select if needed
            if (agent1Select && agent1Select.options.length === 1) {
                var agentIds = """ + json.dumps(agent_ids).replace('</script>', '<\\/script>') + """;
                agentIds.forEach(function(agentId) {
                    var option1 = document.createElement('option');
                    option1.value = agentId;
                    option1.textContent = agentId;
                    agent1Select.appendChild(option1);
                });
            }
            
            // Hide all trajectory divs
            var allDivs = container.querySelectorAll('[id^="traj-"]');
            for (var i = 0; i < allDivs.length; i++) {
                allDivs[i].style.display = 'none';
            }
            
            // Show selected view
            if (viewType === 'all') {
                // Show all views: combined + individual
                var allView = document.getElementById('traj-all-combined');
                if (allView) allView.style.display = 'block';
                
                var allIndiv = container.querySelectorAll('[id^="traj-indiv-"]');
                for (var i = 0; i < allIndiv.length; i++) {
                    allIndiv[i].style.display = 'block';
                }
            } else if (viewType === 'single') {
                var agent1 = agent1Select.value;
                if (agent1) {
                    var agentView = document.getElementById('traj-indiv-' + agent1);
                if (agentView) agentView.style.display = 'block';
                }
            }
            
            // Apply instrument filter after showing the view
            updateTrajectoryInstrumentFilter();
            updateTraitSelectorVisibility();
        }
        
        
        function updateTraitSelectorVisibility() {
            var instrumentFilter = document.getElementById('instrument-filter');
            var traitContainer = document.getElementById('trait-selector-container');
            var selectedInstrument = instrumentFilter ? instrumentFilter.value : 'all';
            
            if (traitContainer) {
                traitContainer.style.display = (selectedInstrument === 'bfi10') ? 'inline-block' : 'none';
            }
        }
        
        function updateTrajectoryInstrumentFilter() {
            var instrumentFilter = document.getElementById('instrument-filter');
            var selectedInstrument = instrumentFilter ? instrumentFilter.value : 'all';
            
            // Get selected traits if BFI-10 is selected
            var selectedTraits = [];
            if (selectedInstrument === 'bfi10') {
                var traitCheckboxes = document.querySelectorAll('.trait-checkbox:checked');
                for (var i = 0; i < traitCheckboxes.length; i++) {
                    selectedTraits.push(traitCheckboxes[i].value);
                }
            }
            
            // Find all visible Plotly graph divs in trajectory charts
            var container = document.getElementById('trajectory-charts-container');
            if (!container) return;
            
            // Get all plotly divs (they have class 'plotly' or id starting with the chart id)
            var plotlyDivs = container.querySelectorAll('.js-plotly-plot, [id*="plotly"]');
            
            // Define score ranges for each instrument
            var yAxisRanges = {
                'all': {min: 0, max: 5},
                'bfi10': {min: 0, max: 5},
                'besa': {min: 0, max: 5},
                'rei': {min: 0, max: 5}
            };
            
            var yAxisRange = yAxisRanges[selectedInstrument] || yAxisRanges['all'];
            
            plotlyDivs.forEach(function(plotDiv) {
                // Get the Plotly graph div (the actual div Plotly uses)
                var graphDiv = plotDiv;
                
                // Wait for Plotly to be ready, then update
                if (typeof Plotly !== 'undefined' && graphDiv.data) {
                    // Determine which traces should be visible based on instrument and traits
                    var visible = [];
                    
                    // Get trace names from the graph
                    for (var i = 0; i < graphDiv.data.length; i++) {
                        var traceName = graphDiv.data[i].name || '';
                        
                        var shouldShow = false;
                        if (selectedInstrument === 'all') {
                            shouldShow = true;
                        } else if (selectedInstrument === 'bfi10') {
                            // BFI-10 scores: check if trace matches any selected trait
                            if (selectedTraits.length === 0) {
                                // If no traits selected, show none
                                shouldShow = false;
                            } else {
                                // Check if trace name includes any selected trait
                                for (var j = 0; j < selectedTraits.length; j++) {
                                    if (traceName.includes(selectedTraits[j])) {
                                        shouldShow = true;
                                        break;
                                    }
                                }
                            }
                        } else if (selectedInstrument === 'besa') {
                            // BES-A score: CE_normalized
                            shouldShow = traceName.includes('CE_normalized');
                        } else if (selectedInstrument === 'rei') {
                            // REI score: RA_normalized
                            shouldShow = traceName.includes('RA_normalized');
                        }
                        visible.push(shouldShow);
                    }
                    
                    // Update visibility and y-axis range
                    Plotly.update(graphDiv, {
                        visible: visible
                    }, {
                        'yaxis.range': [yAxisRange.min, yAxisRange.max]
                    });
                }
            });
        }
        
        // Initialize trait selector visibility on page load
        document.addEventListener('DOMContentLoaded', function() {
            updateTraitSelectorVisibility();
        });
        
        function updateDistributionQuestionLevelView() {
            const dataType = document.getElementById('distribution-data-type').value;
            const rawView = document.getElementById('distribution-question-view-raw');
            const distilledView = document.getElementById('distribution-question-view-distilled');
            
            if (rawView) rawView.style.display = (dataType === 'raw') ? 'block' : 'none';
            if (distilledView) distilledView.style.display = (dataType === 'distilled') ? 'block' : 'none';
        }
        
        function updateQuestionLevelView() {
            var dataType = document.getElementById('data-type').value;
            var viewType = document.getElementById('question-view').value;
            var container = document.getElementById('question-level-content-container');
            
            // Hide all views
            var allViews = container.querySelectorAll('[id^="question-view-"]');
            for (var i = 0; i < allViews.length; i++) {
                allViews[i].style.display = 'none';
            }
            
            // Map view type to container ID
            var viewTypeMap = {
                'all': 'all',
                'question-agg': 'question-agg',
                'agent-agg': 'agent-agg'
            };
            
            // Show selected view based on data type and view type
            var mappedViewType = viewTypeMap[viewType] || viewType;
            var viewId = 'question-view-' + dataType + '-' + mappedViewType;
            var selectedView = document.getElementById(viewId);
            if (selectedView) {
                selectedView.style.display = 'block';
                
                // Load charts dynamically if needed (for non-default views)
                if (viewId !== 'question-view-raw-all' && typeof chartData !== 'undefined') {
                    loadChartsForView(dataType, mappedViewType);
                }
            }
        }
    """
    
    # HTML template
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Survey Consistency Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
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
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .tabs {{
            display: flex;
            border-bottom: 2px solid #ddd;
            margin-bottom: 20px;
        }}
        .tab {{
            padding: 10px 20px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 16px;
            color: #666;
        }}
        .tab.active {{
            color: #4CAF50;
            border-bottom: 3px solid #4CAF50;
            font-weight: bold;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
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
    </style>
    <script>
{js_code}
    </script>
</head>
<body>
    <div class="container">
        <h1>Survey Consistency Analysis Report</h1>
        <p><strong>Agents:</strong> {', '.join(agent_ids)}</p>
        <p><strong>Attempts:</strong> {min(attempt_ids)}-{max(attempt_ids)}</p>
        <p><strong>Total Responses:</strong> {len(df_all)}</p>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('summary', event)">Summary</button>
            <button class="tab" onclick="showTab('question-level', event)">Question Level Consistency</button>
            <button class="tab" onclick="showTab('trajectories', event)">Trajectories</button>
            <button class="tab" onclick="showTab('metric-diagnostics', event)">Metric Diagnostics</button>""" + ("""
            <button class="tab" onclick="showTab('distribution-summary', event)">Distribution Summary</button>
            <button class="tab" onclick="showTab('distribution-question-level', event)">Question Level Distribution</button>""" if (raw_distribution_results is not None or distilled_distribution_results is not None) else "") + """
        </div>
        
        <div id="summary" class="tab-content active">
            <h2>Overall Summary</h2>
            <div id="summary-metrics"></div>
            <div id="summary-charts"></div>
            """ + ("""
            <h3>Distilled Scores Summary</h3>
            <div id="distilled-summary-metrics"></div>
            <div id="distilled-summary-charts"></div>
            """ if distilled_results is not None else "") + """
        </div>
        
        <div id="question-level" class="tab-content">
            <h2>Question Level Consistency</h2>
            <div class="filter-controls" style="margin: 20px 0; padding: 15px; background-color: #f9f9f9; border-radius: 5px;">
                <label for="data-type" style="font-weight: bold; margin-right: 10px;">Data Type:</label>
                <select id="data-type" onchange="updateQuestionLevelView()" style="padding: 5px 10px; font-size: 14px; margin-right: 20px;">
                    <option value="raw">Raw Responses</option>""" + ("""
                    <option value="distilled">Distilled Scores</option>""" if distilled_results is not None else "") + """
                </select>
                <label for="question-view" style="font-weight: bold; margin-right: 10px;">View:</label>
                <select id="question-view" onchange="updateQuestionLevelView()" style="padding: 5px 10px; font-size: 14px;">
                    <option value="all">All (All Agent-Question Pairs)</option>
                    <option value="question-agg">Aggregated by Question</option>
                    <option value="agent-agg">Aggregated by Agent</option>
                </select>
            </div>
            <div id="question-level-content-container"></div>
        </div>
        
        <div id="trajectories" class="tab-content">
            <h2>Distilled Scores Trajectories Across Attempts</h2>
            """ + ("""
            <div style="margin: 20px 0;">
                <label for="trajectory-view" style="font-weight: bold; margin-right: 10px;">View:</label>
                <select id="trajectory-view" onchange="updateTrajectoryView()" style="padding: 5px 10px; font-size: 14px; margin-right: 20px;">
                    <option value="all">All Agents</option>
                    <option value="single">Single Agent</option>
                </select>
                <label for="instrument-filter" style="font-weight: bold; margin-right: 10px; margin-left: 20px;">Filter by Instrument:</label>
                <select id="instrument-filter" onchange="updateTrajectoryInstrumentFilter(); updateTraitSelectorVisibility();" style="padding: 5px 10px; font-size: 14px;">
                    <option value="all">All Scores</option>
                    <option value="bfi10">BFI-10 Only</option>
                    <option value="besa">BES-A Only</option>
                    <option value="rei">REI Only</option>
                </select>
                <div id="trait-selector-container" style="display: none; margin-left: 20px; margin-top: 10px;">
                    <label style="font-weight: bold; margin-right: 10px;">Select Traits:</label>
                    <label style="margin-right: 15px;"><input type="checkbox" class="trait-checkbox" value="Extraversion" checked onchange="updateTrajectoryInstrumentFilter()"> Extraversion</label>
                    <label style="margin-right: 15px;"><input type="checkbox" class="trait-checkbox" value="Agreeableness" checked onchange="updateTrajectoryInstrumentFilter()"> Agreeableness</label>
                    <label style="margin-right: 15px;"><input type="checkbox" class="trait-checkbox" value="Conscientiousness" checked onchange="updateTrajectoryInstrumentFilter()"> Conscientiousness</label>
                    <label style="margin-right: 15px;"><input type="checkbox" class="trait-checkbox" value="Neuroticism" checked onchange="updateTrajectoryInstrumentFilter()"> Neuroticism</label>
                    <label style="margin-right: 15px;"><input type="checkbox" class="trait-checkbox" value="Openness" checked onchange="updateTrajectoryInstrumentFilter()"> Openness</label>
                </div>
            </div>
            <div id="trajectory-charts-container"></div>
            """ if distilled_results is not None else """
            <p>Distilled scores not available. Please run the translation script first.</p>
            """) + """
        </div>
        
        <div id="metric-diagnostics" class="tab-content">
            <h2>Metric Diagnostics</h2>
            
            <h3>ICC (Intraclass Correlation Coefficient)</h3>
            <p><strong>MS_R (Mean Square Rows):</strong> Variance component for agents (between-agent variance). Measures how much agents differ from each other.</p>
            <p><strong>MS_C (Mean Square Columns):</strong> Variance component for attempts (between-attempt variance). Measures systematic differences across attempts.</p>
            <p><strong>MS_E (Mean Square Error):</strong> Residual variance. Measures variability that cannot be explained by agents or attempts.</p>
            
            <p><strong>ICC(2,k) Formula:</strong></p>
            <p style="font-family: monospace; background-color: #f5f5f5; padding: 10px; border-radius: 5px;">
                ICC(2,k) = (MS_R - MS_E) / (MS_R + (MS_C - MS_E) / n)
            </p>
            <p>Where n = number of agents. This measures absolute agreement between agents, accounting for both systematic and random error.</p>
            
            <p><strong>ICC(3,k) Formula:</strong></p>
            <p style="font-family: monospace; background-color: #f5f5f5; padding: 10px; border-radius: 5px;">
                ICC(3,k) = (MS_R - MS_E) / MS_R
            </p>
            <p>This measures consistency between agents, allowing for systematic differences (only random error matters).</p>
            
            <h3>Stability Index</h3>
            <p><strong>Formula:</strong></p>
            <p style="font-family: monospace; background-color: #f5f5f5; padding: 10px; border-radius: 5px;">
                Stability Index = 1 - (variance / (variance + expected_variance_uniform))
            </p>
            <p>Where expected_variance_uniform = (5-1)²/12 = 1.33 for Likert scale (1-5).</p>
            <p>This is a variance-based approximation for single-item test-retest reliability. Measures consistency of responses across attempts for a single question.</p>
            
            <h3>Cronbach's Alpha</h3>
            <p><strong>Standard Formula:</strong></p>
            <p style="font-family: monospace; background-color: #f5f5f5; padding: 10px; border-radius: 5px;">
                α = (k / (k-1)) × (1 - Σσⱼ² / σₜ²)
            </p>
            <p>Where:</p>
            <ul>
                <li>k = number of attempts (treated as items)</li>
                <li>σⱼ² = variance of attempt j across all agents</li>
                <li>σₜ² = variance of total scores (sum of all attempts for each agent)</li>
            </ul>
            <p>This measures internal consistency across attempts, treating each attempt as an item in a scale.</p>
            
            <h3>Diagnostic Tables</h3>
            <div id="metric-diagnostics-raw-container"></div>
            """ + ("""
            <div id="metric-diagnostics-distilled-container"></div>
            """ if distilled_results is not None else "") + """
        </div>
        
        """ + ("""
        <div id="distribution-summary" class="tab-content">
            <h2>Distribution Summary</h2>
            <div id="distribution-summary-stats"></div>
            <div id="distribution-summary-edge-cases"></div>
        </div>
        
        <div id="distribution-question-level" class="tab-content">
            <h2>Question Level Distribution</h2>
            <div class="filter-controls" style="margin: 20px 0; padding: 15px; background-color: #f9f9f9; border-radius: 5px;">
                <label for="distribution-data-type" style="font-weight: bold; margin-right: 10px;">Data Type:</label>
                <select id="distribution-data-type" onchange="updateDistributionQuestionLevelView()" style="padding: 5px 10px; font-size: 14px;">
                    <option value="raw">Raw Responses</option>""" + ("""
                    <option value="distilled">Distilled Scores</option>""" if distilled_distribution_results is not None else "") + """
                </select>
            </div>
            
            <div id="distribution-question-view-raw" class="distribution-question-view" style="display: block;">
                <div id="distribution-raw-content"></div>
            </div>
            """ + ("""
            <div id="distribution-question-view-distilled" class="distribution-question-view" style="display: none;">
                <div id="distribution-distilled-content"></div>
            </div>
            """ if distilled_distribution_results is not None else "") + """
        </div>
        """ if (raw_distribution_results is not None or distilled_distribution_results is not None) else "") + """
    </div>
"""
    
    # Add Plotly visualizations for raw responses
    # 1. Summary metrics (raw)
    overall_section = raw_results['overall_section']
    question_aggregated = raw_results.get('question_aggregated', pd.DataFrame())
    
    # Stability Index (from overall_section)
    mean_stability_index = overall_section['mean_stability_index'].mean() if not overall_section['mean_stability_index'].isna().all() else 0
    
    # Standard ICC (from question_aggregated)
    mean_icc_2k = question_aggregated['icc_2k'].mean() if not question_aggregated.empty and not question_aggregated['icc_2k'].isna().all() else 0
    mean_icc_3k = question_aggregated['icc_3k'].mean() if not question_aggregated.empty and not question_aggregated['icc_3k'].isna().all() else 0
    
    # Cronbach's Alpha (standard formula from aggregated question level)
    mean_alpha = question_aggregated['cronbach_alpha'].mean() if not question_aggregated.empty and not question_aggregated['cronbach_alpha'].isna().all() else 0
    
    summary_html = f"""
            <h3>Raw Responses Summary</h3>
            <div class="metric-box">
                <div class="metric-value">{mean_stability_index:.3f}</div>
                <div class="metric-label">Mean Stability Index</div>
            </div>
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
    """
    
    # Add distilled summary if available
    distilled_summary_html = ""
    if distilled_results is not None:
        distilled_overall = distilled_results['overall_section']
        score_aggregated = distilled_results.get('score_aggregated', pd.DataFrame())
        
        # Stability Index (from overall_section)
        dist_mean_stability_index = distilled_overall['mean_stability_index'].mean() if not distilled_overall['mean_stability_index'].isna().all() else 0
        
        # Standard ICC (from score_aggregated)
        dist_mean_icc_2k = score_aggregated['icc_2k'].mean() if not score_aggregated.empty and not score_aggregated['icc_2k'].isna().all() else 0
        dist_mean_icc_3k = score_aggregated['icc_3k'].mean() if not score_aggregated.empty and not score_aggregated['icc_3k'].isna().all() else 0
        
        # Cronbach's Alpha (standard formula from aggregated score level)
        dist_mean_alpha = score_aggregated['cronbach_alpha'].mean() if not score_aggregated.empty and not score_aggregated['cronbach_alpha'].isna().all() else 0
        
        distilled_summary_html = f"""
            <div class="metric-box">
                <div class="metric-value">{dist_mean_stability_index:.3f}</div>
                <div class="metric-label">Mean Stability Index</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{dist_mean_icc_2k:.3f}</div>
                <div class="metric-label">Mean ICC(2,k)</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{dist_mean_icc_3k:.3f}</div>
                <div class="metric-label">Mean ICC(3,k)</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{dist_mean_alpha:.3f}</div>
                <div class="metric-label">Mean Cronbach's α</div>
            </div>
        """
    
    # 2. Section comparison chart (using Stability Index)
    fig_sections = go.Figure()
    sections = overall_section['section_id'].tolist()
    
    # Handle NaN values
    stability_index_vals = overall_section['mean_stability_index'].fillna(0).tolist()
    alpha_vals = question_aggregated['cronbach_alpha'].fillna(0).tolist()
    
    fig_sections.add_trace(go.Bar(
        x=sections,
        y=stability_index_vals,
        name='Stability Index',
        marker_color='#4CAF50',
        hovertemplate='Section: %{x}<br>Stability Index: %{y:.3f}<extra></extra>'
    ))
    fig_sections.add_trace(go.Bar(
        x=sections,
        y=alpha_vals,
        name="Cronbach's α",
        marker_color='#45a049',
        hovertemplate='Section: %{x}<br>Cronbach\'s α: %{y:.3f}<extra></extra>'
    ))
    fig_sections.update_layout(
        title='Consistency by Section (Stability Index & Cronbach\'s α)',
        xaxis_title='Section',
        yaxis_title='Value',
        yaxis=dict(range=[0, 1]),
        barmode='group',
        height=400,
        hovermode='x unified'
    )
    
    # 3. Question-level visualizations (raw) - NEW STRUCTURE
    question_df = raw_results['question_level']
    
    # Sort question IDs by numeric part
    unique_questions = sorted(question_df['question_id'].unique(), key=extract_question_number)
    
    # Create pivot table, handling missing values
    # Use stability_index for heatmaps (All view)
    try:
        pivot_icc = question_df.pivot_table(
            index='question_id',
            columns='agent_id',
            values='stability_index',
            aggfunc='mean'
        )
        
        # Sort by question ID (numeric)
        pivot_icc = pivot_icc.reindex(sorted(pivot_icc.index, key=extract_question_number))
        # Sort columns (agent IDs) for consistent ordering
        pivot_icc = pivot_icc.sort_index(axis=1)
        
        # Fill NaN with 0 for visualization
        pivot_icc = pivot_icc.fillna(0)
        
        # RAW DATA: Generate all three views
        # View 1: All (heatmap) - split by 50 agents
        agent_ids_list = pivot_icc.columns.tolist()
        question_ids_list = pivot_icc.index.tolist()
        
        raw_heatmap_figs = []
        chunk_size = 50
        num_chunks = (len(agent_ids_list) + chunk_size - 1) // chunk_size
        
        # Option A: If only 1 chunk, show only overall. If multiple chunks, show subgraphs + overall
        if num_chunks == 1:
            # Single chunk: show only overall graph
            hovertemplate_str_all = '<b>Agent ID:</b> %{x}<br><b>Question:</b> %{y}<br><b>Stability Index:</b> %{z:.3f}<extra></extra>'
            fig_heatmap_all = go.Figure(data=go.Heatmap(
                z=pivot_icc.values.tolist(),
                x=agent_ids_list,
                y=question_ids_list,
                colorscale='RdYlGn',
                colorbar=dict(title="Stability Index"),
                hovertemplate=hovertemplate_str_all,
                showscale=True
            ))
            fig_heatmap_all.update_layout(
                title=f'All Agent-Question Pairs (All {len(agent_ids_list)} Agents)',
                xaxis=dict(
                    title='Agent ID',
                    type='category',
                    tickangle=-45
                ),
                yaxis=dict(
                    title='Question ID',
                    type='category'
                ),
                height=600
            )
            raw_heatmap_figs.append(fig_heatmap_all)
        else:
            # Multiple chunks: show subgraphs + overall
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(agent_ids_list))
                
                chunk_agent_ids = agent_ids_list[start_idx:end_idx]
                chunk_pivot = pivot_icc[chunk_agent_ids]
                
                hovertemplate_str = '<b>Agent ID:</b> %{x}<br><b>Question:</b> %{y}<br><b>Stability Index:</b> %{z:.3f}<extra></extra>'
        
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=chunk_pivot.values.tolist(),
                    x=chunk_agent_ids,
                    y=question_ids_list,
            colorscale='RdYlGn',
                    colorbar=dict(title="Stability Index"),
                    hovertemplate=hovertemplate_str,
                    showscale=True
                ))

                title_suffix = f" (Agents {chunk_agent_ids[0]} to {chunk_agent_ids[-1]})"
                fig_heatmap.update_layout(
                    title=f'All Agent-Question Pairs{title_suffix}',
                    xaxis=dict(
                        title='Agent ID',
                        type='category',
                        tickangle=-45
                    ),
                    yaxis=dict(
                        title='Question ID',
                        type='category'
                    ),
            height=600
        )
                raw_heatmap_figs.append(fig_heatmap)
            
            # Add overall heatmap at the end
            hovertemplate_str_all = '<b>Agent ID:</b> %{x}<br><b>Question:</b> %{y}<br><b>Stability Index:</b> %{z:.3f}<extra></extra>'
            fig_heatmap_all = go.Figure(data=go.Heatmap(
                z=pivot_icc.values.tolist(),
                x=agent_ids_list,
                y=question_ids_list,
                colorscale='RdYlGn',
                colorbar=dict(title="Stability Index"),
                hovertemplate=hovertemplate_str_all,
                showscale=True
            ))
            fig_heatmap_all.update_layout(
                title=f'All Agent-Question Pairs (All {len(agent_ids_list)} Agents)',
                xaxis=dict(
                    title='Agent ID',
                    type='category',
                    tickangle=-45
                ),
                yaxis=dict(
                    title='Question ID',
                    type='category'
                ),
                height=600
            )
            raw_heatmap_figs.append(fig_heatmap_all)
        
        # View 2: Aggregated by question
        question_agg_df = raw_results.get('question_aggregated', pd.DataFrame())
        raw_question_agg_figs = []
        
        if not question_agg_df.empty:
            # Sort by question ID
            question_agg_df = question_agg_df.sort_values('question_id', key=lambda x: x.map(extract_question_number))
            question_agg_sorted = question_agg_df.copy()
            question_agg_sorted['icc_2k'] = question_agg_sorted['icc_2k'].fillna(0)
            
            # Split if more than 50 questions
            q_chunk_size = 50
            num_q_chunks = (len(question_agg_sorted) + q_chunk_size - 1) // q_chunk_size
            
            for q_chunk_idx in range(num_q_chunks):
                start_idx = q_chunk_idx * q_chunk_size
                end_idx = min((q_chunk_idx + 1) * q_chunk_size, len(question_agg_sorted))
                
                chunk_data = question_agg_sorted.iloc[start_idx:end_idx]
                chunk_question_ids = chunk_data['question_id'].tolist()
                chunk_icc_values = chunk_data['icc_2k'].tolist()
                
                fig_q_agg = go.Figure()
                fig_q_agg.add_trace(go.Bar(
                    x=chunk_question_ids,
                    y=chunk_icc_values,
                    marker_color='#4CAF50',
                    hovertemplate='Question: %{x}<br>Mean ICC(2,k): %{y:.3f}<extra></extra>',
                    text=chunk_icc_values,
                    texttemplate='%{{text:.3f}}',
                    textposition='outside',
                    width=0.8
                ))
                
                title_suffix = f" (Questions {chunk_question_ids[0]} to {chunk_question_ids[-1]})" if num_q_chunks > 1 else ""
                fig_q_agg.update_layout(
                    title=f'Aggregated by Question (Average Across Agents){title_suffix}',
                    xaxis_title='Question ID',
                    yaxis_title='Mean ICC(2,k)',
                    yaxis=dict(range=[0, 1.1]),
                    height=max(500, len(chunk_data) * 25),
                    xaxis=dict(tickangle=-45),
                    bargap=0.1
                )
                raw_question_agg_figs.append(fig_q_agg)
            
            # Add overall if multiple chunks
            if num_q_chunks > 1:
                fig_q_agg_all = go.Figure()
                fig_q_agg_all.add_trace(go.Bar(
                    x=question_agg_sorted['question_id'].tolist(),
                    y=question_agg_sorted['icc_2k'].tolist(),
                    marker_color='#4CAF50',
                    hovertemplate='Question: %{x}<br>Mean ICC(2,k): %{y:.3f}<extra></extra>',
                    text=question_agg_sorted['icc_2k'].tolist(),
                    texttemplate='%{{text:.3f}}',
                    textposition='outside',
                    width=0.8
                ))
                fig_q_agg_all.update_layout(
                    title=f'Aggregated by Question (Average Across Agents) - All {len(question_agg_sorted)} Questions',
                    xaxis_title='Question ID',
                    yaxis_title='Mean ICC(2,k)',
                    yaxis=dict(range=[0, 1.1]),
                    height=max(500, len(question_agg_sorted) * 25),
                    xaxis=dict(tickangle=-45),
                    bargap=0.1
                )
                raw_question_agg_figs.append(fig_q_agg_all)
        
        # View 3: Aggregated by agent (using Stability Index)
            agent_df = raw_results['agent_level']
        agent_df_sorted = agent_df.sort_values('agent_id').copy()
        agent_df_sorted['stability_index'] = agent_df_sorted['stability_index'].fillna(0)
        
        raw_agent_agg_figs = []
        a_chunk_size = 50
        num_a_chunks = (len(agent_df_sorted) + a_chunk_size - 1) // a_chunk_size
        
        # Option A: If only 1 chunk, show only overall. If multiple chunks, show subgraphs + overall
        if num_a_chunks == 1:
            # Single chunk: show only overall graph
            fig_a_agg_all = go.Figure()
            fig_a_agg_all.add_trace(go.Bar(
                x=agent_df_sorted['agent_id'].tolist(),
                y=agent_df_sorted['stability_index'].tolist(),
        marker_color='#4CAF50',
                hovertemplate='Agent: %{x}<br>Mean Stability Index: %{y:.3f}<extra></extra>',
                text=agent_df_sorted['stability_index'].tolist(),
                texttemplate='%{{text:.3f}}',
                textposition='outside',
                width=0.8
            ))
            fig_a_agg_all.update_layout(
                title=f'Aggregated by Agent (Average Across Questions) - All {len(agent_df_sorted)} Agents',
        xaxis_title='Agent ID',
                yaxis_title='Mean Stability Index',
                yaxis=dict(range=[0, 1.1]),
                height=max(500, len(agent_df_sorted) * 15),
                xaxis=dict(tickangle=-45),
                bargap=0.1
            )
            raw_agent_agg_figs.append(fig_a_agg_all)
        else:
            # Multiple chunks: show subgraphs + overall
            for a_chunk_idx in range(num_a_chunks):
                start_idx = a_chunk_idx * a_chunk_size
                end_idx = min((a_chunk_idx + 1) * a_chunk_size, len(agent_df_sorted))
                
                chunk_data = agent_df_sorted.iloc[start_idx:end_idx]
                chunk_agent_ids = chunk_data['agent_id'].tolist()
                chunk_stability_values = chunk_data['stability_index'].tolist()
                
                fig_a_agg = go.Figure()
                fig_a_agg.add_trace(go.Bar(
                    x=chunk_agent_ids,
                    y=chunk_stability_values,
                    marker_color='#4CAF50',
                    hovertemplate='Agent: %{x}<br>Mean Stability Index: %{y:.3f}<extra></extra>',
                    text=chunk_stability_values,
                    texttemplate='%{{text:.3f}}',
                    textposition='outside',
                    width=0.8
                ))
                
                title_suffix = f" (Agents {chunk_agent_ids[0]} to {chunk_agent_ids[-1]})"
                fig_a_agg.update_layout(
                    title=f'Aggregated by Agent (Average Across Questions){title_suffix}',
                    xaxis_title='Agent ID',
                    yaxis_title='Mean Stability Index',
                    yaxis=dict(range=[0, 1.1]),
                    height=max(500, len(chunk_data) * 15),
                    xaxis=dict(tickangle=-45),
                    bargap=0.1
                )
                raw_agent_agg_figs.append(fig_a_agg)
            
            # Add overall agent aggregated chart at the end
            fig_a_agg_all = go.Figure()
            fig_a_agg_all.add_trace(go.Bar(
                x=agent_df_sorted['agent_id'].tolist(),
                y=agent_df_sorted['stability_index'].tolist(),
                marker_color='#4CAF50',
                hovertemplate='Agent: %{x}<br>Mean Stability Index: %{y:.3f}<extra></extra>',
                text=agent_df_sorted['stability_index'].tolist(),
                texttemplate='%{{text:.3f}}',
                textposition='outside',
                width=0.8
            ))
            fig_a_agg_all.update_layout(
                title=f'Aggregated by Agent (Average Across Questions) - All {len(agent_df_sorted)} Agents',
                xaxis_title='Agent ID',
                yaxis_title='Mean Stability Index',
                yaxis=dict(range=[0, 1.1]),
                height=max(500, len(agent_df_sorted) * 15),
                xaxis=dict(tickangle=-45),
                bargap=0.1
            )
            raw_agent_agg_figs.append(fig_a_agg_all)
        
    except Exception as e:
        print(f"Warning: Could not create raw visualizations: {e}")
        raw_heatmap_figs = [go.Figure()]
        raw_question_agg_figs = [go.Figure()]
        raw_agent_agg_figs = [go.Figure()]
    
    # Generate distilled visualizations if available
    distilled_heatmap_figs = []
    distilled_score_agg_figs = []
    distilled_agent_agg_figs = []
    
    if distilled_results is not None and df_distilled is not None:
        try:
            # DISTILLED DATA: Generate all three views
            score_df = distilled_results['score_level']
            
            # Sort score names (they should already be sorted, but ensure consistency)
            unique_scores = sorted(score_df['score_name'].unique())
            
            # Create pivot table for distilled scores (using Stability Index for heatmaps)
            pivot_score_icc = score_df.pivot_table(
                index='score_name',
                columns='agent_id',
                values='stability_index',
                aggfunc='mean'
            )
            
            # Sort columns (agent IDs) for consistent ordering
            pivot_score_icc = pivot_score_icc.sort_index(axis=1)
            pivot_score_icc = pivot_score_icc.fillna(0)
            
            # View 1: All (heatmap) - split by 50 agents
            dist_agent_ids_list = pivot_score_icc.columns.tolist()
            dist_score_names_list = pivot_score_icc.index.tolist()
            
            dist_chunk_size = 50
            dist_num_chunks = (len(dist_agent_ids_list) + dist_chunk_size - 1) // dist_chunk_size
            
            # Option A: If only 1 chunk, show only overall. If multiple chunks, show subgraphs + overall
            if dist_num_chunks == 1:
                # Single chunk: show only overall graph
                hovertemplate_str_all = '<b>Agent ID:</b> %{x}<br><b>Score:</b> %{y}<br><b>Stability Index:</b> %{z:.3f}<extra></extra>'
                fig_dist_heatmap_all = go.Figure(data=go.Heatmap(
                    z=pivot_score_icc.values.tolist(),
                    x=dist_agent_ids_list,
                    y=dist_score_names_list,
                    colorscale='Blues',
                    colorbar=dict(title="Stability Index"),
                    hovertemplate=hovertemplate_str_all,
                    showscale=True
                ))
                fig_dist_heatmap_all.update_layout(
                    title=f'All Agent-Score Pairs (All {len(dist_agent_ids_list)} Agents)',
                    xaxis=dict(
                        title='Agent ID',
                        type='category',
                        tickangle=-45
                    ),
                    yaxis=dict(
                        title='Score Name',
                        type='category'
                    ),
                    height=600
                )
                distilled_heatmap_figs.append(fig_dist_heatmap_all)
            else:
                # Multiple chunks: show subgraphs + overall
                for dist_chunk_idx in range(dist_num_chunks):
                    start_idx = dist_chunk_idx * dist_chunk_size
                    end_idx = min((dist_chunk_idx + 1) * dist_chunk_size, len(dist_agent_ids_list))
                    
                    chunk_agent_ids = dist_agent_ids_list[start_idx:end_idx]
                    chunk_pivot = pivot_score_icc[chunk_agent_ids]
                    
                    hovertemplate_str = '<b>Agent ID:</b> %{x}<br><b>Score:</b> %{y}<br><b>Stability Index:</b> %{z:.3f}<extra></extra>'
                    
                    fig_dist_heatmap = go.Figure(data=go.Heatmap(
                        z=chunk_pivot.values.tolist(),
                        x=chunk_agent_ids,
                        y=dist_score_names_list,
                        colorscale='Blues',
                        colorbar=dict(title="Stability Index"),
                        hovertemplate=hovertemplate_str,
                        showscale=True
                    ))
                    
                    title_suffix = f" (Agents {chunk_agent_ids[0]} to {chunk_agent_ids[-1]})"
                    fig_dist_heatmap.update_layout(
                        title=f'All Agent-Score Pairs{title_suffix}',
                        xaxis=dict(
                            title='Agent ID',
                            type='category',
                            tickangle=-45
                        ),
                        yaxis=dict(
                            title='Score Name',
                            type='category'
                        ),
                        height=600
                    )
                    distilled_heatmap_figs.append(fig_dist_heatmap)
                
                # Add overall heatmap at the end
                hovertemplate_str_all = '<b>Agent ID:</b> %{x}<br><b>Score:</b> %{y}<br><b>Stability Index:</b> %{z:.3f}<extra></extra>'
                fig_dist_heatmap_all = go.Figure(data=go.Heatmap(
                    z=pivot_score_icc.values.tolist(),
                    x=dist_agent_ids_list,
                    y=dist_score_names_list,
                    colorscale='Blues',
                    colorbar=dict(title="Stability Index"),
                    hovertemplate=hovertemplate_str_all,
                    showscale=True
                ))
                fig_dist_heatmap_all.update_layout(
                    title=f'All Agent-Score Pairs (All {len(dist_agent_ids_list)} Agents)',
                    xaxis=dict(
                        title='Agent ID',
                        type='category',
                        tickangle=-45
                    ),
                    yaxis=dict(
                        title='Score Name',
                        type='category'
                    ),
                    height=600
                )
                distilled_heatmap_figs.append(fig_dist_heatmap_all)
            
            # View 2: Aggregated by score
            score_agg_df = distilled_results.get('score_aggregated', pd.DataFrame())
            if not score_agg_df.empty:
                score_agg_sorted = score_agg_df.sort_values('score_name').copy()
                score_agg_sorted['icc_2k'] = score_agg_sorted['icc_2k'].fillna(0)
                
                # Split if more than 50 scores
                s_chunk_size = 50
                num_s_chunks = (len(score_agg_sorted) + s_chunk_size - 1) // s_chunk_size
                
                for s_chunk_idx in range(num_s_chunks):
                    start_idx = s_chunk_idx * s_chunk_size
                    end_idx = min((s_chunk_idx + 1) * s_chunk_size, len(score_agg_sorted))
                    
                    chunk_data = score_agg_sorted.iloc[start_idx:end_idx]
                    chunk_score_names = chunk_data['score_name'].tolist()
                    chunk_icc_values = chunk_data['icc_2k'].tolist()
                    
                    fig_s_agg = go.Figure()
                    fig_s_agg.add_trace(go.Bar(
                        x=chunk_score_names,
                        y=chunk_icc_values,
                        marker_color='#2196F3',
                        hovertemplate='Score: %{x}<br>Mean ICC(2,k): %{y:.3f}<extra></extra>',
                        text=chunk_icc_values,
                        texttemplate='%{{text:.3f}}',
                        textposition='outside',
                        width=0.8
                    ))
                    
                    title_suffix = f" (Scores {chunk_score_names[0]} to {chunk_score_names[-1]})" if num_s_chunks > 1 else ""
                    fig_s_agg.update_layout(
                        title=f'Aggregated by Score (Average Across Agents){title_suffix}',
                        xaxis_title='Score Name',
                        yaxis_title='Mean ICC(2,k)',
                        yaxis=dict(range=[0, 1.1]),
                        height=max(500, len(chunk_data) * 25),
                        xaxis=dict(tickangle=-45),
                        bargap=0.1
                    )
                    distilled_score_agg_figs.append(fig_s_agg)
                
                # Add overall if multiple chunks
                if num_s_chunks > 1:
                    fig_s_agg_all = go.Figure()
                    fig_s_agg_all.add_trace(go.Bar(
                        x=score_agg_sorted['score_name'].tolist(),
                        y=score_agg_sorted['icc_2k'].tolist(),
                        marker_color='#2196F3',
                        hovertemplate='Score: %{x}<br>Mean ICC(2,k): %{y:.3f}<extra></extra>',
                        text=score_agg_sorted['icc_2k'].tolist(),
                        texttemplate='%{{text:.3f}}',
                        textposition='outside',
                        width=0.8
                    ))
                    fig_s_agg_all.update_layout(
                        title=f'Aggregated by Score (Average Across Agents) - All {len(score_agg_sorted)} Scores',
                        xaxis_title='Score Name',
                        yaxis_title='Mean ICC(2,k)',
                        yaxis=dict(range=[0, 1.1]),
                        height=max(500, len(score_agg_sorted) * 25),
                        xaxis=dict(tickangle=-45),
                        bargap=0.1
                    )
                    distilled_score_agg_figs.append(fig_s_agg_all)
            
            # View 3: Aggregated by agent (using Stability Index)
            dist_agent_df = distilled_results['agent_level']
            dist_agent_df_sorted = dist_agent_df.sort_values('agent_id').copy()
            dist_agent_df_sorted['stability_index'] = dist_agent_df_sorted['stability_index'].fillna(0)
            
            dist_a_chunk_size = 50
            dist_num_a_chunks = (len(dist_agent_df_sorted) + dist_a_chunk_size - 1) // dist_a_chunk_size
            
            # Option A: If only 1 chunk, show only overall. If multiple chunks, show subgraphs + overall
            if dist_num_a_chunks == 1:
                # Single chunk: show only overall graph
                fig_dist_a_agg_all = go.Figure()
                fig_dist_a_agg_all.add_trace(go.Bar(
                    x=dist_agent_df_sorted['agent_id'].tolist(),
                    y=dist_agent_df_sorted['stability_index'].tolist(),
                    marker_color='#2196F3',
                    hovertemplate='Agent: %{x}<br>Mean Stability Index: %{y:.3f}<extra></extra>',
                    text=dist_agent_df_sorted['stability_index'].tolist(),
                    texttemplate='%{{text:.3f}}',
                    textposition='outside',
                    width=0.8
                ))
                fig_dist_a_agg_all.update_layout(
                    title=f'Aggregated by Agent (Average Across Scores) - All {len(dist_agent_df_sorted)} Agents',
                    xaxis_title='Agent ID',
                    yaxis_title='Mean Stability Index',
                    yaxis=dict(range=[0, 1.1]),
                    height=max(500, len(dist_agent_df_sorted) * 15),
                    xaxis=dict(tickangle=-45),
                    bargap=0.1
                )
                distilled_agent_agg_figs.append(fig_dist_a_agg_all)
            else:
                # Multiple chunks: show subgraphs + overall
                for dist_a_chunk_idx in range(dist_num_a_chunks):
                    start_idx = dist_a_chunk_idx * dist_a_chunk_size
                    end_idx = min((dist_a_chunk_idx + 1) * dist_a_chunk_size, len(dist_agent_df_sorted))
                    
                    chunk_data = dist_agent_df_sorted.iloc[start_idx:end_idx]
                    chunk_agent_ids = chunk_data['agent_id'].tolist()
                    chunk_stability_values = chunk_data['stability_index'].tolist()
                    
                    fig_dist_a_agg = go.Figure()
                    fig_dist_a_agg.add_trace(go.Bar(
                        x=chunk_agent_ids,
                        y=chunk_stability_values,
                        marker_color='#2196F3',
                        hovertemplate='Agent: %{x}<br>Mean Stability Index: %{y:.3f}<extra></extra>',
                        text=chunk_stability_values,
                        texttemplate='%{{text:.3f}}',
                        textposition='outside',
                        width=0.8
                    ))
                    
                    title_suffix = f" (Agents {chunk_agent_ids[0]} to {chunk_agent_ids[-1]})"
                    fig_dist_a_agg.update_layout(
                        title=f'Aggregated by Agent (Average Across Scores){title_suffix}',
                        xaxis_title='Agent ID',
                        yaxis_title='Mean Stability Index',
                        yaxis=dict(range=[0, 1.1]),
                        height=max(500, len(chunk_data) * 15),
                        xaxis=dict(tickangle=-45),
                        bargap=0.1
                    )
                    distilled_agent_agg_figs.append(fig_dist_a_agg)
                
                # Add overall agent aggregated chart at the end
                fig_dist_a_agg_all = go.Figure()
                fig_dist_a_agg_all.add_trace(go.Bar(
                    x=dist_agent_df_sorted['agent_id'].tolist(),
                    y=dist_agent_df_sorted['stability_index'].tolist(),
                    marker_color='#2196F3',
                    hovertemplate='Agent: %{x}<br>Mean Stability Index: %{y:.3f}<extra></extra>',
                    text=dist_agent_df_sorted['stability_index'].tolist(),
                    texttemplate='%{{text:.3f}}',
                    textposition='outside',
                    width=0.8
                ))
                fig_dist_a_agg_all.update_layout(
                    title=f'Aggregated by Agent (Average Across Scores) - All {len(dist_agent_df_sorted)} Agents',
                    xaxis_title='Agent ID',
                    yaxis_title='Mean Stability Index',
                    yaxis=dict(range=[0, 1.1]),
                    height=max(500, len(dist_agent_df_sorted) * 15),
                    xaxis=dict(tickangle=-45),
                    bargap=0.1
                )
                distilled_agent_agg_figs.append(fig_dist_a_agg_all)
        
        except Exception as e:
            print(f"Warning: Could not create distilled visualizations: {e}")
            distilled_heatmap_figs = [go.Figure()]
            distilled_score_agg_figs = [go.Figure()]
            distilled_agent_agg_figs = [go.Figure()]
    
    # Generate Plotly HTML divs
    import re
    
    def extract_div(html_str):
        """Extract the div content from Plotly HTML output."""
        div_match = re.search(r'<div[^>]*id="[^"]*"[^>]*>.*?</div>', html_str, re.DOTALL)
        if div_match:
            return div_match.group(0)
        return html_str
    
    def extract_scripts(html_str):
        """Extract script tags from HTML."""
        script_pattern = r'<script[^>]*>.*?</script>'
        return re.findall(script_pattern, html_str, re.DOTALL)
    
    # Summary chart (include plotlyjs)
    summary_chart_html = fig_sections.to_html(full_html=False, include_plotlyjs='cdn', div_id='summary-charts')
    
    # Convert raw visualizations to HTML (only default view: raw-all)
    # For other views, we'll store the underlying data and generate charts client-side
    raw_heatmap_htmls = []
    for idx, fig in enumerate(raw_heatmap_figs):
        div_id = f'raw-heatmap-{idx}'
        html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)
        raw_heatmap_htmls.append(html)
    
    # For non-default views, create placeholder divs (charts will be generated client-side from data)
    raw_question_agg_htmls = []
    num_q_chunks = len(raw_question_agg_figs)
    for idx in range(num_q_chunks):
        div_id = f'raw-question-agg-{idx}'
        html = f'<div id="{div_id}" style="min-height: 500px;"></div>'
        raw_question_agg_htmls.append(html)
    
    raw_agent_agg_htmls = []
    num_a_chunks = len(raw_agent_agg_figs)
    for idx in range(num_a_chunks):
        div_id = f'raw-agent-agg-{idx}'
        html = f'<div id="{div_id}" style="min-height: 500px;"></div>'
        raw_agent_agg_htmls.append(html)
    
    # Convert distilled visualizations (all are non-default, so just placeholders)
    distilled_heatmap_htmls = []
    distilled_score_agg_htmls = []
    distilled_agent_agg_htmls = []
    
    if distilled_results is not None:
        num_dist_heatmap_chunks = len(distilled_heatmap_figs)
        for idx in range(num_dist_heatmap_chunks):
            div_id = f'distilled-heatmap-{idx}'
            html = f'<div id="{div_id}" style="min-height: 500px;"></div>'
            distilled_heatmap_htmls.append(html)
        
        num_dist_score_chunks = len(distilled_score_agg_figs)
        for idx in range(num_dist_score_chunks):
            div_id = f'distilled-score-agg-{idx}'
            html = f'<div id="{div_id}" style="min-height: 500px;"></div>'
            distilled_score_agg_htmls.append(html)
        
        num_dist_agent_chunks = len(distilled_agent_agg_figs)
        for idx in range(num_dist_agent_chunks):
            div_id = f'distilled-agent-agg-{idx}'
            html = f'<div id="{div_id}" style="min-height: 500px;"></div>'
            distilled_agent_agg_htmls.append(html)
    
    # Extract divs and create containers
    # For default view (raw heatmaps), extract full div with scripts
    raw_heatmap_divs = [extract_div(html) for html in raw_heatmap_htmls]
    # For non-default views, we already have placeholder divs
    raw_question_agg_divs = raw_question_agg_htmls  # These are already divs
    raw_agent_agg_divs = raw_agent_agg_htmls  # These are already divs
    
    # Create HTML containers for raw data
    raw_heatmap_container = '<div style="display: flex; flex-wrap: wrap; gap: 20px;">'
    for idx, div in enumerate(raw_heatmap_divs):
        is_overall = (idx == len(raw_heatmap_divs) - 1 and len(raw_heatmap_divs) > 1)
        width_style = 'width: 100%;' if is_overall else ('width: calc(50% - 10px);' if len(raw_heatmap_divs) > 1 else 'width: 100%;')
        raw_heatmap_container += f'<div style="{width_style} margin-bottom: 20px;">{div}</div>'
    raw_heatmap_container += '</div>'
    
    raw_question_agg_container = '<div style="display: flex; flex-wrap: wrap; gap: 20px;">'
    for idx, div in enumerate(raw_question_agg_divs):
        is_overall = (idx == len(raw_question_agg_divs) - 1 and len(raw_question_agg_divs) > 1)
        width_style = 'width: 100%;' if is_overall else ('width: calc(50% - 10px);' if len(raw_question_agg_divs) > 1 else 'width: 100%;')
        raw_question_agg_container += f'<div style="{width_style} margin-bottom: 20px;">{div}</div>'
    raw_question_agg_container += '</div>'
    
    raw_agent_agg_container = '<div style="display: flex; flex-wrap: wrap; gap: 20px;">'
    for idx, div in enumerate(raw_agent_agg_divs):
        is_overall = (idx == len(raw_agent_agg_divs) - 1 and len(raw_agent_agg_divs) > 1)
        width_style = 'width: 100%;' if is_overall else ('width: calc(50% - 10px);' if len(raw_agent_agg_divs) > 1 else 'width: 100%;')
        raw_agent_agg_container += f'<div style="{width_style} margin-bottom: 20px;">{div}</div>'
    raw_agent_agg_container += '</div>'
    
    # Create HTML containers for distilled data
    distilled_heatmap_container = '<div style="display: flex; flex-wrap: wrap; gap: 20px;">'
    distilled_question_agg_container = '<div style="display: flex; flex-wrap: wrap; gap: 20px;">'
    distilled_agent_agg_container = '<div style="display: flex; flex-wrap: wrap; gap: 20px;">'
    
    if distilled_results is not None:
        # For distilled views, we already have placeholder divs
        distilled_heatmap_divs = distilled_heatmap_htmls
        distilled_score_agg_divs = distilled_score_agg_htmls
        distilled_agent_agg_divs = distilled_agent_agg_htmls
        
        for idx, div in enumerate(distilled_heatmap_divs):
            is_overall = (idx == len(distilled_heatmap_divs) - 1 and len(distilled_heatmap_divs) > 1)
            width_style = 'width: 100%;' if is_overall else ('width: calc(50% - 10px);' if len(distilled_heatmap_divs) > 1 else 'width: 100%;')
            distilled_heatmap_container += f'<div style="{width_style} margin-bottom: 20px;">{div}</div>'
        distilled_heatmap_container += '</div>'
        
        for idx, div in enumerate(distilled_score_agg_divs):
            is_overall = (idx == len(distilled_score_agg_divs) - 1 and len(distilled_score_agg_divs) > 1)
            width_style = 'width: 100%;' if is_overall else ('width: calc(50% - 10px);' if len(distilled_score_agg_divs) > 1 else 'width: 100%;')
            distilled_question_agg_container += f'<div style="{width_style} margin-bottom: 20px;">{div}</div>'
        distilled_question_agg_container += '</div>'
        
        for idx, div in enumerate(distilled_agent_agg_divs):
            is_overall = (idx == len(distilled_agent_agg_divs) - 1 and len(distilled_agent_agg_divs) > 1)
            width_style = 'width: 100%;' if is_overall else ('width: calc(50% - 10px);' if len(distilled_agent_agg_divs) > 1 else 'width: 100%;')
            distilled_agent_agg_container += f'<div style="{width_style} margin-bottom: 20px;">{div}</div>'
        distilled_agent_agg_container += '</div>'
    
    # Collect scripts only for default view (raw heatmaps)
    all_scripts = []
    for html in raw_heatmap_htmls:
        all_scripts.extend(extract_scripts(html))
    
    # Store underlying data (DataFrames) as JSON for client-side chart generation
    # This is much smaller than storing full chart JSON
    import json as json_module
    
    # Prepare data for client-side chart generation
    raw_question_agg_data = raw_results.get('question_aggregated', pd.DataFrame())
    raw_agent_agg_data = raw_results.get('agent_level', pd.DataFrame())
    
    # Convert DataFrames to JSON (only the columns we need)
    # question_aggregated uses standard ICC(2,k), agent_aggregated uses Stability Index
    raw_question_agg_json = raw_question_agg_data[['question_id', 'section_id', 'icc_2k']].to_dict('records') if not raw_question_agg_data.empty else []
    raw_agent_agg_json = raw_agent_agg_data[['agent_id', 'stability_index']].to_dict('records') if not raw_agent_agg_data.empty else []
    
    chart_data_js = f"""
        // Chart data for dynamic loading (stored as lightweight data, not full chart JSON)
        var chartData = {{
            raw: {{
                questionAgg: {json_module.dumps(raw_question_agg_json)},
                agentAgg: {json_module.dumps(raw_agent_agg_json)}
            }}
    """
    
    if distilled_results is not None:
        # For distilled, we need the score-level data for heatmaps
        distilled_score_level_data = distilled_results.get('score_level', pd.DataFrame())
        distilled_score_agg_data = distilled_results.get('score_aggregated', pd.DataFrame())
        distilled_agent_agg_data = distilled_results.get('agent_level', pd.DataFrame())
        
        # For heatmaps, we need pivot table data (using Stability Index)
        distilled_heatmap_data = []
        if not distilled_score_level_data.empty:
            # Create a simplified representation of the pivot table
            pivot_data = distilled_score_level_data.pivot_table(
                index='score_name',
                columns='agent_id',
                values='stability_index',
                aggfunc='mean'
            ).fillna(0)
            # Convert to a more compact format
            distilled_heatmap_data = {
                'scores': pivot_data.index.tolist(),
                'agents': pivot_data.columns.tolist(),
                'values': pivot_data.values.tolist()
            }
        
        # score_aggregated uses standard ICC(2,k), agent_aggregated uses Stability Index
        distilled_score_agg_json = distilled_score_agg_data[['score_name', 'section_id', 'icc_2k']].to_dict('records') if not distilled_score_agg_data.empty else []
        distilled_agent_agg_json = distilled_agent_agg_data[['agent_id', 'stability_index']].to_dict('records') if not distilled_agent_agg_data.empty else []
        
        chart_data_js += f""",
            distilled: {{
                heatmap: {json_module.dumps(distilled_heatmap_data)},
                scoreAgg: {json_module.dumps(distilled_score_agg_json)},
                agentAgg: {json_module.dumps(distilled_agent_agg_json)}
            }}
        }};
        
        // Function to generate bar chart from data
        function generateBarChart(divId, data, xKey, yKey, title, color) {{
            var plotDiv = document.getElementById(divId);
            if (!plotDiv || !data || data.length === 0) return;
            
            var xValues = data.map(function(d) {{ return d[xKey]; }});
            var yValues = data.map(function(d) {{ return d[yKey] || 0; }});
            
            // Determine y-axis label based on yKey
            var yAxisLabel = (yKey === 'icc_2k') ? 'Mean ICC(2,k)' : 'Mean Stability Index';
            var hoverLabel = (yKey === 'icc_2k') ? 'Mean ICC(2,k)' : 'Mean Stability Index';
            
            var trace = {{
                x: xValues,
                y: yValues,
                type: 'bar',
                marker: {{color: color}},
                hovertemplate: xKey + ': %{{x}}<br>' + hoverLabel + ': %{{y:.3f}}<extra></extra>',
                text: yValues,
                texttemplate: '%{{text:.3f}}',
                textposition: 'outside',
                width: 0.8
            }};
            
            var layout = {{
                title: {{text: title}},
                xaxis: {{title: xKey, tickangle: -45}},
                yaxis: {{title: yAxisLabel, range: [0, 1.1]}},
                height: Math.max(500, data.length * 25),
                bargap: 0.1
            }};
            
            Plotly.newPlot(divId, [trace], layout);
        }}
        
        // Function to generate heatmap from pivot data
        function generateHeatmap(divId, pivotData, title, colorScale) {{
            var plotDiv = document.getElementById(divId);
            if (!plotDiv || !pivotData || !pivotData.values) return;
            
            var trace = {{
                z: pivotData.values,
                x: pivotData.agents,
                y: pivotData.scores,
                type: 'heatmap',
                colorscale: colorScale,
                colorbar: {{title: "Stability Index"}},
                hovertemplate: '<b>Agent ID:</b> %{{x}}<br><b>Score:</b> %{{y}}<br><b>Stability Index:</b> %{{z:.3f}}<extra></extra>'
            }};
            
            var layout = {{
                title: {{text: title}},
                xaxis: {{title: 'Agent ID', type: 'category', tickangle: -45}},
                yaxis: {{title: 'Score Name', type: 'category'}},
                height: 600
            }};
            
            Plotly.newPlot(divId, [trace], layout);
        }}
        
        // Function to load charts for a view
        function loadChartsForView(dataType, viewType) {{
            if (typeof chartData === 'undefined') return;
            
            if (dataType === 'raw') {{
                if (viewType === 'question-agg') {{
                    var container = document.getElementById('question-view-raw-question-agg');
                    var divs = container.querySelectorAll('[id^="raw-question-agg-"]');
                    var data = chartData.raw.questionAgg;
                    
                    // Split data into chunks of 50
                    var chunkSize = 50;
                    for (var i = 0; i < divs.length; i++) {{
                        var startIdx = i * chunkSize;
                        var endIdx = Math.min((i + 1) * chunkSize, data.length);
                        var chunkData = data.slice(startIdx, endIdx);
                        
                        var isOverall = (i === divs.length - 1 && divs.length > 1);
                        var title = isOverall ? 
                            'Aggregated by Question (Average Across Agents) - All ' + data.length + ' Questions' :
                            'Aggregated by Question (Average Across Agents) (Questions ' + chunkData[0].question_id + ' to ' + chunkData[chunkData.length - 1].question_id + ')';
                        
                        generateBarChart(divs[i].id, chunkData, 'question_id', 'icc_2k', title, '#4CAF50');
                    }}
                }} else if (viewType === 'agent-agg') {{
                    var container = document.getElementById('question-view-raw-agent-agg');
                    var divs = container.querySelectorAll('[id^="raw-agent-agg-"]');
                    var data = chartData.raw.agentAgg;
                    
                    // Split data into chunks of 50
                    var chunkSize = 50;
                    for (var i = 0; i < divs.length; i++) {{
                        var isOverall = (i === divs.length - 1 && divs.length > 1);
                        var chunkData, title;
                        
                        if (isOverall) {{
                            // Overall graph: use all data
                            chunkData = data;
                            title = 'Aggregated by Agent (Average Across Questions) - All ' + data.length + ' Agents';
                        }} else {{
                            // Subgraph: use chunk of 50
                            var startIdx = i * chunkSize;
                            var endIdx = Math.min((i + 1) * chunkSize, data.length);
                            chunkData = data.slice(startIdx, endIdx);
                            title = 'Aggregated by Agent (Average Across Questions) (Agents ' + chunkData[0].agent_id + ' to ' + chunkData[chunkData.length - 1].agent_id + ')';
                        }}
                        
                        generateBarChart(divs[i].id, chunkData, 'agent_id', 'stability_index', title, '#4CAF50');
                    }}
                }}
            }} else if (dataType === 'distilled' && typeof chartData.distilled !== 'undefined') {{
                if (viewType === 'all') {{
                    var container = document.getElementById('question-view-distilled-all');
                    var divs = container.querySelectorAll('[id^="distilled-heatmap-"]');
                    var pivotData = chartData.distilled.heatmap;
                    
                    if (pivotData && pivotData.values) {{
                        // Split agents into chunks of 50
                        var chunkSize = 50;
                        var numChunks = Math.ceil(pivotData.agents.length / chunkSize);
                        
                        for (var i = 0; i < divs.length; i++) {{
                            var isOverall = (i === divs.length - 1 && divs.length > 1);
                            var chunkAgents, chunkValues, title;
                            
                            if (isOverall) {{
                                // Overall graph: use all agents and all values
                                chunkAgents = pivotData.agents;
                                chunkValues = pivotData.values;
                                title = 'All Agent-Score Pairs (All ' + pivotData.agents.length + ' Agents)';
                            }} else {{
                                // Subgraph: use chunk of 50 agents
                                var startIdx = i * chunkSize;
                                var endIdx = Math.min((i + 1) * chunkSize, pivotData.agents.length);
                                chunkAgents = pivotData.agents.slice(startIdx, endIdx);
                                chunkValues = pivotData.values.map(function(row) {{
                                    return row.slice(startIdx, endIdx);
                                }});
                                title = 'All Agent-Score Pairs (Agents ' + chunkAgents[0] + ' to ' + chunkAgents[chunkAgents.length - 1] + ')';
                            }}
                            
                            generateHeatmap(divs[i].id, {{
                                scores: pivotData.scores,
                                agents: chunkAgents,
                                values: chunkValues
                            }}, title, 'Blues');
                        }}
                    }}
                }} else if (viewType === 'question-agg') {{
                    var container = document.getElementById('question-view-distilled-question-agg');
                    var divs = container.querySelectorAll('[id^="distilled-score-agg-"]');
                    var data = chartData.distilled.scoreAgg;
                    
                    var chunkSize = 50;
                    for (var i = 0; i < divs.length; i++) {{
                        var startIdx = i * chunkSize;
                        var endIdx = Math.min((i + 1) * chunkSize, data.length);
                        var chunkData = data.slice(startIdx, endIdx);
                        
                        var isOverall = (i === divs.length - 1 && divs.length > 1);
                        var title = isOverall ? 
                            'Aggregated by Score (Average Across Agents) - All ' + data.length + ' Scores' :
                            'Aggregated by Score (Average Across Agents) (Scores ' + chunkData[0].score_name + ' to ' + chunkData[chunkData.length - 1].score_name + ')';
                        
                        generateBarChart(divs[i].id, chunkData, 'score_name', 'icc_2k', title, '#2196F3');
                    }}
                }} else if (viewType === 'agent-agg') {{
                    var container = document.getElementById('question-view-distilled-agent-agg');
                    var divs = container.querySelectorAll('[id^="distilled-agent-agg-"]');
                    var data = chartData.distilled.agentAgg;
                    
                    var chunkSize = 50;
                    for (var i = 0; i < divs.length; i++) {{
                        var isOverall = (i === divs.length - 1 && divs.length > 1);
                        var chunkData, title;
                        
                        if (isOverall) {{
                            // Overall graph: use all data
                            chunkData = data;
                            title = 'Aggregated by Agent (Average Across Scores) - All ' + data.length + ' Agents';
                        }} else {{
                            // Subgraph: use chunk of 50
                            var startIdx = i * chunkSize;
                            var endIdx = Math.min((i + 1) * chunkSize, data.length);
                            chunkData = data.slice(startIdx, endIdx);
                            title = 'Aggregated by Agent (Average Across Scores) (Agents ' + chunkData[0].agent_id + ' to ' + chunkData[chunkData.length - 1].agent_id + ')';
                        }}
                        
                        generateBarChart(divs[i].id, chunkData, 'agent_id', 'stability_index', title, '#2196F3');
                    }}
                }}
            }}
        }}
    """
    question_level_content = f'''
        <div id="question-view-raw-all" style="display: block;">{raw_heatmap_container}</div>
        <div id="question-view-raw-question-agg" style="display: none;">{raw_question_agg_container}</div>
        <div id="question-view-raw-agent-agg" style="display: none;">{raw_agent_agg_container}</div>
    '''
    if distilled_results is not None:
        question_level_content += f'''
        <div id="question-view-distilled-all" style="display: none;">{distilled_heatmap_container}</div>
        <div id="question-view-distilled-question-agg" style="display: none;">{distilled_question_agg_container}</div>
        <div id="question-view-distilled-agent-agg" style="display: none;">{distilled_agent_agg_container}</div>
        '''
    
    # Build final HTML
    # Replace js_code placeholder (in case f-string interpolation didn't work)
    html_content = html_content.replace('{js_code}', js_code)
    # Generate Metric Diagnostics tables
    import json as json_module
    
    # Raw questions diagnostic table
    raw_question_agg_data = raw_results.get('question_aggregated', pd.DataFrame())
    metric_diagnostics_raw_rows = []
    if not raw_question_agg_data.empty:
        for _, row in raw_question_agg_data.iterrows():
            metric_diagnostics_raw_rows.append({
                'Question ID': row['question_id'],
                'Section': row['section_id'],
                'MS_R': f"{row['ms_rows']:.2f}" if not pd.isna(row['ms_rows']) else 'N/A',
                'MS_C': f"{row['ms_cols']:.2f}" if not pd.isna(row['ms_cols']) else 'N/A',
                'MS_E': f"{row['ms_error']:.2f}" if not pd.isna(row['ms_error']) else 'N/A',
                'ICC(2,k)': f"{row['icc_2k']:.4f}" if not pd.isna(row['icc_2k']) else 'N/A',
                'ICC(3,k)': f"{row['icc_3k']:.4f}" if not pd.isna(row['icc_3k']) else 'N/A',
                'Stability Index': f"{row['mean_stability_index']:.4f}" if not pd.isna(row['mean_stability_index']) else 'N/A',
                "Cronbach's α": f"{row['cronbach_alpha']:.4f}" if not pd.isna(row['cronbach_alpha']) else 'N/A',
                'Min': f"{row['min_value']:.2f}" if not pd.isna(row['min_value']) else 'N/A',
                'Max': f"{row['max_value']:.2f}" if not pd.isna(row['max_value']) else 'N/A',
                'N Agents': int(row['n_agents']) if not pd.isna(row['n_agents']) else 'N/A'
            })
        metric_diagnostics_raw_df = pd.DataFrame(metric_diagnostics_raw_rows)
        # Sort by ICC(2,k) to see lowest values first
        metric_diagnostics_raw_df['ICC(2,k)_num'] = pd.to_numeric(metric_diagnostics_raw_df['ICC(2,k)'], errors='coerce')
        metric_diagnostics_raw_df = metric_diagnostics_raw_df.sort_values('ICC(2,k)_num', na_position='last')
        metric_diagnostics_raw_df = metric_diagnostics_raw_df.drop('ICC(2,k)_num', axis=1)
        metric_diagnostics_raw_table = metric_diagnostics_raw_df.to_html(classes='data-table', index=False, escape=False)
        metric_diagnostics_raw_table_escaped = json_module.dumps(metric_diagnostics_raw_table).replace('</script>', '<\\/script>')
    else:
        metric_diagnostics_raw_table_escaped = json_module.dumps('<p>No data available</p>').replace('</script>', '<\\/script>')
    
    # Distilled scores diagnostic table
    metric_diagnostics_distilled_table_escaped = None
    if distilled_results is not None:
        distilled_score_agg_data = distilled_results.get('score_aggregated', pd.DataFrame())
        metric_diagnostics_distilled_rows = []
        if not distilled_score_agg_data.empty:
            for _, row in distilled_score_agg_data.iterrows():
                metric_diagnostics_distilled_rows.append({
                    'Score Name': row['score_name'],
                    'Section': row['section_id'],
                    'MS_R': f"{row['ms_rows']:.2f}" if not pd.isna(row['ms_rows']) else 'N/A',
                    'MS_C': f"{row['ms_cols']:.2f}" if not pd.isna(row['ms_cols']) else 'N/A',
                    'MS_E': f"{row['ms_error']:.2f}" if not pd.isna(row['ms_error']) else 'N/A',
                    'ICC(2,k)': f"{row['icc_2k']:.4f}" if not pd.isna(row['icc_2k']) else 'N/A',
                    'ICC(3,k)': f"{row['icc_3k']:.4f}" if not pd.isna(row['icc_3k']) else 'N/A',
                    'Stability Index': f"{row['mean_stability_index']:.4f}" if not pd.isna(row['mean_stability_index']) else 'N/A',
                    "Cronbach's α": f"{row['cronbach_alpha']:.4f}" if not pd.isna(row['cronbach_alpha']) else 'N/A',
                    'Min': f"{row['min_value']:.2f}" if not pd.isna(row['min_value']) else 'N/A',
                    'Max': f"{row['max_value']:.2f}" if not pd.isna(row['max_value']) else 'N/A',
                    'N Agents': int(row['n_agents']) if not pd.isna(row['n_agents']) else 'N/A'
                })
            metric_diagnostics_distilled_df = pd.DataFrame(metric_diagnostics_distilled_rows)
            # Sort by ICC(2,k) to see lowest values first
            metric_diagnostics_distilled_df['ICC(2,k)_num'] = pd.to_numeric(metric_diagnostics_distilled_df['ICC(2,k)'], errors='coerce')
            metric_diagnostics_distilled_df = metric_diagnostics_distilled_df.sort_values('ICC(2,k)_num', na_position='last')
            metric_diagnostics_distilled_df = metric_diagnostics_distilled_df.drop('ICC(2,k)_num', axis=1)
            metric_diagnostics_distilled_table = metric_diagnostics_distilled_df.to_html(classes='data-table', index=False, escape=False)
            metric_diagnostics_distilled_table_escaped = json_module.dumps(metric_diagnostics_distilled_table).replace('</script>', '<\\/script>')
        else:
            metric_diagnostics_distilled_table_escaped = json_module.dumps('<p>No data available</p>').replace('</script>', '<\\/script>')
    
    html_content = html_content.replace('<div id="summary-metrics"></div>', f'<div id="summary-metrics">{summary_html}</div>')
    html_content = html_content.replace('<div id="summary-charts"></div>', summary_chart_html)
    html_content = html_content.replace('<div id="question-level-content-container"></div>', f'<div id="question-level-content-container">{question_level_content}</div>')
    html_content = html_content.replace('<div id="metric-diagnostics-raw-container"></div>', f'<h4>Raw Questions</h4><div id="metric-diagnostics-raw-table"></div>')
    if distilled_results is not None:
        html_content = html_content.replace('<div id="metric-diagnostics-distilled-container"></div>', f'<h4>Distilled Scores</h4><div id="metric-diagnostics-distilled-table"></div>')
    
    # Add scripts for raw heatmaps (default view) - these scripts render the graphs
    # Also add chart data JavaScript for dynamic loading
    # Escape script tags in the JavaScript code to prevent syntax errors when embedded in HTML
    chart_data_js_escaped = chart_data_js.replace('</script>', '<\\/script>')
    
    # Wrap raw heatmap scripts in DOMContentLoaded to ensure DOM is ready
    # The scripts from Plotly check for element existence, but they execute in <head> before body is parsed
    # Extract just the JavaScript content (strip the <script> tags)
    wrapped_scripts = []
    if all_scripts:
        for script in all_scripts:
            # Extract JavaScript content from <script>...</script> tag
            # Remove the opening and closing script tags
            script_content = re.sub(r'^<script[^>]*>', '', script, flags=re.IGNORECASE | re.DOTALL)
            script_content = re.sub(r'</script>\s*$', '', script_content, flags=re.IGNORECASE | re.DOTALL)
            # Wrap the JavaScript content in DOMContentLoaded
            wrapped_script = f'''<script type="text/javascript">
        document.addEventListener('DOMContentLoaded', function() {{
            {script_content}
        }});
    </script>'''
            wrapped_scripts.append(wrapped_script)
    
    # Combine wrapped scripts with chart data JavaScript
    scripts_to_insert = []
    if wrapped_scripts:
        scripts_to_insert.extend(wrapped_scripts)
    scripts_to_insert.append(f'<script>\n{chart_data_js_escaped}\n    </script>')
    
    # Insert all scripts right after the Plotly CDN script tag
    all_scripts_html = '\n    '.join(scripts_to_insert)
    html_content = html_content.replace(
        '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
        '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>\n    ' + all_scripts_html,
        1
    )
    
    # Add distilled visualizations if available
    if distilled_results is not None and df_distilled is not None:
        # Distilled section comparison chart (using Stability Index)
        dist_overall_section = distilled_results['overall_section']
        dist_sections = dist_overall_section['section_id'].tolist()
        dist_stability_index_vals = dist_overall_section['mean_stability_index'].fillna(0).tolist()
        dist_alpha_vals = score_aggregated['cronbach_alpha'].fillna(0).tolist()
        
        fig_dist_sections = go.Figure()
        fig_dist_sections.add_trace(go.Bar(
            x=dist_sections,
            y=dist_stability_index_vals,
            name='Stability Index',
            marker_color='#2196F3',
            hovertemplate='Section: %{x}<br>Stability Index: %{y:.3f}<extra></extra>'
                ))
        fig_dist_sections.add_trace(go.Bar(
            x=dist_sections,
            y=dist_alpha_vals,
            name="Cronbach's α",
            marker_color='#1976D2',
            hovertemplate='Section: %{x}<br>Cronbach\'s α: %{y:.3f}<extra></extra>'
                ))
        fig_dist_sections.update_layout(
            title='Distilled Scores Consistency by Section (Stability Index & Cronbach\'s α)',
            xaxis_title='Section',
            yaxis_title='Value',
            yaxis=dict(range=[0, 1]),
            barmode='group',
            height=400,
            hovermode='x unified'
        )
        
        # Distilled trajectories - comprehensive view
        # Define all 7 scores with distinct colors
        score_colors = {
            'Extraversion': '#1f77b4',      # Blue
            'Agreeableness': '#ff7f0e',     # Orange
            'Conscientiousness': '#2ca02c', # Green
            'Neuroticism': '#d62728',       # Red
            'Openness': '#9467bd',          # Purple
            'CE_normalized': '#8c564b',     # Brown
            'RA_normalized': '#e377c2'      # Pink
        }
        
        all_score_names = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness', 'CE_normalized', 'RA_normalized']
        
        # Organize scores by instrument
        bfi10_scores = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
        besa_scores = ['CE_normalized']
        rei_scores = ['RA_normalized']
        
        # Collect all trajectory data
        trajectory_data_by_agent = {}
        for agent_id in agent_ids:
            agent_data = df_distilled[df_distilled['agent_id'] == agent_id]
            agent_scores = {}
            
            for score_name in all_score_names:
                score_data = agent_data[agent_data['score_name'] == score_name]
                if not score_data.empty:
                    score_data = score_data.sort_values('attempt_id')
                    score_data = score_data[~score_data['score_value'].isna()]
                    if len(score_data) > 0:
                        agent_scores[score_name] = {
                            'attempts': score_data['attempt_id'].tolist(),
                            'scores': score_data['score_value'].tolist()
                        }
            
            if agent_scores:
                trajectory_data_by_agent[agent_id] = agent_scores
        
        # 1. Create "View All" combined chart (all agents, all scores) with instrument filtering
        fig_all_combined = go.Figure()
        trace_indices = {'bfi10': [], 'besa': [], 'rei': []}
        trace_idx = 0
        
        for agent_id in agent_ids:
            if agent_id in trajectory_data_by_agent:
                agent_scores = trajectory_data_by_agent[agent_id]
                for score_name in all_score_names:
                    if score_name in agent_scores:
                        data = agent_scores[score_name]
                        # Use different line styles to distinguish agents
                        dash_style = 'solid' if agent_id == agent_ids[0] else 'dot'
                        fig_all_combined.add_trace(go.Scatter(
                            x=data['attempts'],
                            y=data['scores'],
                            mode='lines+markers',
                            name=f"{agent_id}-{score_name}",
                            line=dict(color=score_colors.get(score_name, '#000000'), width=2, dash=dash_style),
                            marker=dict(size=5),
                            legendgroup=agent_id,
                            showlegend=True
                        ))
                        # Track which traces belong to which instrument
                        if score_name in bfi10_scores:
                            trace_indices['bfi10'].append(trace_idx)
                        elif score_name in besa_scores:
                            trace_indices['besa'].append(trace_idx)
                        elif score_name in rei_scores:
                            trace_indices['rei'].append(trace_idx)
                        trace_idx += 1
        
        # Create visibility lists for filtering
        total_traces = len(fig_all_combined.data)
        all_visible = [True] * total_traces
        bfi10_visible = [False] * total_traces
        besa_visible = [False] * total_traces
        rei_visible = [False] * total_traces
        
        for idx in trace_indices['bfi10']:
            bfi10_visible[idx] = True
        for idx in trace_indices['besa']:
            besa_visible[idx] = True
        for idx in trace_indices['rei']:
            rei_visible[idx] = True
        
        fig_all_combined.update_layout(
            title='All Agents: Distilled Scores Trajectories (Combined View)',
            xaxis_title='Attempt ID',
            yaxis_title='Score Value',
            yaxis=dict(range=[0, 5]),
            height=600,
            hovermode='closest',
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        
        # 2. Create individual charts for each agent
        individual_charts_html = []
        agent_options_html = []
        all_traj_scripts = []
        
        for agent_id in agent_ids:
            if agent_id in trajectory_data_by_agent:
                agent_options_html.append(f'                    <option value="{agent_id}">Agent {agent_id}</option>')
                
                fig_agent = go.Figure()
                agent_scores = trajectory_data_by_agent[agent_id]
                
                trace_indices_agent = {'bfi10': [], 'besa': [], 'rei': []}
                trace_idx_agent = 0
                
                for score_name in all_score_names:
                    if score_name in agent_scores:
                        data = agent_scores[score_name]
                        fig_agent.add_trace(go.Scatter(
                            x=data['attempts'],
                            y=data['scores'],
                            mode='lines+markers',
                            name=score_name,
                            line=dict(color=score_colors.get(score_name, '#000000'), width=3),
                            marker=dict(size=8)
                        ))
                        # Track which traces belong to which instrument
                        if score_name in bfi10_scores:
                            trace_indices_agent['bfi10'].append(trace_idx_agent)
                        elif score_name in besa_scores:
                            trace_indices_agent['besa'].append(trace_idx_agent)
                        elif score_name in rei_scores:
                            trace_indices_agent['rei'].append(trace_idx_agent)
                        trace_idx_agent += 1
                
                # Create visibility lists for this agent's chart
                total_traces_agent = len(fig_agent.data)
                all_visible_agent = [True] * total_traces_agent
                bfi10_visible_agent = [False] * total_traces_agent
                besa_visible_agent = [False] * total_traces_agent
                rei_visible_agent = [False] * total_traces_agent
                
                for idx in trace_indices_agent['bfi10']:
                    bfi10_visible_agent[idx] = True
                for idx in trace_indices_agent['besa']:
                    besa_visible_agent[idx] = True
                for idx in trace_indices_agent['rei']:
                    rei_visible_agent[idx] = True
                
                fig_agent.update_layout(
                    title=f'Agent {agent_id}: All Distilled Scores Trajectories',
                    xaxis_title='Attempt ID',
                    yaxis_title='Score Value',
                    yaxis=dict(range=[0, 5]),
                    height=500,
                    hovermode='closest',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                # Generate HTML for this agent's chart (include full HTML with scripts)
                agent_chart_html = fig_agent.to_html(full_html=False, include_plotlyjs=False, div_id=f'traj-indiv-{agent_id}')
                # Wrap in container div with display control
                individual_charts_html.append(f'            <div id="traj-indiv-{agent_id}" style="display: none; margin-bottom: 30px;">{agent_chart_html}</div>')
        
        # Generate HTML for combined chart (include full HTML with scripts)
        all_combined_html = fig_all_combined.to_html(full_html=False, include_plotlyjs=False, div_id='traj-all-combined')
        
        # Update datalist options in HTML
        agent_options_str = '\n'.join(agent_options_html)
        html_content = html_content.replace(
            '<datalist id="agent-list">\n                    <option value="all">View All</option>\n                </datalist>',
            f'<datalist id="agent-list">\n                    <option value="all">View All</option>\n{agent_options_str}\n                </datalist>'
        )
        
        # Add trajectory charts to HTML (with scripts included in each div)
        trajectory_charts_html = f'''            <div id="traj-all-combined" style="display: block; margin-bottom: 40px;">
                {all_combined_html}
            </div>
{''.join(individual_charts_html)}
        '''
        html_content = html_content.replace('<div id="trajectory-charts-container"></div>', f'<div id="trajectory-charts-container">\n{trajectory_charts_html}</div>')
        
        # Generate HTML for distilled summary charts
        dist_summary_chart_html = fig_dist_sections.to_html(full_html=False, include_plotlyjs=False, div_id='distilled-summary-charts')
        
        # Replace distilled summary placeholders
        html_content = html_content.replace('<div id="distilled-summary-metrics"></div>', f'<div id="distilled-summary-metrics">{distilled_summary_html}</div>')
        html_content = html_content.replace('<div id="distilled-summary-charts"></div>', dist_summary_chart_html)
    
    # Add additional scripts if needed
    if all_scripts:
        script_section = '\n'.join(all_scripts)
        html_content = html_content.replace('</body>', f'{script_section}\n</body>')
    
    # Add JavaScript to populate metric diagnostics tables
    metric_diagnostics_js = f"""
        document.getElementById('metric-diagnostics-raw-table').innerHTML = {metric_diagnostics_raw_table_escaped};
        """ + (f"""
        document.getElementById('metric-diagnostics-distilled-table').innerHTML = {metric_diagnostics_distilled_table_escaped};
        """ if metric_diagnostics_distilled_table_escaped is not None else "") + """
    """
    html_content = html_content.replace('</body>', f'<script>{metric_diagnostics_js}</script>\n</body>')
    
    # Add distribution analysis content if available
    if raw_distribution_results is not None or distilled_distribution_results is not None:
        # Load survey questions for labels
        survey_data = load_survey_questions()
        question_labels = {}
        if survey_data:
            for section in survey_data.get("sections", []):
                for q in section.get("questions", []):
                    question_labels[q["question_id"]] = q["question_text"]
        
        # Populate distribution summary tab
        dist_summary_html, dist_edge_cases_html = generate_distribution_summary_html(raw_distribution_results, distilled_distribution_results)
        html_content = html_content.replace('<div id="distribution-summary-stats"></div>', f'<div id="distribution-summary-stats">{dist_summary_html}</div>')
        html_content = html_content.replace('<div id="distribution-summary-edge-cases"></div>', f'<div id="distribution-summary-edge-cases">{dist_edge_cases_html}</div>')
        
        # Populate distribution question level tab with visualizations
        # Import plotly here since it's needed for visualizations
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            if raw_distribution_results:
                # Generate raw distribution visualizations
                raw_dist_html = generate_raw_distribution_visualizations(raw_distribution_results, question_labels)
                html_content = html_content.replace('<div id="distribution-raw-content"></div>', f'<div id="distribution-raw-content">{raw_dist_html}</div>')
            
            if distilled_distribution_results:
                # Generate distilled distribution visualizations
                distilled_dist_html = generate_distilled_distribution_visualizations(distilled_distribution_results)
                html_content = html_content.replace('<div id="distribution-distilled-content"></div>', f'<div id="distribution-distilled-content">{distilled_dist_html}</div>')
        except ImportError:
            # If plotly not available, add basic message
            if raw_distribution_results:
                html_content = html_content.replace('<div id="distribution-raw-content"></div>', '<div id="distribution-raw-content"><p>Plotly not available for visualizations.</p></div>')
            if distilled_distribution_results:
                html_content = html_content.replace('<div id="distribution-distilled-content"></div>', '<div id="distribution-distilled-content"><p>Plotly not available for visualizations.</p></div>')
    
    html_content += """
    </div>
</body>
</html>
"""
    
    # Validate that JavaScript was properly inserted
    if "{js_code}" in html_content:
        raise ValueError("ERROR: js_code placeholder was not replaced in HTML template. Check f-string interpolation.")
    if "function showTab" not in html_content:
        raise ValueError("ERROR: showTab function not found in generated HTML.")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report saved to: {output_path}")


def generate_basic_html_report(raw_results: Dict, df_all: pd.DataFrame, output_path: str,
                               agent_ids: List[str], attempt_ids: List[int],
                               distilled_results: Optional[Dict] = None, df_distilled: Optional[pd.DataFrame] = None):
    """Generate basic HTML report without Plotly (fallback)."""
    distilled_section = ""
    if distilled_results is not None:
        distilled_section = f"""
    <h2>Distilled Scores: Score-Level Results</h2>
    {distilled_results['score_level'].to_html(index=False, na_rep='N/A')}
    
    <h2>Distilled Scores: Section-Level Results</h2>
    {distilled_results['section_level'].to_html(index=False, na_rep='N/A')}
    
    <h2>Distilled Scores: Agent-Level Results</h2>
    {distilled_results['agent_level'].to_html(index=False, na_rep='N/A')}
    
    <h2>Distilled Scores: Overall Section Summary</h2>
    {distilled_results['overall_section'].to_html(index=False, na_rep='N/A')}
"""
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Survey Consistency Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <h1>Survey Consistency Analysis Report</h1>
    <p><strong>Agents:</strong> {', '.join(agent_ids)}</p>
    <p><strong>Attempts:</strong> {min(attempt_ids)}-{max(attempt_ids)}</p>
    
    <h2>Raw Responses: Question-Level Results</h2>
    {raw_results['question_level'].to_html(index=False, na_rep='N/A')}
    
    <h2>Raw Responses: Section-Level Results</h2>
    {raw_results['section_level'].to_html(index=False, na_rep='N/A')}
    
    <h2>Raw Responses: Agent-Level Results</h2>
    {raw_results['agent_level'].to_html(index=False, na_rep='N/A')}
    
    <h2>Raw Responses: Overall Section Summary</h2>
    {raw_results['overall_section'].to_html(index=False, na_rep='N/A')}
{distilled_section}
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Basic HTML report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze consistency of survey responses across multiple attempts"
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
        default='consistency_report.html',
        help='Output HTML file path (default: consistency_report.html)'
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
        raw_results, df_all, distilled_results, df_distilled, raw_distribution_results, distilled_distribution_results = analyze_consistency(
            args.base, agent_ids, attempt_ids, include_distilled=True
        )
        
        # Generate HTML report
        generate_html_report(raw_results, df_all, args.output, agent_ids, attempt_ids,
                            distilled_results, df_distilled, raw_distribution_results, distilled_distribution_results)
        
        # Save CSV if requested
        if args.csv:
            os.makedirs(args.csv, exist_ok=True)
            # Raw results
            raw_results['question_level'].to_csv(f"{args.csv}/raw_question_level.csv", index=False)
            raw_results['section_level'].to_csv(f"{args.csv}/raw_section_level.csv", index=False)
            raw_results['agent_level'].to_csv(f"{args.csv}/raw_agent_level.csv", index=False)
            raw_results['overall_section'].to_csv(f"{args.csv}/raw_overall_section.csv", index=False)
            
            # Distilled results if available
            if distilled_results is not None:
                distilled_results['score_level'].to_csv(f"{args.csv}/distilled_score_level.csv", index=False)
                distilled_results['section_level'].to_csv(f"{args.csv}/distilled_section_level.csv", index=False)
                distilled_results['agent_level'].to_csv(f"{args.csv}/distilled_agent_level.csv", index=False)
                distilled_results['overall_section'].to_csv(f"{args.csv}/distilled_overall_section.csv", index=False)
            
            print(f"CSV files saved to: {args.csv}")
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

