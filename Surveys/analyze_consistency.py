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


def compute_icc(data: np.ndarray) -> Tuple[float, float]:
    """
    Compute ICC(2,k) and ICC(3,k) for multiple attempts.
    
    For single item across attempts (test-retest reliability):
    - Uses variance-based approach
    - ICC measures consistency of responses across attempts
    
    Args:
        data: 1D array of responses across attempts (for single item)
    
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
    
    # For single item test-retest reliability
    # Use variance-based ICC: ICC = 1 - (variance / (variance + mean^2/scale_range))
    # Or simpler: use coefficient of variation approach
    
    mean_val = np.mean(data)
    var_val = np.var(data, ddof=1)
    
    if var_val == 0:
        # Perfect consistency
        return 1.0, 1.0
    
    # For Likert scale (1-5), use scale-relative ICC
    # ICC(2,k) - absolute agreement: considers both systematic and random error
    # ICC(3,k) - consistency: considers only random error (systematic differences don't matter)
    
    # Simplified ICC for single item:
    # ICC = 1 - (variance / expected_variance_if_random)
    # Expected variance for uniform distribution on [1,5] = (5-1)^2/12 = 1.33
    expected_var_uniform = ((5 - 1) ** 2) / 12
    
    # ICC(2,k) - absolute agreement (stricter)
    icc_2k = 1 - (var_val / (var_val + expected_var_uniform))
    
    # ICC(3,k) - consistency (more lenient, allows systematic shifts)
    # For consistency, we normalize by the observed variance range
    icc_3k = 1 - (var_val / (var_val + mean_val * 0.1))  # Simplified consistency measure
    
    # Alternative: Use correlation-based approach if we had multiple items
    # For now, use variance-based which is appropriate for test-retest
    
    return max(0, min(1, icc_2k)), max(0, min(1, icc_3k))


def compute_cronbach_alpha(responses: List[float]) -> float:
    """
    Compute Cronbach's Alpha for single item across attempts.
    
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
    responses_array = responses_array[~np.isnan(responses_array)]  # Remove NaN
    
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


def analyze_consistency(base_path: str, agent_ids: List[str], attempt_ids: List[int], 
                        include_distilled: bool = True) -> Tuple[Dict, pd.DataFrame, Optional[Dict], Optional[pd.DataFrame]]:
    """
    Main analysis function.
    
    Returns:
        Tuple of (raw_results, raw_df, distilled_results, distilled_df)
        If distilled data is not available, distilled_results and distilled_df will be None
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
    
    # Analyze raw responses
    print(f"Analyzing raw response consistency metrics...")
    raw_results = analyze_raw_consistency(df_all, attempt_ids)
    
    # Analyze distilled responses if available
    distilled_results = None
    df_distilled_all = None
    if has_distilled and all_distilled_data:
        df_distilled_all = pd.concat(all_distilled_data, ignore_index=True)
        print(f"Loaded {len(df_distilled_all)} distilled score records")
        print(f"Analyzing distilled score consistency metrics...")
        distilled_results = analyze_distilled_consistency(df_distilled_all, attempt_ids)
    
    return raw_results, df_all, distilled_results, df_distilled_all


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
    
    # 3. Per-agent consistency (average ICC across questions for each agent)
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
            'icc_2k': agent_questions['icc_2k'].mean(),
            'icc_3k': agent_questions['icc_3k'].mean(),
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
            'mean_icc_2k': section_questions['icc_2k'].mean(),
            'mean_icc_3k': section_questions['icc_3k'].mean(),
            'mean_cronbach_alpha': section_questions['cronbach_alpha'].mean(),
            'mean_cv': section_questions['cv'].mean(),
            'mean_sd': section_questions['sd'].mean(),
            'n_questions': len(section_questions)
        })
    
    results['overall_section'] = pd.DataFrame(overall_section)
    
    # 5. Aggregated question-level metrics (average across agents for each question)
    question_aggregated = []
    for question_id, group in results['question_level'].groupby('question_id'):
        # Cache first row to avoid multiple column accesses
        first_row = group.iloc[0] if len(group) > 0 else None
        question_aggregated.append({
            'question_id': question_id,
            'section_id': first_row['section_id'] if first_row is not None else None,
            'icc_2k': group['icc_2k'].mean(),
            'icc_3k': group['icc_3k'].mean(),
            'cronbach_alpha': group['cronbach_alpha'].mean(),
            'cv': group['cv'].mean(),
            'sd': group['sd'].mean(),
            'range': group['range'].mean(),
            'mean': group['mean'].mean(),
            'n_agents': len(group),
            'n_attempts': first_row['n_attempts'] if first_row is not None else 0
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
    
    # 3. Per-agent consistency (average ICC across scores for each agent)
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
            'icc_2k': agent_scores['icc_2k'].mean(),
            'icc_3k': agent_scores['icc_3k'].mean(),
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
            'mean_icc_2k': section_scores['icc_2k'].mean(),
            'mean_icc_3k': section_scores['icc_3k'].mean(),
            'mean_cronbach_alpha': section_scores['cronbach_alpha'].mean(),
            'mean_cv': section_scores['cv'].mean(),
            'mean_sd': section_scores['sd'].mean(),
            'n_scores': len(section_scores)
        })
    
    results['overall_section'] = pd.DataFrame(overall_section)
    
    # 5. Aggregated score-level metrics (average across agents for each score)
    score_aggregated = []
    for score_name, group in results['score_level'].groupby('score_name'):
        # Cache first row to avoid multiple column accesses
        first_row = group.iloc[0] if len(group) > 0 else None
        score_aggregated.append({
            'score_name': score_name,
            'section_id': first_row['section_id'] if first_row is not None else None,
            'score_type': first_row['score_type'] if first_row is not None else None,
            'icc_2k': group['icc_2k'].mean(),
            'icc_3k': group['icc_3k'].mean(),
            'cronbach_alpha': group['cronbach_alpha'].mean(),
            'cv': group['cv'].mean(),
            'sd': group['sd'].mean(),
            'range': group['range'].mean(),
            'mean': group['mean'].mean(),
            'n_agents': len(group),
            'n_attempts': first_row['n_attempts'] if first_row is not None else 0
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
                        distilled_results: Optional[Dict] = None, df_distilled: Optional[pd.DataFrame] = None):
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
    </div>
"""
    
    # Add Plotly visualizations for raw responses
    # 1. Summary metrics (raw)
    overall_section = raw_results['overall_section']
    
    mean_icc_2k = overall_section['mean_icc_2k'].mean() if not overall_section['mean_icc_2k'].isna().all() else 0
    mean_icc_3k = overall_section['mean_icc_3k'].mean() if not overall_section['mean_icc_3k'].isna().all() else 0
    mean_alpha = overall_section['mean_cronbach_alpha'].mean() if not overall_section['mean_cronbach_alpha'].isna().all() else 0
    
    summary_html = f"""
            <h3>Raw Responses Summary</h3>
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
        dist_mean_icc_2k = distilled_overall['mean_icc_2k'].mean() if not distilled_overall['mean_icc_2k'].isna().all() else 0
        dist_mean_icc_3k = distilled_overall['mean_icc_3k'].mean() if not distilled_overall['mean_icc_3k'].isna().all() else 0
        dist_mean_alpha = distilled_overall['mean_cronbach_alpha'].mean() if not distilled_overall['mean_cronbach_alpha'].isna().all() else 0
        
        distilled_summary_html = f"""
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
    
    # 2. Section comparison chart
    fig_sections = go.Figure()
    sections = overall_section['section_id'].tolist()
    
    # Handle NaN values
    icc_2k_vals = overall_section['mean_icc_2k'].fillna(0).tolist()
    icc_3k_vals = overall_section['mean_icc_3k'].fillna(0).tolist()
    
    fig_sections.add_trace(go.Bar(
        x=sections,
        y=icc_2k_vals,
        name='ICC(2,k)',
        marker_color='#4CAF50',
        hovertemplate='Section: %{x}<br>ICC(2,k): %{y:.3f}<extra></extra>'
    ))
    fig_sections.add_trace(go.Bar(
        x=sections,
        y=icc_3k_vals,
        name='ICC(3,k)',
        marker_color='#45a049',
        hovertemplate='Section: %{x}<br>ICC(3,k): %{y:.3f}<extra></extra>'
    ))
    fig_sections.update_layout(
        title='Consistency by Section',
        xaxis_title='Section',
        yaxis_title='ICC Value',
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
    try:
        pivot_icc = question_df.pivot_table(
            index='question_id',
            columns='agent_id',
            values='icc_2k',
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
            hovertemplate_str_all = '<b>Agent ID:</b> %{x}<br><b>Question:</b> %{y}<br><b>ICC(2,k):</b> %{z:.3f}<extra></extra>'
            fig_heatmap_all = go.Figure(data=go.Heatmap(
                z=pivot_icc.values.tolist(),
                x=agent_ids_list,
                y=question_ids_list,
                colorscale='RdYlGn',
                colorbar=dict(title="ICC(2,k)"),
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
                
                hovertemplate_str = '<b>Agent ID:</b> %{x}<br><b>Question:</b> %{y}<br><b>ICC(2,k):</b> %{z:.3f}<extra></extra>'
        
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=chunk_pivot.values.tolist(),
                    x=chunk_agent_ids,
                    y=question_ids_list,
                    colorscale='RdYlGn',
                    colorbar=dict(title="ICC(2,k)"),
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
            hovertemplate_str_all = '<b>Agent ID:</b> %{x}<br><b>Question:</b> %{y}<br><b>ICC(2,k):</b> %{z:.3f}<extra></extra>'
            fig_heatmap_all = go.Figure(data=go.Heatmap(
                z=pivot_icc.values.tolist(),
                x=agent_ids_list,
                y=question_ids_list,
                colorscale='RdYlGn',
                colorbar=dict(title="ICC(2,k)"),
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
        
        # View 3: Aggregated by agent
        agent_df = raw_results['agent_level']
        agent_df_sorted = agent_df.sort_values('agent_id').copy()
        agent_df_sorted['icc_2k'] = agent_df_sorted['icc_2k'].fillna(0)
        
        raw_agent_agg_figs = []
        a_chunk_size = 50
        num_a_chunks = (len(agent_df_sorted) + a_chunk_size - 1) // a_chunk_size
        
        # Option A: If only 1 chunk, show only overall. If multiple chunks, show subgraphs + overall
        if num_a_chunks == 1:
            # Single chunk: show only overall graph
            fig_a_agg_all = go.Figure()
            fig_a_agg_all.add_trace(go.Bar(
                x=agent_df_sorted['agent_id'].tolist(),
                y=agent_df_sorted['icc_2k'].tolist(),
        marker_color='#4CAF50',
                hovertemplate='Agent: %{x}<br>Mean ICC(2,k): %{y:.3f}<extra></extra>',
                text=agent_df_sorted['icc_2k'].tolist(),
                texttemplate='%{{text:.3f}}',
                textposition='outside',
                width=0.8
            ))
            fig_a_agg_all.update_layout(
                title=f'Aggregated by Agent (Average Across Questions) - All {len(agent_df_sorted)} Agents',
        xaxis_title='Agent ID',
                yaxis_title='Mean ICC(2,k)',
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
                chunk_icc_values = chunk_data['icc_2k'].tolist()
                
                fig_a_agg = go.Figure()
                fig_a_agg.add_trace(go.Bar(
                    x=chunk_agent_ids,
                    y=chunk_icc_values,
                    marker_color='#4CAF50',
                    hovertemplate='Agent: %{x}<br>Mean ICC(2,k): %{y:.3f}<extra></extra>',
                    text=chunk_icc_values,
                    texttemplate='%{{text:.3f}}',
                    textposition='outside',
                    width=0.8
                ))
                
                title_suffix = f" (Agents {chunk_agent_ids[0]} to {chunk_agent_ids[-1]})"
                fig_a_agg.update_layout(
                    title=f'Aggregated by Agent (Average Across Questions){title_suffix}',
                    xaxis_title='Agent ID',
                    yaxis_title='Mean ICC(2,k)',
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
                y=agent_df_sorted['icc_2k'].tolist(),
                marker_color='#4CAF50',
                hovertemplate='Agent: %{x}<br>Mean ICC(2,k): %{y:.3f}<extra></extra>',
                text=agent_df_sorted['icc_2k'].tolist(),
                texttemplate='%{{text:.3f}}',
                textposition='outside',
                width=0.8
            ))
            fig_a_agg_all.update_layout(
                title=f'Aggregated by Agent (Average Across Questions) - All {len(agent_df_sorted)} Agents',
                xaxis_title='Agent ID',
                yaxis_title='Mean ICC(2,k)',
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
            
            # Create pivot table for distilled scores
            pivot_score_icc = score_df.pivot_table(
                index='score_name',
                columns='agent_id',
                values='icc_2k',
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
                hovertemplate_str_all = '<b>Agent ID:</b> %{x}<br><b>Score:</b> %{y}<br><b>ICC(2,k):</b> %{z:.3f}<extra></extra>'
                fig_dist_heatmap_all = go.Figure(data=go.Heatmap(
                    z=pivot_score_icc.values.tolist(),
                    x=dist_agent_ids_list,
                    y=dist_score_names_list,
                    colorscale='Blues',
                    colorbar=dict(title="ICC(2,k)"),
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
                    
                    hovertemplate_str = '<b>Agent ID:</b> %{x}<br><b>Score:</b> %{y}<br><b>ICC(2,k):</b> %{z:.3f}<extra></extra>'
                    
                    fig_dist_heatmap = go.Figure(data=go.Heatmap(
                        z=chunk_pivot.values.tolist(),
                        x=chunk_agent_ids,
                        y=dist_score_names_list,
                        colorscale='Blues',
                        colorbar=dict(title="ICC(2,k)"),
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
                hovertemplate_str_all = '<b>Agent ID:</b> %{x}<br><b>Score:</b> %{y}<br><b>ICC(2,k):</b> %{z:.3f}<extra></extra>'
                fig_dist_heatmap_all = go.Figure(data=go.Heatmap(
                    z=pivot_score_icc.values.tolist(),
                    x=dist_agent_ids_list,
                    y=dist_score_names_list,
                    colorscale='Blues',
                    colorbar=dict(title="ICC(2,k)"),
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
            
            # View 3: Aggregated by agent
            dist_agent_df = distilled_results['agent_level']
            dist_agent_df_sorted = dist_agent_df.sort_values('agent_id').copy()
            dist_agent_df_sorted['icc_2k'] = dist_agent_df_sorted['icc_2k'].fillna(0)
            
            dist_a_chunk_size = 50
            dist_num_a_chunks = (len(dist_agent_df_sorted) + dist_a_chunk_size - 1) // dist_a_chunk_size
            
            # Option A: If only 1 chunk, show only overall. If multiple chunks, show subgraphs + overall
            if dist_num_a_chunks == 1:
                # Single chunk: show only overall graph
                fig_dist_a_agg_all = go.Figure()
                fig_dist_a_agg_all.add_trace(go.Bar(
                    x=dist_agent_df_sorted['agent_id'].tolist(),
                    y=dist_agent_df_sorted['icc_2k'].tolist(),
                    marker_color='#2196F3',
                    hovertemplate='Agent: %{x}<br>Mean ICC(2,k): %{y:.3f}<extra></extra>',
                    text=dist_agent_df_sorted['icc_2k'].tolist(),
                    texttemplate='%{{text:.3f}}',
                    textposition='outside',
                    width=0.8
                ))
                fig_dist_a_agg_all.update_layout(
                    title=f'Aggregated by Agent (Average Across Scores) - All {len(dist_agent_df_sorted)} Agents',
                    xaxis_title='Agent ID',
                    yaxis_title='Mean ICC(2,k)',
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
                    chunk_icc_values = chunk_data['icc_2k'].tolist()
                    
                    fig_dist_a_agg = go.Figure()
                    fig_dist_a_agg.add_trace(go.Bar(
                        x=chunk_agent_ids,
                        y=chunk_icc_values,
                        marker_color='#2196F3',
                        hovertemplate='Agent: %{x}<br>Mean ICC(2,k): %{y:.3f}<extra></extra>',
                        text=chunk_icc_values,
                        texttemplate='%{{text:.3f}}',
                        textposition='outside',
                        width=0.8
                    ))
                    
                    title_suffix = f" (Agents {chunk_agent_ids[0]} to {chunk_agent_ids[-1]})"
                    fig_dist_a_agg.update_layout(
                        title=f'Aggregated by Agent (Average Across Scores){title_suffix}',
                        xaxis_title='Agent ID',
                        yaxis_title='Mean ICC(2,k)',
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
                    y=dist_agent_df_sorted['icc_2k'].tolist(),
                    marker_color='#2196F3',
                    hovertemplate='Agent: %{x}<br>Mean ICC(2,k): %{y:.3f}<extra></extra>',
                    text=dist_agent_df_sorted['icc_2k'].tolist(),
                    texttemplate='%{{text:.3f}}',
                    textposition='outside',
                    width=0.8
                ))
                fig_dist_a_agg_all.update_layout(
                    title=f'Aggregated by Agent (Average Across Scores) - All {len(dist_agent_df_sorted)} Agents',
                    xaxis_title='Agent ID',
                    yaxis_title='Mean ICC(2,k)',
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
    raw_question_agg_json = raw_question_agg_data[['question_id', 'section_id', 'icc_2k']].to_dict('records') if not raw_question_agg_data.empty else []
    raw_agent_agg_json = raw_agent_agg_data[['agent_id', 'icc_2k']].to_dict('records') if not raw_agent_agg_data.empty else []
    
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
        
        # For heatmaps, we need pivot table data
        distilled_heatmap_data = []
        if not distilled_score_level_data.empty:
            # Create a simplified representation of the pivot table
            pivot_data = distilled_score_level_data.pivot_table(
                index='score_name',
                columns='agent_id',
                values='icc_2k',
                aggfunc='mean'
            ).fillna(0)
            # Convert to a more compact format
            distilled_heatmap_data = {
                'scores': pivot_data.index.tolist(),
                'agents': pivot_data.columns.tolist(),
                'values': pivot_data.values.tolist()
            }
        
        distilled_score_agg_json = distilled_score_agg_data[['score_name', 'section_id', 'icc_2k']].to_dict('records') if not distilled_score_agg_data.empty else []
        distilled_agent_agg_json = distilled_agent_agg_data[['agent_id', 'icc_2k']].to_dict('records') if not distilled_agent_agg_data.empty else []
        
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
            
            var trace = {{
                x: xValues,
                y: yValues,
                type: 'bar',
                marker: {{color: color}},
                hovertemplate: xKey + ': %{{x}}<br>Mean ICC(2,k): %{{y:.3f}}<extra></extra>',
                text: yValues,
                texttemplate: '%{{text:.3f}}',
                textposition: 'outside',
                width: 0.8
            }};
            
            var layout = {{
                title: {{text: title}},
                xaxis: {{title: xKey, tickangle: -45}},
                yaxis: {{title: 'Mean ICC(2,k)', range: [0, 1.1]}},
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
                colorbar: {{title: "ICC(2,k)"}},
                hovertemplate: '<b>Agent ID:</b> %{{x}}<br><b>Score:</b> %{{y}}<br><b>ICC(2,k):</b> %{{z:.3f}}<extra></extra>'
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
                        
                        generateBarChart(divs[i].id, chunkData, 'agent_id', 'icc_2k', title, '#4CAF50');
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
                        
                        generateBarChart(divs[i].id, chunkData, 'agent_id', 'icc_2k', title, '#2196F3');
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
    html_content = html_content.replace('<div id="summary-metrics"></div>', f'<div id="summary-metrics">{summary_html}</div>')
    html_content = html_content.replace('<div id="summary-charts"></div>', summary_chart_html)
    html_content = html_content.replace('<div id="question-level-content-container"></div>', f'<div id="question-level-content-container">{question_level_content}</div>')
    
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
        # Distilled section comparison chart
        dist_overall_section = distilled_results['overall_section']
        dist_sections = dist_overall_section['section_id'].tolist()
        dist_icc_2k_vals = dist_overall_section['mean_icc_2k'].fillna(0).tolist()
        dist_icc_3k_vals = dist_overall_section['mean_icc_3k'].fillna(0).tolist()
        
        fig_dist_sections = go.Figure()
        fig_dist_sections.add_trace(go.Bar(
            x=dist_sections,
            y=dist_icc_2k_vals,
            name='ICC(2,k)',
            marker_color='#2196F3',
            hovertemplate='Section: %{x}<br>ICC(2,k): %{y:.3f}<extra></extra>'
        ))
        fig_dist_sections.add_trace(go.Bar(
            x=dist_sections,
            y=dist_icc_3k_vals,
            name='ICC(3,k)',
            marker_color='#1976D2',
            hovertemplate='Section: %{x}<br>ICC(3,k): %{y:.3f}<extra></extra>'
        ))
        fig_dist_sections.update_layout(
            title='Distilled Scores Consistency by Section',
            xaxis_title='Section',
            yaxis_title='ICC Value',
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
        raw_results, df_all, distilled_results, df_distilled = analyze_consistency(
            args.base, agent_ids, attempt_ids, include_distilled=True
        )
        
        # Generate HTML report
        generate_html_report(raw_results, df_all, args.output, agent_ids, attempt_ids,
                            distilled_results, df_distilled)
        
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

