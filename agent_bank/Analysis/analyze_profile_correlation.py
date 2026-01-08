#!/usr/bin/env python3
"""
Agent Profile vs. Behavior Correlation Analysis
-----------------------------------------------
This script implements a 3-phase statistical analysis to study the relationship 
between Agent Profiles (7 traits) and Game Behavioral Choices.

Phases:
1. Single Trait Analysis: Linear/Non-linear correlations.
2. Similarity Analysis: Euclidean distance correlation (Mantel Test approach).
3. Multivariate Regression: OLS modeling of Profile -> Behavior.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not found. Plots will not be generated.")

# --- DATA LOADING & PREPARATION (Phase 0) ---

def load_agent_profiles(base_path: Path) -> pd.DataFrame:
    """Load and aggregate 7 personality traits for each agent."""
    agent_profiles = []
    
    for agent_dir in base_path.iterdir():
        if not agent_dir.is_dir(): continue
        
        distilled_path = agent_dir / "survey_distilled_responses.json"
        if not distilled_path.exists(): continue
        
        try:
            with open(distilled_path, 'r') as f:
                data = json.load(f)
            
            # Aggregate scores across attempts (using mean if multiple attempts exist)
            traits = {'agent_id': agent_dir.name}
            temp_scores = {
                'Extraversion': [], 'Agreeableness': [], 'Conscientiousness': [],
                'Neuroticism': [], 'Openness': [], 'BES-A': [], 'REI': []
            }
            
            for attempt in data.get('attempts', []):
                for section_id, content in attempt.get('sections', {}).items():
                    if section_id == 'BFI-10':
                        for k, v in content.get('trait_scores', {}).items():
                            if v is not None: temp_scores[k].append(float(v))
                    elif section_id == 'BES-A':
                        v = content.get('normalized_score')
                        if v is not None: temp_scores['BES-A'].append(float(v))
                    elif section_id == 'REI':
                        v = content.get('normalized_score')
                        if v is not None: temp_scores['REI'].append(float(v))
            
            # Calculate mean for each trait
            has_data = True
            for k, v_list in temp_scores.items():
                if not v_list:
                    has_data = False
                    break
                traits[k] = np.mean(v_list)
                
            if has_data:
                agent_profiles.append(traits)
                
        except Exception as e:
            print(f"Error loading profile for {agent_dir.name}: {e}")
            continue
            
    return pd.DataFrame(agent_profiles).set_index('agent_id')

def load_game_behavior(base_path: Path) -> pd.DataFrame:
    """Load game choices (0-100) per header for each agent."""
    game_data = []
    
    for agent_dir in base_path.iterdir():
        if not agent_dir.is_dir(): continue
        
        game_path = agent_dir / "wavelength_responses.json"
        if not game_path.exists(): continue
        
        try:
            with open(game_path, 'r') as f:
                data = json.load(f)
            
            # Extract responses per header
            for attempt in data.get('attempts', []):
                for header_id, content in attempt.get('headers', {}).items():
                    val = content.get('response')
                    if val is not None:
                        game_data.append({
                            'agent_id': agent_dir.name,
                            'header_id': header_id,
                            'cue': content.get('cue', header_id),
                            'response': float(val)
                        })
                        
        except Exception as e:
            print(f"Error loading game data for {agent_dir.name}: {e}")
            
    df = pd.DataFrame(game_data)
    # Aggregate by averaging attempts if an agent played the same header multiple times
    if not df.empty:
        df = df.groupby(['agent_id', 'header_id', 'cue'])['response'].mean().reset_index()
    return df

# --- STATISTICAL ANALYSIS TOOLS ---

def calculate_correlations(df: pd.DataFrame, traits: List[str], target: str) -> pd.DataFrame:
    """Phase 1: Calculate Pearson and Spearman correlations."""
    results = []
    for trait in traits:
        # Pearson (Linear)
        r_p, p_p = stats.pearsonr(df[trait], df[target])
        # Spearman (Monotonic)
        r_s, p_s = stats.spearmanr(df[trait], df[target])
        
        results.append({
            'Trait': trait,
            'Pearson_r': r_p,
            'Pearson_p': p_p,
            'Spearman_r': r_s,
            'Spearman_p': p_s,
            'Correlation_Strength': abs(r_s) # Use magnitude for sorting
        })
    return pd.DataFrame(results).sort_values('Correlation_Strength', ascending=False)

def run_mantel_test(X: np.ndarray, Y: np.ndarray, permutations=1000) -> Tuple[float, float]:
    """Phase 2: Simple Mantel Test implementation for Distance Correlation."""
    # 1. Calculate Distance Matrices (Euclidean)
    # Using efficient broadcasting for pairwise distance
    # shape: (N, N)
    n = X.shape[0]
    if n < 3: return np.nan, np.nan
    
    dist_X = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1))
    dist_Y = np.abs(Y[:, None] - Y[None, :]).flatten() # Y is 1D usually
    
    # Flatten upper triangles (excluding diagonal)
    i_upper = np.triu_indices(n, k=1)
    flat_X = dist_X[i_upper]
    flat_Y = np.sqrt((Y[:, None] - Y[None, :]) ** 2)[i_upper] # Re-calc Y dist for consistency
    
    # 2. Observed Correlation
    obs_r, _ = stats.pearsonr(flat_X, flat_Y)
    
    # 3. Permutation Test
    # Permute Y distances effectively by shuffling the indices of the original Y array
    # Note: Full Mantel permutes the matrix rows/cols, which is equivalent to shuffling Y values here
    perm_r = []
    y_shuffled = Y.copy()
    
    for _ in range(permutations):
        np.random.shuffle(y_shuffled)
        # Recalculate dist_Y for shuffled data
        flat_Y_shuffled = np.sqrt((y_shuffled[:, None] - y_shuffled[None, :]) ** 2)[i_upper]
        r, _ = stats.pearsonr(flat_X, flat_Y_shuffled)
        perm_r.append(r)
        
    perm_r = np.array(perm_r)
    p_value = np.sum(perm_r >= obs_r) / permutations
    
    return obs_r, p_value

def run_ols_regression(X: pd.DataFrame, y: pd.Series) -> Dict:
    """Phase 3: Multivariate OLS Regression (using numpy/scipy to avoid statsmodels dependency)."""
    # X: Features (Traits), y: Target (Game Choice)
    # Standardize X and y for standardized coefficients (Beta)
    X_std = (X - X.mean()) / X.std()
    y_std = (y - y.mean()) / y.std()
    
    # Add intercept
    X_design = np.column_stack([np.ones(X_std.shape[0]), X_std.values])
    
    try:
        # Solve OLS: (X'X)^-1 X'y
        beta_hat, residuals, rank, s = np.linalg.lstsq(X_design, y_std.values, rcond=None)
        
        # Calculate R-squared
        y_pred = X_design @ beta_hat
        ss_res = np.sum((y_std.values - y_pred) ** 2)
        ss_tot = np.sum((y_std.values - np.mean(y_std.values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Adjust R-squared
        n, p = X.shape
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
        
        # Map coefficients to trait names (skipping intercept at index 0)
        feature_importance = {}
        for idx, col in enumerate(X.columns):
            feature_importance[col] = beta_hat[idx+1]
            
        return {
            'R2': r_squared,
            'Adj_R2': adj_r_squared,
            'Coefficients': feature_importance
        }
    except Exception as e:
        return {'R2': np.nan, 'Adj_R2': np.nan, 'Coefficients': {}}

# --- MAIN ANALYSIS LOOP ---

def main():
    parser = argparse.ArgumentParser(description="Analyze Agent Profile vs. Behavior")
    parser.add_argument('--base', required=True, help="Path to agent bank")
    parser.add_argument('--output', default='analysis_results', help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading data from {args.base}...")
    df_profiles = load_agent_profiles(Path(args.base))
    df_game = load_game_behavior(Path(args.base))
    
    print(f"Loaded {len(df_profiles)} profiles and {len(df_game)} game records.")
    
    if df_profiles.empty or df_game.empty:
        print("Insufficient data to proceed.")
        return

    # Prepare Traits List
    traits = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness', 'BES-A', 'REI']
    traits = [t for t in traits if t in df_profiles.columns]
    
    # Store aggregated results
    all_correlations = []
    regression_summary = []
    
    unique_headers = df_game['header_id'].unique()
    print(f"Analyzing {len(unique_headers)} unique game headers...")
    
    for header_id in unique_headers:
        # Filter game data for this header
        header_data = df_game[df_game['header_id'] == header_id]
        cue_text = header_data.iloc[0]['cue']
        
        # Merge with profiles
        merged = header_data.merge(df_profiles, on='agent_id', how='inner')
        
        if len(merged) < 10:
            print(f"Skipping {header_id}: Too few data points ({len(merged)})")
            continue
            
        print(f"\nProcessing Header: {header_id} ('{cue_text}') [N={len(merged)}]")
        
        # --- PHASE 1: Single Trait Correlations ---
        corr_df = calculate_correlations(merged, traits, 'response')
        corr_df['header_id'] = header_id
        corr_df['cue'] = cue_text
        all_correlations.append(corr_df)
        
        top_trait = corr_df.iloc[0]
        print(f"  > Top Correlated Trait: {top_trait['Trait']} (r={top_trait['Spearman_r']:.2f})")
        
        # --- PHASE 2: Similarity (Mantel Test) ---
        X_traits = merged[traits].values
        Y_behavior = merged['response'].values
        mantel_r, mantel_p = run_mantel_test(X_traits, Y_behavior)
        print(f"  > Profile Similarity vs. Choice Similarity: Mantel r={mantel_r:.2f} (p={mantel_p:.3f})")
        
        # --- PHASE 3: Multivariate Regression ---
        reg_result = run_ols_regression(merged[traits], merged['response'])
        print(f"  > Regression Model R²: {reg_result['R2']:.2f}")
        
        reg_summary_row = {
            'header_id': header_id,
            'cue': cue_text,
            'N': len(merged),
            'Mantel_r': mantel_r,
            'Mantel_p': mantel_p,
            'R2': reg_result['R2'],
            'Adj_R2': reg_result['Adj_R2']
        }
        # Add regression coefficients to summary
        for trait, coef in reg_result['Coefficients'].items():
            reg_summary_row[f'Beta_{trait}'] = coef
            
        regression_summary.append(reg_summary_row)

        # Plotting (Optional) - Save top correlation plot
        if PLOTTING_AVAILABLE:
            plt.figure(figsize=(6, 4))
            sns.regplot(data=merged, x=top_trait['Trait'], y='response', scatter_kws={'alpha':0.5})
            plt.title(f"Header: {cue_text}\nTop Trait: {top_trait['Trait']} (r={top_trait['Pearson_r']:.2f})")
            plt.tight_layout()
            plt.savefig(output_dir / f"scatter_{header_id}.png")
            plt.close()

    # --- SAVE RESULTS ---
    if all_correlations:
        full_corr_df = pd.concat(all_correlations)
        full_corr_df.to_csv(output_dir / 'phase1_correlations.csv', index=False)
        
    if regression_summary:
        reg_df = pd.DataFrame(regression_summary)
        reg_df.to_csv(output_dir / 'phase2_3_multivariate_results.csv', index=False)
        
        # Print Executive Summary
        print("\n" + "="*50)
        print("EXECUTIVE SUMMARY")
        print("="*50)
        print(f"High Correlation Headers (Mantel p < 0.05): {len(reg_df[reg_df['Mantel_p'] < 0.05])}/{len(reg_df)}")
        print(f"High Explanatory Power (R² > 0.3): {len(reg_df[reg_df['R2'] > 0.3])}/{len(reg_df)}")
        print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()