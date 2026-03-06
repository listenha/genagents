import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_data(output_dir='my_correlation_results'):
    """加载之前分析生成的CSV结果"""
    path = Path(output_dir)
    try:
        df_corr = pd.read_csv(path / 'phase1_correlations.csv')
        df_model = pd.read_csv(path / 'phase2_3_multivariate_results.csv')
        return df_corr, df_model
    except FileNotFoundError:
        print(f"Error: Could not find result files in {output_dir}. Please run analysis script first.")
        return None, None

def plot_performance_metrics_v2(df_model, output_file='plot_model_performance_v2.png'):
    """
    绘制 R2 和 Mantel P-value 的双柱状图 (改进版)
    改进点：P-value 使用 -log10 变换，使显著结果更突出
    """
    # 按 R2 排序
    df_sorted = df_model.sort_values('R2', ascending=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    
    # --- Chart 1: R-squared (解释力) ---
    # 颜色：R2 > 0.3 为深绿
    colors = ['#2ca02c' if x > 0.3 else '#bababa' for x in df_sorted['R2']]
    axes[0].barh(df_sorted['cue'], df_sorted['R2'], color=colors)
    axes[0].set_title('Explanatory Power ($R^2$)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('$R^2$ Value (Variance Explained)', fontsize=12)
    axes[0].axvline(0.3, color='black', linestyle='--', alpha=0.5, label='Strong Effect (>0.3)')
    axes[0].legend(loc='lower right')

    # --- Chart 2: Mantel Test Significance (-log10 P-value) ---
    # 转换 P-value 为 -log10(P)
    # 避免 log(0) 错误：将 0 替换为一个极小值 (如 1e-10)
    p_values = df_sorted['Mantel_p'].replace(0, 1e-10)
    log_p = -np.log10(p_values)
    
    # 阈值：P < 0.05 对应 -log10(0.05) ≈ 1.301
    threshold_val = -np.log10(0.05)
    
    # 颜色：显著的 (值 > 1.3) 为红色
    p_colors = ['#d62728' if x > threshold_val else '#bababa' for x in log_p]
    
    axes[1].barh(df_sorted['cue'], log_p, color=p_colors)
    axes[1].set_title('Profile-Behavior Alignment Significance', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Significance Strength (-log10 p-value)', fontsize=12)
    
    # 绘制阈值线
    axes[1].axvline(threshold_val, color='black', linestyle='--', linewidth=2, label='p < 0.05 Threshold')
    
    # 添加直观的标签：越高越显著
    axes[1].text(threshold_val + 0.1, 0, 'Significant ->', color='black', fontsize=10, va='bottom')
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved chart: {output_file}")
    plt.close()

def plot_trait_heatmap_v2(df_corr, output_file='plot_trait_heatmap_v2.png'):
    """
    绘制特质-话题相关性热力图 (改进版)
    改进点：使用绝对值，使用单色系 (Greens) 强调强度
    """
    # 转换数据格式
    pivot_df = df_corr.pivot(index='cue', columns='Trait', values='Spearman_r')
    
    # 取绝对值
    pivot_df_abs = pivot_df.abs()
    
    # 排序：按 Agreeableness 的强度排序
    if 'Agreeableness' in pivot_df_abs.columns:
        pivot_df_abs['sort_key'] = pivot_df_abs['Agreeableness']
        pivot_df_abs = pivot_df_abs.sort_values('sort_key', ascending=False).drop('sort_key', axis=1)

    plt.figure(figsize=(10, 12))
    
    # 使用 'Greens' 色系，范围 0-1
    sns.heatmap(pivot_df_abs, annot=True, fmt=".2f", cmap="Greens", vmin=0, vmax=1,
                cbar_kws={'label': 'Correlation Strength (|r|)'})
    
    plt.title('Correlation Strength Heatmap (Absolute Value)', fontsize=16, pad=20)
    plt.ylabel('Game Header (Cue)', fontsize=12)
    plt.xlabel('Personality Trait', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved chart: {output_file}")
    plt.close()

def main():
    df_corr, df_model = load_data('my_correlation_results')
    if df_corr is None: return

    # 运行改进版绘图函数
    plot_performance_metrics_v2(df_model)
    plot_trait_heatmap_v2(df_corr)

if __name__ == "__main__":
    main()