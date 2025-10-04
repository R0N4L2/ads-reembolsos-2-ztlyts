"""
Visualization script for World Cup Predictor
Creates charts and plots for simulation results
Note: Requires matplotlib and seaborn (optional dependencies)
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict
import json

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
    if SEABORN_AVAILABLE:
        sns.set_style("whitegrid")
except ImportError:
    SEABORN_AVAILABLE = False


def plot_championship_probabilities(champions: List[str], 
                                    top_n: int = 10,
                                    save_path: str = None):
    """
    Create bar chart of championship probabilities.
    
    Args:
        champions: List of tournament champions from simulations
        top_n: Number of top teams to display
        save_path: Path to save figure (optional)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib required for plotting")
        return
    
    champion_counts = Counter(champions)
    top_teams = champion_counts.most_common(top_n)
    
    teams = [team for team, _ in top_teams]
    probabilities = [count / len(champions) * 100 for _, count in top_teams]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(teams)))
    bars = ax.barh(teams, probabilities, color=colors)
    
    ax.set_xlabel('Championship Probability (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Team', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Teams - Championship Probabilities\n({len(champions):,} Simulations)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        ax.text(prob + 0.3, i, f'{prob:.1f}%', 
               va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlim(0, max(probabilities) * 1.15)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Championship probability chart saved to {save_path}")
    
    plt.show()


def plot_probability_distribution(champions: List[str],
                                  runners_up: List[str],
                                  semi_finalists: List[str],
                                  top_n: int = 8,
                                  save_path: str = None):
    """
    Create grouped bar chart showing probabilities across tournament stages.
    
    Args:
        champions: List of champions
        runners_up: List of runners-up
        semi_finalists: List of semi-finalists
        top_n: Number of top teams to show
        save_path: Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib required for plotting")
        return
    
    # Get top teams by championship
    champion_counts = Counter(champions)
    top_teams = [team for team, _ in champion_counts.most_common(top_n)]
    
    # Calculate probabilities for each stage
    champ_probs = [champion_counts.get(team, 0) / len(champions) * 100 
                   for team in top_teams]
    runner_probs = [Counter(runners_up).get(team, 0) / len(runners_up) * 100 
                    for team in top_teams]
    semi_probs = [Counter(semi_finalists).get(team, 0) / len(semi_finalists) * 100 
                  for team in top_teams]
    
    x = np.arange(len(top_teams))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars1 = ax.bar(x - width, champ_probs, width, label='Champion', color='gold', alpha=0.8)
    bars2 = ax.bar(x, runner_probs, width, label='Runner-up', color='silver', alpha=0.8)
    bars3 = ax.bar(x + width, semi_probs, width, label='Semi-finalist', color='#CD7F32', alpha=0.8)
    
    ax.set_xlabel('Team', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability (%)', fontsize=12, fontweight='bold')
    ax.set_title('Tournament Success Probabilities by Stage', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(top_teams, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Stage probability chart saved to {save_path}")
    
    plt.show()


def plot_confederation_performance(champions: List[str],
                                   teams_df: pd.DataFrame,
                                   save_path: str = None):
    """
    Create pie chart showing championship distribution by confederation.
    
    Args:
        champions: List of champions
        teams_df: DataFrame with team information
        save_path: Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib required for plotting")
        return
    
    # Create team to confederation mapping
    team_to_conf = dict(zip(teams_df['fifa_code'], teams_df['confederation']))
    
    # Count championships by confederation
    conf_counts = Counter([team_to_conf.get(team, 'Unknown') for team in champions])
    
    confederations = list(conf_counts.keys())
    counts = list(conf_counts.values())
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(confederations)))
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    wedges, texts, autotexts = ax.pie(counts, 
                                       labels=confederations,
                                       autopct='%1.1f%%',
                                       colors=colors,
                                       startangle=90,
                                       textprops={'fontsize': 11, 'fontweight': 'bold'})
    
    ax.set_title('Championship Distribution by Confederation\n' + 
                f'({len(champions):,} Simulations)',
                fontsize=14, fontweight='bold', pad=20)
    
    # Add legend with counts
    legend_labels = [f'{conf}: {count}' for conf, count in zip(confederations, counts)]
    ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confederation chart saved to {save_path}")
    
    plt.show()


def plot_simulation_convergence(simulation_results: List[str],
                                window_size: int = 50,
                                save_path: str = None):
    """
    Plot probability convergence over simulation iterations.
    
    Args:
        simulation_results: List of champions in order of simulation
        window_size: Window for rolling average
        save_path: Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib required for plotting")
        return
    
    # Get top 5 teams overall
    champion_counts = Counter(simulation_results)
    top_teams = [team for team, _ in champion_counts.most_common(5)]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for team in top_teams:
        # Calculate rolling probability
        is_team = [1 if champ == team else 0 for champ in simulation_results]
        cumulative_prob = np.cumsum(is_team) / np.arange(1, len(is_team) + 1) * 100
        
        ax.plot(range(1, len(cumulative_prob) + 1), cumulative_prob, 
               label=team, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Simulation Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Championship Probability (%)', fontsize=12, fontweight='bold')
    ax.set_title('Probability Convergence Over Simulations\n(Top 5 Teams)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence chart saved to {save_path}")
    
    plt.show()


def create_comparison_table(teams: List[str],
                           champions: List[str],
                           runners_up: List[str],
                           semi_finalists: List[str],
                           quarter_finalists: List[str]) -> pd.DataFrame:
    """
    Create comparison table with probabilities for multiple stages.
    
    Args:
        teams: List of teams to include
        champions: List of champions
        runners_up: List of runners-up
        semi_finalists: List of semi-finalists
        quarter_finalists: List of quarter-finalists
        
    Returns:
        DataFrame with probabilities
    """
    n_sims = len(champions)
    
    data = []
    for team in teams:
        data.append({
            'Team': team,
            'Champion %': Counter(champions).get(team, 0) / n_sims * 100,
            'Runner-up %': Counter(runners_up).get(team, 0) / n_sims * 100,
            'Semi-final %': Counter(semi_finalists).get(team, 0) / (n_sims * 4) * 100,
            'Quarter-final %': Counter(quarter_finalists).get(team, 0) / (n_sims * 8) * 100,
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('Champion %', ascending=False)
    
    return df


def export_visualizations(champions: List[str],
                         runners_up: List[str],
                         semi_finalists: List[str],
                         quarter_finalists: List[str],
                         teams_df: pd.DataFrame,
                         output_dir: str = 'visualizations'):
    """
    Export all visualizations to files.
    
    Args:
        champions: List of champions
        runners_up: List of runners-up
        semi_finalists: List of semi-finalists
        quarter_finalists: List of quarter-finalists
        teams_df: Teams DataFrame
        output_dir: Directory to save visualizations
    """
    import os
    
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib required for exporting visualizations")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating visualizations...")
    
    # Championship probabilities
    plot_championship_probabilities(
        champions, 
        save_path=f'{output_dir}/championship_probabilities.png'
    )
    
    # Stage probabilities
    plot_probability_distribution(
        champions, runners_up, semi_finalists,
        save_path=f'{output_dir}/stage_probabilities.png'
    )
    
    # Confederation performance
    plot_confederation_performance(
        champions, teams_df,
        save_path=f'{output_dir}/confederation_performance.png'
    )
    
    # Convergence
    plot_simulation_convergence(
        champions,
        save_path=f'{output_dir}/simulation_convergence.png'
    )
    
    # Export comparison table
    champion_counts = Counter(champions)
    top_teams = [team for team, _ in champion_counts.most_common(15)]
    
    comparison_df = create_comparison_table(
        top_teams, champions, runners_up, semi_finalists, quarter_finalists
    )
    comparison_df.to_csv(f'{output_dir}/probability_comparison.csv', index=False)
    print(f"Comparison table saved to {output_dir}/probability_comparison.csv")
    
    print(f"\nAll visualizations exported to {output_dir}/")


def print_ascii_chart(data: Dict[str, float], title: str = "", max_width: int = 50):
    """
    Print ASCII bar chart to terminal.
    
    Args:
        data: Dictionary of team: probability
        title: Chart title
        max_width: Maximum width of bars in characters
    """
    if title:
        print(f"\n{title}")
        print("=" * 60)
    
    if not data:
        print("No data to display")
        return
    
    max_value = max(data.values())
    
    for team, value in sorted(data.items(), key=lambda x: x[1], reverse=True):
        bar_length = int((value / max_value) * max_width)
        bar = '█' * bar_length
        print(f"{team:15s} {value:6.2f}% {bar}")
    
    print()


if __name__ == "__main__":
    # Example usage with dummy data
    print("Visualization Module - Example Usage")
    print("=" * 60)
    
    if not MATPLOTLIB_AVAILABLE:
        print("\nTo use visualizations, install matplotlib:")
        print("pip install matplotlib")
        print("\nOptionally install seaborn for better styling:")
        print("pip install seaborn")
    else:
        print("\nMatplotlib is installed. Visualizations available!")
        
        # Create example data
        example_champions = ['BRA'] * 187 + ['GER'] * 156 + ['ESP'] * 134 + \
                          ['FRA'] * 121 + ['ARG'] * 108 + ['BEL'] * 84 + \
                          ['ENG'] * 67 + ['POR'] * 52 + ['URU'] * 41 + \
                          ['CRO'] * 23 + ['COL'] * 12 + ['POL'] * 8 + \
                          ['SUI'] * 4 + ['DEN'] * 2 + ['ISL'] * 1
        
        print(f"\nExample: Plotting championship probabilities for {len(example_champions)} simulations")
        print("Close the plot window to continue...")
        
        # Uncomment to show example plot:
        # plot_championship_probabilities(example_champions, top_n=10)
        
    # ASCII chart example (works without matplotlib)
    print("\nASCII Chart Example:")
    example_data = {
        'BRA': 18.7,
        'GER': 15.6,
        'ESP': 13.4,
        'FRA': 12.1,
        'ARG': 10.8,
    }
    print_ascii_chart(example_data, "Championship Probabilities (%)")
