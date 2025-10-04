"""
Utility functions for World Cup Predictor
Provides helper functions for data processing, visualization, and analysis
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
import json


def load_data(matches_file: str = 'data/matches.csv',
              teams_file: str = 'data/teams.csv',
              qualified_file: str = 'data/qualified.csv') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all required data files.
    
    Args:
        matches_file: Path to matches CSV
        teams_file: Path to teams CSV
        qualified_file: Path to qualified teams CSV
        
    Returns:
        Tuple of (matches_df, teams_df, qualified_df)
    """
    matches = pd.read_csv(matches_file)
    teams = pd.read_csv(teams_file)
    qualified = pd.read_csv(qualified_file)
    
    return matches, teams, qualified


def validate_data(matches: pd.DataFrame, teams: pd.DataFrame, qualified: pd.DataFrame) -> Dict:
    """
    Validate loaded data for completeness and quality.
    
    Args:
        matches: Matches DataFrame
        teams: Teams DataFrame
        qualified: Qualified teams DataFrame
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'matches': {
            'total_rows': len(matches),
            'missing_scores': matches[['team1Score', 'team2Score']].isnull().any(axis=1).sum(),
            'valid_matches': len(matches[matches['team1Score'].notna() & matches['team2Score'].notna()]),
            'date_range': f"{matches['date'].min()} to {matches['date'].max()}",
            'unique_teams': len(set(matches['team1'].unique()) | set(matches['team2'].unique()))
        },
        'teams': {
            'total_teams': len(teams),
            'confederations': teams['confederation'].value_counts().to_dict()
        },
        'qualified': {
            'total_qualified': len(qualified),
            'groups': len(qualified['draw'].str[0].unique())
        }
    }
    
    return validation


def get_team_name_mapping(teams: pd.DataFrame) -> Dict[str, str]:
    """
    Create mapping from FIFA code to full team name.
    
    Args:
        teams: Teams DataFrame
        
    Returns:
        Dictionary mapping FIFA code to team name
    """
    return dict(zip(teams['fifa_code'], teams['name']))


def calculate_team_form(matches: pd.DataFrame, team_code: str, n_matches: int = 10) -> float:
    """
    Calculate recent form for a specific team.
    
    Args:
        matches: Matches DataFrame
        team_code: FIFA code of the team
        n_matches: Number of recent matches to consider
        
    Returns:
        Form score (0-3 scale, where 3 = all wins)
    """
    team_matches = matches[
        (matches['team1'] == team_code) | (matches['team2'] == team_code)
    ].copy()
    
    team_matches = team_matches[
        team_matches['team1Score'].notna() & team_matches['team2Score'].notna()
    ].sort_values('date', ascending=False).head(n_matches)
    
    if len(team_matches) == 0:
        return 1.0  # Neutral form
    
    points = []
    for _, match in team_matches.iterrows():
        if match['team1'] == team_code:
            own_score, opp_score = match['team1Score'], match['team2Score']
        else:
            own_score, opp_score = match['team2Score'], match['team1Score']
        
        if own_score > opp_score:
            points.append(3)
        elif own_score == opp_score:
            points.append(1)
        else:
            points.append(0)
    
    return sum(points) / len(points)


def export_results_to_json(champions: List[str], 
                           runners_up: List[str],
                           semi_finalists: List[str],
                           quarter_finalists: List[str],
                           output_file: str = 'simulation_results.json'):
    """
    Export simulation results to JSON file.
    
    Args:
        champions: List of tournament champions
        runners_up: List of runners-up
        semi_finalists: List of semi-finalists
        quarter_finalists: List of quarter-finalists
        output_file: Path to output JSON file
    """
    results = {
        'num_simulations': len(champions),
        'championship_probabilities': {
            team: count / len(champions) 
            for team, count in Counter(champions).most_common()
        },
        'runner_up_probabilities': {
            team: count / len(runners_up)
            for team, count in Counter(runners_up).most_common()
        },
        'semi_final_probabilities': {
            team: count / len(semi_finalists)
            for team, count in Counter(semi_finalists).most_common()
        },
        'quarter_final_probabilities': {
            team: count / len(quarter_finalists)
            for team, count in Counter(quarter_finalists).most_common()
        },
        'statistics': {
            'unique_champions': len(set(champions)),
            'unique_finalists': len(set(champions + runners_up)),
            'most_likely_champion': Counter(champions).most_common(1)[0]
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results exported to {output_file}")


def export_results_to_csv(champions: List[str],
                          output_file: str = 'championship_probabilities.csv'):
    """
    Export championship probabilities to CSV.
    
    Args:
        champions: List of tournament champions
        output_file: Path to output CSV file
    """
    champion_counts = Counter(champions)
    
    results_df = pd.DataFrame([
        {
            'Team': team,
            'Championships': count,
            'Probability': count / len(champions),
            'Percentage': f"{count / len(champions) * 100:.2f}%"
        }
        for team, count in champion_counts.most_common()
    ])
    
    results_df.to_csv(output_file, index=False)
    print(f"Championship probabilities exported to {output_file}")


def get_match_statistics(matches: pd.DataFrame) -> Dict:
    """
    Calculate overall statistics from historical matches.
    
    Args:
        matches: Matches DataFrame
        
    Returns:
        Dictionary with match statistics
    """
    valid_matches = matches[matches['team1Score'].notna() & matches['team2Score'].notna()]
    
    total_goals = (valid_matches['team1Score'] + valid_matches['team2Score']).sum()
    
    wins = len(valid_matches[valid_matches['team1Score'] > valid_matches['team2Score']])
    draws = len(valid_matches[valid_matches['team1Score'] == valid_matches['team2Score']])
    losses = len(valid_matches[valid_matches['team1Score'] < valid_matches['team2Score']])
    
    stats = {
        'total_matches': len(valid_matches),
        'total_goals': int(total_goals),
        'avg_goals_per_match': total_goals / len(valid_matches),
        'home_wins': wins,
        'draws': draws,
        'away_wins': losses,
        'home_win_percentage': wins / len(valid_matches) * 100,
        'draw_percentage': draws / len(valid_matches) * 100,
        'away_win_percentage': losses / len(valid_matches) * 100,
        'competitions': matches['CupName'].value_counts().to_dict()
    }
    
    return stats


def print_team_profile(matches: pd.DataFrame, teams: pd.DataFrame, team_code: str):
    """
    Print detailed profile of a specific team.
    
    Args:
        matches: Matches DataFrame
        teams: Teams DataFrame
        team_code: FIFA code of the team
    """
    team_info = teams[teams['fifa_code'] == team_code]
    
    if len(team_info) == 0:
        print(f"Team {team_code} not found")
        return
    
    team_matches = matches[
        (matches['team1'] == team_code) | (matches['team2'] == team_code)
    ]
    
    valid_matches = team_matches[
        team_matches['team1Score'].notna() & team_matches['team2Score'].notna()
    ]
    
    wins = 0
    draws = 0
    losses = 0
    goals_for = 0
    goals_against = 0
    wc_matches = 0
    
    for _, match in valid_matches.iterrows():
        is_team1 = match['team1'] == team_code
        own_score = match['team1Score'] if is_team1 else match['team2Score']
        opp_score = match['team2Score'] if is_team1 else match['team1Score']
        
        goals_for += own_score
        goals_against += opp_score
        
        if own_score > opp_score:
            wins += 1
        elif own_score == opp_score:
            draws += 1
        else:
            losses += 1
        
        if 'World Cup' in str(match['CupName']):
            wc_matches += 1
    
    print("=" * 60)
    print(f"TEAM PROFILE: {team_info.iloc[0]['name']} ({team_code})")
    print("=" * 60)
    print(f"Confederation: {team_info.iloc[0]['confederation']}")
    print(f"\nAll-Time Record:")
    print(f"  Total Matches: {len(valid_matches)}")
    print(f"  Wins: {wins} ({wins/len(valid_matches)*100:.1f}%)")
    print(f"  Draws: {draws} ({draws/len(valid_matches)*100:.1f}%)")
    print(f"  Losses: {losses} ({losses/len(valid_matches)*100:.1f}%)")
    print(f"\nGoals:")
    print(f"  Scored: {int(goals_for)} (avg {goals_for/len(valid_matches):.2f})")
    print(f"  Conceded: {int(goals_against)} (avg {goals_against/len(valid_matches):.2f})")
    print(f"  Difference: {int(goals_for - goals_against)}")
    print(f"\nWorld Cup:")
    print(f"  Matches: {wc_matches}")
    print(f"\nRecent Form (last 10 matches): {calculate_team_form(matches, team_code, 10):.2f}/3.00")
    print("=" * 60)


def compare_teams(matches: pd.DataFrame, teams: pd.DataFrame, team1_code: str, team2_code: str):
    """
    Compare two teams head-to-head and overall stats.
    
    Args:
        matches: Matches DataFrame
        teams: Teams DataFrame
        team1_code: FIFA code of first team
        team2_code: FIFA code of second team
    """
    # Head to head
    h2h = matches[
        ((matches['team1'] == team1_code) & (matches['team2'] == team2_code)) |
        ((matches['team1'] == team2_code) & (matches['team2'] == team1_code))
    ]
    
    h2h_valid = h2h[h2h['team1Score'].notna() & h2h['team2Score'].notna()]
    
    team1_wins = 0
    team2_wins = 0
    draws = 0
    
    for _, match in h2h_valid.iterrows():
        if match['team1'] == team1_code:
            if match['team1Score'] > match['team2Score']:
                team1_wins += 1
            elif match['team1Score'] < match['team2Score']:
                team2_wins += 1
            else:
                draws += 1
        else:
            if match['team1Score'] > match['team2Score']:
                team2_wins += 1
            elif match['team1Score'] < match['team2Score']:
                team1_wins += 1
            else:
                draws += 1
    
    team1_name = teams[teams['fifa_code'] == team1_code]['name'].values[0] if len(teams[teams['fifa_code'] == team1_code]) > 0 else team1_code
    team2_name = teams[teams['fifa_code'] == team2_code]['name'].values[0] if len(teams[teams['fifa_code'] == team2_code]) > 0 else team2_code
    
    print("=" * 60)
    print(f"HEAD-TO-HEAD: {team1_name} vs {team2_name}")
    print("=" * 60)
    print(f"Total Meetings: {len(h2h_valid)}")
    print(f"{team1_name} Wins: {team1_wins}")
    print(f"{team2_name} Wins: {team2_wins}")
    print(f"Draws: {draws}")
    
    if len(h2h_valid) > 0:
        print(f"\n{team1_name} Win Rate: {team1_wins/len(h2h_valid)*100:.1f}%")
        print(f"{team2_name} Win Rate: {team2_wins/len(h2h_valid)*100:.1f}%")
    
    print(f"\nRecent Form:")
    print(f"{team1_name}: {calculate_team_form(matches, team1_code):.2f}/3.00")
    print(f"{team2_name}: {calculate_team_form(matches, team2_code):.2f}/3.00")
    print("=" * 60)


def create_group_stage_table(group_name: str, teams_in_group: List[str]) -> str:
    """
    Create a formatted table for group stage standings.
    
    Args:
        group_name: Name of the group (e.g., 'A')
        teams_in_group: List of team codes in the group
        
    Returns:
        Formatted string table
    """
    table = f"\nGroup {group_name}\n"
    table += "-" * 50 + "\n"
    table += f"{'Pos':<5}{'Team':<20}{'Pts':<6}{'GD':<6}{'GF':<6}{'GA':<6}\n"
    table += "-" * 50 + "\n"
    
    for i, team in enumerate(teams_in_group, 1):
        # Placeholder values - would be filled by actual simulation
        table += f"{i:<5}{team:<20}{'?':<6}{'?':<6}{'?':<6}{'?':<6}\n"
    
    return table


def print_progress_bar(iteration: int, total: int, prefix: str = '', 
                      suffix: str = '', length: int = 50, fill: str = '█'):
    """
    Print a progress bar to terminal.
    
    Args:
        iteration: Current iteration
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        length: Character length of bar
        fill: Bar fill character
    """
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    
    if iteration == total:
        print()


if __name__ == "__main__":
    # Example usage
    print("Loading data...")
    matches, teams, qualified = load_data()
    
    print("\nValidating data...")
    validation = validate_data(matches, teams, qualified)
    print(json.dumps(validation, indent=2))
    
    print("\nMatch Statistics:")
    stats = get_match_statistics(matches)
    for key, value in stats.items():
        if key != 'competitions':
            print(f"  {key}: {value}")
    
    print("\nExample: Brazil Profile")
    print_team_profile(matches, teams, 'BRA')
    
    print("\nExample: Brazil vs Argentina")
    compare_teams(matches, teams, 'BRA', 'ARG')
