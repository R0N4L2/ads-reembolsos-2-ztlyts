"""
Interactive script for predicting individual match outcomes
Usage: python predict_match.py [TEAM1] [TEAM2]
Example: python predict_match.py BRA ARG
"""

import sys
import pandas as pd
from main import MatchPredictor, TeamStrengthCalculator
from utils import get_team_name_mapping


def predict_single_match(team1_code: str, team2_code: str, 
                        predictor: MatchPredictor = None,
                        teams_df: pd.DataFrame = None,
                        verbose: bool = True):
    """
    Predict outcome of a single match.
    
    Args:
        team1_code: FIFA code of team 1
        team2_code: FIFA code of team 2
        predictor: Trained MatchPredictor (will load if None)
        teams_df: Teams DataFrame for names
        verbose: Print detailed output
        
    Returns:
        Dictionary with prediction results
    """
    # Load and train model if not provided
    if predictor is None:
        print("Loading data and training model...")
        matches = pd.read_csv('data/matches.csv')
        teams = pd.read_csv('data/teams.csv')
        predictor = MatchPredictor(matches, teams)
        predictor.train()
        teams_df = teams
        print()
    
    # Get team names
    if teams_df is not None:
        name_mapping = get_team_name_mapping(teams_df)
        team1_name = name_mapping.get(team1_code, team1_code)
        team2_name = name_mapping.get(team2_code, team2_code)
    else:
        team1_name = team1_code
        team2_name = team2_code
    
    # Make prediction
    prediction = predictor.predict_match(team1_code, team2_code)
    
    if verbose:
        print("=" * 70)
        print(f"MATCH PREDICTION: {team1_name} vs {team2_name}")
        print("=" * 70)
        print()
        
        # Probabilities
        print("OUTCOME PROBABILITIES:")
        print("-" * 70)
        print(f"{team1_name:30s} Win:  {prediction['team1_win_prob']*100:5.1f}%  {'█' * int(prediction['team1_win_prob']*50)}")
        print(f"{'Draw':30s}      {prediction['draw_prob']*100:5.1f}%  {'█' * int(prediction['draw_prob']*50)}")
        print(f"{team2_name:30s} Win:  {prediction['team2_win_prob']*100:5.1f}%  {'█' * int(prediction['team2_win_prob']*50)}")
        print()
        
        # Predicted score
        score1, score2 = prediction['predicted_score']
        print("PREDICTED SCORE:")
        print("-" * 70)
        print(f"{team1_name:30s} {score1}")
        print(f"{team2_name:30s} {score2}")
        print()
        
        # Most likely outcome
        if prediction['team1_win_prob'] > max(prediction['team2_win_prob'], prediction['draw_prob']):
            likely_outcome = f"{team1_name} victory"
        elif prediction['team2_win_prob'] > max(prediction['team1_win_prob'], prediction['draw_prob']):
            likely_outcome = f"{team2_name} victory"
        else:
            likely_outcome = "Draw"
        
        print(f"MOST LIKELY OUTCOME: {likely_outcome}")
        print()
        
        # Additional stats
        team1_stats = predictor.strength_calc.team_stats.get(team1_code, {})
        team2_stats = predictor.strength_calc.team_stats.get(team2_code, {})
        
        print("TEAM STATISTICS:")
        print("-" * 70)
        print(f"{'Metric':<30s} {team1_name:>18s} {team2_name:>18s}")
        print("-" * 70)
        print(f"{'Strength Rating':<30s} {team1_stats.get('strength_rating', 0):>18.3f} {team2_stats.get('strength_rating', 0):>18.3f}")
        print(f"{'Win Rate':<30s} {team1_stats.get('win_rate', 0)*100:>17.1f}% {team2_stats.get('win_rate', 0)*100:>17.1f}%")
        print(f"{'Avg Goals For':<30s} {team1_stats.get('avg_goals_for', 0):>18.2f} {team2_stats.get('avg_goals_for', 0):>18.2f}")
        print(f"{'World Cup Matches':<30s} {team1_stats.get('wc_matches', 0):>18d} {team2_stats.get('wc_matches', 0):>18d}")
        
        # Head to head
        h2h = predictor.strength_calc.get_head_to_head(team1_code, team2_code)
        print()
        print("HEAD-TO-HEAD RECORD:")
        print("-" * 70)
        print(f"Total Meetings: {h2h['matches']}")
        if h2h['matches'] > 0:
            print(f"{team1_name} Wins: {h2h['team1_wins']} ({h2h['team1_wins']/h2h['matches']*100:.1f}%)")
            print(f"{team2_name} Wins: {h2h['team2_wins']} ({h2h['team2_wins']/h2h['matches']*100:.1f}%)")
            print(f"Draws: {h2h['draws']} ({h2h['draws']/h2h['matches']*100:.1f}%)")
        else:
            print("No previous meetings found in dataset")
        
        print("=" * 70)
    
    return {
        'team1': team1_code,
        'team2': team2_code,
        'team1_name': team1_name,
        'team2_name': team2_name,
        'prediction': prediction
    }


def interactive_mode():
    """Run interactive mode for match predictions."""
    print("=" * 70)
    print("WORLD CUP MATCH PREDICTOR - Interactive Mode")
    print("=" * 70)
    print()
    
    # Load data and train model once
    print("Loading data and training model (this may take a minute)...")
    matches = pd.read_csv('data/matches.csv')
    teams = pd.read_csv('data/teams.csv')
    
    predictor = MatchPredictor(matches, teams)
    predictor.train()
    
    print()
    print("Model ready!")
    print()
    print("Available teams (FIFA codes):")
    print("-" * 70)
    
    # Show some popular teams
    popular_teams = ['BRA', 'ARG', 'GER', 'ESP', 'FRA', 'ENG', 'ITA', 'NED', 
                    'POR', 'URU', 'BEL', 'CRO', 'MEX', 'USA', 'COL', 'CHL']
    
    name_mapping = get_team_name_mapping(teams)
    
    for i, code in enumerate(popular_teams):
        name = name_mapping.get(code, code)
        print(f"{code} ({name})", end="  ")
        if (i + 1) % 4 == 0:
            print()
    print()
    print()
    print("Enter 'list' to see all teams, or 'quit' to exit")
    print()
    
    while True:
        print("-" * 70)
        team1 = input("Enter Team 1 FIFA code (or 'quit'): ").strip().upper()
        
        if team1.lower() == 'quit':
            print("Goodbye!")
            break
        
        if team1.lower() == 'list':
            print("\nAll available teams:")
            all_codes = sorted(teams['fifa_code'].unique())
            for i, code in enumerate(all_codes):
                name = name_mapping.get(code, code)
                print(f"{code:4s} ({name:25s})", end="  ")
                if (i + 1) % 3 == 0:
                    print()
            print("\n")
            continue
        
        team2 = input("Enter Team 2 FIFA code: ").strip().upper()
        
        if team2.lower() == 'quit':
            print("Goodbye!")
            break
        
        print()
        
        # Validate teams
        if team1 not in teams['fifa_code'].values:
            print(f"Warning: {team1} not found in dataset. Using default stats.")
        
        if team2 not in teams['fifa_code'].values:
            print(f"Warning: {team2} not found in dataset. Using default stats.")
        
        # Make prediction
        try:
            predict_single_match(team1, team2, predictor, teams)
        except Exception as e:
            print(f"Error making prediction: {e}")
        
        print()


def batch_predict(matchups_file: str, output_file: str = None):
    """
    Predict multiple matches from a CSV file.
    
    Args:
        matchups_file: CSV file with columns: team1, team2
        output_file: Output CSV file for predictions
    """
    # Load matchups
    matchups = pd.read_csv(matchups_file)
    
    if 'team1' not in matchups.columns or 'team2' not in matchups.columns:
        print("Error: CSV must have 'team1' and 'team2' columns")
        return
    
    # Load and train model
    print("Loading data and training model...")
    matches = pd.read_csv('data/matches.csv')
    teams = pd.read_csv('data/teams.csv')
    predictor = MatchPredictor(matches, teams)
    predictor.train()
    print()
    
    # Make predictions
    results = []
    
    for i, row in matchups.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        
        print(f"Predicting {team1} vs {team2}...")
        
        prediction = predictor.predict_match(team1, team2)
        score1, score2 = prediction['predicted_score']
        
        results.append({
            'team1': team1,
            'team2': team2,
            'team1_win_prob': prediction['team1_win_prob'],
            'draw_prob': prediction['draw_prob'],
            'team2_win_prob': prediction['team2_win_prob'],
            'predicted_team1_score': score1,
            'predicted_team2_score': score2
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save or display
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"\nPredictions saved to {output_file}")
    else:
        print("\nPrediction Results:")
        print(results_df.to_string(index=False))


def main():
    """Main function for command-line usage."""
    if len(sys.argv) == 1:
        # Interactive mode
        interactive_mode()
    
    elif len(sys.argv) == 3:
        # Single match prediction
        team1 = sys.argv[1].upper()
        team2 = sys.argv[2].upper()
        predict_single_match(team1, team2)
    
    elif len(sys.argv) == 4 and sys.argv[1] == '--batch':
        # Batch mode
        batch_predict(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    
    else:
        print("Usage:")
        print("  Interactive mode:    python predict_match.py")
        print("  Single match:        python predict_match.py TEAM1 TEAM2")
        print("  Batch mode:          python predict_match.py --batch input.csv output.csv")
        print()
        print("Examples:")
        print("  python predict_match.py BRA ARG")
        print("  python predict_match.py GER ESP")
        print("  python predict_match.py --batch matchups.csv predictions.csv")


if __name__ == "__main__":
    main()
