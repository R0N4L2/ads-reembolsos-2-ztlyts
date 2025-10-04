"""
Configuration file for World Cup Predictor
Centralized settings for model parameters and file paths
"""

import os

# Data paths
DATA_DIR = 'data'
MATCHES_FILE = os.path.join(DATA_DIR, 'matches.csv')
TEAMS_FILE = os.path.join(DATA_DIR, 'teams.csv')
QUALIFIED_FILE = os.path.join(DATA_DIR, 'qualified.csv')

# Model parameters
MODEL_CONFIG = {
    # Random Forest Classifier for match outcomes
    'outcome_classifier': {
        'n_estimators': 200,
        'max_depth': 10,
        'random_state': 42,
        'n_jobs': -1  # Use all CPU cores
    },
    
    # Random Forest Regressor for goal prediction
    'goal_regressor': {
        'n_estimators': 100,
        'max_depth': 8,
        'random_state': 42,
        'n_jobs': -1
    }
}

# Feature weights for team strength calculation
STRENGTH_WEIGHTS = {
    'win_rate': 0.3,
    'recent_form': 0.3,
    'goal_difference': 0.2,
    'wc_experience': 0.2
}

# Simulation parameters
SIMULATION_CONFIG = {
    'num_simulations': 1000,
    'random_seed': 42,
    'verbose_frequency': 100  # Print progress every N simulations
}

# Features used for prediction
FEATURE_COLUMNS = [
    'team1_strength',
    'team2_strength',
    'team1_win_rate',
    'team2_win_rate',
    'team1_recent_form',
    'team2_recent_form',
    'team1_avg_goals',
    'team2_avg_goals',
    'team1_wc_experience',
    'team2_wc_experience',
    'strength_diff',
    'h2h_team1_wins',
    'h2h_team2_wins',
    'h2h_draws'
]

# Recent form window (number of recent matches to consider)
RECENT_FORM_WINDOW = 10

# World Cup experience normalization
WC_EXPERIENCE_NORMALIZER = 20  # Divide WC matches by this for experience metric

# Default strength for unknown teams
DEFAULT_TEAM_STRENGTH = 0.3

# Output settings
OUTPUT_CONFIG = {
    'show_top_n_champions': 10,
    'show_top_n_runners_up': 10,
    'show_top_n_semi_finalists': 10,
    'show_top_n_quarter_finalists': 15,
    'results_file': 'RESULTS.md',
    'progress_bar_width': 50
}

# Validation settings
VALIDATION_CONFIG = {
    'min_matches_for_stats': 1,
    'test_size': 0.2,  # For train/test split if doing validation
    'cross_validation_folds': 5
}
