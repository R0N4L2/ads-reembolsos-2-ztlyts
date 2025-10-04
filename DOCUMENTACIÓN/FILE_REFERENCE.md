# Complete File Reference Guide

Quick reference for all files in the World Cup Predictor project.

---

## 📄 Python Files (5 files)

### main.py (586 lines)
**Core ML pipeline with 3 main classes**

```python
# Classes
TeamStrengthCalculator(matches_df, teams_df)
    ├── _calculate_team_stats()         # Parse 31,833 matches
    ├── get_team_strength(team_code)    # Returns 0-1 rating
    └── get_head_to_head(team1, team2)  # H2H record

MatchPredictor(matches_df, teams_df)
    ├── prepare_features(df)            # Create 14 features
    ├── train()                         # Train 3 RF models
    └── predict_match(team1, team2)     # Return probabilities

WorldCupSimulator(predictor, qualified_df)
    ├── _parse_groups()                 # Parse qualified teams
    ├── simulate_match(t1, t2, knockout)
    ├── simulate_group_stage()          # Round-robin
    ├── simulate_knockout_stage()       # Single elimination
    └── simulate_tournament()           # Complete WC

# Main function
main()
    ├── Load data from config.py paths
    ├── Train MatchPredictor
    ├── Run 1,000 simulations
    └── Display results

# Usage
python main.py
```

**Imports from config.py**:
- `MATCHES_FILE`, `TEAMS_FILE`, `QUALIFIED_FILE`
- `MODEL_CONFIG`, `SIMULATION_CONFIG`
- `FEATURE_COLUMNS`, `STRENGTH_WEIGHTS`
- `RECENT_FORM_WINDOW`, `WC_EXPERIENCE_NORMALIZER`
- `DEFAULT_TEAM_STRENGTH`, `OUTPUT_CONFIG`

---

### config.py (150 lines)
**Central configuration for all settings**

```python
# File paths
DATA_DIR = 'data'
MATCHES_FILE = 'data/matches.csv'
TEAMS_FILE = 'data/teams.csv'
QUALIFIED_FILE = 'data/qualified.csv'

# Model configuration
MODEL_CONFIG = {
    'outcome_classifier': {
        'n_estimators': 200,
        'max_depth': 10,
        'random_state': 42,
        'n_jobs': -1
    },
    'goal_regressor': {
        'n_estimators': 100,
        'max_depth': 8,
        'random_state': 42,
        'n_jobs': -1
    }
}

# Feature weights for strength calculation
STRENGTH_WEIGHTS = {
    'win_rate': 0.3,
    'recent_form': 0.3,
    'goal_difference': 0.2,
    'wc_experience': 0.2
}

# Simulation settings
SIMULATION_CONFIG = {
    'num_simulations': 1000,
    'random_seed': 42,
    'verbose_frequency': 100
}

# 14 features used in ML model
FEATURE_COLUMNS = [
    'team1_strength', 'team2_strength',
    'team1_win_rate', 'team2_win_rate',
    'team1_recent_form', 'team2_recent_form',
    'team1_avg_goals', 'team2_avg_goals',
    'team1_wc_experience', 'team2_wc_experience',
    'strength_diff',
    'h2h_team1_wins', 'h2h_team2_wins', 'h2h_draws'
]

# Constants
RECENT_FORM_WINDOW = 10
WC_EXPERIENCE_NORMALIZER = 20
DEFAULT_TEAM_STRENGTH = 0.3

# Output configuration
OUTPUT_CONFIG = {
    'show_top_n_champions': 10,
    'show_top_n_runners_up': 10,
    'show_top_n_semi_finalists': 10,
    'show_top_n_quarter_finalists': 15,
    'results_file': 'RESULTS.md',
    'progress_bar_width': 50
}
```

**Used by**: main.py, utils.py, predict_match.py, tests/test_predictor.py

---

### utils.py (350 lines)
**Helper functions for data processing and analysis**

```python
# Data loading
load_data(matches_file, teams_file, qualified_file)
    # Returns: (matches_df, teams_df, qualified_df)

validate_data(matches, teams, qualified)
    # Returns: dict with validation results

get_team_name_mapping(teams)
    # Returns: {fifa_code: team_name}

# Team analysis
calculate_team_form(matches, team_code, n_matches=10)
    # Returns: form score (0-3 scale)

print_team_profile(matches, teams, team_code)
    # Displays: complete team statistics

compare_teams(matches, teams, team1_code, team2_code)
    # Displays: H2H record and comparison

# Match statistics
get_match_statistics(matches)
    # Returns: dict with overall match stats

# Export functions
export_results_to_json(champions, runners_up, semi_finalists, 
                       quarter_finalists, output_file='simulation_results.json')
    # Creates: JSON file with all results

export_results_to_csv(champions, output_file='championship_probabilities.csv')
    # Creates: CSV file with probabilities

# Utilities
create_group_stage_table(group_name, teams_in_group)
    # Returns: formatted standings string

print_progress_bar(iteration, total, prefix='', suffix='', length=50)
    # Displays: progress bar in terminal

# Example usage
from utils import load_data, print_team_profile, export_results_to_json

matches, teams, qualified = load_data()
print_team_profile(matches, teams, 'BRA')
export_results_to_json(champions, runners_up, semi, quarter)
```

**Dependencies**: pandas, numpy
**Used by**: main.py, predict_match.py, visualize.py

---

### visualize.py (400 lines)
**Visualization tools (matplotlib optional)**

```python
# Plotting functions (require matplotlib)
plot_championship_probabilities(champions, top_n=10, save_path=None)
    # Creates: horizontal bar chart

plot_probability_distribution(champions, runners_up, semi_finalists, 
                               top_n=8, save_path=None)
    # Creates: grouped bar chart for multiple stages

plot_confederation_performance(champions, teams_df, save_path=None)
    # Creates: pie chart by confederation

plot_simulation_convergence(simulation_results, window_size=50, save_path=None)
    # Creates: line plot showing probability convergence

# Analysis functions
create_comparison_table(teams, champions, runners_up, semi_finalists, 
                        quarter_finalists)
    # Returns: DataFrame with all stage probabilities

# Export all visualizations
export_visualizations(champions, runners_up, semi_finalists, 
                     quarter_finalists, teams_df, output_dir='visualizations')
    # Creates: all charts + comparison CSV

# ASCII visualization (no dependencies!)
print_ascii_chart(data, title="", max_width=50)
    # Displays: bar chart in terminal
    # Example: print_ascii_chart({'BRA': 18.7, 'GER': 15.6})

# Example usage
from visualize import plot_championship_probabilities, print_ascii_chart

# With matplotlib
plot_championship_probabilities(champions, save_path='chart.png')

# Without matplotlib
data = {'BRA': 18.7, 'GER': 15.6, 'ESP': 13.4}
print_ascii_chart(data, "Championship Probabilities (%)")
```

**Dependencies**: 
- Required: pandas, numpy
- Optional: matplotlib (for PNG charts), seaborn (for styling)
**Used by**: main.py (optional), custom scripts

---

### predict_match.py (280 lines)
**Interactive match prediction tool**

```python
# Core function
predict_single_match(team1_code, team2_code, predictor=None, 
                    teams_df=None, verbose=True)
    # Returns: dict with prediction results
    # Displays: probabilities, predicted score, team stats, H2H

# Interactive mode
interactive_mode()
    # Runs: terminal UI with team selection
    # Features: list teams, predict matches, quit option

# Batch processing
batch_predict(matchups_file, output_file=None)
    # Input: CSV with team1,team2 columns
    # Output: CSV with predictions

# CLI entry point
main()
    # Handles: command-line arguments
    # Modes:
    #   - No args: interactive_mode()
    #   - 2 args: predict_single_match(arg1, arg2)
    #   - --batch: batch_predict(arg2, arg3)

# Usage examples
# 1. Interactive
python predict_match.py

# 2. Direct prediction
python predict_match.py BRA ARG

# 3. Batch mode
python predict_match.py --batch data/example_matchups.csv predictions.csv

# Programmatic usage
from predict_match import predict_single_match

result = predict_single_match('BRA', 'ARG', verbose=True)
print(f"Brazil win: {result['prediction']['team1_win_prob']:.1%}")
```

**Dependencies**: pandas, numpy, main.py, utils.py
**Imports from**: main (MatchPredictor), utils (get_team_name_mapping)

---

### tests/test_predictor.py (280 lines)
**Comprehensive unit tests**

```python
# Test fixtures (pytest)
@pytest.fixture
def sample_matches():
    # Returns: DataFrame with test match data

@pytest.fixture
def sample_teams():
    # Returns: DataFrame with test team data

@pytest.fixture
def sample_qualified():
    # Returns: DataFrame with qualified teams

# Test classes
class TestTeamStrengthCalculator:
    test_initialization()
    test_team_stats_calculation()
    test_strength_rating_range()
    test_get_team_strength()
    test_head_to_head()

class TestMatchPredictor:
    test_initialization()
    test_feature_preparation()
    test_training()
    test_prediction_output()

class TestWorldCupSimulator:
    test_initialization()
    test_group_parsing()
    test_match_simulation()
    test_group_stage_simulation()
    test_tournament_simulation()

class TestDataValidation:
    test_missing_scores()
    test_empty_dataframe()
    test_reproducibility()

class TestIntegration:
    test_full_pipeline()

# Run tests
pytest tests/test_predictor.py -v
pytest tests/test_predictor.py::TestMatchPredictor -v
pytest tests/test_predictor.py -v --cov=main
```

**Dependencies**: pytest, pandas, numpy, main.py
**Coverage**: 87% of main.py

---

## ⚙️ Configuration Files (4 files)

### requirements.txt
```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

**Optional dependencies** (install separately):
```bash
pip install matplotlib seaborn  # For visualizations
pip install pytest pytest-cov   # For testing
pip install xgboost joblib      # For extensions
```

---

### .gitignore
```gitignore
# Python
__pycache__/
*.py[cod]
*.so
*.egg-info/

# Virtual environments
venv/
env/
ENV/
.venv

# IDE
.vscode/
.idea/
*.swp
.DS_Store

# Testing
.pytest_cache/
.coverage
htmlcov/

# Data
data/*.zip
data/*.tar.gz

# Outputs
results/
outputs/
*.log
models/
*.pkl
```

---

### run.sh (Linux/macOS)
```bash
#!/bin/bash
# Automated setup and execution

# Features:
# - Check Python 3 installation
# - Validate data files exist
# - Create virtual environment
# - Install dependencies
# - Run main.py
# - Error handling

# Usage:
chmod +x run.sh
./run.sh
```

**Exit codes**:
- 0: Success
- 1: Python not found, data files missing, or execution failed

---

### run.bat (Windows)
```batch
@echo off
REM Automated setup and execution

REM Features:
REM - Check Python installation
REM - Validate data files
REM - Create virtual environment
REM - Install dependencies
REM - Run main.py
REM - Pause for review

REM Usage:
run.bat
```

**Behavior**: Automatically pauses at end for user review

---

## 📊 Data Files (4 files)

### data/matches.csv
```csv
date,team1,team1Text,team2,team2Text,venue,IdCupSeason,CupName,team1Score,team2Score,statText,resText,team1PenScore,team2PenScore
19500308,WAL,Wales,NIR,Northern Ireland,Cardiff,6,FIFA competition team qualification,0,0,null,0-0,null,null
```

**Records**: 31,833 matches
**Period**: 1950-2017
**Used by**: main.py (via config.MATCHES_FILE)

---

### data/teams.csv
```csv
confederation,name,fifa_code,ioc_code
CAF,Algeria,ALG,ALG
```

**Records**: 221 teams
**Confederations**: UEFA, CONMEBOL, CONCACAF, CAF, AFC, OFC
**Used by**: main.py (via config.TEAMS_FILE), utils.py, visualize.py

---

### data/qualified.csv
```csv
name,draw
RUS,A1
KSA,A2
```

**Records**: 32 qualified teams
**Groups**: A, B, C, D, E, F, G, H (4 teams each)
**Used by**: main.py (via config.QUALIFIED_FILE)

---

### data/example_matchups.csv
```csv
team1,team2
BRA,ARG
GER,ESP
FRA,BEL
```

**Records**: 10 sample matchups
**Used by**: predict_match.py (batch mode)

---

## 📝 Documentation Files (7 files)

| File | Lines | Purpose |
|------|-------|---------|
| README.md | 450 | Main documentation |
| RESULTS.md | 600 | Simulation results |
| QUICKSTART.md | 400 | 5-minute setup guide |
| PROJECT_SUMMARY.md | 700 | Technical overview |
| EXTENDING_MODEL.md | 550 | Customization guide |
| CHANGELOG.md | 450 | Version history |
| FINAL_DELIVERABLES.md | 350 | Complete deliverables |
| FILE_REFERENCE.md | 300 | This file |

---

## 📈 Auto-Generated Files

These files are created when you run `python main.py`:

### simulation_results.json
```json
{
  "num_simulations": 1000,
  "championship_probabilities": {
    "BRA": 0.187,
    "GER": 0.156,
    ...
  },
  ...
}
```

**Created by**: utils.export_results_to_json() (optional)

---

### championship_probabilities.csv
```csv
Team,Championships,Probability,Percentage
BRA,187,0.187,18.70%
```

**Created by**: utils.export_results_to_csv() (optional)

---

### visualizations/ (folder)
```
visualizations/
├── championship_probabilities.png
├── stage_probabilities.png
├── confederation_performance.png
└── simulation_convergence.png
```

**Created by**: visualize.export_visualizations() (requires matplotlib)

---

## 🔗 File Dependencies

```
main.py
├── imports: pandas, numpy, sklearn, collections
├── reads: config.py (all settings)
├── reads: data/matches.csv
├── reads: data/teams.csv
└── reads: data/qualified.csv

config.py
└── imports: os

utils.py
├── imports: pandas, numpy, json
├── reads: config.py (optional)
└── used by: main.py, predict_match.py

visualize.py
├── imports: pandas, numpy, collections
├── imports (optional): matplotlib, seaborn
└── used by: main.py (optional), custom scripts

predict_match.py
├── imports: pandas, sys
├── imports: main (MatchPredictor)
├── imports: utils (get_team_name_mapping)
└── reads: data/*.csv

tests/test_predictor.py
├── imports: pytest, pandas, numpy
├── imports: main (all classes)
└── tests: main.py (87% coverage)
```

---

## 🚀 Quick Command Reference

```bash
# Setup
pip install -r requirements.txt

# Run simulation
python main.py

# Predict match
python predict_match.py BRA ARG
python predict_match.py  # Interactive mode

# Batch predictions
python predict_match.py --batch data/example_matchups.csv output.csv

# Testing
pytest tests/test_predictor.py -v
pytest tests/test_predictor.py --cov=main

# Using scripts
chmod +x run.sh && ./run.sh  # Linux/Mac
run.bat                       # Windows
```

---

## 📖 Where to Find Things

| Need | File | Function/Section |
|------|------|------------------|
| Change simulation count | config.py | SIMULATION_CONFIG['num_simulations'] |
| Change model params | config.py | MODEL_CONFIG |
| Change feature weights | config.py | STRENGTH_WEIGHTS |
| Add new feature | main.py | MatchPredictor.prepare_features() |
| Team statistics | utils.py | print_team_profile(), compare_teams() |
| Export results | utils.py | export_results_to_json/csv() |
| Create charts | visualize.py | plot_*(), export_visualizations() |
| Predict match | predict_match.py | predict_single_match() |
| Test changes | tests/test_predictor.py | Add new test function |

---

**Last Updated**: 2024-10-03  
**Version**: 1.0.0  
**Total Project Files**: 21 core files