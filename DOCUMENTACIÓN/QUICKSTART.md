# Quick Start Guide

Get up and running with the World Cup Predictor in 5 minutes.

## 📋 Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- The data files in `data/` folder:
  - `matches.csv`
  - `teams.csv`
  - `qualified.csv`

## 🚀 Installation

### Option 1: Quick Setup (Recommended)

```bash
# 1. Navigate to project directory
cd worldcup-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the simulation
python main.py
```

### Option 2: Using Virtual Environment

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run simulation
python main.py
```

### Option 3: Using Run Scripts

The project includes automated setup scripts:

**Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```

**Windows:**
```bash
run.bat
```

These scripts will:
- Create virtual environment automatically
- Install all dependencies
- Run the simulation
- Show results

## ▶️ Running the Simulation

### Basic Usage

```bash
python main.py
```

**Expected runtime**: 1-2 minutes on modern hardware

This will:
1. Load 31,833 historical matches
2. Calculate team statistics
3. Train ML models
4. Run 1,000 World Cup simulations
5. Display championship probabilities

### Sample Output

```
==============================================================
FIFA World Cup 2018 Prediction Model
==============================================================

Loading data...
Loaded 31833 historical matches
Loaded 221 teams
Loaded 32 qualified teams for 2018 WC

Training prediction model...
Model trained on 31233 matches
Feature importance (top 5):
  team1_strength: 0.187
  team2_strength: 0.183
  ...

Running 1,000 tournament simulations...
Completed 100 simulations...
Completed 200 simulations...
...
Completed 1,000 simulations...

CHAMPIONSHIP PROBABILITIES:
------------------------------------------------------------
BRA                   18.70%  █████████
GER                   15.60%  ███████
ESP                   13.40%  ██████
...

TOP 5 FAVORITES TO WIN:
  1. BRA             - 18.7%
  2. GER             - 15.6%
  3. ESP             - 13.4%
  4. FRA             - 12.1%
  5. ARG             - 10.8%
```

## 🎯 Other Use Cases

### 1. Predict Individual Matches

**Interactive mode:**
```bash
python predict_match.py
```

Follow prompts to enter team codes (e.g., BRA, ARG).

**Direct prediction:**
```bash
python predict_match.py BRA ARG
```

Output:
```
==============================================================
MATCH PREDICTION: Brazil vs Argentina
==============================================================

OUTCOME PROBABILITIES:
--------------------------------------------------------------
Brazil                     Win:  52.3%  ██████████████
Draw                            12.1%  ██████
Argentina                  Win:  35.6%  █████████

PREDICTED SCORE:
--------------------------------------------------------------
Brazil                     2
Argentina                  1

MOST LIKELY OUTCOME: Brazil victory
```

### 2. Batch Predictions

Predict multiple matches from CSV:

```bash
python predict_match.py --batch data/example_matchups.csv predictions.csv
```

Input file format (`example_matchups.csv`):
```csv
team1,team2
BRA,ARG
GER,ESP
FRA,BEL
```

### 3. Run Tests

Verify installation:

```bash
pip install pytest
python -m pytest tests/ -v
```

Expected: 15+ tests pass

## 📊 Understanding Results

### Terminal Output

The main script displays:
- **Championship probabilities**: Top 10 teams
- **Runner-up probabilities**: Likely finalists
- **Semi-final rates**: Teams reaching top 4
- **Quarter-final rates**: Teams reaching top 8
- **Key statistics**: Unique champions, favorites

### Generated Files

After running `main.py`, these files are created:

```
worldcup-predictor/
└── outputs/
    ├── simulation_results.json         # Detailed JSON data
    └── championship_probabilities.csv  # Exportable table
```

### Documentation

Check these files for more details:
- **RESULTS.md**: Complete statistical analysis
- **README.md**: Full documentation
- **PROJECT_SUMMARY.md**: Technical overview

## ⚙️ Customization

### Change Number of Simulations

Edit `config.py`, line ~38:
```python
SIMULATION_CONFIG = {
    'num_simulations': 5000,  # Change from 1000 to 5000
    'verbose_frequency': 100,
    'random_seed': 42
}
```

Then run:
```bash
python main.py
```

### Adjust Model Parameters

Edit `config.py`, line ~17:
```python
MODEL_CONFIG = {
    'outcome_classifier': {
        'n_estimators': 300,  # Increase from 200
        'max_depth': 15,      # Increase from 10
        'random_state': 42,
        'n_jobs': -1
    },
    'goal_regressor': {
        'n_estimators': 150,  # Increase from 100
        'max_depth': 10,      # Increase from 8
        'random_state': 42,
        'n_jobs': -1
    }
}
```

### Change Feature Weights

Edit `config.py`, line ~30:
```python
STRENGTH_WEIGHTS = {
    'win_rate': 0.4,          # Change from 0.3
    'recent_form': 0.3,
    'goal_difference': 0.2,
    'wc_experience': 0.1      # Change from 0.2
}
```

### Use Different Random Seed

Edit `config.py`:
```python
SIMULATION_CONFIG = {
    'num_simulations': 1000,
    'verbose_frequency': 100,
    'random_seed': 123  # Change from 42 for different results
}
```

## 🧪 Advanced Usage

### Using utils.py Functions

```python
from utils import (
    load_data,
    validate_data,
    print_team_profile,
    compare_teams,
    export_results_to_json
)

# Load data
matches, teams, qualified = load_data(
    matches_file='data/matches.csv',
    teams_file='data/teams.csv',
    qualified_file='data/qualified.csv'
)

# Validate data
validation = validate_data(matches, teams, qualified)
print(validation)

# Team analysis
print_team_profile(matches, teams, 'BRA')
compare_teams(matches, teams, 'BRA', 'ARG')
```

### Using visualize.py Functions

```python
from visualize import (
    plot_championship_probabilities,
    plot_probability_distribution,
    export_visualizations
)
import pandas as pd
from main import MatchPredictor, WorldCupSimulator

# Run simulation
matches = pd.read_csv('data/matches.csv')
teams = pd.read_csv('data/teams.csv')
qualified = pd.read_csv('data/qualified.csv')

predictor = MatchPredictor(matches, teams)
predictor.train()

simulator = WorldCupSimulator(predictor, qualified)

champions = []
for i in range(100):
    result = simulator.simulate_tournament()
    champions.append(result['champion'])

# Create visualizations
plot_championship_probabilities(champions, top_n=10)
```

### Using predict_match.py Programmatically

```python
from predict_match import predict_single_match
import pandas as pd

# Predict match
result = predict_single_match('BRA', 'ARG', verbose=True)

# Access probabilities
print(f"Brazil win: {result['prediction']['team1_win_prob']:.1%}")
print(f"Draw: {result['prediction']['draw_prob']:.1%}")
print(f"Argentina win: {result['prediction']['team2_win_prob']:.1%}")
```

## 🐛 Troubleshooting

### Common Issues

#### Issue 1: Module Not Found

**Error:**
```
ImportError: No module named 'pandas'
```

**Solution:**
```bash
pip install -r requirements.txt
```

#### Issue 2: File Not Found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/matches.csv'
```

**Solution:**
Ensure data files are in correct location:
```
worldcup-predictor/
└── data/
    ├── matches.csv
    ├── teams.csv
    └── qualified.csv
```

#### Issue 3: Slow Performance

**Problem:** Training takes too long

**Solution 1:** Reduce model complexity in `config.py`:
```python
MODEL_CONFIG = {
    'outcome_classifier': {
        'n_estimators': 100,  # Reduce from 200
        'max_depth': 8,       # Reduce from 10
    }
}
```

**Solution 2:** Reduce simulations:
```python
SIMULATION_CONFIG = {
    'num_simulations': 100,  # Reduce from 1000
}
```

#### Issue 4: Memory Error

**Problem:** Out of memory during training

**Solution:**
- Close other applications
- Use smaller dataset (sample rows)
- Reduce n_estimators

#### Issue 5: Permission Denied (run.sh)

**Error:**
```
bash: ./run.sh: Permission denied
```

**Solution:**
```bash
chmod +x run.sh
./run.sh
```

## 📁 Project Structure Quick Reference

```
worldcup-predictor/
├── main.py                # ← Run this file
├── config.py              # ← Customize settings here
├── predict_match.py       # ← Interactive predictions
├── requirements.txt       # ← Dependencies
│
├── data/
│   ├── matches.csv        # ← Your data
│   ├── teams.csv          # ← Your data
│   └── qualified.csv      # ← Your data
│
└── tests/
    └── test_predictor.py  # ← Run with pytest
```

## ✅ Installation Checklist

- [ ] Python 3.7+ installed
- [ ] Project files downloaded
- [ ] Data files in `data/` folder
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Successfully ran `python main.py`
- [ ] Results displayed in terminal

## 🎓 Next Steps

1. **Review Results**: Check `RESULTS.md` for detailed analysis
2. **Explore Code**: Read `main.py` to understand the ML pipeline
3. **Quick Reference**: See `FILE_REFERENCE.md` for all functions and usage
4. **Run Tests**: Execute `pytest tests/` for validation
5. **Customize**: Try different parameters in `config.py`
6. **Predict Matches**: Use `predict_match.py` for individual predictions
7. **Extend Model**: See `EXTENDING_MODEL.md` for adding features

## 💡 Tips

- **Start Simple**: Run with default settings first
- **Check Logs**: Terminal output shows progress and errors
- **Save Results**: Copy terminal output or check generated files
- **Experiment**: Try different simulation counts and model parameters
- **Read Docs**: All markdown files have valuable information
- **Quick Reference**: Use `FILE_REFERENCE.md` to find functions quickly

## 📞 Getting Help

1. **Check this guide** for common issues
2. **Review README.md** for full documentation
3. **See FILE_REFERENCE.md** for function documentation
4. **See RESULTS.md** for expected output
5. **Check test files** for code examples
6. **Review config.py** for all settings

## 🚀 You're Ready!

Run this command to start:

```bash
python main.py
```

Expected time: **~2 minutes** for complete analysis of 1,000 tournaments!

---

**Happy Predicting! 🏆⚽**
