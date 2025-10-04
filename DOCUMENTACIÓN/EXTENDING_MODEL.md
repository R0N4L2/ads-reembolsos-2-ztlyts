# Extending the World Cup Predictor Model

This guide explains how to customize and extend the prediction model with new features, different algorithms, and enhanced functionality.

---

## Table of Contents

1. [Adding New Features](#adding-new-features)
2. [Changing ML Algorithms](#changing-ml-algorithms)
3. [Incorporating External Data](#incorporating-external-data)
4. [Custom Simulation Logic](#custom-simulation-logic)
5. [Performance Optimization](#performance-optimization)
6. [Advanced Techniques](#advanced-techniques)

---

## Adding New Features

### 1. Team-Level Features

Add new features based on team statistics in `main.py`:

```python
# In TeamStrengthCalculator._calculate_team_stats()

# Example: Add home advantage metric
s['home_win_rate'] = home_wins / max(home_matches, 1)

# Example: Add recent goals metric
recent_goals = sum(recent_goals_list[-5:])
s['recent_goals'] = recent_goals / 5

# Example: Add consistency metric
s['consistency'] = 1 - np.std(recent_results)
```

### 2. Match-Level Features

Add contextual features for specific matchups in `main.py`:

```python
# In MatchPredictor.prepare_features()

# Venue advantage
feature_dict['venue_advantage'] = 1 if venue == team1_country else 0

# Continental matchup
team1_conf = self.teams[self.teams['fifa_code'] == team1]['confederation'].values[0]
team2_conf = self.teams[self.teams['fifa_code'] == team2]['confederation'].values[0]
feature_dict['continental_matchup'] = 1 if team1_conf == team2_conf else 0

# Rest days difference (if available)
feature_dict['rest_days_diff'] = team1_rest - team2_rest
```

### 3. Update Configuration

After adding features, update `config.py`:

```python
# In config.py

FEATURE_COLUMNS = [
    # ... existing features ...
    'team1_strength',
    'team2_strength',
    # ... add new features here ...
    'venue_advantage',
    'continental_matchup',
    'rest_days_diff',
]
```

### 4. Test New Features

Always test after adding features:

```python
# In tests/test_predictor.py

def test_new_features():
    """Test new feature implementation."""
    matches, teams = load_test_data()
    predictor = MatchPredictor(matches, teams)
    predictor.train()
    
    # Verify new features exist
    features = predictor.prepare_features(matches.head(10))
    assert 'venue_advantage' in features.columns
    assert 'continental_matchup' in features.columns
    
    # Test prediction still works
    result = predictor.predict_match('BRA', 'ARG')
    assert 'team1_win_prob' in result
    
    print("✓ New features working correctly")
```

---

## Changing ML Algorithms

### Using Gradient Boosting

Replace Random Forest with XGBoost in `main.py`:

```python
# At the top of main.py
from xgboost import XGBClassifier, XGBRegressor

# In MatchPredictor.train() method
self.model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

self.goal_model_team1 = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

self.goal_model_team2 = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
```

**Installation:**
```bash
pip install xgboost
```

**Advantages:**
- Better accuracy on structured data
- Faster training
- Better handling of feature interactions

### Using Neural Networks

For deep learning approach in `main.py`:

```python
from sklearn.neural_network import MLPClassifier, MLPRegressor

# In MatchPredictor.train()
self.model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

self.goal_model_team1 = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)
```

### Ensemble Multiple Models

Combine predictions from multiple algorithms in `main.py`:

```python
from sklearn.ensemble import VotingClassifier, VotingRegressor

# In MatchPredictor.train()

# Create individual models
rf_model = RandomForestClassifier(**MODEL_CONFIG['outcome_classifier'])
xgb_model = XGBClassifier(n_estimators=200, max_depth=6, random_state=42)

# Ensemble for outcome
self.model = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model),
    ],
    voting='soft'  # Use probability averages
)

self.model.fit(X, y_outcome)
```

**Update config.py** to support multiple models:

```python
# In config.py

MODEL_CONFIG = {
    'ensemble': True,  # New flag
    'outcome_classifier': {
        'n_estimators': 200,
        'max_depth': 10,
        'random_state': 42
    },
    'xgboost_classifier': {  # New section
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    }
}
```

---

## Incorporating External Data

### FIFA Rankings

Add current FIFA rankings as a feature:

```python
# Load FIFA rankings
fifa_rankings = pd.read_csv('fifa_rankings.csv')  # rank, team, points

# In prepare_features()
team1_rank = fifa_rankings[fifa_rankings['team'] == team1]['rank'].values[0]
team2_rank = fifa_rankings[fifa_rankings['team'] == team2]['rank'].values[0]

feature_dict['team1_fifa_rank'] = team1_rank
feature_dict['team2_fifa_rank'] = team2_rank
feature_dict['rank_difference'] = team1_rank - team2_rank
```

### Market Values

Integrate player market values from Transfermarkt:

```python
# Load market values
market_values = pd.read_csv('market_values.csv')  # team, total_value

feature_dict['team1_market_value'] = get_market_value(team1)
feature_dict['team2_market_value'] = get_market_value(team2)
feature_dict['value_ratio'] = team1_value / max(team2_value, 1)
```

### Weather Data

For venue-specific predictions:

```python
# If match venue is known
weather = get_weather_forecast(venue, match_date)

feature_dict['temperature'] = weather['temp']
feature_dict['humidity'] = weather['humidity']
feature_dict['is_rainy'] = 1 if weather['precipitation'] > 0 else 0
```

### Social Media Sentiment

Experimental feature using sentiment analysis:

```python
from textblob import TextBlob

# Analyze tweets about teams
team1_sentiment = analyze_twitter_sentiment(team1)
team2_sentiment = analyze_twitter_sentiment(team2)

feature_dict['team1_sentiment'] = team1_sentiment
feature_dict['team2_sentiment'] = team2_sentiment
```

---

## Custom Simulation Logic

### Alternative Group Stage Rules

Modify ranking criteria in `main.py`:

```python
# In WorldCupSimulator.simulate_group_stage()

# FIFA official tiebreaker rules
for team in standings:
    standings[team]['gd'] = standings[team]['gf'] - standings[team]['ga']

sorted_teams = sorted(
    standings.items(),
    key=lambda x: (
        x[1]['points'],           # Primary: Points
        x[1]['gd'],               # Secondary: Goal difference  
        x[1]['gf'],               # Tertiary: Goals scored
        x[1].get('h2h_points', 0), # Quaternary: Head-to-head
        x[1].get('fair_play', 0)   # Quinary: Fair play
    ),
    reverse=True
)

group_results[group_name] = [team for team, _ in sorted_teams[:2]]
```

### Dynamic Bracket Generation

Support different tournament formats by modifying `main.py`:

```python
# Add to WorldCupSimulator class

def simulate_custom_tournament(self, format_type='world_cup_2018'):
    """
    Simulate tournaments with different formats.
    
    Args:
        format_type: 'world_cup_2018', 'knockout_32', 'round_robin'
    """
    if format_type == 'world_cup_2018':
        # Standard: Group stage + knockout
        group_results = self.simulate_group_stage()
        return self.simulate_knockout_stage(group_results)
    
    elif format_type == 'knockout_32':
        # Single elimination from start
        return self.simulate_single_elimination_32()
    
    elif format_type == 'round_robin':
        # All teams play each other
        return self.simulate_round_robin()

def simulate_single_elimination_32(self):
    """32-team single elimination."""
    teams = list(self.qualified['name'])
    
    # Round of 32
    round_32_winners = []
    for i in range(0, 32, 2):
        winner, _ = self.simulate_match(teams[i], teams[i+1], knockout=True)
        round_32_winners.append(winner)
    
    # Continue with round of 16, etc...
    # ... (similar logic)
```

### Injury and Fatigue Simulation

Model player availability in `main.py`:

```python
# Add to WorldCupSimulator class

def __init__(self, predictor, qualified_df):
    self.predictor = predictor
    self.qualified = qualified_df
    self.groups = self._parse_groups()
    self.team_fatigue = {}  # Track fatigue
    self.last_match_date = {}  # Track rest days

def simulate_match_with_fatigue(self, team1, team2, days_since_last_match, knockout=False):
    """
    Adjust prediction based on rest days.
    
    Args:
        team1: Team 1 code
        team2: Team 2 code
        days_since_last_match: Dict with rest days for each team
        knockout: Whether this is knockout stage
    """
    base_prediction = self.predictor.predict_match(team1, team2)
    
    # Apply fatigue penalty
    team1_rest = days_since_last_match.get(team1, 7)
    team2_rest = days_since_last_match.get(team2, 7)
    
    # Reduce win probability if insufficient rest
    if team1_rest < 3:
        fatigue_factor = 0.9  # 10% performance reduction
        base_prediction['team1_win_prob'] *= fatigue_factor
        base_prediction['draw_prob'] *= 1.05
    
    if team2_rest < 3:
        fatigue_factor = 0.9
        base_prediction['team2_win_prob'] *= fatigue_factor
        base_prediction['draw_prob'] *= 1.05
    
    # Renormalize probabilities
    total = sum([
        base_prediction['team1_win_prob'],
        base_prediction['draw_prob'],
        base_prediction['team2_win_prob']
    ])
    
    base_prediction['team1_win_prob'] /= total
    base_prediction['draw_prob'] /= total
    base_prediction['team2_win_prob'] /= total
    
    # Simulate match with adjusted probabilities
    rand = np.random.random()
    
    if rand < base_prediction['team1_win_prob']:
        winner = team1
    elif rand < base_prediction['team1_win_prob'] + base_prediction['draw_prob']:
        winner = team2 if knockout and np.random.random() < 0.5 else 'draw'
    else:
        winner = team2
    
    return winner, base_prediction['predicted_score']
```

Update your simulation loop to use fatigue:

```python
# In simulate_tournament()

# Track rest days
rest_days = {team: 7 for team in all_teams}

for team1, team2 in group_stage_matches:
    winner, score = self.simulate_match_with_fatigue(
        team1, team2, rest_days, knockout=False
    )
    
    # Update rest days
    rest_days[team1] = 0
    rest_days[team2] = 0
```

---

## Performance Optimization

### Parallel Simulation

Speed up Monte Carlo with multiprocessing. Create a new file `parallel_simulator.py`:

```python
"""
Parallel simulation using multiprocessing.
Run with: python parallel_simulator.py
"""

from multiprocessing import Pool, cpu_count
import numpy as np
from main import MatchPredictor, WorldCupSimulator
import pandas as pd

def run_single_simulation(seed):
    """Run one tournament simulation with given seed."""
    np.random.seed(seed)
    
    # Load data
    matches = pd.read_csv('data/matches.csv')
    teams = pd.read_csv('data/teams.csv')
    qualified = pd.read_csv('data/qualified.csv')
    
    # Create predictor (should be cached)
    predictor = MatchPredictor(matches, teams)
    predictor.train()
    
    # Run simulation
    simulator = WorldCupSimulator(predictor, qualified)
    results = simulator.simulate_tournament()
    
    return results

def parallel_simulate(n_simulations=1000):
    """Run simulations in parallel."""
    n_cores = cpu_count()
    print(f"Using {n_cores} CPU cores")
    
    with Pool(n_cores) as pool:
        results = pool.map(run_single_simulation, range(n_simulations))
    
    return results

if __name__ == "__main__":
    print("Starting parallel simulation...")
    results = parallel_simulate(1000)
    
    # Extract champions
    champions = [r['champion'] for r in results]
    
    # Display results
    from collections import Counter
    champion_counts = Counter(champions)
    print("\nTop 5 Champions:")
    for team, count in champion_counts.most_common(5):
        print(f"{team}: {count/1000*100:.1f}%")
```

**Performance Gain**: 4-8x speedup on multi-core systems

### Model Caching

Save trained models to avoid retraining. Add to `main.py` or create `cache_model.py`:

```python
"""
Model caching for faster repeated use.
"""

import joblib
import os

# After training in main.py
def save_models(predictor, cache_dir='models'):
    """Save trained models to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    
    joblib.dump(predictor.model, f'{cache_dir}/outcome_model.pkl')
    joblib.dump(predictor.goal_model_team1, f'{cache_dir}/goal_model_1.pkl')
    joblib.dump(predictor.goal_model_team2, f'{cache_dir}/goal_model_2.pkl')
    joblib.dump(predictor.strength_calc, f'{cache_dir}/strength_calc.pkl')
    
    print(f"Models saved to {cache_dir}/")

def load_models(cache_dir='models'):
    """Load pre-trained models from disk."""
    from main import MatchPredictor
    import pandas as pd
    
    # Create empty predictor
    matches = pd.read_csv('data/matches.csv')
    teams = pd.read_csv('data/teams.csv')
    predictor = MatchPredictor(matches, teams)
    
    # Load trained models
    predictor.model = joblib.load(f'{cache_dir}/outcome_model.pkl')
    predictor.goal_model_team1 = joblib.load(f'{cache_dir}/goal_model_1.pkl')
    predictor.goal_model_team2 = joblib.load(f'{cache_dir}/goal_model_2.pkl')
    predictor.strength_calc = joblib.load(f'{cache_dir}/strength_calc.pkl')
    
    print("Models loaded from cache")
    return predictor

# Usage in main.py
if os.path.exists('models/outcome_model.pkl'):
    print("Loading cached models...")
    predictor = load_models()
else:
    print("Training new models...")
    predictor = MatchPredictor(matches, teams)
    predictor.train()
    save_models(predictor)
```

**Installation**:
```bash
pip install joblib
```

### Feature Engineering Optimization

Pre-compute features for known matchups. Add to `utils.py`:

```python
"""
Feature caching for faster predictions.
"""

import pickle
import hashlib

class FeatureCache:
    """Cache computed features for team matchups."""
    
    def __init__(self, cache_file='feature_cache.pkl'):
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk."""
        try:
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}
    
    def _save_cache(self):
        """Save cache to disk."""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def get_key(self, team1, team2):
        """Generate cache key for team matchup."""
        # Sort teams to make key order-independent
        teams = sorted([team1, team2])
        return f"{teams[0]}_vs_{teams[1]}"
    
    def get_features(self, team1, team2):
        """Get cached features or return None."""
        key = self.get_key(team1, team2)
        return self.cache.get(key)
    
    def set_features(self, team1, team2, features):
        """Cache features for team matchup."""
        key = self.get_key(team1, team2)
        self.cache[key] = features
        self._save_cache()

# Use in main.py
feature_cache = FeatureCache()

def predict_match_cached(self, team1, team2):
    """Predict match with feature caching."""
    # Try to get from cache
    cached_features = feature_cache.get_features(team1, team2)
    
    if cached_features is not None:
        X = pd.DataFrame([cached_features])
    else:
        # Compute features
        features = self._compute_features(team1, team2)
        feature_cache.set_features(team1, team2, features)
        X = pd.DataFrame([features])
    
    # Make prediction
    probs = self.model.predict_proba(X)[0]
    # ... rest of prediction logic
```

### Memory Optimization

For large-scale simulations, reduce memory usage. Add to `config.py`:

```python
# Memory-efficient settings
MEMORY_OPTIMIZATION = {
    'use_sparse_matrices': False,  # Use sparse matrices for features
    'batch_size': 100,              # Process in batches
    'clear_cache_interval': 500,    # Clear cache every N simulations
}
```

Implement in `main.py`:

```python
# In main() function

champions = []
gc_counter = 0

for i in range(num_sims):
    results = simulator.simulate_tournament()
    champions.append(results['champion'])
    
    # Periodic garbage collection
    gc_counter += 1
    if gc_counter % 500 == 0:
        import gc
        gc.collect()
        print(f"Memory cleared at simulation {i+1}")
```

---

## Advanced Techniques

### Bayesian Inference

Use probabilistic programming for uncertainty quantification:

```python
import pymc3 as pm

with pm.Model() as model:
    # Priors
    team_strength = pm.Normal('team_strength', mu=0, sd=1, shape=n_teams)
    home_advantage = pm.Normal('home_advantage', mu=0.3, sd=0.1)
    
    # Likelihood
    goal_rate = pm.math.exp(
        team_strength[team1_idx] - team_strength[team2_idx] + home_advantage
    )
    
    goals = pm.Poisson('goals', mu=goal_rate, observed=observed_goals)
    
    # Inference
    trace = pm.sample(2000, tune=1000)
```

### Transfer Learning

Use pre-trained models from other sports:

```python
# Load model trained on club football
base_model = load_pretrained_model('club_football_model.h5')

# Fine-tune on international matches
base_model.compile(optimizer='adam', loss='categorical_crossentropy')
base_model.fit(X_international, y_international, epochs=10)
```

### Time Series Forecasting

Model team strength evolution:

```python
from statsmodels.tsa.arima.model import ARIMA

# Forecast team strength for 2018
team_strengths_history = get_historical_strengths(team_code)

model = ARIMA(team_strengths_history, order=(2, 1, 2))
fitted = model.fit()

# Predict future strength
forecast = fitted.forecast(steps=12)  # Next 12 months
```

### Graph Neural Networks

Model team relationships:

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TeamGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 3)  # Win/Draw/Loss
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.softmax(x, dim=1)
```

### Causal Inference

Identify true causal effects:

```python
from econml.dml import DML

# Estimate effect of home advantage
dml = DML(
    model_y=RandomForestRegressor(),
    model_t=RandomForestClassifier(),
    discrete_treatment=True
)

dml.fit(Y=goals_scored, T=is_home, X=features)
treatment_effect = dml.effect(X_test)
```

---

## Example: Complete Feature Addition

Here's a complete example of adding FIFA rankings:

### Step 1: Obtain Data

```python
# fifa_rankings.csv
# date,team,rank,points
# 2018-06-01,BRA,1,1742
# 2018-06-01,GER,2,1714
# ...
```

### Step 2: Modify TeamStrengthCalculator

```python
class TeamStrengthCalculator:
    def __init__(self, matches_df, teams_df, fifa_rankings_df=None):
        self.matches = matches_df.copy()
        self.teams = teams_df.copy()
        self.fifa_rankings = fifa_rankings_df
        self.team_stats = self._calculate_team_stats()
```

### Step 3: Update Feature Preparation

```python
def prepare_features(self, df):
    # ... existing code ...
    
    if self.strength_calc.fifa_rankings is not None:
        team1_rank = self._get_fifa_rank(team1, match['date'])
        team2_rank = self._get_fifa_rank(team2, match['date'])
        
        feature_dict['team1_fifa_rank'] = team1_rank
        feature_dict['team2_fifa_rank'] = team2_rank
        feature_dict['fifa_rank_diff'] = team1_rank - team2_rank
```

### Step 4: Update Config

```python
# config.py
FEATURE_COLUMNS = [
    # ... existing features ...
    'team1_fifa_rank',
    'team2_fifa_rank',
    'fifa_rank_diff'
]
```

### Step 5: Test and Validate

```python
# Test with new features
matches, teams, qualified = load_data()
fifa_rankings = pd.read_csv('fifa_rankings.csv')

predictor = MatchPredictor(matches, teams, fifa_rankings)
predictor.train()

# Check feature importance
print("Feature importance:")
for feature, importance in zip(
    FEATURE_COLUMNS,
    predictor.model.feature_importances_
):
    print(f"{feature}: {importance:.3f}")
```

---

## Testing Extensions

Always test new features thoroughly:

```python
def test_new_feature():
    """Test new feature implementation."""
    # Load data
    matches, teams = load_test_data()
    
    # Train with new feature
    predictor = MatchPredictor(matches, teams)
    predictor.train()
    
    # Validate
    test_matches = matches.sample(100)
    predictions = []
    
    for _, match in test_matches.iterrows():
        pred = predictor.predict_match(match['team1'], match['team2'])
        predictions.append(pred)
    
    # Check accuracy
    accuracy = calculate_accuracy(predictions, test_matches)
    assert accuracy > 0.65, f"Accuracy too low: {accuracy}"
    
    print(f"✓ Test passed with accuracy: {accuracy:.2%}")
```

---

## Best Practices

1. **Version Control**: Track all changes with git
2. **Documentation**: Document new features and parameters
3. **Testing**: Add unit tests for new functionality
4. **Validation**: Use cross-validation to verify improvements
5. **Logging**: Log experiments and results
6. **Reproducibility**: Set random seeds for consistency

---

## Resources

- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Feature Engineering Book](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [Kaggle Competitions](https://www.kaggle.com/competitions) for inspiration

---

**Ready to extend?** Start with small changes, test thoroughly, and iterate!
