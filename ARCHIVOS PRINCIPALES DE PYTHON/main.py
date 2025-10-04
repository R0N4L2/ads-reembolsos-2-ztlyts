"""
FIFA World Cup 2018 Match Outcome Predictor
===========================================
This module builds an ML model to predict match outcomes and simulates
the 2018 World Cup tournament 1,000 times.

Author: ADS Team Technical Assessment
Date: 2024
Version: 1.0
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Import configuration
try:
    from config import (
        MATCHES_FILE, TEAMS_FILE, QUALIFIED_FILE,
        MODEL_CONFIG, SIMULATION_CONFIG, FEATURE_COLUMNS,
        STRENGTH_WEIGHTS, RECENT_FORM_WINDOW, WC_EXPERIENCE_NORMALIZER,
        DEFAULT_TEAM_STRENGTH, OUTPUT_CONFIG
    )
except ImportError:
    # Fallback to defaults if config.py not found
    MATCHES_FILE = 'matches.csv'
    TEAMS_FILE = 'teams.csv'
    QUALIFIED_FILE = 'qualified.csv'
    MODEL_CONFIG = {
        'outcome_classifier': {'n_estimators': 200, 'max_depth': 10, 'random_state': 42},
        'goal_regressor': {'n_estimators': 100, 'max_depth': 8, 'random_state': 42}
    }
    SIMULATION_CONFIG = {'num_simulations': 1000, 'random_seed': 42}
    FEATURE_COLUMNS = None
    STRENGTH_WEIGHTS = {'win_rate': 0.3, 'recent_form': 0.3, 'goal_difference': 0.2, 'wc_experience': 0.2}
    RECENT_FORM_WINDOW = 10
    WC_EXPERIENCE_NORMALIZER = 20
    DEFAULT_TEAM_STRENGTH = 0.3
    OUTPUT_CONFIG = {'show_top_n_champions': 10}

np.random.seed(SIMULATION_CONFIG.get('random_seed', 42))


class TeamStrengthCalculator:
    """Calculates team strength metrics from historical matches."""
    
    def __init__(self, matches_df, teams_df):
        self.matches = matches_df.copy()
        self.teams = teams_df.copy()
        self.team_stats = self._calculate_team_stats()
        
    def _calculate_team_stats(self):
        """Calculate comprehensive statistics for each team."""
        stats = defaultdict(lambda: {
            'matches': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_for': 0,
            'goals_against': 0,
            'recent_form': [],
            'wc_matches': 0,
            'wc_wins': 0
        })
        
        for _, match in self.matches.iterrows():
            team1, team2 = match['team1'], match['team2']
            score1, score2 = match['team1Score'], match['team2Score']
            
            if pd.isna(score1) or pd.isna(score2):
                continue
                
            # Update stats for both teams
            for team, own_score, opp_score, is_team1 in [
                (team1, score1, score2, True),
                (team2, score2, score1, False)
            ]:
                stats[team]['matches'] += 1
                stats[team]['goals_for'] += own_score
                stats[team]['goals_against'] += opp_score
                
                # Determine result
                if own_score > opp_score:
                    stats[team]['wins'] += 1
                    stats[team]['recent_form'].append(3)
                elif own_score == opp_score:
                    stats[team]['draws'] += 1
                    stats[team]['recent_form'].append(1)
                else:
                    stats[team]['losses'] += 1
                    stats[team]['recent_form'].append(0)
                
                # World Cup specific stats
                if 'World Cup' in str(match['CupName']):
                    stats[team]['wc_matches'] += 1
                    if own_score > opp_score:
                        stats[team]['wc_wins'] += 1
        
        # Calculate derived metrics
        for team in stats:
            s = stats[team]
            s['win_rate'] = s['wins'] / max(s['matches'], 1)
            s['goal_diff'] = s['goals_for'] - s['goals_against']
            s['avg_goals_for'] = s['goals_for'] / max(s['matches'], 1)
            s['avg_goals_against'] = s['goals_against'] / max(s['matches'], 1)
            
            # Recent form (last N matches based on config)
            recent = s['recent_form'][-RECENT_FORM_WINDOW:] if len(s['recent_form']) >= RECENT_FORM_WINDOW else s['recent_form']
            s['recent_form_score'] = sum(recent) / max(len(recent), 1)
            
            # World Cup experience (normalized)
            s['wc_win_rate'] = s['wc_wins'] / max(s['wc_matches'], 1)
            s['wc_experience'] = min(s['wc_matches'] / WC_EXPERIENCE_NORMALIZER, 1)
            
            # Overall strength rating using configured weights
            s['strength_rating'] = (
                s['win_rate'] * STRENGTH_WEIGHTS['win_rate'] +
                s['recent_form_score'] / 3 * STRENGTH_WEIGHTS['recent_form'] +
                min(s['goal_diff'] / 100, 1) * STRENGTH_WEIGHTS['goal_difference'] +
                s['wc_experience'] * STRENGTH_WEIGHTS['wc_experience']
            )
        
        return dict(stats)
    
    def get_team_strength(self, team_code):
        """Get strength rating for a team."""
        if team_code in self.team_stats:
            return self.team_stats[team_code]['strength_rating']
        return DEFAULT_TEAM_STRENGTH  # Default for unknown teams
    
    def get_head_to_head(self, team1, team2):
        """Get head-to-head record between two teams."""
        h2h = {'team1_wins': 0, 'team2_wins': 0, 'draws': 0, 'matches': 0}
        
        for _, match in self.matches.iterrows():
            if pd.isna(match['team1Score']) or pd.isna(match['team2Score']):
                continue
                
            if (match['team1'] == team1 and match['team2'] == team2):
                h2h['matches'] += 1
                if match['team1Score'] > match['team2Score']:
                    h2h['team1_wins'] += 1
                elif match['team1Score'] < match['team2Score']:
                    h2h['team2_wins'] += 1
                else:
                    h2h['draws'] += 1
                    
            elif (match['team1'] == team2 and match['team2'] == team1):
                h2h['matches'] += 1
                if match['team1Score'] > match['team2Score']:
                    h2h['team2_wins'] += 1
                elif match['team1Score'] < match['team2Score']:
                    h2h['team1_wins'] += 1
                else:
                    h2h['draws'] += 1
        
        return h2h


class MatchPredictor:
    """ML model for predicting match outcomes."""
    
    def __init__(self, matches_df, teams_df):
        self.matches = matches_df
        self.teams = teams_df
        self.strength_calc = TeamStrengthCalculator(matches_df, teams_df)
        self.model = None
        self.goal_model_team1 = None
        self.goal_model_team2 = None
        
    def prepare_features(self, df):
        """Create features for ML model."""
        features = []
        
        for _, match in df.iterrows():
            if pd.isna(match['team1Score']) or pd.isna(match['team2Score']):
                continue
            
            team1, team2 = match['team1'], match['team2']
            
            # Team strength features
            team1_stats = self.strength_calc.team_stats.get(team1, {})
            team2_stats = self.strength_calc.team_stats.get(team2, {})
            
            # Head-to-head
            h2h = self.strength_calc.get_head_to_head(team1, team2)
            
            feature_dict = {
                'team1_strength': team1_stats.get('strength_rating', 0.3),
                'team2_strength': team2_stats.get('strength_rating', 0.3),
                'team1_win_rate': team1_stats.get('win_rate', 0.3),
                'team2_win_rate': team2_stats.get('win_rate', 0.3),
                'team1_recent_form': team1_stats.get('recent_form_score', 1),
                'team2_recent_form': team2_stats.get('recent_form_score', 1),
                'team1_avg_goals': team1_stats.get('avg_goals_for', 1),
                'team2_avg_goals': team2_stats.get('avg_goals_for', 1),
                'team1_wc_experience': team1_stats.get('wc_experience', 0),
                'team2_wc_experience': team2_stats.get('wc_experience', 0),
                'strength_diff': team1_stats.get('strength_rating', 0.3) - team2_stats.get('strength_rating', 0.3),
                'h2h_team1_wins': h2h['team1_wins'],
                'h2h_team2_wins': h2h['team2_wins'],
                'h2h_draws': h2h['draws'],
                'team1_score': match['team1Score'],
                'team2_score': match['team2Score']
            }
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def train(self):
        """Train the prediction models."""
        print("Preparing training data...")
        feature_df = self.prepare_features(self.matches)
        
        if len(feature_df) == 0:
            raise ValueError("No valid training data")
        
        # Prepare target variables
        feature_df['outcome'] = feature_df.apply(
            lambda x: 'team1_win' if x['team1_score'] > x['team2_score']
            else ('team2_win' if x['team1_score'] < x['team2_score'] else 'draw'),
            axis=1
        )
        
        # Features for prediction
        feature_cols = [
            'team1_strength', 'team2_strength', 'team1_win_rate', 'team2_win_rate',
            'team1_recent_form', 'team2_recent_form', 'team1_avg_goals', 'team2_avg_goals',
            'team1_wc_experience', 'team2_wc_experience', 'strength_diff',
            'h2h_team1_wins', 'h2h_team2_wins', 'h2h_draws'
        ]
        
        X = feature_df[feature_cols]
        y_outcome = feature_df['outcome']
        y_team1_goals = feature_df['team1_score']
        y_team2_goals = feature_df['team2_score']
        
        # Train outcome classifier
        print("Training outcome prediction model...")
        self.model = RandomForestClassifier(**MODEL_CONFIG['outcome_classifier'])
        self.model.fit(X, y_outcome)
        
        # Train goal prediction models
        print("Training goal prediction models...")
        self.goal_model_team1 = RandomForestRegressor(**MODEL_CONFIG['goal_regressor'])
        self.goal_model_team2 = RandomForestRegressor(**MODEL_CONFIG['goal_regressor'])
        self.goal_model_team1.fit(X, y_team1_goals)
        self.goal_model_team2.fit(X, y_team2_goals)
        
        print(f"Model trained on {len(feature_df)} matches")
        print(f"Feature importance (top 5):")
        importance = sorted(zip(feature_cols, self.model.feature_importances_), 
                          key=lambda x: x[1], reverse=True)
        for feat, imp in importance[:5]:
            print(f"  {feat}: {imp:.3f}")
    
    def predict_match(self, team1, team2):
        """Predict outcome and score for a match."""
        team1_stats = self.strength_calc.team_stats.get(team1, {})
        team2_stats = self.strength_calc.team_stats.get(team2, {})
        h2h = self.strength_calc.get_head_to_head(team1, team2)
        
        features = {
            'team1_strength': team1_stats.get('strength_rating', DEFAULT_TEAM_STRENGTH),
            'team2_strength': team2_stats.get('strength_rating', DEFAULT_TEAM_STRENGTH),
            'team1_win_rate': team1_stats.get('win_rate', DEFAULT_TEAM_STRENGTH),
            'team2_win_rate': team2_stats.get('win_rate', DEFAULT_TEAM_STRENGTH),
            'team1_recent_form': team1_stats.get('recent_form_score', 1),
            'team2_recent_form': team2_stats.get('recent_form_score', 1),
            'team1_avg_goals': team1_stats.get('avg_goals_for', 1),
            'team2_avg_goals': team2_stats.get('avg_goals_for', 1),
            'team1_wc_experience': team1_stats.get('wc_experience', 0),
            'team2_wc_experience': team2_stats.get('wc_experience', 0),
            'strength_diff': team1_stats.get('strength_rating', DEFAULT_TEAM_STRENGTH) - team2_stats.get('strength_rating', DEFAULT_TEAM_STRENGTH),
            'h2h_team1_wins': h2h['team1_wins'],
            'h2h_team2_wins': h2h['team2_wins'],
            'h2h_draws': h2h['draws']
        }
        
        X = pd.DataFrame([features])
        
        # Get probabilities
        probs = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        prob_dict = dict(zip(classes, probs))
        
        # Predict goals
        team1_goals = max(0, round(self.goal_model_team1.predict(X)[0]))
        team2_goals = max(0, round(self.goal_model_team2.predict(X)[0]))
        
        return {
            'team1_win_prob': prob_dict.get('team1_win', 0),
            'team2_win_prob': prob_dict.get('team2_win', 0),
            'draw_prob': prob_dict.get('draw', 0),
            'predicted_score': (team1_goals, team2_goals)
        }


class WorldCupSimulator:
    """Simulates the World Cup tournament."""
    
    def __init__(self, predictor, qualified_df):
        self.predictor = predictor
        self.qualified = qualified_df
        self.groups = self._parse_groups()
        
    def _parse_groups(self):
        """Parse qualified teams into groups."""
        groups = defaultdict(list)
        for _, row in self.qualified.iterrows():
            group_letter = row['draw'][0]
            groups[group_letter].append(row['name'])
        return dict(groups)
    
    def simulate_match(self, team1, team2, knockout=False):
        """Simulate a single match."""
        prediction = self.predictor.predict_match(team1, team2)
        
        # Use probabilities to determine outcome
        rand = np.random.random()
        
        if rand < prediction['team1_win_prob']:
            return team1, prediction['predicted_score']
        elif rand < prediction['team1_win_prob'] + prediction['draw_prob']:
            # Draw
            if knockout:
                # In knockout, use penalty shootout (50-50)
                winner = team1 if np.random.random() < 0.5 else team2
                return winner, prediction['predicted_score']
            return 'draw', prediction['predicted_score']
        else:
            return team2, prediction['predicted_score']
    
    def simulate_group_stage(self):
        """Simulate group stage matches."""
        group_results = {}
        
        for group_name, teams in self.groups.items():
            standings = {team: {'points': 0, 'gf': 0, 'ga': 0, 'gd': 0} for team in teams}
            
            # Each team plays each other once
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    team1, team2 = teams[i], teams[j]
                    winner, (score1, score2) = self.simulate_match(team1, team2)
                    
                    standings[team1]['gf'] += score1
                    standings[team1]['ga'] += score2
                    standings[team2]['gf'] += score2
                    standings[team2]['ga'] += score1
                    
                    if winner == team1:
                        standings[team1]['points'] += 3
                    elif winner == team2:
                        standings[team2]['points'] += 3
                    else:  # draw
                        standings[team1]['points'] += 1
                        standings[team2]['points'] += 1
            
            # Calculate goal difference
            for team in standings:
                standings[team]['gd'] = standings[team]['gf'] - standings[team]['ga']
            
            # Sort by points, then goal difference, then goals scored
            sorted_teams = sorted(
                standings.items(),
                key=lambda x: (x[1]['points'], x[1]['gd'], x[1]['gf']),
                reverse=True
            )
            
            group_results[group_name] = [team for team, _ in sorted_teams[:2]]
        
        return group_results
    
    def simulate_knockout_stage(self, qualified_teams):
        """Simulate knockout rounds."""
        # Round of 16 matchups based on group positions
        r16_matchups = [
            (qualified_teams['A'][0], qualified_teams['B'][1]),  # Match 1
            (qualified_teams['C'][0], qualified_teams['D'][1]),  # Match 2
            (qualified_teams['E'][0], qualified_teams['F'][1]),  # Match 3
            (qualified_teams['G'][0], qualified_teams['H'][1]),  # Match 4
            (qualified_teams['B'][0], qualified_teams['A'][1]),  # Match 5
            (qualified_teams['D'][0], qualified_teams['C'][1]),  # Match 6
            (qualified_teams['F'][0], qualified_teams['E'][1]),  # Match 7
            (qualified_teams['H'][0], qualified_teams['G'][1]),  # Match 8
        ]
        
        # Round of 16
        quarter_finalists = []
        for team1, team2 in r16_matchups:
            winner, _ = self.simulate_match(team1, team2, knockout=True)
            quarter_finalists.append(winner)
        
        # Quarter finals
        semi_finalists = []
        for i in range(0, 8, 2):
            winner, _ = self.simulate_match(quarter_finalists[i], quarter_finalists[i+1], knockout=True)
            semi_finalists.append(winner)
        
        # Semi finals
        finalists = []
        third_place_teams = []
        for i in range(0, 4, 2):
            winner, _ = self.simulate_match(semi_finalists[i], semi_finalists[i+1], knockout=True)
            finalists.append(winner)
            loser = semi_finalists[i+1] if winner == semi_finalists[i] else semi_finalists[i]
            third_place_teams.append(loser)
        
        # Third place match
        third_place, _ = self.simulate_match(third_place_teams[0], third_place_teams[1], knockout=True)
        
        # Final
        champion, _ = self.simulate_match(finalists[0], finalists[1], knockout=True)
        runner_up = finalists[1] if champion == finalists[0] else finalists[0]
        
        return {
            'champion': champion,
            'runner_up': runner_up,
            'third_place': third_place,
            'semi_finalists': semi_finalists,
            'quarter_finalists': quarter_finalists
        }
    
    def simulate_tournament(self):
        """Simulate entire tournament."""
        group_results = self.simulate_group_stage()
        knockout_results = self.simulate_knockout_stage(group_results)
        return knockout_results


def main():
    """Main execution function."""
    print("=" * 60)
    print("FIFA World Cup 2018 Prediction Model")
    print("=" * 60)
    print()
    
    # Load data
    print("Loading data...")
    matches = pd.read_csv(MATCHES_FILE)
    teams = pd.read_csv(TEAMS_FILE)
    qualified = pd.read_csv(QUALIFIED_FILE)
    
    print(f"Loaded {len(matches)} historical matches")
    print(f"Loaded {len(teams)} teams")
    print(f"Loaded {len(qualified)} qualified teams for 2018 WC")
    print()
    
    # Train model
    print("Training prediction model...")
    predictor = MatchPredictor(matches, teams)
    predictor.train()
    print()
    
    # Get simulation parameters
    num_sims = SIMULATION_CONFIG.get('num_simulations', 1000)
    verbose_freq = SIMULATION_CONFIG.get('verbose_frequency', 100)
    
    # Run simulations
    print("=" * 60)
    print(f"Running {num_sims:,} tournament simulations...")
    print("=" * 60)
    print()
    
    simulator = WorldCupSimulator(predictor, qualified)
    
    champions = []
    runners_up = []
    third_places = []
    semi_finalists = []
    quarter_finalists = []
    
    for i in range(num_sims):
        if (i + 1) % verbose_freq == 0:
            print(f"Completed {i + 1:,} simulations...")
        
        results = simulator.simulate_tournament()
        champions.append(results['champion'])
        runners_up.append(results['runner_up'])
        third_places.append(results['third_place'])
        semi_finalists.extend(results['semi_finalists'])
        quarter_finalists.extend(results['quarter_finalists'])
    
    print()
    print("=" * 60)
    print(f"SIMULATION RESULTS ({num_sims:,} tournaments)")
    print("=" * 60)
    print()
    
    # Championship probabilities
    print("CHAMPIONSHIP PROBABILITIES:")
    print("-" * 60)
    champion_counts = Counter(champions)
    top_n = OUTPUT_CONFIG.get('show_top_n_champions', 10)
    for team, count in champion_counts.most_common(top_n):
        percentage = (count / num_sims) * 100
        print(f"{team:20s} {percentage:6.2f}%  {'█' * int(percentage/2)}")
    print()
    
    # Runner-up probabilities
    print("RUNNER-UP PROBABILITIES:")
    print("-" * 60)
    runner_up_counts = Counter(runners_up)
    top_n = OUTPUT_CONFIG.get('show_top_n_runners_up', 10)
    for team, count in runner_up_counts.most_common(top_n):
        percentage = (count / num_sims) * 100
        print(f"{team:20s} {percentage:6.2f}%")
    print()
    
    # Semi-final probabilities
    print("SEMI-FINAL APPEARANCE PROBABILITIES:")
    print("-" * 60)
    semi_counts = Counter(semi_finalists)
    top_n = OUTPUT_CONFIG.get('show_top_n_semi_finalists', 10)
    for team, count in semi_counts.most_common(top_n):
        percentage = (count / (num_sims * 4)) * 100  # 4 semi-finalists per tournament
        print(f"{team:20s} {percentage:6.2f}%")
    print()
    
    # Quarter-final probabilities
    print("QUARTER-FINAL APPEARANCE PROBABILITIES:")
    print("-" * 60)
    quarter_counts = Counter(quarter_finalists)
    top_n = OUTPUT_CONFIG.get('show_top_n_quarter_finalists', 15)
    for team, count in quarter_counts.most_common(top_n):
        percentage = (count / (num_sims * 8)) * 100  # 8 quarter-finalists per tournament
        print(f"{team:20s} {percentage:6.2f}%")
    print()
    
    # Summary statistics
    print("=" * 60)
    print("KEY FINDINGS:")
    print("=" * 60)
    most_likely_champion = champion_counts.most_common(1)[0]
    print(f"Most likely champion: {most_likely_champion[0]} ({most_likely_champion[1]*100/num_sims:.1f}%)")
    print(f"Total unique champions: {len(champion_counts)}")
    print(f"Total unique finalists: {len(set(champions + runners_up))}")
    print()
    
    # Top 5 favorites
    print("TOP 5 FAVORITES TO WIN:")
    for i, (team, count) in enumerate(champion_counts.most_common(5), 1):
        print(f"  {i}. {team:15s} - {count*100/num_sims:.1f}%")


if __name__ == "__main__":
    main()
