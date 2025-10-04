"""
Unit tests for World Cup Predictor
Run with: python -m pytest tests/test_predictor.py -v
"""

import pytest
import pandas as pd
import numpy as np
from main import TeamStrengthCalculator, MatchPredictor, WorldCupSimulator


@pytest.fixture
def sample_matches():
    """Create sample match data for testing."""
    return pd.DataFrame({
        'date': [20170101, 20170102, 20170103, 20170104],
        'team1': ['BRA', 'GER', 'BRA', 'FRA'],
        'team1Text': ['Brazil', 'Germany', 'Brazil', 'France'],
        'team2': ['ARG', 'FRA', 'GER', 'ESP'],
        'team2Text': ['Argentina', 'France', 'Germany', 'Spain'],
        'venue': ['Stadium A', 'Stadium B', 'Stadium C', 'Stadium D'],
        'IdCupSeason': [1, 1, 1, 1],
        'CupName': ['Friendly', 'World Cup', 'Friendly', 'World Cup'],
        'team1Score': [2.0, 1.0, 1.0, 2.0],
        'team2Score': [1.0, 1.0, 2.0, 2.0],
        'statText': [None, None, None, None],
        'resText': ['2-1', '1-1', '1-2', '2-2'],
        'team1PenScore': [None, None, None, None],
        'team2PenScore': [None, None, None, None]
    })


@pytest.fixture
def sample_teams():
    """Create sample team data for testing."""
    return pd.DataFrame({
        'confederation': ['CONMEBOL', 'UEFA', 'CONMEBOL', 'UEFA', 'UEFA'],
        'name': ['Brazil', 'Germany', 'Argentina', 'France', 'Spain'],
        'fifa_code': ['BRA', 'GER', 'ARG', 'FRA', 'ESP'],
        'ioc_code': ['BRA', 'GER', 'ARG', 'FRA', 'ESP']
    })


@pytest.fixture
def sample_qualified():
    """Create sample qualified teams for testing."""
    return pd.DataFrame({
        'name': ['BRA', 'GER', 'ARG', 'FRA'],
        'draw': ['A1', 'A2', 'B1', 'B2']
    })


class TestTeamStrengthCalculator:
    """Test cases for TeamStrengthCalculator class."""
    
    def test_initialization(self, sample_matches, sample_teams):
        """Test that calculator initializes correctly."""
        calc = TeamStrengthCalculator(sample_matches, sample_teams)
        assert calc is not None
        assert len(calc.team_stats) > 0
    
    def test_team_stats_calculation(self, sample_matches, sample_teams):
        """Test that team statistics are calculated correctly."""
        calc = TeamStrengthCalculator(sample_matches, sample_teams)
        
        # Brazil played 2 matches: won 1, lost 1
        brazil_stats = calc.team_stats['BRA']
        assert brazil_stats['matches'] == 2
        assert brazil_stats['wins'] == 1
        assert brazil_stats['losses'] == 1
        assert brazil_stats['goals_for'] == 3
        assert brazil_stats['goals_against'] == 3
    
    def test_strength_rating_range(self, sample_matches, sample_teams):
        """Test that strength ratings are in valid range."""
        calc = TeamStrengthCalculator(sample_matches, sample_teams)
        
        for team, stats in calc.team_stats.items():
            assert 0 <= stats['strength_rating'] <= 1, \
                f"Team {team} has invalid strength rating: {stats['strength_rating']}"
    
    def test_get_team_strength(self, sample_matches, sample_teams):
        """Test getting strength for known and unknown teams."""
        calc = TeamStrengthCalculator(sample_matches, sample_teams)
        
        # Known team
        brazil_strength = calc.get_team_strength('BRA')
        assert isinstance(brazil_strength, float)
        assert 0 <= brazil_strength <= 1
        
        # Unknown team
        unknown_strength = calc.get_team_strength('XXX')
        assert unknown_strength == 0.3  # Default value
    
    def test_head_to_head(self, sample_matches, sample_teams):
        """Test head-to-head record calculation."""
        calc = TeamStrengthCalculator(sample_matches, sample_teams)
        
        # Brazil vs Germany: 1 match (1-2)
        h2h = calc.get_head_to_head('BRA', 'GER')
        assert h2h['matches'] == 1
        assert h2h['team1_wins'] == 0  # Brazil lost
        assert h2h['team2_wins'] == 1  # Germany won
        
        # Teams with no matches
        h2h_none = calc.get_head_to_head('BRA', 'ESP')
        assert h2h_none['matches'] == 0


class TestMatchPredictor:
    """Test cases for MatchPredictor class."""
    
    def test_initialization(self, sample_matches, sample_teams):
        """Test that predictor initializes correctly."""
        predictor = MatchPredictor(sample_matches, sample_teams)
        assert predictor is not None
        assert predictor.model is None  # Not trained yet
    
    def test_feature_preparation(self, sample_matches, sample_teams):
        """Test feature preparation from matches."""
        predictor = MatchPredictor(sample_matches, sample_teams)
        features = predictor.prepare_features(sample_matches)
        
        assert len(features) == 4  # 4 valid matches
        assert 'team1_strength' in features.columns
        assert 'team2_strength' in features.columns
        assert 'strength_diff' in features.columns
    
    def test_training(self, sample_matches, sample_teams):
        """Test model training."""
        predictor = MatchPredictor(sample_matches, sample_teams)
        predictor.train()
        
        assert predictor.model is not None
        assert predictor.goal_model_team1 is not None
        assert predictor.goal_model_team2 is not None
    
    def test_prediction_output(self, sample_matches, sample_teams):
        """Test that predictions have correct format."""
        predictor = MatchPredictor(sample_matches, sample_teams)
        predictor.train()
        
        prediction = predictor.predict_match('BRA', 'ARG')
        
        # Check that all required keys exist
        assert 'team1_win_prob' in prediction
        assert 'team2_win_prob' in prediction
        assert 'draw_prob' in prediction
        assert 'predicted_score' in prediction
        
        # Check probability constraints
        total_prob = (prediction['team1_win_prob'] + 
                     prediction['team2_win_prob'] + 
                     prediction['draw_prob'])
        assert 0.99 <= total_prob <= 1.01  # Account for floating point
        
        # Check score format
        assert isinstance(prediction['predicted_score'], tuple)
        assert len(prediction['predicted_score']) == 2
        assert all(isinstance(s, (int, np.integer)) for s in prediction['predicted_score'])
        assert all(s >= 0 for s in prediction['predicted_score'])


class TestWorldCupSimulator:
    """Test cases for WorldCupSimulator class."""
    
    def test_initialization(self, sample_matches, sample_teams, sample_qualified):
        """Test simulator initialization."""
        predictor = MatchPredictor(sample_matches, sample_teams)
        predictor.train()
        
        simulator = WorldCupSimulator(predictor, sample_qualified)
        assert simulator is not None
        assert len(simulator.groups) == 2  # Groups A and B
    
    def test_group_parsing(self, sample_matches, sample_teams, sample_qualified):
        """Test that groups are parsed correctly."""
        predictor = MatchPredictor(sample_matches, sample_teams)
        predictor.train()
        
        simulator = WorldCupSimulator(predictor, sample_qualified)
        
        assert 'A' in simulator.groups
        assert 'B' in simulator.groups
        assert len(simulator.groups['A']) == 2
        assert len(simulator.groups['B']) == 2
    
    def test_match_simulation(self, sample_matches, sample_teams, sample_qualified):
        """Test single match simulation."""
        predictor = MatchPredictor(sample_matches, sample_teams)
        predictor.train()
        
        simulator = WorldCupSimulator(predictor, sample_qualified)
        
        # Regular match
        winner, score = simulator.simulate_match('BRA', 'ARG', knockout=False)
        assert winner in ['BRA', 'ARG', 'draw']
        assert isinstance(score, tuple)
        assert len(score) == 2
        
        # Knockout match (no draws allowed)
        winner, score = simulator.simulate_match('BRA', 'ARG', knockout=True)
        assert winner in ['BRA', 'ARG']
        assert winner != 'draw'
    
    def test_group_stage_simulation(self, sample_matches, sample_teams, sample_qualified):
        """Test group stage simulation."""
        predictor = MatchPredictor(sample_matches, sample_teams)
        predictor.train()
        
        simulator = WorldCupSimulator(predictor, sample_qualified)
        group_results = simulator.simulate_group_stage()
        
        # Check that 2 teams advance from each group
        for group, teams in group_results.items():
            assert len(teams) == 2
            # Verify advancing teams were in original group
            assert all(team in simulator.groups[group] for team in teams)
    
    def test_tournament_simulation(self, sample_matches, sample_teams, sample_qualified):
        """Test full tournament simulation."""
        predictor = MatchPredictor(sample_matches, sample_teams)
        predictor.train()
        
        simulator = WorldCupSimulator(predictor, sample_qualified)
        results = simulator.simulate_tournament()
        
        # Check required fields
        assert 'champion' in results
        assert 'runner_up' in results
        assert 'third_place' in results
        
        # Champion should be from qualified teams
        all_qualified = sample_qualified['name'].tolist()
        assert results['champion'] in all_qualified
        assert results['runner_up'] in all_qualified
        assert results['third_place'] in all_qualified
        
        # Champion and runner-up should be different
        assert results['champion'] != results['runner_up']


class TestDataValidation:
    """Test cases for data validation and edge cases."""
    
    def test_missing_scores(self, sample_teams):
        """Test handling of matches with missing scores."""
        matches_with_nulls = pd.DataFrame({
            'date': [20170101, 20170102],
            'team1': ['BRA', 'GER'],
            'team2': ['ARG', 'FRA'],
            'team1Score': [2.0, None],  # Missing score
            'team2Score': [1.0, None],
            'CupName': ['Friendly', 'Friendly']
        })
        
        calc = TeamStrengthCalculator(matches_with_nulls, sample_teams)
        
        # Brazil should have 1 match (the valid one)
        assert calc.team_stats['BRA']['matches'] == 1
        # Germany should have 0 matches (score was null)
        assert calc.team_stats.get('GER', {}).get('matches', 0) == 0
    
    def test_empty_dataframe(self, sample_teams):
        """Test handling of empty match data."""
        empty_matches = pd.DataFrame(columns=[
            'date', 'team1', 'team2', 'team1Score', 'team2Score', 'CupName'
        ])
        
        calc = TeamStrengthCalculator(empty_matches, sample_teams)
        assert len(calc.team_stats) == 0
    
    def test_reproducibility(self, sample_matches, sample_teams, sample_qualified):
        """Test that results are reproducible with same random seed."""
        np.random.seed(42)
        predictor1 = MatchPredictor(sample_matches, sample_teams)
        predictor1.train()
        simulator1 = WorldCupSimulator(predictor1, sample_qualified)
        
        np.random.seed(42)
        predictor2 = MatchPredictor(sample_matches, sample_teams)
        predictor2.train()
        simulator2 = WorldCupSimulator(predictor2, sample_qualified)
        
        # Run same simulation twice
        np.random.seed(123)
        result1 = simulator1.simulate_tournament()
        
        np.random.seed(123)
        result2 = simulator2.simulate_tournament()
        
        # Results should be identical
        assert result1['champion'] == result2['champion']
        assert result1['runner_up'] == result2['runner_up']


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_full_pipeline(self, sample_matches, sample_teams, sample_qualified):
        """Test the complete workflow from data to predictions."""
        # Initialize
        predictor = MatchPredictor(sample_matches, sample_teams)
        
        # Train
        predictor.train()
        
        # Simulate
        simulator = WorldCupSimulator(predictor, sample_qualified)
        
        # Run multiple tournaments
        champions = []
        for _ in range(10):
            result = simulator.simulate_tournament()
            champions.append(result['champion'])
        
        # Should have at least one champion
        assert len(champions) == 10
        # All champions should be from qualified teams
        assert all(c in sample_qualified['name'].tolist() for c in champions)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
