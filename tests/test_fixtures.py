"""
Unit tests for test fixtures.

Tests verify that fixture files are properly formatted and loadable.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFixtures:
    """Test suite for fixture files."""
    
    def test_toy_game_inputs_exists(self):
        """Test that toy_game_inputs.json exists."""
        fixture_path = Path("fixtures/toy_game_inputs.json")
        assert fixture_path.exists(), "toy_game_inputs.json not found"
    
    def test_toy_game_inputs_loadable(self):
        """Test that toy_game_inputs.json can be loaded."""
        fixture_path = Path("fixtures/toy_game_inputs.json")
        
        with open(fixture_path, 'r') as f:
            data = json.load(f)
        
        assert isinstance(data, dict)
        assert "game_id" in data
        assert "players" in data
        assert "opponent" in data
    
    def test_toy_game_inputs_structure(self):
        """Test that toy_game_inputs.json has correct structure."""
        fixture_path = Path("fixtures/toy_game_inputs.json")
        
        with open(fixture_path, 'r') as f:
            data = json.load(f)
        
        # Check game context
        assert data["game_id"] == "TEST_001"
        assert data["team_id"] == "GSW"
        assert data["opponent_id"] == "LAL"
        assert "pace" in data
        
        # Check players
        assert len(data["players"]) > 0
        
        for player in data["players"]:
            assert "player_id" in player
            assert "role" in player
            assert "posterior" in player
            
            # Check posterior structure
            posterior = player["posterior"]
            assert "mu" in posterior
            assert "Sigma" in posterior
            assert "feature_names" in posterior
            
            # Check dimensions match
            mu_len = len(posterior["mu"])
            sigma_shape = (len(posterior["Sigma"]), len(posterior["Sigma"][0]))
            feature_len = len(posterior["feature_names"])
            
            assert mu_len == feature_len, f"mu length {mu_len} != features length {feature_len}"
            assert sigma_shape[0] == feature_len, f"Sigma rows {sigma_shape[0]} != features length {feature_len}"
            assert sigma_shape[1] == feature_len, f"Sigma cols {sigma_shape[1]} != features length {feature_len}"
        
        # Check opponent
        opponent = data["opponent"]
        assert "opponent_id" in opponent
        assert "scheme_drop_rate" in opponent
        assert "pace" in opponent
    
    def test_small_eval_window_exists(self):
        """Test that small_eval_window.parquet exists."""
        fixture_path = Path("fixtures/small_eval_window.parquet")
        assert fixture_path.exists(), "small_eval_window.parquet not found"
    
    def test_small_eval_window_loadable(self):
        """Test that small_eval_window.parquet can be loaded."""
        fixture_path = Path("fixtures/small_eval_window.parquet")
        
        df = pd.read_parquet(fixture_path)
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
    
    def test_small_eval_window_structure(self):
        """Test that small_eval_window.parquet has correct structure."""
        fixture_path = Path("fixtures/small_eval_window.parquet")
        
        df = pd.read_parquet(fixture_path)
        
        # Check required columns
        required_cols = ['player_id', 'game_id', 'Date', 'PTS', 'TRB', 'AST']
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(df['Date'])
        assert pd.api.types.is_numeric_dtype(df['PTS'])
        assert pd.api.types.is_numeric_dtype(df['TRB'])
        assert pd.api.types.is_numeric_dtype(df['AST'])
        
        # Check data ranges
        assert df['PTS'].min() >= 0, "PTS should be non-negative"
        assert df['TRB'].min() >= 0, "TRB should be non-negative"
        assert df['AST'].min() >= 0, "AST should be non-negative"
        
        # Check we have multiple players and games
        assert df['player_id'].nunique() > 1, "Should have multiple players"
        assert df['game_id'].nunique() > 1, "Should have multiple games"
    
    def test_small_eval_window_statistics(self):
        """Test that small_eval_window.parquet has reasonable statistics."""
        fixture_path = Path("fixtures/small_eval_window.parquet")
        
        df = pd.read_parquet(fixture_path)
        
        # Check box stats are in reasonable ranges
        assert df['PTS'].max() < 100, "PTS should be realistic"
        assert df['TRB'].max() < 30, "TRB should be realistic"
        assert df['AST'].max() < 20, "AST should be realistic"
        assert df['MP'].max() < 50, "MP should be realistic"
        
        # Check percentages are in [0, 1] range
        pct_cols = ['FG%', '3P%', 'FT%', 'TS%', 'USG%', 'AST%', 'TOV%', 'TRB%', 'STL%', 'BLK%']
        for col in pct_cols:
            if col in df.columns:
                assert df[col].min() >= 0, f"{col} should be >= 0"
                assert df[col].max() <= 1, f"{col} should be <= 1"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
