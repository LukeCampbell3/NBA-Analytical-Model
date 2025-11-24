"""
Test script to verify results.txt generation
"""

import json
from pathlib import Path

def test_results_txt():
    """Test that results.txt was generated correctly."""
    results_file = Path("results/results.txt")
    
    # Check file exists
    assert results_file.exists(), "results.txt not found"
    
    # Read content
    with open(results_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Verify key sections exist
    required_sections = [
        "NBA PREDICTION SYSTEM - COMPREHENSIVE STUDY RESULTS",
        "METHODOLOGY",
        "PLAYER RANKINGS (BY METRIC)",
        "DETAILED PLAYER ANALYSIS",
        "CAPABILITY REGION GEOMETRY",
        "Ellipsoid Axes (Principal Components)",
        "STUDY SUMMARY",
        "KEY FINDINGS"
    ]
    
    for section in required_sections:
        assert section in content, f"Missing section: {section}"
    
    # Verify all players are included
    players = [
        "Stephen_Curry",
        "LeBron_James",
        "Nikola_Jokic",
        "Giannis_Antetokounmpo",
        "Luka_Doncic"
    ]
    
    for player in players:
        assert player in content, f"Missing player: {player}"
    
    # Verify percentile information is present
    assert "Percentile:" in content, "Missing percentile information"
    
    # Verify geometry details are present
    assert "Ellipsoid Volume:" in content, "Missing ellipsoid volume"
    assert "Direction =" in content, "Missing axis directions"
    
    print("✓ All tests passed!")
    print(f"✓ File size: {len(content):,} characters")
    print(f"✓ File location: {results_file.absolute()}")
    
    # Count lines
    lines = content.split('\n')
    print(f"✓ Total lines: {len(lines)}")
    
    return True


def test_json_results():
    """Test that JSON results are valid."""
    json_file = Path("results/final_results.json")
    
    assert json_file.exists(), "final_results.json not found"
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Verify structure
    assert 'timestamp' in data, "Missing timestamp"
    assert 'season' in data, "Missing season"
    assert 'players' in data, "Missing players"
    
    # Verify player data
    assert len(data['players']) == 5, "Expected 5 players"
    
    for player in data['players']:
        required_fields = [
            'player', 'games', 'avg_pts', 'avg_reb', 'avg_ast',
            'avg_pts_percentile', 'avg_reb_percentile', 'avg_ast_percentile',
            'posterior_center', 'region_alpha', 'region_dim'
        ]
        for field in required_fields:
            assert field in player, f"Missing field {field} for player {player.get('player', 'unknown')}"
    
    print("✓ JSON results valid!")
    print(f"✓ Players: {', '.join(p['player'] for p in data['players'])}")
    
    return True


if __name__ == "__main__":
    print("Testing results generation...\n")
    test_results_txt()
    print()
    test_json_results()
    print("\n✓ All tests passed successfully!")
