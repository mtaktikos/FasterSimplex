"""
Verify Chaterjee's algorithm by comparing with Nash equilibrium.
This script tests whether the EPA algorithm produces the same game value as Nash equilibrium.
"""

import numpy as np
import nashpy as nash
from chaterjee_algorithm import chaterjee_epa_algorithm


def compute_nash_equilibrium(payoff_matrix):
    """Compute Nash equilibrium using nashpy."""
    game = nash.Game(payoff_matrix)
    equilibria = list(game.support_enumeration())
    
    if equilibria:
        row_strategy, col_strategy = equilibria[0]
        value = row_strategy @ payoff_matrix @ col_strategy
        return row_strategy, col_strategy, value
    else:
        return None, None, None


def test_paper_example_4v4():
    """Test the exact 4v4 example from the paper (Table 1)."""
    print("=" * 70)
    print("Test: Paper Example - 4 Guerrillas vs 4 Police (Table 1)")
    print("=" * 70)
    
    # From Table 1 in the paper (3x3 after eliminating dominated strategies)
    payoff_matrix = np.array([
        [0.5, 1.0, 1.0],
        [1.0, 0.5, 1.0],
        [1.0, 1.0, 0.0]
    ])
    
    print("Payoff matrix (3x3 from paper):")
    print(payoff_matrix)
    print()
    
    # Compute Nash equilibrium
    print("Computing Nash equilibrium...")
    row_nash, col_nash, value_nash = compute_nash_equilibrium(payoff_matrix)
    print(f"Nash row strategy: {row_nash}")
    print(f"Nash col strategy: {col_nash}")
    print(f"Nash game value: {value_nash:.6f}")
    print()
    
    # Compute with EPA
    print("Computing with Chaterjee's EPA...")
    row_epa, col_epa, value_epa = chaterjee_epa_algorithm(payoff_matrix, verbose=False)
    print(f"EPA row strategy: {row_epa}")
    print(f"EPA col strategy: {col_epa}")
    print(f"EPA game value: {value_epa:.6f}")
    print()
    
    # Compare
    print("Comparison:")
    print(f"Nash value: {value_nash:.6f}")
    print(f"EPA value:  {value_epa:.6f}")
    print(f"Match: {abs(value_nash - value_epa) < 1e-6}")
    print()
    
    return abs(value_nash - value_epa) < 1e-6


def test_symmetric_game():
    """Test a simple symmetric 2x2 game."""
    print("=" * 70)
    print("Test: Simple 2x2 Symmetric Game")
    print("=" * 70)
    
    # Rock-Paper-Scissors style payoff
    payoff_matrix = np.array([
        [0.5, 1.0],
        [0.0, 0.5]
    ])
    
    print("Payoff matrix:")
    print(payoff_matrix)
    print()
    
    # Nash equilibrium
    row_nash, col_nash, value_nash = compute_nash_equilibrium(payoff_matrix)
    print(f"Nash equilibrium:")
    print(f"  Row strategy: {row_nash}")
    print(f"  Col strategy: {col_nash}")
    print(f"  Game value: {value_nash:.6f}")
    print()
    
    # EPA
    row_epa, col_epa, value_epa = chaterjee_epa_algorithm(payoff_matrix, verbose=False)
    print(f"EPA algorithm:")
    print(f"  Row strategy: {row_epa}")
    print(f"  Col strategy: {col_epa}")
    print(f"  Game value: {value_epa:.6f}")
    print()
    
    # Compare
    print(f"Values match: {abs(value_nash - value_epa) < 1e-6}")
    print()
    
    return abs(value_nash - value_epa) < 1e-6


def test_asymmetric_game():
    """Test an asymmetric game where EPA might fail."""
    print("=" * 70)
    print("Test: Asymmetric 3x3 Game")
    print("=" * 70)
    
    # A game with no dominated strategies but unequal Nash probabilities
    payoff_matrix = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0]
    ])
    
    print("Payoff matrix:")
    print(payoff_matrix)
    print()
    
    # Nash equilibrium
    row_nash, col_nash, value_nash = compute_nash_equilibrium(payoff_matrix)
    print(f"Nash equilibrium:")
    print(f"  Row strategy: {row_nash}")
    print(f"  Col strategy: {col_nash}")
    print(f"  Game value: {value_nash:.6f}")
    print()
    
    # EPA
    row_epa, col_epa, value_epa = chaterjee_epa_algorithm(payoff_matrix, verbose=False)
    print(f"EPA algorithm:")
    print(f"  Row strategy: {row_epa}")
    print(f"  Col strategy: {col_epa}")
    print(f"  Game value: {value_epa:.6f}")
    print()
    
    # Compare
    diff = abs(value_nash - value_epa)
    print(f"Difference: {diff:.6f}")
    print(f"Values match: {diff < 1e-6}")
    print()
    
    return abs(value_nash - value_epa) < 1e-6


def find_counterexample():
    """
    Try to construct a counterexample where Chaterjee's EPA gives a different value
    than Nash equilibrium.
    """
    print("=" * 70)
    print("Searching for Counterexamples")
    print("=" * 70)
    print()
    
    # Test various game types
    test_cases = []
    
    # Case 1: Game where column player has strong preference
    print("Case 1: Asymmetric preferences")
    payoff1 = np.array([
        [3.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    row_nash1, col_nash1, val_nash1 = compute_nash_equilibrium(payoff1)
    row_epa1, col_epa1, val_epa1 = chaterjee_epa_algorithm(payoff1, verbose=False)
    
    print(f"  Nash value: {val_nash1:.6f}, EPA value: {val_epa1:.6f}")
    print(f"  Difference: {abs(val_nash1 - val_epa1):.6f}")
    
    if abs(val_nash1 - val_epa1) > 0.01:
        print("  *** COUNTEREXAMPLE FOUND! ***")
        print(f"  Nash col strategy: {col_nash1}")
        print(f"  EPA col strategy: {col_epa1}")
        test_cases.append(("Asymmetric preferences", payoff1, False))
    else:
        print("  Values match")
        test_cases.append(("Asymmetric preferences", payoff1, True))
    print()
    
    # Case 2: Game with tie outcomes
    print("Case 2: Game with ties")
    payoff2 = np.array([
        [0.5, 1.0, 0.0],
        [0.0, 0.5, 1.0],
        [1.0, 0.0, 0.5]
    ])
    
    row_nash2, col_nash2, val_nash2 = compute_nash_equilibrium(payoff2)
    row_epa2, col_epa2, val_epa2 = chaterjee_epa_algorithm(payoff2, verbose=False)
    
    print(f"  Nash value: {val_nash2:.6f}, EPA value: {val_epa2:.6f}")
    print(f"  Difference: {abs(val_nash2 - val_epa2):.6f}")
    
    if abs(val_nash2 - val_epa2) > 0.01:
        print("  *** COUNTEREXAMPLE FOUND! ***")
        print(f"  Nash col strategy: {col_nash2}")
        print(f"  EPA col strategy: {col_epa2}")
        test_cases.append(("Game with ties", payoff2, False))
    else:
        print("  Values match")
        test_cases.append(("Game with ties", payoff2, True))
    print()
    
    # Summary
    print("-" * 70)
    print("Summary:")
    matches = sum(1 for _, _, match in test_cases if match)
    print(f"  {matches}/{len(test_cases)} test cases matched")
    
    if matches < len(test_cases):
        print("\n  Counterexamples found! Chaterjee's algorithm does NOT always work.")
    else:
        print("\n  No counterexamples found in these tests.")
    print()
    
    return test_cases


if __name__ == "__main__":
    # Run tests
    test1 = test_paper_example_4v4()
    test2 = test_symmetric_game()
    test3 = test_asymmetric_game()
    
    # Search for counterexamples
    counterexample_results = find_counterexample()
    
    # Final summary
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Paper example (4v4): {'PASS' if test1 else 'FAIL'}")
    print(f"Symmetric game: {'PASS' if test2 else 'FAIL'}")
    print(f"Asymmetric game: {'PASS' if test3 else 'FAIL'}")
    print()
    
    all_passed = all([test1, test2, test3])
    counterexamples_found = any(not match for _, _, match in counterexample_results)
    
    if all_passed and not counterexamples_found:
        print("Conclusion: Chaterjee's EPA algorithm appears to work correctly")
        print("in all tested cases.")
    else:
        print("Conclusion: Chaterjee's EPA algorithm does NOT always produce")
        print("the same game value as Nash equilibrium.")
