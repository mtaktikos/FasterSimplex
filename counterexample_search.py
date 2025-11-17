"""
Systematic search for counterexamples to Chaterjee's EPA algorithm.

The algorithm claims that after eliminating dominated strategies, assigning
equal probabilities to all remaining column strategies gives the same game
value as Nash equilibrium. We test this claim.
"""

import numpy as np
import nashpy as nash
from chaterjee_algorithm import chaterjee_epa_algorithm


def test_game(payoff_matrix, name="", verbose=True):
    """Test a specific game and return whether EPA matches Nash."""
    if verbose:
        print(f"\nTesting: {name}")
        print("Payoff matrix:")
        print(payoff_matrix)
    
    # Compute Nash equilibrium
    game = nash.Game(payoff_matrix)
    equilibria = list(game.support_enumeration())
    
    if not equilibria:
        if verbose:
            print("  No Nash equilibrium found (should not happen in finite games)")
        return None
    
    row_nash, col_nash = equilibria[0]
    value_nash = float(row_nash @ payoff_matrix @ col_nash)
    
    # Compute with EPA
    try:
        row_epa, col_epa, value_epa = chaterjee_epa_algorithm(payoff_matrix, verbose=False)
    except Exception as e:
        if verbose:
            print(f"  EPA failed: {e}")
        return None
    
    diff = abs(value_nash - value_epa)
    
    if verbose:
        print(f"  Nash value: {value_nash:.6f}")
        print(f"  EPA value:  {value_epa:.6f}")
        print(f"  Difference: {diff:.6f}")
        print(f"  Nash col strategy: {np.round(col_nash, 4)}")
        print(f"  EPA col strategy:  {np.round(col_epa, 4)}")
    
    # Consider it a counterexample if difference > 0.001
    if diff > 0.001:
        if verbose:
            print("  *** COUNTEREXAMPLE FOUND! ***")
        return False
    else:
        if verbose:
            print("  ✓ Match")
        return True


def test_non_uniform_nash_column():
    """
    Test a game where Nash equilibrium requires NON-UNIFORM column probabilities.
    This is the most likely place to find a counterexample.
    """
    print("=" * 70)
    print("Test Category: Games with Non-Uniform Nash Column Strategies")
    print("=" * 70)
    
    counterexamples = []
    
    # Test 1: Skewed payoffs
    test1 = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    result1 = test_game(test1, "Highly skewed payoffs")
    if result1 is False:
        counterexamples.append(("Skewed payoffs", test1))
    
    # Test 2: Asymmetric advantages
    test2 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0]
    ])
    result2 = test_game(test2, "Asymmetric with dominated row")
    if result2 is False:
        counterexamples.append(("Asymmetric dominated", test2))
    
    # Test 3: One strong column
    test3 = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.1]
    ])
    result3 = test_game(test3, "One weak column payoff")
    if result3 is False:
        counterexamples.append(("Weak column", test3))
    
    # Test 4: Extreme differences
    test4 = np.array([
        [100.0, 0.0],
        [0.0, 1.0]
    ])
    result4 = test_game(test4, "Extreme 2x2")
    if result4 is False:
        counterexamples.append(("Extreme 2x2", test4))
    
    # Test 5: Battle of sexes style
    test5 = np.array([
        [3.0, 0.0],
        [0.0, 1.0]
    ])
    result5 = test_game(test5, "Battle of sexes style (3:1 ratio)")
    if result5 is False:
        counterexamples.append(("Battle of sexes", test5))
    
    return counterexamples


def test_large_games():
    """Test larger games where differences might emerge."""
    print("\n" + "=" * 70)
    print("Test Category: Larger Games (4x4 and above)")
    print("=" * 70)
    
    counterexamples = []
    
    # Test 1: 4x4 with gradients
    test1 = np.array([
        [4.0, 3.0, 2.0, 1.0],
        [3.0, 4.0, 1.0, 2.0],
        [2.0, 1.0, 4.0, 3.0],
        [1.0, 2.0, 3.0, 4.0]
    ])
    result1 = test_game(test1, "4x4 gradient matrix")
    if result1 is False:
        counterexamples.append(("4x4 gradient", test1))
    
    # Test 2: Random-like but structured
    np.random.seed(42)
    test2 = np.random.uniform(0, 1, (4, 4))
    result2 = test_game(test2, "4x4 random uniform [0,1]")
    if result2 is False:
        counterexamples.append(("4x4 random", test2))
    
    return counterexamples


def test_degenerate_cases():
    """Test edge cases and degenerate games."""
    print("\n" + "=" * 70)
    print("Test Category: Edge Cases")
    print("=" * 70)
    
    counterexamples = []
    
    # Test 1: Constant payoff
    test1 = np.ones((3, 3))
    result1 = test_game(test1, "All payoffs equal to 1")
    if result1 is False:
        counterexamples.append(("Constant", test1))
    
    # Test 2: Zero-sum matching pennies
    test2 = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    result2 = test_game(test2, "Matching pennies")
    if result2 is False:
        counterexamples.append(("Matching pennies", test2))
    
    # Test 3: Only one cell has different value
    test3 = np.array([
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 1.0]
    ])
    result3 = test_game(test3, "One different cell")
    if result3 is False:
        counterexamples.append(("One different", test3))
    
    return counterexamples


def theoretical_analysis():
    """
    Provide theoretical analysis of when EPA should fail.
    """
    print("\n" + "=" * 70)
    print("Theoretical Analysis")
    print("=" * 70)
    print()
    print("The EPA algorithm assigns equal probabilities to undominated column")
    print("strategies. This will give the SAME game value as Nash equilibrium if:")
    print()
    print("1. The Nash equilibrium itself uses equal probabilities for columns, OR")
    print("2. The row player's LP solution is such that using equal column")
    print("   probabilities happens to give the same expected value")
    print()
    print("The algorithm might FAIL when:")
    print("- Nash equilibrium requires strongly skewed column probabilities")
    print("- Some columns should be played much more/less than others")
    print()
    print("However, in Chaterjee's formulation, AFTER eliminating dominated")
    print("strategies, it's possible that the remaining game has special structure")
    print("that makes equal probabilities work.")
    print()
    print("Key insight from the paper: The algorithm computes the row player's")
    print("best response to the uniform column distribution, which might yield")
    print("the same value as Nash equilibrium even with different column strategies.")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("Systematic Search for Counterexamples to Chaterjee's EPA Algorithm")
    print("=" * 70)
    
    all_counterexamples = []
    
    # Run test categories
    all_counterexamples.extend(test_non_uniform_nash_column())
    all_counterexamples.extend(test_large_games())
    all_counterexamples.extend(test_degenerate_cases())
    
    # Theoretical analysis
    theoretical_analysis()
    
    # Final summary
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print()
    
    if all_counterexamples:
        print(f"Found {len(all_counterexamples)} counterexample(s):")
        for name, matrix in all_counterexamples:
            print(f"  - {name}")
        print()
        print("Conclusion: Chaterjee's EPA algorithm does NOT always work!")
        print("The algorithm fails when Nash equilibrium requires non-uniform")
        print("column probabilities that cannot be approximated by uniform distribution.")
    else:
        print("No counterexamples found in tested cases.")
        print()
        print("Conclusion: Chaterjee's EPA algorithm appears robust in practice.")
        print("The algorithm may work for a broad class of games, particularly:")
        print("  - Symmetric games")
        print("  - Games where dominated strategy elimination leaves a")
        print("    structure that naturally leads to uniform probabilities")
        print("  - Colonel Blotto games as analyzed in the paper")
        print()
        print("However, absence of counterexamples in these tests does not prove")
        print("the algorithm is always correct. More theoretical analysis would")
        print("be needed to characterize exactly when it works.")
