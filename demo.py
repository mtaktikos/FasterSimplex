"""
Simple demonstration of Chaterjee's EPA algorithm.

This script shows how to use the algorithm to solve zero-sum games
and Colonel Blotto problems.
"""

import numpy as np
from chaterjee_algorithm import (
    chaterjee_epa_algorithm,
    generate_colonel_blotto_payoffs
)


def demo_simple_game():
    """Demonstrate on a simple 2x2 game."""
    print("=" * 70)
    print("Demo 1: Simple 2x2 Game")
    print("=" * 70)
    print()
    
    print("Consider a simple matching pennies variant:")
    print()
    payoff = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    
    print("Payoff matrix (row player's payoffs):")
    print(payoff)
    print()
    print("Row player wins 1 if both choose same, 0 otherwise")
    print("Column player wants to minimize row's payoff")
    print()
    
    row_strat, col_strat, value = chaterjee_epa_algorithm(payoff, verbose=True)
    
    print("Solution:")
    print(f"  Row strategy: {row_strat}")
    print(f"  Column strategy: {col_strat}")
    print(f"  Game value: {value:.3f}")
    print()
    print("Interpretation: Each player should randomize 50-50, giving")
    print("                expected payoff of 0.5 to row player.")
    print()


def demo_colonel_blotto():
    """Demonstrate on a Colonel Blotto game."""
    print("=" * 70)
    print("Demo 2: Colonel Blotto Game - 5 vs 5")
    print("=" * 70)
    print()
    
    print("Scenario: 5 guerrillas attack, 5 police defend 2 arsenals")
    print("Guerrillas win if they outnumber police at ANY arsenal")
    print("Police win if they defend BOTH arsenals")
    print()
    
    payoff, g_allocs, p_allocs = generate_colonel_blotto_payoffs(5, 5, 2)
    
    print(f"Number of strategies: {len(g_allocs)} (guerrillas) × {len(p_allocs)} (police)")
    print()
    print("Guerrilla allocations:")
    for i, alloc in enumerate(g_allocs):
        print(f"  {i}: {alloc}")
    print()
    print("Police allocations:")
    for i, alloc in enumerate(p_allocs):
        print(f"  {i}: {alloc}")
    print()
    
    row_strat, col_strat, value = chaterjee_epa_algorithm(payoff, verbose=False)
    
    print("Optimal strategies:")
    print()
    print("Guerrilla strategy (probability for each allocation):")
    for i, (alloc, prob) in enumerate(zip(g_allocs, row_strat)):
        if prob > 0.001:  # Only show non-zero probabilities
            print(f"  {alloc}: {prob:.4f}")
    print()
    
    print("Police strategy (probability for each allocation):")
    for i, (alloc, prob) in enumerate(zip(p_allocs, col_strat)):
        if prob > 0.001:
            print(f"  {alloc}: {prob:.4f}")
    print()
    
    print(f"Game value: {value:.4f}")
    print()
    print(f"Interpretation: Guerrillas have {value*100:.1f}% chance of winning")
    print()


def demo_asymmetric_blotto():
    """Demonstrate on an asymmetric Colonel Blotto game."""
    print("=" * 70)
    print("Demo 3: Asymmetric Colonel Blotto - 6 Guerrillas vs 8 Police")
    print("=" * 70)
    print()
    
    print("Scenario: Larger police force (8) vs guerrillas (6)")
    print()
    
    payoff, g_allocs, p_allocs = generate_colonel_blotto_payoffs(6, 8, 2)
    
    print(f"Matrix size: {payoff.shape[0]} × {payoff.shape[1]}")
    print()
    
    row_strat, col_strat, value = chaterjee_epa_algorithm(payoff, verbose=False)
    
    print(f"Game value: {value:.4f}")
    print(f"Guerrillas' winning probability: {value*100:.1f}%")
    print(f"Police's winning probability: {(1-value)*100:.1f}%")
    print()
    print("Note: Despite being outnumbered, guerrillas still have a")
    print("      reasonable chance due to the asymmetric objective")
    print("      (they only need to win ONE arsenal, police need BOTH)")
    print()


def demo_custom_game():
    """Allow user to define a custom game."""
    print("=" * 70)
    print("Demo 4: Custom Payoff Matrix")
    print("=" * 70)
    print()
    
    # Example from game theory textbook
    payoff = np.array([
        [3.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    print("Payoff matrix:")
    print(payoff)
    print()
    print("This represents a game where:")
    print("  - Row player prefers outcome (0,0) worth 3")
    print("  - Next best is (1,1) worth 2")
    print("  - Least preferred is (2,2) worth 1")
    print("  - All other outcomes worth 0")
    print()
    
    row_strat, col_strat, value = chaterjee_epa_algorithm(payoff, verbose=False)
    
    print("Solution:")
    print(f"  Row strategy: {np.round(row_strat, 4)}")
    print(f"  Column strategy: {np.round(col_strat, 4)}")
    print(f"  Game value: {value:.4f}")
    print()


def compare_game_sizes():
    """Compare performance on different game sizes."""
    print("=" * 70)
    print("Demo 5: Scaling Behavior")
    print("=" * 70)
    print()
    
    import time
    
    sizes = [(3, 3), (5, 5), (8, 9), (10, 12)]
    
    print("Testing Colonel Blotto games of different sizes:")
    print()
    print(f"{'Guerrillas':<12} {'Police':<10} {'Matrix Size':<15} {'Time (s)':<12} {'Value':<10}")
    print("-" * 70)
    
    for g, p in sizes:
        payoff, _, _ = generate_colonel_blotto_payoffs(g, p, 2)
        
        start = time.time()
        _, _, value = chaterjee_epa_algorithm(payoff, verbose=False)
        elapsed = time.time() - start
        
        matrix_size = f"{payoff.shape[0]}×{payoff.shape[1]}"
        print(f"{g:<12} {p:<10} {matrix_size:<15} {elapsed:<12.6f} {value:<10.4f}")
    
    print()
    print("Note: EPA algorithm handles these efficiently!")
    print()


if __name__ == "__main__":
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + " " * 15 + "CHATERJEE'S EPA ALGORITHM - DEMO" + " " * 21 + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print()
    
    demo_simple_game()
    input("Press Enter to continue...")
    
    demo_colonel_blotto()
    input("Press Enter to continue...")
    
    demo_asymmetric_blotto()
    input("Press Enter to continue...")
    
    demo_custom_game()
    input("Press Enter to continue...")
    
    compare_game_sizes()
    
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print()
    print("For more details, see:")
    print("  - chaterjee_algorithm.py: Core implementation")
    print("  - verify_algorithm.py: Verification against Nash equilibrium")
    print("  - counterexample_search.py: Search for counterexamples")
    print("  - theoretical_analysis.py: Theoretical investigation")
    print("  - README.md: Full documentation")
    print()
