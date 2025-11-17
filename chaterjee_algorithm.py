"""
Implementation of Chaterjee's Equal Probabilities Approach (EPA) Algorithm
for solving two-person zero-sum games.

Reference: "Computing two-person zero sum games at multiple times the speed 
of Linear Programming solvers" by Chaterjee

Key idea: After eliminating weakly dominated strategies, the column player
assigns equal probability (1/n) to all remaining strategies, while the row
player still computes optimal strategy via LP.
"""

import numpy as np
from typing import Tuple, List
import itertools


def eliminate_weakly_dominated_rows(matrix: np.ndarray) -> List[int]:
    """
    Identify rows that are weakly dominated.
    
    A row i is weakly dominated by row j if:
    - matrix[i, k] <= matrix[j, k] for all k, AND
    - matrix[i, k] < matrix[j, k] for at least one k
    
    Returns:
        List of 0/1 flags: 1 if row should be kept, 0 if dominated
    """
    keep = [1] * matrix.shape[0]
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if i == j:
                continue
            # Check if row i is weakly dominated by row j
            if np.all(matrix[i, :] <= matrix[j, :]) and np.any(matrix[i, :] < matrix[j, :]):
                keep[i] = 0
                break
    
    return keep


def eliminate_weakly_dominated_cols(matrix: np.ndarray) -> List[int]:
    """
    Identify columns that are weakly dominated.
    
    For a zero-sum game, column player wants to MINIMIZE row player's payoff.
    Column j is weakly dominated by column k if:
    - matrix[i, j] >= matrix[i, k] for all i, AND
    - matrix[i, j] > matrix[i, k] for at least one i
    
    Returns:
        List of 0/1 flags: 1 if column should be kept, 0 if dominated
    """
    keep = [1] * matrix.shape[1]
    
    for j in range(matrix.shape[1]):
        for k in range(matrix.shape[1]):
            if j == k:
                continue
            # Check if column j is weakly dominated by column k
            if np.all(matrix[:, j] >= matrix[:, k]) and np.any(matrix[:, j] > matrix[:, k]):
                keep[j] = 0
                break
    
    return keep


def iterated_elimination_dominated_strategies(matrix: np.ndarray) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Iteratively eliminate weakly dominated strategies for both players.
    
    Args:
        matrix: Row player's payoff matrix (m x n)
    
    Returns:
        reduced_matrix: Matrix after elimination
        row_flags: 1 if row kept, 0 if eliminated
        col_flags: 1 if column kept, 0 if eliminated
    """
    A = matrix.copy()
    orig_rows, orig_cols = matrix.shape
    row_flags = [1] * orig_rows
    col_flags = [1] * orig_cols
    
    active_rows = list(range(orig_rows))
    active_cols = list(range(orig_cols))
    
    changed = True
    while changed:
        changed = False
        
        # Eliminate weakly dominated rows
        current_matrix = A[np.ix_(active_rows, active_cols)]
        row_keep = eliminate_weakly_dominated_rows(current_matrix)
        
        if sum(row_keep) < len(row_keep):
            new_active_rows = [r for r, keep in zip(active_rows, row_keep) if keep == 1]
            for r in active_rows:
                if r not in new_active_rows:
                    row_flags[r] = 0
            active_rows = new_active_rows
            changed = True
        
        # Eliminate weakly dominated columns
        current_matrix = A[np.ix_(active_rows, active_cols)]
        col_keep = eliminate_weakly_dominated_cols(current_matrix)
        
        if sum(col_keep) < len(col_keep):
            new_active_cols = [c for c, keep in zip(active_cols, col_keep) if keep == 1]
            for c in active_cols:
                if c not in new_active_cols:
                    col_flags[c] = 0
            active_cols = new_active_cols
            changed = True
    
    reduced_matrix = A[np.ix_(active_rows, active_cols)]
    return reduced_matrix, row_flags, col_flags


def solve_row_player_lp(payoff_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Solve the row player's LP problem:
    
    maximize V
    subject to:
        sum_i p_i * payoff[i, j] >= V  for all j
        sum_i p_i = 1
        p_i >= 0
    
    Args:
        payoff_matrix: Row player's payoff matrix (reduced)
    
    Returns:
        row_strategy: Optimal mixed strategy for row player
        game_value: Value of the game
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        raise ImportError("scipy is required for LP solving")
    
    m, n = payoff_matrix.shape
    
    # Variables: [p_0, p_1, ..., p_{m-1}, V]
    # Objective: maximize V => minimize -V
    c = np.zeros(m + 1)
    c[-1] = -1.0  # Maximize V
    
    # Inequality constraints: sum_i p_i * payoff[i,j] >= V for all j
    # Rewritten: -sum_i p_i * payoff[i,j] + V <= 0
    A_ub = np.zeros((n, m + 1))
    for j in range(n):
        A_ub[j, :m] = -payoff_matrix[:, j]
        A_ub[j, m] = 1.0
    b_ub = np.zeros(n)
    
    # Equality constraint: sum_i p_i = 1
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])
    
    # Bounds: p_i >= 0, V unbounded
    bounds = [(0, None)] * m + [(None, None)]
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                     bounds=bounds, method='highs')
    
    if not result.success:
        raise ValueError(f"LP solver failed: {result.message}")
    
    row_strategy = result.x[:m]
    game_value = result.x[m]
    
    return row_strategy, game_value


def chaterjee_epa_algorithm(payoff_matrix: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Chaterjee's Equal Probabilities Approach (EPA) Algorithm.
    
    Algorithm:
    1. Eliminate weakly dominated strategies
    2. Solve LP for row player's optimal strategy
    3. Assign equal probabilities to all undominated column strategies
    4. Compute game value
    
    Args:
        payoff_matrix: Row player's payoff matrix (m x n)
        verbose: Whether to print debug information
    
    Returns:
        row_strategy: Full-size row player strategy (with 0s for dominated)
        col_strategy: Full-size column player strategy (equal probs for undominated)
        game_value: Expected value of the game for row player
    """
    if verbose:
        print("Original payoff matrix:")
        print(payoff_matrix)
        print()
    
    # Step 1: Eliminate dominated strategies
    reduced_matrix, row_flags, col_flags = iterated_elimination_dominated_strategies(payoff_matrix)
    
    if verbose:
        print("After eliminating dominated strategies:")
        print(f"Reduced matrix shape: {reduced_matrix.shape}")
        print(f"Active rows: {sum(row_flags)} / {len(row_flags)}")
        print(f"Active cols: {sum(col_flags)} / {len(col_flags)}")
        print("Reduced matrix:")
        print(reduced_matrix)
        print()
    
    # Step 2: Solve for row player's optimal strategy
    row_strategy_reduced, game_value = solve_row_player_lp(reduced_matrix)
    
    if verbose:
        print(f"Row player strategy (reduced): {row_strategy_reduced}")
        print(f"Game value: {game_value}")
        print()
    
    # Step 3: Assign equal probabilities to undominated column strategies
    num_active_cols = sum(col_flags)
    col_strategy_reduced = np.ones(num_active_cols) / num_active_cols
    
    if verbose:
        print(f"Column player strategy (reduced, equal probs): {col_strategy_reduced}")
        print()
    
    # Step 4: Expand strategies to full size
    row_strategy_full = np.zeros(len(row_flags))
    idx = 0
    for i, flag in enumerate(row_flags):
        if flag == 1:
            row_strategy_full[i] = row_strategy_reduced[idx]
            idx += 1
    
    col_strategy_full = np.zeros(len(col_flags))
    idx = 0
    for i, flag in enumerate(col_flags):
        if flag == 1:
            col_strategy_full[i] = col_strategy_reduced[idx]
            idx += 1
    
    # Verify game value with equal probabilities
    game_value_epa = np.dot(row_strategy_full, np.dot(payoff_matrix, col_strategy_full))
    
    if verbose:
        print(f"Game value with EPA: {game_value_epa}")
        print()
    
    return row_strategy_full, col_strategy_full, game_value_epa


def generate_colonel_blotto_payoffs(guerrillas: int, police: int, num_arsenals: int = 2) -> Tuple[np.ndarray, List, List]:
    """
    Generate payoff matrix for Colonel Blotto game.
    
    Game: Guerrillas try to capture at least one arsenal, police try to defend both.
    Guerrillas win if g > p at any arsenal.
    
    Args:
        guerrillas: Number of guerrillas
        police: Number of police
        num_arsenals: Number of arsenals (battlefields)
    
    Returns:
        payoff_matrix: Payoff matrix (guerrilla strategies x police strategies)
        guerrilla_allocations: List of guerrilla allocations
        police_allocations: List of police allocations
    """
    # Generate all possible allocations
    guerrilla_allocations = []
    for allocation in itertools.combinations_with_replacement(range(guerrillas + 1), num_arsenals):
        if sum(allocation) == guerrillas:
            # Generate all permutations of this allocation
            for perm in set(itertools.permutations(allocation)):
                guerrilla_allocations.append(perm)
    
    police_allocations = []
    for allocation in itertools.combinations_with_replacement(range(police + 1), num_arsenals):
        if sum(allocation) == police:
            for perm in set(itertools.permutations(allocation)):
                police_allocations.append(perm)
    
    # Sort for consistency
    guerrilla_allocations.sort()
    police_allocations.sort()
    
    # Generate payoff matrix
    payoff_matrix = np.zeros((len(guerrilla_allocations), len(police_allocations)))
    
    for i, g_alloc in enumerate(guerrilla_allocations):
        for j, p_alloc in enumerate(police_allocations):
            # Guerrillas win (payoff 1) if they outnumber police at ANY arsenal
            # Police win (payoff 0) only if they match or outnumber guerrillas at ALL arsenals
            guerrillas_win = any(g > p for g, p in zip(g_alloc, p_alloc))
            
            if guerrillas_win:
                payoff_matrix[i, j] = 1.0
            else:
                # If forces are equal at all arsenals, it's a tie
                all_equal = all(g == p for g, p in zip(g_alloc, p_alloc))
                if all_equal:
                    payoff_matrix[i, j] = 0.5
                else:
                    payoff_matrix[i, j] = 0.0
    
    return payoff_matrix, guerrilla_allocations, police_allocations


if __name__ == "__main__":
    print("=" * 70)
    print("Chaterjee's EPA Algorithm - Colonel Blotto Game Examples")
    print("=" * 70)
    print()
    
    # Example 1: 4 Guerrillas vs 4 Police (from paper)
    print("Example 1: 4 Guerrillas vs 4 Police")
    print("-" * 70)
    
    payoff, g_allocs, p_allocs = generate_colonel_blotto_payoffs(4, 4, 2)
    
    print(f"Number of guerrilla strategies: {len(g_allocs)}")
    print(f"Number of police strategies: {len(p_allocs)}")
    print(f"Guerrilla allocations: {g_allocs}")
    print(f"Police allocations: {p_allocs}")
    print()
    
    row_strat, col_strat, value = chaterjee_epa_algorithm(payoff, verbose=True)
    
    print("Final strategies:")
    print(f"Row (Guerrilla) strategy: {np.round(row_strat, 3)}")
    print(f"Column (Police) strategy: {np.round(col_strat, 3)}")
    print(f"Game value (EPA): {value:.6f}")
    print()
    
    # Example 2: 8 Guerrillas vs 9 Police (from paper)
    print("=" * 70)
    print("Example 2: 8 Guerrillas vs 9 Police")
    print("-" * 70)
    
    payoff2, g_allocs2, p_allocs2 = generate_colonel_blotto_payoffs(8, 9, 2)
    
    print(f"Number of guerrilla strategies: {len(g_allocs2)}")
    print(f"Number of police strategies: {len(p_allocs2)}")
    print()
    
    row_strat2, col_strat2, value2 = chaterjee_epa_algorithm(payoff2, verbose=True)
    
    print("Final strategies:")
    print(f"Game value (EPA): {value2:.6f}")
    print()
