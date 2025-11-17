# Chaterjee's Equal Probabilities Approach (EPA) Algorithm

This repository contains an implementation and analysis of Chaterjee's EPA algorithm for solving two-person zero-sum games, as described in the paper "Computing two-person zero sum games at multiple times the speed of Linear Programming solvers."

## What is Chaterjee's Algorithm?

Chaterjee proposes a faster algorithm for computing the value of two-person zero-sum games (including Colonel Blotto games). The key innovation is in how the column player's (defender's) strategy is computed:

### Traditional Approach (Nash Equilibrium via LP)
- Both players solve linear programs to find optimal mixed strategies
- Computationally expensive, especially for large games
- Time complexity: O((n+m)^3.5 * log L) for Karmarkar's algorithm

### Chaterjee's EPA Algorithm
1. **Eliminate dominated strategies**: Iteratively remove weakly dominated strategies for both players
2. **Row player**: Solve LP to find optimal strategy (same as traditional)
3. **Column player**: Assign **equal probability (1/n)** to all remaining undominated strategies
4. **Compute game value**: Calculate expected payoff using these strategies

### Key Claim
The algorithm claims that using equal probabilities for the column player (after eliminating dominated strategies) gives the **same game value** as computing the full Nash equilibrium, while being computationally much simpler.

## Implementation

### Files
- `chaterjee_algorithm.py`: Implementation of the EPA algorithm
- `verify_algorithm.py`: Verification tests comparing EPA with Nash equilibrium
- `counterexample_search.py`: Systematic search for counterexamples

### Core Functions

```python
from chaterjee_algorithm import chaterjee_epa_algorithm

# Example: Solve a zero-sum game
payoff_matrix = np.array([
    [0.5, 1.0, 1.0],
    [1.0, 0.5, 1.0],
    [1.0, 1.0, 0.0]
])

row_strategy, col_strategy, game_value = chaterjee_epa_algorithm(payoff_matrix)
```

## Testing and Verification

### Examples from the Paper

The implementation successfully reproduces the examples from Chaterjee's paper:

**Example 1: 4 Guerrillas vs 4 Police**
- Payoff matrix from Table 1
- Nash equilibrium: Row [0.4, 0.4, 0.2], Col [0.4, 0.4, 0.2]
- EPA algorithm: Row [0.4, 0.4, 0.2], Col [0.33, 0.33, 0.33]
- **Game value: Both give 0.8** ✓

**Example 2: 8 Guerrillas vs 9 Police**
- After eliminating dominated strategies: 5 row × 6 col matrix
- **Game value: Both give 0.8** ✓

### Counterexample Search

I conducted an extensive search for counterexamples where Chaterjee's algorithm would give a different game value than Nash equilibrium:

**Test Categories:**
1. Games with non-uniform Nash column strategies
2. Highly skewed payoffs (10:5:1 ratios)
3. Large games (4×4 and above)
4. Edge cases (constant payoffs, matching pennies, etc.)

**Result: No counterexamples found!**

Even in cases where Nash equilibrium uses very different column probabilities (e.g., [0.0099, 0.9901] vs EPA's [0.5, 0.5]), the game value remains identical.

## Analysis: Why Does It Work?

This is a surprising result that deserves explanation. Here's what's happening:

### Mathematical Insight

For a two-person zero-sum game with payoff matrix A:
- Row player wants to **maximize** expected payoff
- Column player wants to **minimize** row player's payoff

At Nash equilibrium:
- Row strategy p* satisfies: p* ∈ argmax_p min_q p^T A q
- Column strategy q* satisfies: q* ∈ argmin_q max_p p^T A q
- Game value v* = p*^T A q*

In Chaterjee's EPA:
- Row strategy p_EPA is optimal against **uniform** column distribution q_uni = [1/n, ..., 1/n]
- p_EPA = argmax_p (p^T A q_uni)
- Game value v_EPA = p_EPA^T A q_uni

### Why They Match

The key insight is that **after eliminating dominated strategies**, the remaining game has special structure:

1. **No dominated strategies remain**: Every remaining strategy must be part of some best response
2. **Row player optimization**: The EPA finds the row strategy that maximizes value against uniform column play
3. **Minimax theorem**: Due to the structure after elimination, this gives the same value as full Nash

This works because:
- The dominated strategy elimination removes all "obvious" asymmetries
- The remaining strategies are in some sense "balanced"
- The row player's best response to uniform column play happens to achieve the minimax value

### When Might It Fail?

Theoretical considerations suggest it could fail when:
- The game has no dominated strategies but still requires highly skewed Nash probabilities
- The game structure is such that uniform column play is exploitable by the row player in a way that changes the game value

However, such counterexamples appear to be rare or non-existent in practice.

## Colonel Blotto Games

The paper focuses on Colonel Blotto games where:
- Two adversaries allocate resources to multiple battlefields
- One side wins if they achieve superiority on at least one battlefield
- The other wins only by defending all battlefields

The EPA algorithm is particularly effective here because:
1. Many allocations are dominated (e.g., putting all resources on one battlefield)
2. After elimination, remaining strategies are symmetric
3. Equal probabilities are often close to or exactly the Nash strategy

## Performance Claims

According to Chaterjee's paper:
- EPA is **4,131,382× faster** than GLOP (Google Linear Optimization) for large games
- EPA is **1.05-1.21× faster** than PSA (Probabilities Summing to 1) approach
- Time complexity: O(m × n × (m + n)) for dominated strategy elimination
- No exponential growth in input bit length L (unlike LP solvers)

## Conclusions

### What Works
✓ The EPA algorithm produces correct game values in all tested cases  
✓ It successfully solves Colonel Blotto games from the paper  
✓ It handles symmetric and asymmetric games  
✓ It works even when Nash equilibrium uses very skewed column probabilities  

### What's Unclear
? Theoretical proof of when/why it always works  
? Characterization of game classes where it's guaranteed to work  
? Whether there exist counterexamples (none found yet)  

### Practical Implications
- The algorithm is correct for the tested game classes
- It could potentially be faster than LP solvers for large games
- The dominated strategy elimination step is crucial
- The equal probability assignment is surprisingly robust

## Usage

```bash
# Run the main algorithm on Colonel Blotto examples
python chaterjee_algorithm.py

# Verify against Nash equilibrium
python verify_algorithm.py

# Search for counterexamples
python counterexample_search.py
```

## Requirements

```bash
pip install numpy scipy nashpy
```

## References

Chaterjee. "Computing two-person zero sum games at multiple times the speed of Linear Programming solvers." Research Square.

## License

This implementation is for educational and research purposes.
