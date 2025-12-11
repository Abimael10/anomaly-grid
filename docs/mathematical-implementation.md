# Mathematical Implementation

How the variable-order Markov model computes probabilities and anomaly scores for finite alphabets.

## Core Algorithm

### Variable-Order Markov Chains

Order ranges from 1 to `max_order`, with a fallback from longer to shorter contexts.

#### State Transition Probability

For a context of length `k` and next state `s`:

```
P(s | context) = (count(context → s) + α) / (count(context) + α × |vocabulary|)
```

Where:
- `α` is the smoothing parameter (configurable, default: 1.0)
- `|vocabulary|` is the size of the alphabet
- `count(context → s)` is the number of times state `s` follows the context
- `count(context)` is the total number of times the context appears

#### Hierarchical Context Selection

Fallback strategy:
1. Try context of length `max_order`
2. If none, try shorter contexts down to length 1
3. If still none, use uniform probability `1 / |vocabulary|`

### Information Theory

#### Shannon Entropy

For a probability distribution P:

```
H(X) = -∑ P(x) log₂ P(x)
```

#### Information Content

For a specific outcome with probability P(x):

```
I(x) = -log₂ P(x)
```

#### Average Information Score

For a sequence of length n:

```
avg_info = (1/n) × ∑ I(xᵢ)
```

### Anomaly Scoring

#### Log-Likelihood Calculation

For a sequence S = [s₁, s₂, ..., sₙ]:

```
log_likelihood = ∑ log P(sᵢ | context_i)
```

#### Anomaly Strength

The implementation combines a surprise term (−ln(likelihood) scaled to [0,1]) and an information term (information_score scaled to [0,1]) using the configured weights. The weighted score is then passed through a calibrated, piecewise scaling to keep values in [0,1] and accentuate higher-risk windows. Default weights: likelihood_weight = 0.7, information_weight = 0.3.

## Memory Optimization

### String Interning

Duplicate strings are stored only once using a string interner, reducing memory usage for repeated elements.

### Trie-Based Storage

Context trees use trie structures to share common prefixes, significantly reducing memory usage for overlapping contexts.

### Lazy Computation

Entropy and information scores are computed on-demand and cached to avoid redundant calculations.

## Complexity Analysis

### Time Complexity

#### Training
- **Best case**: O(n × max_order) where n is sequence length
- **Worst case**: O(n × max_order × |alphabet|) with hash collisions
- **Average case**: O(n × max_order × log(contexts))

#### Detection
- **Per window**: O(max_order × log(contexts))
- **Full sequence**: O(m × max_order × log(contexts)) where m is test sequence length

### Space Complexity

#### Theoretical Maximum
```
max_contexts = ∑(k=1 to max_order) |alphabet|^k
```

#### Practical Usage
Actual memory usage is typically much lower due to:
- Not all possible contexts appear in training data
- String interning reduces duplicate storage
- Trie structure shares common prefixes

#### Memory Estimation
```rust
fn estimate_memory_usage(contexts: usize, avg_context_length: usize) -> usize {
    let base_overhead = 64; // bytes per context node
    let string_storage = avg_context_length * 8; // estimated string storage
    contexts * (base_overhead + string_storage)
}
```

## Numerical Considerations

### Probability Bounds

Probabilities are kept in [0, 1] with smoothing to avoid zeros and log-space math to avoid underflow.

### Smoothing Strategies

#### Laplace Smoothing (Default)
```
P_smooth(s | context) = (count + α) / (total + α × |vocab|)
```

#### Add-k Smoothing
Configurable α parameter allows for different smoothing strengths:
- α = 0: No smoothing (may cause zero probabilities)
- α = 1: Laplace smoothing (default)
- α > 1: Stronger smoothing (more uniform distribution)

### Numerical Stability

- Likelihood calculations use log-space with clamping for underflow/overflow protection.
- Anomaly strengths are kept in [0, 1] using a calibrated piecewise scaling of the weighted score.

## Validation and Testing

Tests cover probability conservation (∑P=1), threshold monotonicity, consistent outputs, and score bounds; domain tests exercise Markov math, information measures, detection logic, and sequence analysis.