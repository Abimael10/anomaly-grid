# Mathematical Implementation

Detailed mathematical foundations of the Anomaly Grid library.

## Core Algorithm

### Variable-Order Markov Chains

The library implements variable-order Markov chains where the order can range from 1 to `max_order`.

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

The algorithm uses a hierarchical fallback strategy:

1. Try context of length `max_order`
2. If no transitions found, try `max_order - 1`
3. Continue until context of length 1
4. If still no match, use uniform probability: `1 / |vocabulary|`

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

#### Normalized Likelihood

```
likelihood = exp(log_likelihood)
```

Note: Clamped to [0, 1] range for numerical stability.

#### Anomaly Strength

The final anomaly strength combines likelihood and information components:

```
combined_score = (log_likelihood_component × likelihood_weight + 
                 info_score × information_weight) / normalization_factor

anomaly_strength = tanh(combined_score)
```

Default weights:
- `likelihood_weight = 0.7`
- `information_weight = 0.3`
- `normalization_factor = 10.0`

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

All probabilities are maintained in the range [0, 1] with special handling for:
- Zero probabilities (smoothing)
- Underflow protection (minimum probability thresholds)
- Overflow protection (log-space calculations)

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

#### Log-Space Calculations
Likelihood calculations use log-space to prevent underflow:
```rust
log_likelihood += log_probability;
likelihood = exp(log_likelihood.clamp(MIN_LOG_PROB, 0.0));
```

#### Clamping and Bounds
- Probabilities clamped to [MIN_PROB, 1.0]
- Log probabilities clamped to [MIN_LOG_PROB, 0.0]
- Anomaly strengths normalized to [0, 1] using tanh

## Validation and Testing

### Mathematical Properties Verified

1. **Probability Conservation**: ∑ P(s | context) = 1 for all contexts
2. **Monotonicity**: Higher thresholds detect fewer anomalies
3. **Consistency**: Same input produces same output
4. **Bounds**: All scores within expected ranges
5. **Convergence**: Training converges to stable state

### Domain Testing

The library includes comprehensive domain testing covering:
- Markov chain mathematics
- Probability theory
- Information theory
- Anomaly detection logic
- Sequence analysis

Each domain has dedicated tests validating mathematical correctness and edge cases.