# Advanced Markov Anomaly Detection Library
## User Manual

```
╔════════════════════════════════════════════════════╗
║                   anomaly-grid                     ║
║                   Library v1.0                     ║
╚════════════════════════════════════════════════════╝
```

## Dependencies

```toml
[dependencies]
nalgebra = "*"
ndarray = "*"
rayon = "*"
```

## Core Functionality

The library implements a variable-order Markov model for sequence anomaly detection with these actual capabilities:

- **Context Tree Building**: Creates contexts up to specified order
- **Probability Calculation**: Computes transition probabilities per context
- **Information Metrics**: Shannon entropy, KL-divergence per context
- **Spectral Analysis**: Eigenvalue decomposition of transition matrix
- **Quantum Representation**: Amplitude encoding with entropy-based phases
- **Multi-Score Anomalies**: 5 different scoring methods per sequence window

## Basic Usage

```rust
use your_lib::AdvancedTransitionModel;

// Initialize
let mut model = AdvancedTransitionModel::new(3); // max context length = 3

// Train on sequence
let sequence: Vec<String> = vec!["A", "B", "C", "A", "B", "D"]
    .into_iter().map(String::from).collect();

model.build_context_tree(&sequence)?;

// Detect anomalies
let test_seq: Vec<String> = vec!["A", "B", "X", "Y"]
    .into_iter().map(String::from).collect();

let anomalies = model.detect_advanced_anomalies(&test_seq, 0.1);
```

## What build_context_tree() Actually Does

1. **Context Creation**: For max_order=3, creates contexts of length 1, 2, and 3
   - Example: sequence ["A","B","C","A"] creates contexts like ["A"], ["A","B"], ["A","B","C"]

2. **Count Collection**: Tracks transition frequencies
   - Context ["A"] → "B" appears X times

3. **Probability Calculation**: `prob = count / total_count` for each context-transition pair

4. **Information Measures**:
   - Shannon entropy: `-Σ(p * log2(p))`
   - Information content: `-log2(entropy)`
   - KL divergence from uniform distribution

5. **Spectral Analysis**: 
   - Builds transition matrix from first-order contexts only
   - Computes eigenvalues via `matrix.complex_eigenvalues()`
   - Finds stationary distribution through power iteration

6. **Quantum State**: Creates equal superposition with entropy-based phases

## Anomaly Detection Process

`detect_advanced_anomalies()` slides a window of size `max_order + 1` across the sequence and scores each window with 5 methods:

### 1. Likelihood Score
```rust
// For each position, finds best probability across all context lengths
let weighted_prob = prob / sqrt(total_support);
likelihood *= weighted_prob;
```
Lower values = more anomalous.

### 2. Information Score
```rust
// Sums context entropy + transition surprise
(total_surprise + context_entropy) / sequence.len()
```
Higher values = more anomalous.

### 3. Spectral Score
```rust
// Deviation from stationary distribution
abs(observed_frequency - expected_stationary_prob)
```

### 4. Quantum Coherence
```rust
// Simplified coherence measure
1.0 - trace / n_states
```

### 5. Topological Signature
- Connected components count (= number of contexts)
- Cycle detection in sequence
- Local clustering coefficient

## Batch Processing

```rust
use your_lib::batch_process_sequences;

let sequences = vec![
    vec!["A".to_string(), "B".to_string()],
    vec!["X".to_string(), "Y".to_string()],
];

let results = batch_process_sequences(&sequences, 2, 0.1);
// Returns Vec<Vec<AnomalyScore>> - one result per input sequence
```

## AnomalyScore Structure

```rust
pub struct AnomalyScore {
    pub state_sequence: Vec<String>,           // The window that was scored
    pub likelihood: f64,                       // Likelihood-based score
    pub information_theoretic_score: f64,      // Information theory score
    pub spectral_anomaly_score: f64,          // Spectral deviation score
    pub quantum_coherence_measure: f64,        // Quantum coherence measure
    pub topological_signature: Vec<f64>,       // [components, cycles, clustering]
    pub confidence_interval: (f64, f64),      // Bayesian confidence bounds
}
```

## Limitations & Realistic Expectations

**Memory Usage**: O(|alphabet|^max_order) contexts stored in HashMap
- alphabet size 10, max_order 3 = up to 1000 contexts
- alphabet size 50, max_order 3 = up to 125,000 contexts

**Training Data Requirements**: Needs sufficient examples of each transition
- Minimum ~10 × max_order sequence length recommended
- Sparse contexts get unreliable probability estimates

**Spectral Analysis Constraints**:
- Only uses first-order transitions for matrix construction
- Stationary distribution may not converge for disconnected graphs
- Large alphabets create large matrices (expensive eigenvalue computation)

**Quantum Features**:
- Simplified implementation, not full quantum computation
- Phase encoding based on entropy values only
- Coherence measure is approximate

**Topological Analysis**:
- Basic cycle detection and clustering only
- Not full persistent homology as suggested by comments

## Performance Characteristics

Tested performance (approximate):
- Training: O(sequence_length × max_order)
- Detection: O(test_length × max_order × log(alphabet_size))
- Memory: O(alphabet_size^max_order)

Parallel processing uses rayon for batch operations across sequences.

## Practical Example

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Network traffic patterns
    let normal_traffic: Vec<String> = vec![
        "connect", "auth", "request", "response", "disconnect"
    ].into_iter().map(String::from).collect();
    
    let mut model = AdvancedTransitionModel::new(2);
    model.build_context_tree(&normal_traffic)?;
    
    // Test suspicious sequence
    let suspicious: Vec<String> = vec![
        "connect", "auth", "admin_panel", "delete_logs", "disconnect"
    ].into_iter().map(String::from).collect();
    
    let anomalies = model.detect_advanced_anomalies(&suspicious, 0.1);
    
    for anomaly in anomalies {
        if anomaly.likelihood < 0.01 {  // Very low likelihood
            println!("High anomaly: {:?}", anomaly.state_sequence);
            println!("Likelihood: {:.6}", anomaly.likelihood);
        }
    }
    
    Ok(())
}
```

## Testing

Run the included tests:
```bash
cargo test test_advanced_anomaly_detection
cargo test test_spectral_analysis
```

Both tests verify basic functionality with small example sequences.

---

This library provides multi-faceted anomaly scoring for sequential data using established mathematical techniques, implemented with reasonable computational complexity for practical use cases.