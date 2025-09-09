# Anomaly Grid

     █████╗ ███╗   ██╗ ██████╗ ███╗   ███╗ █████╗ ██╗  ██╗   ██╗
    ██╔══██╗████╗  ██║██╔═══██╗████╗ ████║██╔══██╗██║  ╚██╗ ██╔╝
    ███████║██╔██╗ ██║██║   ██║██╔████╔██║███████║██║   ╚████╔╝ 
    ██╔══██║██║╚██╗██║██║   ██║██║╚██╔╝██║██╔══██║██║    ╚██╔╝  
    ██║  ██║██║ ╚████║╚██████╔╝██║ ╚═╝ ██║██║  ██║███████╗██║   
    ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   
    [ANOMALY-GRID v0.2.1] - SEQUENCE ANOMALY DETECTION ENGINE

[![Crates.io](https://img.shields.io/crates/v/anomaly-grid.svg)](https://crates.io/crates/anomaly-grid)
[![Documentation](https://docs.rs/anomaly-grid/badge.svg)](https://docs.rs/anomaly-grid)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Rust library implementing variable-order Markov chains for sequence anomaly detection in finite alphabets.

## What This Library Does

- **Variable-Order Markov Models**: Builds contexts of length 1 to max_order from training sequences
- **Hierarchical Context Selection**: Uses longest available context, falls back to shorter contexts
- **Configurable Laplace Smoothing**: Adjustable α parameter for probability estimation (default: 1.0)
- **Information-Theoretic Scoring**: Shannon entropy and KL divergence with lazy computation
- **Memory-Optimized Storage**: String interning, trie-based contexts, and small collections
- **Sliding Window Detection**: Analyzes test sequences using overlapping windows
- **Parallel Batch Processing**: Processes multiple sequences using Rayon

## Known Limitations So Far

### Memory Complexity: O(|alphabet|^max_order) theoretical, optimized in practice
**Measured results with substantial datasets:**

| Scenario | Alphabet Size | Max Order | Elements Processed | Theoretical Max | Actual Contexts | Memory Usage | Efficiency |
|----------|---------------|-----------|-------------------|-----------------|-----------------|--------------|------------|
| IoT Monitoring | 20 | 3 | 1.6M | 8,420 | 830 | 17.4 MB | 9.9% |
| Security Analysis | 50 | 3 | 1.8M | 127,550 | 19,867 | 352.7 MB | 15.6% |
| Enterprise Monitor | 100 | 2 | 2.4M | 10,100 | 2,892 | 89.9 MB | 28.6% |
| Pattern Detection | 30 | 4 | 2.2M | 837,930 | 8,028 | 135.5 MB | 1.0% |

**Memory optimizations reduce actual usage:**
- **String interning** reduces duplicate string storage
- **Trie-based storage** shares common context prefixes  
- **On-demand computation** avoids storing precomputed values
- **Small collections** use memory-efficient data structures

### Algorithm Limitations
- **Standard Markov Model**: Basic implementation, no theoretical optimality guarantees
- **Greedy Context Selection**: Uses longest available context (hierarchical fallback)
- **Configurable Smoothing**: Laplace smoothing with configurable α parameter (default: 1.0)
- **Prefix Compression**: Uses trie-based storage for memory efficiency

### Practical Constraints
- **Memory Bound**: Will fail on large state spaces
- **Sequential Processing**: Not designed for real-time high-volume streams (No plans to change this by my own, it is too much)

## Installation

```toml
[dependencies]
anomaly-grid = "0.2.1"
```

## Basic Usage

```rust
use anomaly_grid::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create detector (max_order determines memory usage)
    let mut detector = AnomalyDetector::new(3)?;
    
    // Train on normal patterns
    let normal_sequence = vec![
        "A".to_string(), "B".to_string(), "C".to_string(),
        "A".to_string(), "B".to_string(), "C".to_string(),
    ];
    detector.train(&normal_sequence)?;
    
    // Detect anomalies
    let test_sequence = vec![
        "A".to_string(), "X".to_string(), "Y".to_string(),
    ];
    let anomalies = detector.detect_anomalies(&test_sequence, 0.1)?;
    
    for anomaly in anomalies {
        println!("Sequence: {:?}, Likelihood: {:.6}", 
                 anomaly.sequence, anomaly.likelihood);
    }
    
    Ok(())
}
```

## API Reference

### AnomalyDetector

```rust
// Constructors - return Result due to validation
pub fn new(max_order: usize) -> AnomalyGridResult<Self>
pub fn with_config(config: AnomalyGridConfig) -> AnomalyGridResult<Self>

// Training - builds context tree from sequence
pub fn train(&mut self, sequence: &[String]) -> AnomalyGridResult<()>

// Detection - sliding window analysis
pub fn detect_anomalies(&self, sequence: &[String], threshold: f64) -> AnomalyGridResult<Vec<AnomalyScore>>
pub fn detect_anomalies_with_monitoring(&mut self, sequence: &[String], threshold: f64) -> AnomalyGridResult<Vec<AnomalyScore>>

// Performance monitoring and optimization
pub fn performance_metrics(&self) -> &PerformanceMetrics
pub fn optimize(&mut self, optimization_config: &OptimizationConfig) -> AnomalyGridResult<()>

// Batch processing - parallel analysis
pub fn batch_process_sequences(
    sequences: &[Vec<String>],
    config: &AnomalyGridConfig,
    threshold: f64,
) -> AnomalyGridResult<Vec<Vec<AnomalyScore>>>
```

### AnomalyScore

```rust
pub struct AnomalyScore {
    pub sequence: Vec<String>,      // The analyzed window
    pub likelihood: f64,            // P(sequence|model) ∈ [0,1]
    pub log_likelihood: f64,        // ln(likelihood)
    pub information_score: f64,     // Average -log₂(P(x))
    pub anomaly_strength: f64,      // Normalized score ∈ [0,1]
}
```

## Mathematical Implementation

### Probability Estimation
```
P(next_state | context) = (count + α) / (total_count + α × vocab_size)
```
where α is configurable (default: 1.0, Laplace smoothing)

### Shannon Entropy
```
H(X) = -∑ P(x) log₂ P(x)
```

### Context Selection
1. Try context of length max_order
2. If no transitions found, try max_order-1
3. Continue until context of length 1
4. If still no match, use uniform probability

### Anomaly Scoring
```
anomaly_strength = tanh((log_likelihood_component × likelihood_weight + info_score × information_weight) / normalization_factor)
```
where weights are configurable (default: likelihood_weight=0.7, information_weight=0.3, normalization_factor=10.0)

## Complexity Analysis

### Time Complexity (Theoretical)
- **Training**: O(n × max_order × |alphabet|)
- **Detection**: O(m × max_order) where m = test sequence length
- **Memory**: O(|alphabet|^max_order)

**Note**: Actual performance depends heavily on your data characteristics, sequence patterns, and system resources. Profile your specific use case.

## Configuration

### Choosing max_order
- **Order 2**: Lower memory usage, captures immediate dependencies
- **Order 3**: Higher memory usage, captures longer patterns
- **Order 4+**: Exponentially higher memory usage

**Recommendation**: Start with order 2, measure memory usage, then increase if needed.

### Choosing threshold
- **Lower values (0.001)**: More sensitive, more detections
- **Higher values (0.1)**: Less sensitive, fewer detections

**Recommendation**: Experiment with your data to find appropriate values.

### Memory Estimation
```rust
fn estimate_max_contexts(alphabet_size: usize, max_order: usize) -> usize {
    (1..=max_order)
        .map(|order| alphabet_size.pow(order as u32))
        .sum()
}
```

**Note**: Actual context count depends on sequence patterns and may be much lower.

## Performance Monitoring

The library includes comprehensive performance monitoring:

```rust
let mut detector = AnomalyDetector::new(3)?;
detector.train(&sequence)?;

// Get performance metrics after training
let metrics = detector.performance_metrics();
println!("Training time: {} ms", metrics.training_time_ms);
println!("Context count: {}", metrics.context_count);
println!("Memory estimate: {} bytes", metrics.estimated_memory_bytes);

// Detection with monitoring (updates metrics)
let anomalies = detector.detect_anomalies_with_monitoring(&test_sequence, 0.1)?;
let updated_metrics = detector.performance_metrics();
println!("Detection time: {} ms", updated_metrics.detection_time_ms);

// Apply memory optimizations
use anomaly_grid::OptimizationConfig;
let opt_config = OptimizationConfig::default();
detector.optimize(&opt_config)?;
```

## Suitable Use Cases

### Potentially Good Fit
- **System logs** with limited event types
- **Network protocols** with small command sets
- **User workflows** with simple action sequences
- **IoT sensors** with categorical states
- **Educational purposes** and algorithm learning

### Likely Poor Fit
- **NLPs** (large vocabulary)
- **High-resolution sensor data** (continuous values)
- **Real-time processing** (high-volume streams)
- **Large state spaces** (many unique states)

**Recommendation**: Test with your specific data to determine suitability.

## Error Handling

All functions return `AnomalyGridResult<T>` which is `Result<T, AnomalyGridError>`.

Common errors:
- `InvalidMaxOrder`: max_order = 0
- `SequenceTooShort`: sequence length < min_sequence_length
- `InvalidThreshold`: threshold not in [0,1]
- `EmptyContextTree`: detection attempted before training
- `MemoryLimitExceeded`: too many contexts created

## Dependencies

```toml
[dependencies]
rayon = "1.10.0"    # Parallel batch processing
smallvec = "1.13.0" # Memory-efficient small collections
```

Minimal dependencies for core functionality and memory optimization.

## Testing

```bash
# Run all tests
cargo test

# Run comprehensive validation
cargo test --test comprehensive_test_runner

# Run examples
cargo run --example quick_start
```

## Version History

### v0.2.1
- **Memory Optimization Sprint**: Comprehensive memory efficiency improvements
  - String interning for duplicate elimination
  - Trie-based context storage with prefix sharing
  - On-demand computation with lazy evaluation and caching
  - Small collections optimization using SmallVec
  - Memory pooling infrastructure
- **Enhanced Configuration**: Configurable smoothing, weights, and memory limits
- **Performance Monitoring**: Comprehensive metrics and optimization tools
- **Expanded Test Suite**: 72 tests with mathematical validation

### v0.2.0
- **Major refactoring**: Encapsulation of the main functions and a bigger set of tests
- **Documentation reduction**: Difficulty to keep documenting everything in text outside the code got chaotic, reduced for simplicity but provided enough examples and comments in the code
- **API consistency**: All constructors return `Result<T, AnomalyGridError>`

## Benchmarking Your Use Case

To evaluate if this library suits your needs:

1. **Start small**: Test with a subset of your data
2. **Monitor memory**: Use the performance metrics
3. **Profile thoroughly**: Test with realistic data volumes
4. **Compare alternatives**: Benchmark against other solutions

## When This Library May Not Be Suitable

- **Production systems** requiring guaranteed performance
- **Applications** with strict memory constraints
- **Real-time systems** with strict latency requirements
- **Systems** requiring theoretical optimality guarantees
- **Applications** needing adaptive or online learning

## License

MIT License - see [LICENSE](LICENSE) file.

---
