# Anomaly Grid

     █████╗ ███╗   ██╗ ██████╗ ███╗   ███╗ █████╗ ██╗  ██╗   ██╗
    ██╔══██╗████╗  ██║██╔═══██╗████╗ ████║██╔══██╗██║  ╚██╗ ██╔╝
    ███████║██╔██╗ ██║██║   ██║██╔████╔██║███████║██║   ╚████╔╝ 
    ██╔══██║██║╚██╗██║██║   ██║██║╚██╔╝██║██╔══██║██║    ╚██╔╝  
    ██║  ██║██║ ╚████║╚██████╔╝██║ ╚═╝ ██║██║  ██║███████╗██║   
    ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   
    [ANOMALY-GRID v0.2.0] - SEQUENCE ANOMALY DETECTION ENGINE

[![Crates.io](https://img.shields.io/crates/v/anomaly-grid.svg)](https://crates.io/crates/anomaly-grid)
[![Documentation](https://docs.rs/anomaly-grid/badge.svg)](https://docs.rs/anomaly-grid)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Rust library implementing variable-order Markov chains for sequence anomaly detection in finite alphabets.

## What This Library Does

- **Variable-Order Markov Models**: Builds contexts of length 1 to max_order from training sequences
- **Hierarchical Context Selection**: Uses longest available context, falls back to shorter contexts
- **Laplace Smoothing**: Adds α=1.0 to all transition counts for probability estimation
- **Shannon Entropy Calculation**: Computes H(X) = -∑ P(x) log₂ P(x) for each context
- **Sliding Window Detection**: Analyzes test sequences using overlapping windows
- **Parallel Batch Processing**: Processes multiple sequences using Rayon

## Known Limitations So Far

### Memory Complexity: O(|alphabet|^max_order)
**This grows exponentially:**

| Alphabet Size | Max Order | Theoretical Contexts |
|---------------|-----------|---------------------|
| 10            | 3         | ~1,100              |
| 20            | 3         | ~8,400              |
| 20            | 4         | ~168,000            |
| 50            | 3         | ~132,500            |
| 100           | 3         | ~1,010,000          |

**Memory usage depends on actual sequence patterns and cannot be predicted without profiling your specific data.**

### Algorithm Limitations
- **Standard Markov Model**: Basic implementation, no theoretical optimality (Will see if it is worth it)
- **Greedy Context Selection**: Uses longest available context for now (not optimal)
- **Fixed Smoothing**: α=1.0 Laplace smoothing (not adaptive for now)
- **No Compression**: Does not implement Context Tree Weighting or similar algorithms for now.

### Practical Constraints
- **Memory Bound**: Will fail on large state spaces
- **Sequential Processing**: Not designed for real-time high-volume streams (No plans to change this by my own, it is too much)

## Installation

```toml
[dependencies]
anomaly-grid = "0.2.0"
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
// Constructor - returns Result due to validation
pub fn new(max_order: usize) -> AnomalyGridResult<Self>

// Training - builds context tree from sequence
pub fn train(&mut self, sequence: &[String]) -> AnomalyGridResult<()>

// Detection - sliding window analysis
pub fn detect_anomalies(&self, sequence: &[String], threshold: f64) -> AnomalyGridResult<Vec<AnomalyScore>>

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
where α = 1.0 (Laplace smoothing)

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
anomaly_strength = tanh((log_likelihood_component × 0.7 + info_score × 0.3) / 10.0)
```

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

The library includes performance monitoring:

```rust
let mut detector = AnomalyDetector::new(3)?;
detector.train(&sequence)?;

// Get performance metrics
let metrics = detector.performance_metrics();
println!("Training time: {} ms", metrics.training_time_ms);
println!("Context count: {}", metrics.context_count);
println!("Memory estimate: {} bytes", metrics.estimated_memory_bytes);

// Detection with monitoring
let anomalies = detector.detect_anomalies_with_monitoring(&test_sequence, 0.1)?;
let updated_metrics = detector.performance_metrics();
println!("Detection time: {} ms", updated_metrics.detection_time_ms);
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
rayon = "1.10.0"  # Parallel processing only
```

Minimal dependencies - only Rayon for optional parallel batch processing.

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

### v0.2.0
- **Major refactoring**: Encapsulation of the main functions and a bigger set of tests.
- **Documentation reduction**: Difficulty to keep documenting everything in text outside the code got chaotic, reduced for simplicity but provided enough examples and comments in the code.
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
