# Performance Guide

Comprehensive guide to performance characteristics and optimization strategies.

## Optimization Strategies

### 1. Choose Appropriate Max Order

**Recommendation**: Start with order 2-3, increase only if needed.

```rust
// Conservative approach - good for most use cases
let detector = AnomalyDetector::new(2)?;

// Higher accuracy but more memory
let detector = AnomalyDetector::new(4)?;
```

**Memory Impact**:
- Order 2: ~O(|alphabet|²) contexts
- Order 3: ~O(|alphabet|³) contexts  
- Order 4+: Exponential growth

### 2. Configure Memory Limits

```rust
let config = AnomalyGridConfig::default()
    .with_max_order(3)?
    .with_memory_limit(100 * 1024 * 1024); // 100MB limit

let detector = AnomalyDetector::with_config(config)?;
```

### 3. Optimize Smoothing Parameters

```rust
// Less smoothing - more sensitive to training data
let config = AnomalyGridConfig::default()
    .with_smoothing_alpha(0.1)?;

// More smoothing - more robust to sparse data
let config = AnomalyGridConfig::default()
    .with_smoothing_alpha(2.0)?;
```

### 4. Use Memory Optimization

```rust
let opt_config = OptimizationConfig {
    prune_low_frequency_contexts: true,
    frequency_threshold: 5,
    prune_low_entropy_contexts: true,
    entropy_threshold: 0.1,
    enable_memory_pooling: true,
};

detector.optimize(&opt_config)?;
```

### 5. Batch Processing for Multiple Sequences

```rust
// Process multiple sequences in parallel
let results = AnomalyDetector::batch_process_sequences(
    &sequences,
    &config,
    threshold
)?;
```

## Performance Monitoring

### Built-in Metrics

```rust
let metrics = detector.performance_metrics();
println!("Training time: {} ms", metrics.training_time_ms);
println!("Detection time: {} ms", metrics.detection_time_ms);
println!("Context count: {}", metrics.context_count);
println!("Memory usage: {} KB", metrics.estimated_memory_bytes / 1024);
```

### Custom Benchmarking

```rust
use std::time::Instant;

fn benchmark_detection(detector: &AnomalyDetector, sequences: &[Vec<String>]) {
    let start = Instant::now();
    
    for sequence in sequences {
        let _ = detector.detect_anomalies(sequence, 0.1);
    }
    
    let duration = start.elapsed();
    let throughput = sequences.len() as f64 / duration.as_secs_f64();
    
    println!("Throughput: {:.0} sequences/second", throughput);
}
```

## Memory Management

### Memory Usage Patterns

1. **Training Phase**: Memory grows as contexts are learned
2. **Detection Phase**: Memory usage remains stable
3. **Optimization Phase**: Memory may decrease after pruning

### Memory Estimation

```rust
fn estimate_memory_usage(alphabet_size: usize, max_order: usize, data_size: usize) -> usize {
    // Rough estimation
    let max_contexts = (1..=max_order)
        .map(|order| alphabet_size.pow(order as u32))
        .sum::<usize>();
    
    // Actual usage is typically much lower
    let estimated_contexts = (max_contexts as f64 * 0.1) as usize; // 10% utilization
    let bytes_per_context = 100; // rough estimate
    
    estimated_contexts * bytes_per_context
}
```

### Memory Optimization Techniques

#### 1. Context Pruning

Remove infrequently used contexts:

```rust
let opt_config = OptimizationConfig {
    prune_low_frequency_contexts: true,
    frequency_threshold: 3, // Remove contexts seen < 3 times
    ..Default::default()
};
```

#### 2. Entropy-Based Pruning

Remove low-information contexts:

```rust
let opt_config = OptimizationConfig {
    prune_low_entropy_contexts: true,
    entropy_threshold: 0.5, // Remove low-entropy contexts
    ..Default::default()
};
```

#### 3. String Interning

Automatically enabled - reduces duplicate string storage.

## Performance Tuning Guidelines

### For High Throughput

1. Use lower max_order (2-3)
2. Enable batch processing
3. Optimize memory regularly
4. Use appropriate thresholds

```rust
let config = AnomalyGridConfig::default()
    .with_max_order(2)?
    .with_smoothing_alpha(1.0)?;

// Process in batches
let results = AnomalyDetector::batch_process_sequences(&sequences, &config, 0.1)?;
```

### For High Accuracy

1. Use higher max_order (4-6)
2. Lower smoothing alpha
3. More training data
4. Fine-tune thresholds

```rust
let config = AnomalyGridConfig::default()
    .with_max_order(5)?
    .with_smoothing_alpha(0.1)?
    .with_memory_limit(500 * 1024 * 1024); // 500MB
```

### For Memory-Constrained Environments

1. Use low max_order (1-2)
2. Set strict memory limits
3. Aggressive optimization
4. Regular pruning

```rust
let config = AnomalyGridConfig::default()
    .with_max_order(2)?
    .with_memory_limit(50 * 1024 * 1024); // 50MB limit

let opt_config = OptimizationConfig {
    prune_low_frequency_contexts: true,
    frequency_threshold: 10,
    prune_low_entropy_contexts: true,
    entropy_threshold: 1.0,
    enable_memory_pooling: true,
};
```

## Profiling and Debugging

### Performance Profiling

```bash
# Profile with cargo
cargo build --release
cargo run --release --example your_example

# Use perf for detailed profiling
perf record --call-graph=dwarf cargo run --release --example your_example
perf report
```

### Memory Profiling

```bash
# Use valgrind for memory analysis
valgrind --tool=massif cargo run --release --example your_example

# Use heaptrack for heap profiling
heaptrack cargo run --release --example your_example
```

### Custom Profiling

```rust
use std::time::Instant;

fn profile_training(data: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut detector = AnomalyDetector::new(3)?;
    
    let start = Instant::now();
    detector.train(data)?;
    let training_time = start.elapsed();
    
    let metrics = detector.performance_metrics();
    
    println!("Training completed:");
    println!("  Time: {:?}", training_time);
    println!("  Contexts: {}", metrics.context_count);
    println!("  Memory: {} KB", metrics.estimated_memory_bytes / 1024);
    println!("  Throughput: {:.0} elements/sec", 
             data.len() as f64 / training_time.as_secs_f64());
    
    Ok(())
}
```

## Common Performance Issues

### Issue 1: High Memory Usage

**Symptoms**: Out of memory errors, slow performance
**Solutions**:
- Reduce max_order
- Set memory limits
- Use optimization
- Prune contexts regularly

### Issue 2: Slow Training

**Symptoms**: Long training times
**Solutions**:
- Reduce alphabet size if possible
- Lower max_order
- Use more efficient data structures
- Profile for bottlenecks

### Issue 3: Slow Detection

**Symptoms**: High detection latency
**Solutions**:
- Optimize context tree
- Use batch processing
- Cache frequently accessed contexts
- Profile detection path

### Issue 4: Memory Leaks

**Symptoms**: Growing memory usage over time
**Solutions**:
- Regular optimization
- Check for reference cycles
- Monitor metrics
- Use memory profiling tools
