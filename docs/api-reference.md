# API Reference

API overview for Anomaly Grid. This is a concise map of the public surface; for behavior details see the crate docs.

## Core Types

### AnomalyDetector

The main detector class for sequence anomaly detection.

#### Constructors

```rust
// Create detector with specified maximum order
pub fn new(max_order: usize) -> AnomalyGridResult<Self>

// Create detector with custom configuration
pub fn with_config(config: AnomalyGridConfig) -> AnomalyGridResult<Self>
```

#### Training

```rust
// Train the detector on a sequence
pub fn train(&mut self, sequence: &[String]) -> AnomalyGridResult<()>

// Train on multiple sequences while preserving boundaries
pub fn train_sequences(&mut self, sequences: &[Vec<String>]) -> AnomalyGridResult<()>
```

#### Detection

```rust
// Detect anomalies in a sequence
pub fn detect_anomalies(&self, sequence: &[String], threshold: f64) -> AnomalyGridResult<Vec<AnomalyScore>>

// Detect anomalies with performance monitoring
pub fn detect_anomalies_with_monitoring(&mut self, sequence: &[String], threshold: f64) -> AnomalyGridResult<Vec<AnomalyScore>>
```

#### Performance and Optimization

```rust
// Get performance metrics
pub fn performance_metrics(&self) -> &PerformanceMetrics

// Optimize memory usage
pub fn optimize(&mut self, optimization_config: &OptimizationConfig) -> AnomalyGridResult<()>

// Get context statistics
pub fn context_statistics(&self) -> ContextStatistics
```

#### Batch Processing

```rust
// Process multiple sequences in parallel
pub fn batch_process_sequences(
    sequences: &[Vec<String>],
    config: &AnomalyGridConfig,
    threshold: f64,
) -> AnomalyGridResult<Vec<Vec<AnomalyScore>>>
```

### AnomalyScore

Result structure for detected anomalies. Values are bounded (likelihood ∈ [0,1], anomaly_strength ∈ [0,1]).

```rust
pub struct AnomalyScore {
    pub sequence: Vec<String>,      // The analyzed window
    pub likelihood: f64,            // P(sequence|model) ∈ [0,1]
    pub log_likelihood: f64,        // ln(likelihood)
    pub information_score: f64,     // Average -log₂(P(x))
    pub anomaly_strength: f64,      // Normalized score ∈ [0,1]
}
```

### AnomalyGridConfig

Configuration structure for the detector.

```rust
pub struct AnomalyGridConfig {
    pub max_order: usize,
    pub smoothing_alpha: f64,
    pub likelihood_weight: f64,
    pub information_weight: f64,
    pub normalization_factor: f64,
    pub memory_limit: Option<usize>,
    pub min_sequence_length: usize,
}
```

#### Configuration Methods

```rust
impl AnomalyGridConfig {
    pub fn default() -> Self
    pub fn with_max_order(self, max_order: usize) -> AnomalyGridResult<Self>
    pub fn with_smoothing_alpha(self, alpha: f64) -> AnomalyGridResult<Self>
    pub fn with_weights(self, likelihood_weight: f64, information_weight: f64) -> AnomalyGridResult<Self>
    pub fn with_memory_limit(self, limit: Option<usize>) -> AnomalyGridResult<Self>
}
```

### PerformanceMetrics

Performance monitoring structure.

```rust
pub struct PerformanceMetrics {
    pub training_time_ms: u64,
    pub detection_time_ms: u64,
    pub context_count: usize,
    pub estimated_memory_bytes: usize,
}
```
Note: `detection_time_ms` is populated when using `detect_anomalies_with_monitoring`.

### OptimizationConfig

Configuration for memory optimization.

```rust
pub struct OptimizationConfig {
    pub enable_pruning: bool,
    pub min_context_count: usize,
    pub min_entropy: f64,
    pub max_contexts: Option<usize>,
    pub enable_monitoring: bool,
}
```

## Usage Examples

### Basic Usage

```rust
use anomaly_grid::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create detector (order-3)
    let mut detector = AnomalyDetector::new(3)?;

    // Train on a richer pattern set
    let mut normal_sequence = Vec::new();
    for _ in 0..30 {
        normal_sequence.extend(["A", "B", "C", "A", "B", "C", "A", "B", "C"].iter().cloned());
    }
    normal_sequence.extend(["A", "B", "A", "C", "A", "B", "C"].iter().cloned());
    normal_sequence.extend(["A", "C", "B", "A", "B", "C"].iter().cloned());
    let normal_sequence = normal_sequence
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    detector.train(&normal_sequence)?;

    // Detect deviations
    let test_data = ["A", "B", "C", "X", "Y", "C", "A", "B", "C"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let anomalies = detector.detect_anomalies(&test_data, 0.2)?;

    for anomaly in anomalies {
        println!(
            "Anomaly window {:?}, Strength: {:.3}",
            anomaly.sequence, anomaly.anomaly_strength
        );
    }

    Ok(())
}
```

### Advanced Configuration

```rust
use anomaly_grid::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create custom configuration
    let config = AnomalyGridConfig::default()
        .with_max_order(4)?
        .with_smoothing_alpha(0.5)?
        .with_weights(0.8, 0.2)?
        .with_memory_limit(Some(100 * 1024 * 1024))?; // 100MB limit
    
    let mut detector = AnomalyDetector::with_config(config)?;
    
    // Training and detection...
    
    Ok(())
}
```

### Performance Monitoring

```rust
use anomaly_grid::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut detector = AnomalyDetector::new(3)?;
    
    // Train with monitoring
    let training_data = generate_training_data();
    detector.train(&training_data)?;
    
    let metrics = detector.performance_metrics();
    println!("Training time: {} ms", metrics.training_time_ms);
    println!("Contexts learned: {}", metrics.context_count);
    println!("Memory usage: {} KB", metrics.estimated_memory_bytes / 1024);
    
    // Optimize if needed
    let opt_config = OptimizationConfig::default();
    detector.optimize(&opt_config)?;
    
    Ok(())
}
```

### Batch Processing

```rust
use anomaly_grid::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sequences = vec![
        vec!["A", "B", "C"].iter().map(|s| s.to_string()).collect(),
        vec!["X", "Y", "Z"].iter().map(|s| s.to_string()).collect(),
        vec!["A", "X", "C"].iter().map(|s| s.to_string()).collect(),
    ];
    
    let config = AnomalyGridConfig::default().with_max_order(2)?;
    let results = AnomalyDetector::batch_process_sequences(&sequences, &config, 0.1)?;
    
    for (i, anomalies) in results.iter().enumerate() {
        println!("Sequence {}: {} anomalies detected", i, anomalies.len());
    }
    
    Ok(())
}
```

## Supporting Types

### ContextTree / ContextNode
Context storage and probability estimation.
```rust
pub struct ContextTree { /* context storage and probability estimation */ }
pub struct ContextNode { /* transition counts and cached entropy/KL */ }
```
Key operations:
- `ContextTree::new(max_order)` to create a tree.
- `build_from_sequence(&sequence, &config)` to ingest a sequence.
- Introspection: `context_count()`, `estimate_memory_usage()`, `interner()`, entropy/likelihood accessors.

### ContextStatistics
Aggregated context information.
```rust
pub struct ContextStatistics {
    pub total_contexts: usize,
    pub total_transitions: usize,
    pub total_entropy: f64,
    pub avg_entropy: f64,
    pub avg_frequency: f64,
    pub min_frequency: usize,
    pub max_frequency: usize,
    pub min_entropy: f64,
    pub max_entropy: f64,
    pub contexts_by_order: HashMap<usize, usize>,
    pub transitions_by_context: HashMap<usize, usize>,
}
```

### TrainingDataAnalysis (validation utilities)
```rust
pub struct TrainingDataAnalysis {
    pub total_elements: usize,
    pub unique_elements: usize,
    pub diversity_ratio: f64,
    pub entropy: f64,
    pub normalized_entropy: f64,
    pub most_common_element: Option<(String, usize)>,
}
```
Helpers:
- `validate_training_data_quality(&[String]) -> Vec<String>`: returns warnings.
- `analyze_training_data_characteristics(&[String]) -> TrainingDataAnalysis`.

### OptimizationConfig helpers
```rust
impl OptimizationConfig {
    pub fn for_low_memory() -> Self
    pub fn for_high_accuracy() -> Self
    pub fn for_balanced_performance() -> Self
}
```

### AnomalyGridError
Structured errors for configuration/training/detection issues.

### Lower-Level Structures (supporting)
- `ContextTrie`/`TrieNode`/`NodeId`: prefix-sharing storage for contexts (used by `ContextTree`).
- `StringInterner`/`StateId`: string interning for states.
- `MemoryPool`/`PoolStats`: pooling for context/trie allocations.
