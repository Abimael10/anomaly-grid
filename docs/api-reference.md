# API Reference

Complete API documentation for Anomaly Grid library.

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

Result structure for detected anomalies.

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
    pub fn with_max_order(mut self, max_order: usize) -> AnomalyGridResult<Self>
    pub fn with_smoothing_alpha(mut self, alpha: f64) -> AnomalyGridResult<Self>
    pub fn with_weights(mut self, likelihood_weight: f64, information_weight: f64) -> AnomalyGridResult<Self>
    pub fn with_memory_limit(mut self, limit: usize) -> Self
    pub fn with_min_sequence_length(mut self, length: usize) -> Self
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
    pub sequences_processed: usize,
}
```

### OptimizationConfig

Configuration for memory optimization.

```rust
pub struct OptimizationConfig {
    pub prune_low_frequency_contexts: bool,
    pub frequency_threshold: usize,
    pub prune_low_entropy_contexts: bool,
    pub entropy_threshold: f64,
    pub enable_memory_pooling: bool,
}
```

## Error Types

### AnomalyGridError

Main error type for the library.

```rust
pub enum AnomalyGridError {
    InvalidMaxOrder(String),
    SequenceTooShort(String),
    InvalidThreshold(String),
    EmptyContextTree(String),
    MemoryLimitExceeded(String),
    InvalidConfiguration(String),
    TrainingError(String),
    DetectionError(String),
    OptimizationError(String),
}
```

## Usage Examples

### Basic Usage

```rust
use anomaly_grid::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create detector
    let mut detector = AnomalyDetector::new(3)?;
    
    // Train on normal patterns
    let training_data = vec!["A", "B", "C", "A", "B", "C"]
        .iter().map(|s| s.to_string()).collect();
    detector.train(&training_data)?;
    
    // Detect anomalies
    let test_data = vec!["A", "X", "Y"]
        .iter().map(|s| s.to_string()).collect();
    let anomalies = detector.detect_anomalies(&test_data, 0.1)?;
    
    for anomaly in anomalies {
        println!("Anomaly: {:?}, Strength: {:.3}", 
                 anomaly.sequence, anomaly.anomaly_strength);
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
        .with_memory_limit(100 * 1024 * 1024) // 100MB limit
        .with_min_sequence_length(3);
    
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