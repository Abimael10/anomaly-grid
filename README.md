# Anomaly Grid

     █████╗ ███╗   ██╗ ██████╗ ███╗   ███╗ █████╗ ██╗  ██╗   ██╗
    ██╔══██╗████╗  ██║██╔═══██╗████╗ ████║██╔══██╗██║  ╚██╗ ██╔╝
    ███████║██╔██╗ ██║██║   ██║██╔████╔██║███████║██║   ╚████╔╝ 
    ██╔══██║██║╚██╗██║██║   ██║██║╚██╔╝██║██╔══██║██║    ╚██╔╝  
    ██║  ██║██║ ╚████║╚██████╔╝██║ ╚═╝ ██║██║  ██║███████╗██║   
    ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   
    [ANOMALY-GRID v0.4.0] - SEQUENCE ANOMALY DETECTION ENGINE

[![Crates.io](https://img.shields.io/crates/v/anomaly-grid.svg)](https://crates.io/crates/anomaly-grid)
[![Downloads](https://img.shields.io/crates/d/anomaly-grid.svg)](https://crates.io/crates/anomaly-grid)
[![PyPI version](https://img.shields.io/pypi/v/anomaly-grid-py.svg)](https://pypi.org/project/anomaly-grid-py/)
[![Documentation](https://docs.rs/anomaly-grid/badge.svg)](https://docs.rs/anomaly-grid)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)

A Rust library implementing variable-order Markov chains for sequence anomaly detection in finite alphabets.

To use a Python wrapper of this library implementations refer, to my other repository at: https://github.com/Abimael10/anomaly-grid-py

## Quick Start

```toml
[dependencies]
anomaly-grid = "0.4.0"
```

```rust
use anomaly_grid::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create detector
    let mut detector = AnomalyDetector::new(3)?;
    
    // Train on normal patterns
    let normal_sequence = vec!["A", "B", "C", "A", "B", "C"]
        .iter().map(|s| s.to_string()).collect();
    detector.train(&normal_sequence)?;
    
    // Detect anomalies
    let test_sequence = vec!["A", "X", "Y"]
        .iter().map(|s| s.to_string()).collect();
    let anomalies = detector.detect_anomalies(&test_sequence, 0.1)?;
    
    for anomaly in anomalies {
        println!("Anomaly: {:?}, Strength: {:.3}", 
                 anomaly.sequence, anomaly.anomaly_strength);
    }
    
    Ok(())
}
```

## What This Library Does

- **Variable-Order Markov Models**: Builds contexts of length 1 to max_order from training sequences with hierarchical context selection
- **Adaptive Context Selection**: Uses longest available context with sufficient data, falls back to shorter contexts automatically
- **Information-Theoretic Scoring**: Shannon entropy and KL divergence calculations with lazy computation and caching
- **Memory-Optimized Storage**: String interning, trie-based context storage with prefix sharing, and SmallVec for efficient small collections
- **Parallel Batch Processing**: Processes multiple sequences concurrently using Rayon for improved throughput
- **Comprehensive Testing**: Extensive unit, integration, domain, and performance validation with mathematical correctness verification

## Configuration

```rust
let config = AnomalyGridConfig::default()
    .with_max_order(4)?                    // Higher order = more memory, better accuracy
    .with_smoothing_alpha(0.5)?            // Lower = more sensitive to training data
    .with_weights(0.8, 0.2)?               // Likelihood vs information weight
    .with_memory_limit(100 * 1024 * 1024); // 100MB memory limit

let detector = AnomalyDetector::with_config(config)?;
```

## Use Cases

### Excellent Fit
- **Software Development Workflows**: Git command sequences, CI/CD pipeline analysis, code review patterns
- **Database Query Optimization**: SQL operation sequences, transaction pattern analysis, N+1 query detection
- **Network Protocol Analysis**: TCP/HTTP/TLS state transitions, protocol compliance verification, traffic flow analysis
- **System Administration**: CLI command sequences, automation pattern detection, user proficiency analysis
- **Creative Pattern Analysis**: Musical composition analysis, artistic workflow patterns, style classification
- **Security Monitoring**: Login sequences, access patterns, behavioral anomaly detection
- **IoT and Sensor Networks**: Device state transitions, sensor reading patterns, equipment health monitoring

### Good Fit
- **Business Process Mining**: Workflow step sequences, process compliance, bottleneck identification
- **User Experience Analysis**: Click sequences, navigation patterns, conversion funnel analysis
- **Manufacturing Quality Control**: Production step sequences, assembly line monitoring, defect pattern detection
- **Financial Transaction Analysis**: Payment sequences, fraud pattern detection, risk assessment
- **Healthcare Workflow Analysis**: Treatment sequences, care pathway optimization, protocol adherence

### Requires Preprocessing
- **Natural Language Processing**: Tokenize to categorical sequences (POS tags, named entities, semantic categories)
- **Time Series Data**: Discretize continuous values into categorical states or trend patterns
- **High-Resolution Sensor Data**: Aggregate into categorical states or pattern classifications
- **Large Vocabularies**: Apply dimensionality reduction or clustering to create manageable alphabets

### Poor Fit
- **Raw Continuous Data**: Unprocessed sensor readings, audio waveforms, high-frequency financial data
- **Extremely Large Alphabets**: >1000 unique states without preprocessing
- **Real-Time Streaming**: Microsecond-latency requirements (though batch processing is efficient)
- **Unstructured Data**: Images, videos, raw binary data without categorical interpretation

## Testing

```bash
# Run all tests
cargo test

# Run specific test suites
cargo test unit_           # Unit tests
cargo test integration_    # Integration tests
cargo test domain_         # Domain tests
cargo test performance_    # Performance tests (run with --release for perf thresholds)

# Run examples
cargo run --example quick_start
cargo run --example network_security_monitoring
cargo run --example financial_fraud_detection
```

## Documentation

- **[Complete Documentation](docs/)** - Comprehensive guides and API reference
- **[API Reference](https://docs.rs/anomaly-grid)** - Online API documentation
- **[Examples](examples/)** - Production-ready examples with validation
- **[Changelog](CHANGELOG.md)** - Version history and changes

## Dependencies

```toml
[dependencies]
rayon = "1.10.0"    # Parallel batch processing
smallvec = "1.13.0" # Memory-efficient small collections
```

Minimal dependencies for core functionality and memory optimization.

## License

MIT License - see [LICENSE](LICENSE) file.

---

**Performance Note**: The library efficiently handles alphabets up to ~100 unique states with excellent memory usage (typically <100MB). For larger alphabets, consider preprocessing techniques like clustering, dimensionality reduction, or hierarchical categorization.
