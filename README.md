# Anomaly Grid

     █████╗ ███╗   ██╗ ██████╗ ███╗   ███╗ █████╗ ██╗  ██╗   ██╗
    ██╔══██╗████╗  ██║██╔═══██╗████╗ ████║██╔══██╗██║  ╚██╗ ██╔╝
    ███████║██╔██╗ ██║██║   ██║██╔████╔██║███████║██║   ╚████╔╝ 
    ██╔══██║██║╚██╗██║██║   ██║██║╚██╔╝██║██╔══██║██║    ╚██╔╝  
    ██║  ██║██║ ╚████║╚██████╔╝██║ ╚═╝ ██║██║  ██║███████╗██║   
    ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   
    [ANOMALY-GRID v0.4.1] - SEQUENCE ANOMALY DETECTION ENGINE

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
anomaly-grid = "0.4.1"
```

```rust
use anomaly_grid::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create detector (order-3)
    let mut detector = AnomalyDetector::new(3)?;

    // Train on a richer pattern set: repeating ABC blocks plus a few benign variants
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
    let test_sequence = ["A", "B", "C", "X", "Y", "C", "A", "B", "C"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let anomalies = detector.detect_anomalies(&test_sequence, 0.2)?;

    for anomaly in anomalies {
        println!(
            "Anomaly window {:?}, Strength: {:.3}",
            anomaly.sequence, anomaly.anomaly_strength
        );
    }

    Ok(())
}
```

Expected output with the above data:
- Two anomaly windows flagged: `["B","C","X","Y"]` (strength ~0.27) and `["C","X","Y","C"]` (strength ~0.39).
- No other windows reported; the rest of the test sequence matches the trained ABC pattern.

## What This Library Does

- **Variable-Order Markov Models**: Builds contexts of length 1 to max_order from training sequences with hierarchical context selection
- **Adaptive Context Selection**: Uses longest available context with sufficient data, falls back to shorter contexts automatically
- **Information-Theoretic Scoring**: Shannon entropy and KL divergence calculations with lazy computation and caching
- **Memory-Optimized Storage**: String interning, trie-based context storage with prefix sharing, and SmallVec for efficient small collections
- **Parallel Batch Processing**: Processes multiple sequences concurrently using Rayon
- **Comprehensive Testing**: Extensive unit, integration, domain, and performance validation

## Configuration

```rust
let config = AnomalyGridConfig::default()
    .with_max_order(4)?                    // Higher order = more memory, better accuracy
    .with_smoothing_alpha(0.5)?            // Lower = more sensitive to training data
    .with_weights(0.8, 0.2)?               // Likelihood vs information weight
    .with_memory_limit(Some(100 * 1024 * 1024))?; // 100MB memory limit

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
cargo run --example communication_protocol_analysis
cargo run --example network_protocol_analysis
cargo run --example protein_folding_sequences
cargo run --example docs_validation
```

## Documentation

- **[Complete Documentation](docs/)** - Comprehensive guides and API reference
- **[API Reference](https://docs.rs/anomaly-grid)** - Online API documentation
- **[Examples](examples/)**
- **[Changelog](CHANGELOG.md)** - Version history and changes

## License

MIT License - see [LICENSE](LICENSE) file.

---
