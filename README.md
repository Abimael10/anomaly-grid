# Anomaly Grid

     █████╗ ███╗   ██╗ ██████╗ ███╗   ███╗ █████╗ ██╗  ██╗   ██╗
    ██╔══██╗████╗  ██║██╔═══██╗████╗ ████║██╔══██╗██║  ╚██╗ ██╔╝
    ███████║██╔██╗ ██║██║   ██║██╔████╔██║███████║██║   ╚████╔╝ 
    ██╔══██║██║╚██╗██║██║   ██║██║╚██╔╝██║██╔══██║██║    ╚██╔╝  
    ██║  ██║██║ ╚████║╚██████╔╝██║ ╚═╝ ██║██║  ██║███████╗██║   
    ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   
    [ANOMALY-GRID v0.3.0] - SEQUENCE ANOMALY DETECTION ENGINE

[![Crates.io](https://img.shields.io/crates/v/anomaly-grid.svg)](https://crates.io/crates/anomaly-grid)
[![Documentation](https://docs.rs/anomaly-grid/badge.svg)](https://docs.rs/anomaly-grid)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-162%20passing-brightgreen.svg)](#testing)

A Rust library implementing variable-order Markov chains for sequence anomaly detection in finite alphabets.

## Quick Start

```toml
[dependencies]
anomaly-grid = "0.3.0"
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
- **Comprehensive Testing**: 162 tests covering unit, integration, domain, and performance validation with mathematical correctness verification

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

### ✅ Good Fit
- System logs with limited event types
- Network protocols with small command sets  
- User workflows with simple action sequences
- IoT sensors with categorical states

### ❌ Poor Fit
- Natural language processing (large vocabulary)
- High-resolution sensor data (continuous values)
- Real-time processing (high-volume streams)
- Large state spaces (many unique states)

## Testing

```bash
# Run all tests (162 tests)
cargo test

# Run specific test suites
cargo test unit_           # Unit tests (39 tests)
cargo test integration_    # Integration tests (24 tests)  
cargo test domain_         # Domain tests (5 tests)
cargo test performance_    # Performance tests (36 tests)

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

**Note**: This library is designed for categorical sequence analysis. For continuous data, consider preprocessing into discrete categories or using specialized time-series anomaly detection libraries.