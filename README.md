# Anomaly Grid

     █████╗ ███╗   ██╗ ██████╗ ███╗   ███╗ █████╗ ██╗  ██╗   ██╗
    ██╔══██╗████╗  ██║██╔═══██╗████╗ ████║██╔══██╗██║  ╚██╗ ██╔╝
    ███████║██╔██╗ ██║██║   ██║██╔████╔██║███████║██║   ╚████╔╝ 
    ██╔══██║██║╚██╗██║██║   ██║██║╚██╔╝██║██╔══██║██║    ╚██╔╝  
    ██║  ██║██║ ╚████║╚██████╔╝██║ ╚═╝ ██║██║  ██║███████╗██║   
    ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   
    [ANOMALY-GRID v0.4.3] - SEQUENCE ANOMALY DETECTION ENGINE

[![Crates.io](https://img.shields.io/crates/v/anomaly-grid.svg)](https://crates.io/crates/anomaly-grid)
[![Downloads](https://img.shields.io/crates/d/anomaly-grid.svg)](https://crates.io/crates/anomaly-grid)
[![Documentation](https://docs.rs/anomaly-grid/badge.svg)](https://docs.rs/anomaly-grid)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Rust library implementing variable-order Markov chains for sequence anomaly detection in finite alphabets.

## Quick Start

```toml
[dependencies]
anomaly-grid = "0.4.3"
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

- Variable-order Markov modeling for finite alphabets (order 1..max_order with fallback).
- On-the-fly scoring: likelihood + information score, combined into an anomaly strength.
- Memory-conscious storage: string interning, trie-based contexts, SmallVec for small counts.
- Batch processing: detect anomalies across many sequences in parallel (Rayon).
- Tunable config: smoothing, weights, memory limit, and optimization helpers for pruning.

## Configuration

```rust
let config = AnomalyGridConfig::default()
    .with_max_order(4)?                    // Higher order = more memory, better accuracy
    .with_smoothing_alpha(0.5)?            // Lower = more sensitive to training data
    .with_weights(0.8, 0.2)?               // Likelihood vs information weight
    .with_memory_limit(Some(100 * 1024 * 1024))?; // 100MB memory limit

let detector = AnomalyDetector::with_config(config)?;
```

## Use Cases (with context)

Markov chains **are not state of the art** for anomaly detection. Modern systems favor deep sequence, probabilistic, and graph-based models. This library remains useful when you need:
- Discrete, low-dimensional states with short contexts.
- Predictable workflows where interpretability matters.
- Ultra-low-latency or resource-constrained inference.

### Practical fits
- **Network/Protocol flows**: Finite state machines, handshake/order violations.
- **Small structured workflows**: Ops runbooks, CLI/session macros, simple ETL steps.
- **Device/state telemetry**: Low-cardinality IoT states, embedded controllers.

### Not a fit without heavy preprocessing
- High-dimensional logs/sensors or complex user behavior with long-range dependencies.
- Large alphabets or non-stationary patterns.
- Continuous/unstructured data (images, audio, raw text) without discretization.

### Current state-of-the-art alternatives
- **Deep sequence models**: LSTM/GRU, Transformers (TFT, Anomaly Transformer, TS foundation models), autoencoders/VAEs.
- **Probabilistic deep models**: Normalizing flows, diffusion, energy-based models.
- **Graph/representation learning**: GNNs, dynamic graph embeddings, contrastive methods.
- **Classical statistical baselines**: HMMs (strong Markovian baseline), GMMs/Bayesian changepoint, ARIMA/VAR/Kalman for continuous signals.
- **TS foundation models (2023–2025)**: TimeGPT, Chronos, MOIRAI, DeepTime.

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
```

## Documentation

- **[Complete Documentation](docs/)** - Comprehensive guides and API reference
- **[API Reference](https://docs.rs/anomaly-grid)** - Online API documentation
- **[Examples](examples/)**
- **[Changelog](CHANGELOG.md)** - Version history and changes

## License

MIT License - see [LICENSE](LICENSE) file.

---
