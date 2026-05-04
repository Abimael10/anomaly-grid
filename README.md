# Anomaly Grid

     █████╗ ███╗   ██╗ ██████╗ ███╗   ███╗ █████╗ ██╗  ██╗   ██╗
    ██╔══██╗████╗  ██║██╔═══██╗████╗ ████║██╔══██╗██║  ╚██╗ ██╔╝
    ███████║██╔██╗ ██║██║   ██║██╔████╔██║███████║██║   ╚████╔╝ 
    ██╔══██║██║╚██╗██║██║   ██║██║╚██╔╝██║██╔══██║██║    ╚██╔╝  
    ██║  ██║██║ ╚████║╚██████╔╝██║ ╚═╝ ██║██║  ██║███████╗██║   
    ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   
    [ANOMALY-GRID v0.6.0] - SEQUENCE ANOMALY DETECTION ENGINE

[![Crates.io](https://img.shields.io/crates/v/anomaly-grid.svg)](https://crates.io/crates/anomaly-grid)
[![Downloads](https://img.shields.io/crates/d/anomaly-grid.svg)](https://crates.io/crates/anomaly-grid)
[![Documentation](https://docs.rs/anomaly-grid/badge.svg)](https://docs.rs/anomaly-grid)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Rust library implementing variable-order Markov chains with **Witten-Bell interpolation** for sequence anomaly detection over finite alphabets. Anomaly strength combines per-symbol surprise and information content, both in bits, squashed by `tanh` into `[0, 1)`.

## Quick Start

```toml
[dependencies]
anomaly-grid = "0.6"
```

### Train once, score many in parallel

```rust
use anomaly_grid::{AnomalyDetector, batch_score};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut detector = AnomalyDetector::new(3)?;

    // Train on a corpus of *known-normal* sequences (a fleet of benign
    // sessions, golden-path traces, etc.).
    let mut normal_sequence = Vec::new();
    for _ in 0..30 {
        normal_sequence.extend(["A", "B", "C", "A", "B", "C", "A", "B", "C"].iter().copied());
    }
    normal_sequence.extend(["A", "B", "A", "C", "A", "B", "C"].iter().copied());
    let normal_sequence: Vec<String> = normal_sequence.into_iter().map(str::to_string).collect();
    detector.train(&normal_sequence)?;

    // Score one sequence:
    let test_sequence: Vec<String> = ["A", "B", "C", "X", "Y", "C", "A", "B", "C"]
        .iter().map(|s| s.to_string()).collect();
    for a in detector.detect_anomalies(&test_sequence, 0.2)? {
        println!("window {:?}, strength {:.3}", a.sequence, a.anomaly_strength);
    }

    // Or batch many in parallel (rayon, lock-free over `&AnomalyDetector`):
    let unknown: Vec<Vec<String>> = vec![test_sequence.clone(), test_sequence];
    let results = batch_score(&detector, &unknown, 0.2)?;
    for (i, scores) in results.iter().enumerate() {
        for s in scores {
            println!("seq {i}: {:?} strength {:.3}", s.sequence, s.anomaly_strength);
        }
    }
    Ok(())
}
```

Expected output for the single-sequence call: two flagged windows
`["B","C","X","Y"]` and `["C","X","Y","C"]`. The rest of the test
sequence matches the trained ABC pattern and falls below `0.2`.

## What This Library Does

- **Variable-order Markov model** with smooth Witten-Bell backoff
  `λ(c) = N(c) / (N(c) + T(c))`. Order-0 base case is Laplace
  (Add-α) over the global alphabet — unseen contexts never collapse to
  zero probability.
- **Information-theoretic scoring** in bits throughout: per-window
  surprise `(−1/(n−1)) Σ log₂ P` and per-symbol information content
  `−log₂ P(xᵢ | context)`.
- **Memory-conscious storage**: `StateId(u32)` interner backed by
  `Arc<str>`, arena trie indexed by `NodeId(u32)` with
  `SmallVec<[(StateId, NodeId); 4]>` children (≤ 4 inline).
  `TransitionCounts` enum stays inline as `SmallVec<[(StateId, usize); 4]>`
  for the typical case and falls back to `HashMap` only when a context
  exceeds 4 distinct continuations.
- **Parallel batch scoring** (`batch_score`) over a shared
  `&AnomalyDetector` via rayon. Lock-free during scoring; deterministic
  across thread-pool sizes.
- **Strict lints**: builds under
  `#![deny(clippy::pedantic, clippy::nursery, clippy::unwrap_used,
  clippy::expect_used, missing_docs)]`.
- **Property-tested invariants**: probability sums = 1, entropy
  bounded, parallel determinism, Unicode round-trip, long-sequence
  finiteness.

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
