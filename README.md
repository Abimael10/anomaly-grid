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

### Catch a privilege-escalation attack in user sessions

Train on benign user sessions, then scan unknown sessions in parallel
and surface only the windows that exceed your tolerance.

```rust
use anomaly_grid::{AnomalyDetector, batch_score};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let s = |w: &[&str]| -> Vec<String> { w.iter().map(|x| x.to_string()).collect() };

    // 30 benign sessions, three legitimate workflow shapes.
    let mut detector = AnomalyDetector::new(3)?;
    let mut benign = Vec::new();
    for _ in 0..30 {
        benign.extend(s(&["LOGIN", "AUTH", "READ", "WRITE", "READ", "LOGOUT"]));
        benign.extend(s(&["LOGIN", "AUTH", "READ", "READ", "WRITE", "LOGOUT"]));
        benign.extend(s(&["LOGIN", "AUTH", "WRITE", "READ", "READ", "LOGOUT"]));
    }
    detector.train(&benign)?;

    // Score four unknown sessions in parallel (lock-free over &detector).
    let candidates = vec![
        s(&["LOGIN", "AUTH", "READ", "WRITE", "READ", "LOGOUT"]),
        s(&["LOGIN", "AUTH", "READ", "READ", "WRITE", "LOGOUT"]),
        s(&["LOGIN", "AUTH", "WRITE", "READ", "READ", "LOGOUT"]),
        s(&["LOGIN", "AUTH", "PRIV_ESCALATE", "EXFIL", "LOGOUT"]), // attack
    ];
    let results = batch_score(&detector, &candidates, 0.3)?;

    for (i, anomalies) in results.iter().enumerate() {
        if anomalies.is_empty() {
            println!("session {i}: clean");
        } else {
            for a in anomalies {
                println!(
                    "session {i}: ANOMALY in {:?} (strength {:.3})",
                    a.sequence, a.anomaly_strength
                );
            }
        }
    }
    Ok(())
}
```

Output:

```text
session 0: clean
session 1: clean
session 2: clean
session 3: ANOMALY in ["LOGIN", "AUTH", "PRIV_ESCALATE", "EXFIL"] (strength 0.481)
session 3: ANOMALY in ["AUTH", "PRIV_ESCALATE", "EXFIL", "LOGOUT"] (strength 0.543)
```

The three benign sessions cap at strength 0.096 across every window,
so threshold `0.3` clears them. The privilege-escalation session
contains the never-before-seen `PRIV_ESCALATE` and `EXFIL` symbols and
both four-grams that touch them break above the threshold.

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
    .with_max_order(4)?              // Higher order = longer context, more memory
    .with_smoothing_alpha(0.5)?      // Lower = more sensitive to training data
    .with_weights(0.8, 0.2)?         // (likelihood + information) — must sum to 1.0
    .with_memory_limit(Some(100_000))?; // Cap at 100k context nodes (default: 1_000_000)

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

The test suite is organised by feature in `tests/`, with shared
fixtures in `tests/common/mod.rs`:

| File | Coverage |
|---|---|
| `api.rs` | Public API smoke (constructors, training, metrics, optimisation) |
| `math.rs` | Markov + Kolmogorov + Witten-Bell + Shannon entropy + KL invariants |
| `detection.rs` | Anomaly-detection contract (score bounds, monotonicity, threshold) |
| `sequences.rs` | Sequence behaviour (window truncation, alphabet scaling, long inputs) |
| `workflow.rs` | End-to-end domain scenarios (network, fraud, IoT, syslog) |
| `errors.rs` | Error-path coverage (`AnomalyGridError` variants) |
| `concurrency.rs` | `Send + Sync` static asserts + parallel determinism |
| `proptest.rs` | Property tests (sums to 1, entropy bounds, Unicode, long-sequence) |
| `regression.rs` | Past-bug regressions |
| `perf_*.rs` | Throughput / memory / scaling — run with `--release` |

```bash
cargo test                      # all tests
cargo test --test math          # one suite
cargo test --release perf_      # performance suites
cargo run --release --example network_protocol_analysis
cargo run --release --example communication_protocol_analysis
cargo run --release --example protein_folding_sequences
```

## Documentation

- [docs.rs/anomaly-grid](https://docs.rs/anomaly-grid) — rustdoc reference
- [docs/api-reference.md](docs/api-reference.md) — public-surface map
- [docs/mathematical-implementation.md](docs/mathematical-implementation.md) — Witten-Bell + entropy + KL
- [docs/performance-guide.md](docs/performance-guide.md) — sizing, pruning, parallel scoring
- [examples/](examples/) — runnable demos (network, comms, protein)
- [CHANGELOG.md](CHANGELOG.md) — version history

## License

MIT — see [LICENCE](LICENCE).
