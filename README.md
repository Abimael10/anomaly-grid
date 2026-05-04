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
  clippy::expect_used)]`.
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

## Use cases

`anomaly-grid` is a fit when your data is a **sequence of discrete
tokens over a finite alphabet** and you have a corpus of known-normal
examples to train on. The detector flags windows whose Markov
likelihood under that corpus drops sharply — locally improbable
*transitions*, even when each individual token is legitimate.

### Concrete fits

- **Protocol / state-machine traces** — TCP session states,
  application handshakes, consensus rounds. Catches sessions that
  skip handshake steps, hit illegal transitions, or reset mid-stream.
  See [`examples/network_protocol_analysis.rs`](examples/network_protocol_analysis.rs)
  (16-state TCP-like flow).

- **System-call / audit-log monitoring** — `open → read → close`,
  `socket → connect → send`. Surfaces fileless malware, shell escapes,
  and privilege-escalation patterns whose individual syscalls are
  legitimate but whose *order* isn't. The quick-start above is a
  minimal version of this.

- **Operational workflows** — runbook steps, CI pipeline ordering,
  CLI session macros. Deviation from the canonical sequence is itself
  the signal. See [`examples/communication_protocol_analysis.rs`](examples/communication_protocol_analysis.rs)
  (12-symbol comms protocol with injected attacks).

- **Bioinformatics motif scanning** — codon triplets in a known
  reading frame, residue patterns in a curated taxon. Frameshifts
  and rare splice variants surface as low-likelihood windows. See
  [`examples/protein_folding_sequences.rs`](examples/protein_folding_sequences.rs)
  (20-residue alphabet).

### Where it doesn't fit

- Continuous or high-dimensional data (images, raw audio, dense
  feature vectors) without discretisation.
- Alphabets above ~1000 symbols at high `max_order` — context-tree
  memory grows as `|Σ|^max_order`.
- Long-range dependencies beyond 4–5 tokens. If the signal lives in
  context spans of dozens of tokens, prefer a Transformer-based
  sequence model (TFT, Anomaly Transformer) or an HMM with explicit
  hidden state.

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
