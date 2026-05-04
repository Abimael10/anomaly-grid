# Performance Guide

Sizing trade-offs and tuning knobs for `anomaly-grid`. The detection
hot path is allocation-free per window; throughput scales with the
chosen `max_order` and the alphabet size of the corpus.

## Sizing rules of thumb

| Knob | Conservative | Default | Aggressive |
|---|---|---|---|
| `max_order` | 2 | 3 | 5 |
| `smoothing_alpha` | 1.0 | 1.0 | 0.1 (less smoothing → more sensitive) |
| `memory_limit` | `Some(50 * 1024 * 1024)` | `Some(1_000_000)` (contexts) | `None` |

Memory grows roughly as `Σ_{k=1..max_order} |alphabet|^k` worst-case;
typical usage is far below the bound because most context combinations
never appear during training.

## Pre-built configuration helpers

`AnomalyGridConfig` and `OptimizationConfig` ship a few presets:

```rust
let cfg = AnomalyGridConfig::for_small_alphabet();   // order 4, 100k contexts
let cfg = AnomalyGridConfig::for_large_alphabet();   // order 2, 50k contexts, alpha 0.5
let cfg = AnomalyGridConfig::for_low_memory();       // order 2, 10k contexts
let cfg = AnomalyGridConfig::for_high_accuracy();    // order 5, 5M contexts, alpha 0.1

let opt = OptimizationConfig::for_low_memory();
let opt = OptimizationConfig::for_high_accuracy();
let opt = OptimizationConfig::balanced();
```

## Pruning

After training, call `AnomalyDetector::optimize` with an
`OptimizationConfig` to drop low-value contexts:

```rust
detector.optimize(&OptimizationConfig {
    enable_pruning: true,
    min_context_count: 3,    // drop contexts seen < 3 times
    min_entropy: 0.1,        // drop near-deterministic contexts
    max_contexts: Some(50_000),
    enable_monitoring: true,
})?;
```

Pruning rebuilds the trie, so it's a one-shot cost. Detection accuracy
on rare contexts may drop slightly; the trade-off is worthwhile when
memory is tight.

## Parallel scoring

Training is single-threaded (`&mut self`). Scoring is shared-read and
parallel via [`batch_score`]:

```rust
use anomaly_grid::{AnomalyDetector, batch_score};

let mut detector = AnomalyDetector::new(3)?;
detector.train_sequences(&normal_corpus)?;

let results = batch_score(&detector, &unknown_sequences, 0.5)?;
```

Internally this is a rayon `par_iter` over the input. Scoring is
deterministic across thread-pool sizes (the alphabet is frozen at the
end of training, so per-thread interning of unseen symbols cannot
perturb probabilities). See `tests/concurrency.rs` for the invariants.

## Benchmarking your own data

Compile-time release flags matter:

```bash
cargo run --release --example network_protocol_analysis
cargo test --release perf_   # the perf_*.rs test suites
```

`PerformanceMetrics` exposes training time, detection time, context
count, and estimated memory bytes:

```rust
let metrics = detector.performance_metrics();
println!("trained in {} ms; {} contexts; ~{} KB",
    metrics.training_time_ms,
    metrics.context_count,
    metrics.estimated_memory_bytes / 1024,
);
```

`detection_time_ms` is only populated by
`detect_anomalies_with_monitoring`; the read-only `detect_anomalies`
path does not record per-call timing.

## Common pitfalls

- **Underflow on long sequences**: use
  `MarkovModel::log_likelihood_bits_per_symbol` (log-space) instead of
  `calculate_likelihood` (joint, exponentiated).
- **Cross-sequence transitions**: when training on multiple
  conceptually-independent sequences, use `train_sequences` (not a
  concatenated `train`) so `A→B` from sequence 1 isn't followed by
  `C→D` from sequence 2 in the model.
- **Stale `last_config`**: every call to `build_from_sequence` updates
  the tree's last-used config; if you query probabilities later with a
  different config, pass it explicitly via
  `get_transition_probability_with_config`.
