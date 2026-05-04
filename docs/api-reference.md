# API Reference

The canonical reference is the rendered rustdoc at
[docs.rs/anomaly-grid](https://docs.rs/anomaly-grid). This page is a
short orientation map of the public surface.

## Public surface

| Type | Module | Purpose |
|---|---|---|
| `AnomalyDetector` | `anomaly_detector` | Train once on known-normal data; score unknown sequences. |
| `AnomalyScore` | `anomaly_detector` | Per-window score: `likelihood`, `log_likelihood`, `information_score`, `anomaly_strength`. |
| `batch_score` | `anomaly_detector` | Score many sequences in parallel against a pre-trained `&AnomalyDetector`. |
| `AnomalyGridConfig` | `config` | Tuneable parameters: `max_order`, `smoothing_alpha`, weights, memory limit. |
| `MarkovModel` | `markov_model` | Variable-order Markov chain with Witten-Bell interpolation (low-level). |
| `ContextTree` / `ContextNode` | `context_tree` | Trie-backed context storage + per-context probability/entropy/KL. |
| `OptimizationConfig` / `optimize_context_tree` | `performance` | Pruning by frequency / entropy / top-N. |
| `PerformanceMetrics` / `ContextStatistics` | `performance` | Training/detection metrics + tree-shape diagnostics. |
| `AnomalyGridError` / `AnomalyGridResult<T>` | `error` | Structured `thiserror`-derived errors. |

## Idiomatic usage

```rust
use anomaly_grid::{AnomalyDetector, batch_score};

let mut detector = AnomalyDetector::new(3)?;
detector.train_sequences(&normal_corpus)?;

// One sequence:
let scores = detector.detect_anomalies(&candidate, 0.5)?;

// Many sequences in parallel:
let results = batch_score(&detector, &candidates, 0.5)?;
```

## Score units

All quantities are in **bits**:

- `information_score` = mean of `−log₂ P_wb(xᵢ | context)` over the window.
- `log_likelihood` = `ln(likelihood)` where `likelihood = ∏ᵢ P_wb(xᵢ | context)`.
- `anomaly_strength = tanh((w_l + w_i) · information_score / normalization_factor)` ∈ \[0, 1).

## Tuning

| Parameter | Default | Effect |
|---|---|---|
| `max_order` | 3 | Higher = longer context, more memory, sharper detection. |
| `smoothing_alpha` | 1.0 | Order-0 Laplace strength. Lower = more sensitive to training data. |
| `likelihood_weight` + `information_weight` | 0.7 + 0.3 | Must sum to 1.0; their combined value scales surprise. |
| `normalization_factor` | 10.0 | Sets the `tanh` saturation point. |
| `memory_limit` | `Some(1_000_000)` | Caps the number of context nodes. |

See [mathematical-implementation.md](mathematical-implementation.md) for
the formulas and [performance-guide.md](performance-guide.md) for sizing
trade-offs.

## Errors

All fallible calls return `AnomalyGridResult<T>`. The `AnomalyGridError`
enum has variants for invalid configuration, threshold, sequence
length, memory limit, untrained detector, and an `Internal` catch-all
for invariant violations (which should never fire in practice).
