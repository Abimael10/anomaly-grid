# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-05-04

### Added
- `batch_score(detector, sequences, threshold)` for parallel scoring
  against a **pre-trained** detector. Replaces the broken
  `batch_process_sequences` which retrained per input. Built on rayon's
  `par_iter` over a shared `&AnomalyDetector` (now `Send + Sync`-asserted
  in `tests/concurrency_invariants.rs`).
- `MarkovModel::log_likelihood_bits_per_symbol[_ids]`: numerically stable
  per-symbol surprise that never multiplies probabilities. Long
  sequences (1k–10k symbols) no longer produce subnormal/zero
  likelihoods on the detection path.
- `ContextTree::alphabet()`: public snapshot of Σ for callers that need
  to iterate the global vocabulary (e.g. probability-sum invariants).
- `AnomalyDetector::training_warnings`: surfaces validation diagnostics
  (monotonous data, very short data, low diversity) that v0.5 dropped on
  the floor.
- Property tests for empty / length-1 / single-symbol-alphabet /
  Unicode / 1k-symbol sequences and **parallel batch determinism**
  (`tests/proptest.rs`).
- `tests/concurrency.rs`: static `Send + Sync` assertions plus
  deterministic-across-thread-pool-sizes test.
- `clippy::expect_used` and `missing_docs` are now denied at the crate
  root in addition to `pedantic`, `nursery`, and `unwrap_used`.
- Shared test fixtures in `tests/common/mod.rs` (`s()`, `trained()`,
  `pattern_abc()`, `max_strength()`).

### Changed
- **BREAKING**: `batch_process_sequences(seqs, config, threshold)` is
  removed. v0.5 trained a fresh detector on each input and then scored
  the same input — degenerate. Use
  `batch_score(&trained_detector, seqs, threshold)` instead.
- **BREAKING**: `AnomalyScore::log_likelihood` is now the natural-log
  of the *joint* `likelihood` (and `f64::NEG_INFINITY` on underflow).
  The v0.5 anomaly-strength formula combined `−ln(L)` (nats) with
  `−log₂ P` (bits), silently injecting a `ln 2 ≈ 0.693` scale factor
  before `tanh`. The v0.6 `calculate_anomaly_strength` is bits-only and
  uses a single combined weight.
- **BREAKING**: `MarkovModel::get_background_probability` no longer
  takes a state argument — the previous signature ignored it. The
  function returns a single `α / (N + α·|Σ|)` scalar regardless of
  state.
- **BREAKING**: `AnomalyGridError::Internal(&'static str)` is now
  produced by `ContextTrie::insert_context_path` and
  `ContextTrie::get_or_create_context_data`, which were `panic!`ing via
  `.expect("Invalid node ID")`. Callers using the trie directly must
  propagate via `?`. Public users of `AnomalyDetector` / `ContextTree`
  see no change — these errors only fire on internal arena corruption.
- Module layout flattened: `mod.rs`-only directories collapsed to
  flat `.rs` files (`anomaly_detector.rs`, `context_tree.rs`,
  `markov_model.rs`).
- Internal modules (`context_trie`, `transition_counts`,
  `string_interner`, `validation`, `constants`) are now `pub(crate)`.
- `MarkovModel::calculate_likelihood` is now computed via
  `Σ log₂ P` then `exp2`, avoiding per-step product underflow on
  moderate sequences. Very long sequences still underflow to zero —
  use `log_likelihood_bits_per_symbol` for those.

### Removed
- Top-level `pub mod context_trie` exposure (now `pub(crate)`).
- `lib::info()` (dead diagnostic helper).
- `ContextTree::with_interner` (dead constructor — internal trie has
  always owned its interner).
- `ContextNode::add_transition(&str)` (dead — only `add_transition_by_id`
  is called on the hot path).
- `StringInterner::try_intern`, `get_arc`, `is_empty`,
  `estimate_memory_usage`, `entries` (latter inlined into
  `ContextTree::alphabet`); `StateIdConversion` trait;
  `strings_to_state_ids` / `state_ids_to_strings` (all dead).
- `validation::{validate_detection_sequence,
  analyze_training_data_characteristics, TrainingDataAnalysis}` (dead).
- All unused `constants::*` modules: only `validation::MIN_THRESHOLD`
  / `MAX_THRESHOLD` remain.
- Obsolete `total_transitions()` method on `ContextNode` (alias for
  `total_count`).
- `ContextStatistics` fields that were never populated (`total_entropy`,
  `avg_entropy`, `min_*`, `max_*`, `transitions_by_context`).
- `std::thread::sleep(10ms)` in the stress-test loop — made the test
  time-sensitive on slow runners.
- Custom `DomainTestResult` scaffolding and verbose `println!` blocks
  from the audit/domain test suite — replaced with idiomatic
  per-property `#[test] fn`s.

### Test layout
- Test files renamed from numbered `domain_N_*` / `unit_*_tests` /
  `integration_*_tests` / `performance_N_*` / `*_invariants` /
  `regression_anomaly_grid` to feature-named files: `api.rs`,
  `math.rs`, `detection.rs`, `sequences.rs`, `workflow.rs`,
  `errors.rs`, `concurrency.rs`, `proptest.rs`, `regression.rs`, and
  `perf_{training,detection,memory,batch,stress}.rs`.
- Shared fixtures live in `tests/common/mod.rs` (idiomatic Rust-library
  layout: cargo recognises the directory as a module rather than a
  separate test binary).

### Docs
- `docs/api-reference.md`, `docs/mathematical-implementation.md`,
  `docs/performance-guide.md`, `docs/README.md` rewritten to match
  v0.6 (Witten-Bell, `batch_score`, bits-only score). Stale references
  to `MemoryPool`, `TrainingDataAnalysis`,
  `analyze_training_data_characteristics`, and
  `batch_process_sequences` removed.
- Root README testing section updated to the new test layout.

### Fixed
- Anomaly strength now combines surprise (bits) with information
  content (bits) instead of mixing nats with bits. The `tanh` envelope
  is preserved; the score is on a single, defined scale.
- `batch_score` is deterministic across rayon thread-pool sizes
  (verified by both proptest and integration test).
- `ContextTrie` no longer panics on internal arena invariant violation;
  errors propagate as `AnomalyGridError::Internal`.

## [0.5.0] - 2026-05-04

### Added
- Property-based tests via proptest: probability normalization, entropy bounds, anomaly strength bounds, backoff monotonicity, training determinism
- Witten-Bell interpolation for variable-order backoff, replacing hard-cutoff heuristic
- `get_transition_probability_by_ids` for fast StateId-based context lookups
- `global_vocab_size()` on `ContextTree` derived from the shared interner

### Changed
- **BREAKING**: Laplace smoothing now normalizes over the global alphabet (`interner.len()`) instead of local context vocabulary — fixes under-smoothing for unseen symbols
- **BREAKING**: Anomaly strength uses `tanh`-based scoring instead of piecewise-linear formula
- **BREAKING**: Deleted `new_v2`, `calculate_likelihood_with_fallback`, `calculate_information_score_enhanced` (dead/duplicate code)
- All probability methods (`get_probability`, `compute_entropy`, `compute_kl_divergence`, `get_all_probabilities`) now take a `global_vocab_size` parameter
- Marginal probability uses smoothed unigram: `P(x) = (count + α) / (N + α·|Σ|)`
- Error types migrated to thiserror with structured `AnomalyGridError` variants
- Lint gates: `#![deny(clippy::pedantic, clippy::nursery, clippy::unwrap_used)]` enforced across all targets

### Removed
- `memory_pool` module (entirely dead code)
- `STATE_ID_BUFFER` thread-local and `RefCell` machinery
- `context_has_sufficient_data` / `context_has_sufficient_data_ids` magic-threshold heuristics
- `get_transition_probability_normalized` / `get_transition_probability_normalized_ids` (replaced by global-alphabet variants)
- `eprintln!` warnings from `train()`
- Uniform-sequence fast path in detector

### Fixed
- Probability distributions now sum to 1.0 over the full global alphabet
- Entropy upper bound correctly bounded by `log₂(|Σ_global|)`
- String interner `try_intern` no longer holds read guard across write acquisition

## [0.4.2] - 2025-12-11

### Added
- Consolidated docs and examples for real-world scenarios (protocol, protein, communication) with count-based reporting.
- Added missing API doc reference, it now cover `train_sequences`, `ContextTree`/`ContextStatistics`, validation helpers, and optimization helpers.

### Changed
- Updated configuration snippets to match current API (`with_memory_limit(Some(..))?`), and refreshed Quick Start to use richer training data.
- Simplified wording in docs and examples to avoid jargon; removed outdated examples in favor of focused finite-alphabet use cases.

### Fixed
- Documentation references now align with code (correct config signatures, `OptimizationConfig` fields, `PerformanceMetrics` fields, anomaly strength description).
- Note in docs about detection timing requiring `detect_anomalies_with_monitoring`.

## [0.4.0] - 2025-12-11

### Added
- Regression coverage for vocab preservation, marginal stability, and longest-context information scoring
- ID-based detection fast path with shared state interner caching for lower overhead
- Context pruning/limiting implementations for frequency, entropy, and top-N retention with trie rebuilds

### Changed
- Vocabulary building now spans multi-sequence training without dropping earlier states
- Information scoring and fallback likelihoods walk contexts longest-to-shortest to honor variable-order modeling
- Marginals derive from raw counts independent of `max_order`; normalized probabilities honor training config
- Batch and uniform-sequence detection paths optimized for stability and throughput; detection uses thread-local ID buffers to avoid contention
- Batch performance tests stabilized with deterministic workloads and warm-ups to reduce variance

### Fixed
- Context probability queries respect last-used config instead of default smoothing
- Adaptive likelihood fallback uses hierarchical contexts instead of single-step backgrounds
- Release perf tests stabilized (throughput/latency/scalability) without relaxing thresholds

## [0.3.0] - 2025-09-11

### Added
- **Test Suite**: 146 tests covering unit, integration, domain, and performance testing
- **Performance Tests**: 5 dedicated performance test suites covering training scalability, detection throughput, memory optimization, batch processing, and stress testing. The edges can be configured manually to test with your own machine.
- **Domain-Driven Testing**: Mathematical validation across 5 core domains (Markov chains, probability theory, information theory, anomaly detection logic, sequence analysis)
- **Some Examples**: 4 sophisticated examples with functionalities validation, ROC analysis, and real-world kind of usage
- **Enhanced API**: Unified `detect_anomalies` method with backward compatibility
- **Configuration System**: Enhanced configuration with memory limits, smoothing parameters, and weights
- **Error Handling**: Detailed error types with proper context and validation

### Changed
- **BREAKING**: Major version bump to 0.3.0 due to significant API and internal changes
- **BREAKING**: Unified API with single `detect_anomalies` method
- **BREAKING**: All constructors return `Result<T, AnomalyGridError>` for proper error handling
- **Documentation Structure**: Moved detailed documentation to `docs/` folder for better organization
- **Test Organization**: Flattened test structure for better cargo test discovery
- **Performance Thresholds**: Realistic performance expectations based on actual benchmarks

### Fixed
- **Threshold Mechanism**: Properly filters results based on anomaly strength values
- **Performance Issues**: Optimized algorithms for large alphabet sizes and long sequences
- **Edge Case Handling**: Robust handling of empty sequences, single elements, and unknown states
- **Numerical Stability**: Improved probability calculations and mathematical operations

### Removed
- **Obsolete Tests**: Cleaned up deprecated test files and structures after some major flaws resolution

## [0.2.2] - Previous Release

### Added
- String interning for duplicate elimination
- Trie-based context storage with prefix sharing
- On-demand computation with lazy evaluation and caching
- Small collections optimization using SmallVec
- Memory pooling infrastructure
- Enhanced configuration system
- Expanded test suite with 72 tests

### Changed
- API consistency improvements
- Enhanced documentation
- Optimized memory usage patterns

### Fixed
- Threshold mechanism issues
- Memory leaks in context tree management
- Performance issues with large alphabet sizes

## [0.2.0] - Previous Release

### Added
- Major refactoring with better encapsulation
- Expanded test suite
- API consistency improvements

### Changed
- Documentation reduction for simplicity
- All constructors return Result types

## [0.1.x] - Initial Releases

### Added
- Basic variable-order Markov chain implementation
- Sequence anomaly detection capabilities
- Initial API design
- Basic documentation and examples
