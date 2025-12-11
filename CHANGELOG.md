# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - 2025-12-11

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
