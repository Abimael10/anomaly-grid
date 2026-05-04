//! Anomaly Detector module for Markov chain-based anomaly detection
//!
//! This module provides anomaly detection functionality using variable-order
//! Markov models with information-theoretic scoring.

use crate::config::AnomalyGridConfig;
use crate::constants::validation::{MAX_THRESHOLD, MIN_THRESHOLD};
use crate::error::{AnomalyGridError, AnomalyGridResult};
use crate::markov_model::MarkovModel;
use crate::performance::{optimize_context_tree, OptimizationConfig, PerformanceMetrics};
use crate::string_interner::StateId;
use std::sync::Arc;
use std::time::Instant;

/// Anomaly score for a sequence window
#[derive(Debug, Clone)]
pub struct AnomalyScore {
    /// The sequence that was analyzed
    pub sequence: Vec<String>,
    /// Likelihood of the sequence under the model
    pub likelihood: f64,
    /// Log-likelihood for numerical stability
    pub log_likelihood: f64,
    /// Information-theoretic anomaly score
    pub information_score: f64,
    /// Combined anomaly strength \[0,1\]
    pub anomaly_strength: f64,
}

impl AnomalyScore {
    /// Create a new anomaly score.
    pub fn new(
        sequence: Vec<String>,
        likelihood: f64,
        log_likelihood: f64,
        information_score: f64,
        config: &AnomalyGridConfig,
    ) -> Self {
        let anomaly_strength =
            Self::calculate_anomaly_strength(likelihood, information_score, config);

        Self {
            sequence,
            likelihood,
            log_likelihood,
            information_score,
            anomaly_strength,
        }
    }

    /// Combined anomaly strength ∈ [0, 1) via `tanh`.
    ///
    /// ```text
    /// s = w_l · (−ln P) + w_i · I
    /// anomaly_strength = tanh(s / normalization_factor)
    /// ```
    ///
    /// Smooth, monotonic, and has a single tunable scale via
    /// `config.normalization_factor` (default 10.0).
    pub(crate) fn calculate_anomaly_strength(
        likelihood: f64,
        information_score: f64,
        config: &AnomalyGridConfig,
    ) -> f64 {
        let surprise = if likelihood > 0.0 {
            -likelihood.ln()
        } else {
            config.normalization_factor * 4.0 // atanh(~1) ≈ 4·scale
        };

        let raw = surprise * config.likelihood_weight
            + information_score.max(0.0) * config.information_weight;

        (raw / config.normalization_factor).tanh()
    }
}

/// Anomaly detector using Markov chain analysis
#[derive(Debug)]
pub struct AnomalyDetector {
    /// The underlying Markov model
    model: MarkovModel,
    /// Performance metrics for monitoring
    metrics: PerformanceMetrics,
}

impl AnomalyDetector {
    /// Create a new anomaly detector with specified maximum order
    pub fn new(max_order: usize) -> AnomalyGridResult<Self> {
        if max_order == 0 {
            return Err(AnomalyGridError::invalid_max_order(max_order));
        }

        Ok(Self {
            model: MarkovModel::new(max_order)?,
            metrics: PerformanceMetrics::new(),
        })
    }

    /// Create a new anomaly detector with custom configuration
    pub fn with_config(config: AnomalyGridConfig) -> AnomalyGridResult<Self> {
        config.validate()?;

        Ok(Self {
            model: MarkovModel::with_config(config)?,
            metrics: PerformanceMetrics::new(),
        })
    }

    /// Train the detector on normal sequences
    ///
    /// # Complexity
    /// - Time: O(n × max_order × |alphabet|) where n = sequence length
    /// - Space: O(|alphabet|^max_order) in worst case
    ///
    /// # Performance Guarantees
    /// - Memory usage is bounded by config.memory_limit if set
    /// - Validates sequence length against config.min_sequence_length
    /// - Updates performance metrics for monitoring
    /// - Provides warnings for edge cases that may limit performance
    pub fn train(&mut self, sequence: &[String]) -> AnomalyGridResult<()> {
        let start_time = Instant::now();

        let result = self.model.train(sequence);

        // Update performance metrics
        self.metrics.training_time_ms = start_time.elapsed().as_millis() as u64;
        self.metrics.context_count = self.model.context_tree().context_count();
        self.metrics.estimated_memory_bytes = self.model.context_tree().estimate_memory_usage();

        result
    }

    /// Train the detector on multiple sequences while preserving sequence boundaries
    ///
    /// This method addresses the sequence vs stream processing mismatch by training
    /// on multiple sequences without learning cross-sequence transitions.
    ///
    /// # Arguments
    /// * `sequences` - A slice of sequences to train on
    ///
    /// # Behavior
    /// - Each sequence is processed independently
    /// - No transitions are learned across sequence boundaries
    /// - Sequence boundary information is preserved
    /// - All sequences contribute to the same model
    ///
    /// # Example
    /// ```rust
    /// use anomaly_grid::*;
    ///
    /// let mut detector = AnomalyDetector::new(2)?;
    /// let sequences = vec![
    ///     vec!["A".to_string(), "B".to_string(), "C".to_string()],
    ///     vec!["D".to_string(), "E".to_string(), "F".to_string()],
    /// ];
    /// detector.train_sequences(&sequences)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Complexity
    /// - Time: O(k × n × max_order × |alphabet|) where k = number of sequences, n = average sequence length
    /// - Space: O(|alphabet|^max_order) in worst case
    pub fn train_sequences(&mut self, sequences: &[Vec<String>]) -> AnomalyGridResult<()> {
        let start_time = Instant::now();

        // Validate that we have sequences to train on
        if sequences.is_empty() {
            return Err(AnomalyGridError::invalid_configuration(
                "sequences",
                "empty",
                "at least one sequence",
            ));
        }

        // Build a unified vocabulary across all sequences to preserve earlier states
        self.model.prepare_state_mapping(sequences);

        // Train on each sequence independently to preserve boundaries
        for (i, sequence) in sequences.iter().enumerate() {
            // Validate sequence length
            if sequence.len() < self.model.config().min_sequence_length {
                return Err(AnomalyGridError::sequence_too_short(
                    self.model.config().min_sequence_length,
                    sequence.len(),
                    "sequence training",
                ));
            }

            // Train on this sequence (accumulates in the same model)
            self.model
                .train_with_existing_vocab(sequence)
                .map_err(|e| {
                    AnomalyGridError::invalid_configuration(
                        "sequence_training",
                        &format!("sequence {i} failed"),
                        &format!("valid sequence: {e}"),
                    )
                })?;
        }

        // Update performance metrics
        self.metrics.training_time_ms = start_time.elapsed().as_millis() as u64;
        self.metrics.context_count = self.model.context_tree().context_count();
        self.metrics.estimated_memory_bytes = self.model.context_tree().estimate_memory_usage();

        Ok(())
    }

    /// Detect anomalies in a sequence using sliding window analysis
    ///
    /// Uses intuitive threshold semantics:
    /// - Higher thresholds are more restrictive (detect fewer anomalies)
    /// - Lower thresholds are less restrictive (detect more anomalies)
    /// - Threshold is compared against anomaly_strength
    ///
    /// # Arguments
    /// * `sequence` - The sequence to analyze for anomalies
    /// * `threshold` - Minimum anomaly strength required (0.0 to 1.0)
    ///   - 0.0: Include all anomalies
    ///   - 0.5: Include moderate to strong anomalies
    ///   - 0.9: Include only very strong anomalies
    ///
    /// # Returns
    /// Vector of anomaly scores where anomaly_strength >= threshold
    ///
    /// # Complexity
    /// - Time: O(m × max_order) where m = test sequence length
    /// - Space: O(1) for detection (excluding result storage)
    ///
    /// # Performance Guarantees
    /// - Validates threshold is in valid range \[0,1\]
    /// - Checks if model has been trained before detection
    ///
    /// # Example
    /// ```rust
    /// use anomaly_grid::*;
    ///
    /// let mut detector = AnomalyDetector::new(2)?;
    /// detector.train(&["A".to_string(), "B".to_string(), "C".to_string()]);
    ///
    /// let test_seq = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];
    /// let strong_anomalies = detector.detect_anomalies(&test_seq, 0.8)?;
    /// let all_anomalies = detector.detect_anomalies(&test_seq, 0.0)?;
    ///
    /// assert!(strong_anomalies.len() <= all_anomalies.len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn detect_anomalies(
        &self,
        sequence: &[String],
        threshold: f64,
    ) -> AnomalyGridResult<Vec<AnomalyScore>> {
        // Validate threshold
        if !threshold.is_finite() || !(MIN_THRESHOLD..=MAX_THRESHOLD).contains(&threshold) {
            return Err(AnomalyGridError::invalid_threshold(threshold));
        }

        // Check if model is trained
        if self.model.context_tree().context_count() == 0 {
            return Err(AnomalyGridError::empty_context_tree());
        }

        // Handle short sequences by using adaptive window sizing before any allocation
        if sequence.len() <= self.model.max_order() {
            // For short sequences, use adaptive detection
            return self.detect_with_adaptive_order(sequence, threshold);
        }

        let interner = Arc::clone(self.model.context_tree().interner());
        let sequence_ids: Vec<StateId> =
            sequence.iter().map(|s| interner.get_or_intern(s)).collect();

        let window_size = self.model.max_order() + 1;
        let mut anomalies = Vec::with_capacity(sequence.len().saturating_sub(window_size) + 1);

        for (window, window_ids) in sequence
            .windows(window_size)
            .zip(sequence_ids.windows(window_size))
        {
            if let Some((likelihood, log_likelihood, information_score, anomaly_strength)) =
                self.compute_anomaly_metrics_with_ids(window, window_ids, false)
            {
                if anomaly_strength >= threshold {
                    anomalies.push(AnomalyScore::new(
                        window.to_vec(),
                        likelihood,
                        log_likelihood,
                        information_score,
                        self.model.config(),
                    ));
                }
            }
        }

        Ok(anomalies)
    }

    /// Detect anomalies with performance monitoring (mutable version)
    ///
    /// This version updates performance metrics and should be used when
    /// performance monitoring is needed.
    pub fn detect_anomalies_with_monitoring(
        &mut self,
        sequence: &[String],
        threshold: f64,
    ) -> AnomalyGridResult<Vec<AnomalyScore>> {
        let start_time = Instant::now();

        let result = self.detect_anomalies(sequence, threshold);

        // Update detection metrics - ensure we always record some time for monitoring
        let elapsed_micros = start_time.elapsed().as_micros() as u64;
        // Convert to milliseconds, but ensure at least 1ms is recorded for monitoring purposes
        self.metrics.detection_time_ms = if elapsed_micros == 0 {
            1
        } else {
            elapsed_micros.div_ceil(1000)
        };

        result
    }

    /// Get the maximum order of the detector
    pub fn max_order(&self) -> usize {
        self.model.max_order()
    }

    /// Get access to the underlying Markov model (for testing)
    pub fn model(&self) -> &MarkovModel {
        &self.model
    }

    /// Get performance metrics
    pub fn performance_metrics(&self) -> &PerformanceMetrics {
        &self.metrics
    }

    /// Apply performance optimizations to the context tree
    ///
    /// This can significantly reduce memory usage by removing low-frequency
    /// or low-entropy contexts while maintaining detection accuracy.
    pub fn optimize(&mut self, optimization_config: &OptimizationConfig) -> AnomalyGridResult<()> {
        // Get mutable access to the context tree through the model
        let context_tree = self.model.context_tree_mut();

        // Apply optimizations
        let optimization_metrics = optimize_context_tree(context_tree, optimization_config)?;

        // Update our metrics
        self.metrics.context_count = optimization_metrics.context_count;
        self.metrics.estimated_memory_bytes = optimization_metrics.estimated_memory_bytes;

        Ok(())
    }

    /// Get context tree statistics for analysis
    pub fn context_statistics(&self) -> crate::performance::ContextStatistics {
        self.model.context_tree().get_context_statistics()
    }

    #[allow(clippy::unnecessary_wraps)]
    fn detect_with_adaptive_order(
        &self,
        sequence: &[String],
        threshold: f64,
    ) -> AnomalyGridResult<Vec<AnomalyScore>> {
        if sequence.len() < 2 {
            return Ok(Vec::new());
        }

        let mut anomalies = Vec::new();

        // For short sequences, use the maximum possible window size
        let max_window_size = sequence.len().min(self.model.max_order() + 1);

        // Start with the largest possible window and work down
        for window_size in (2..=max_window_size).rev() {
            if sequence.len() >= window_size {
                for window in sequence.windows(window_size) {
                    if let Some((likelihood, log_likelihood, information_score, anomaly_strength)) =
                        self.compute_anomaly_metrics(window, true)
                    {
                        if anomaly_strength >= threshold {
                            anomalies.push(AnomalyScore::new(
                                window.to_vec(),
                                likelihood,
                                log_likelihood,
                                information_score,
                                self.model.config(),
                            ));
                        }
                    }
                }
                // If we found anomalies with this window size, we're done
                if !anomalies.is_empty() {
                    break;
                }
            }
        }

        // If no anomalies found with windowing, try the whole sequence
        if anomalies.is_empty() && sequence.len() >= 2 {
            if let Some((likelihood, log_likelihood, information_score, anomaly_strength)) =
                self.compute_anomaly_metrics(sequence, true)
            {
                if anomaly_strength >= threshold {
                    anomalies.push(AnomalyScore::new(
                        sequence.to_vec(),
                        likelihood,
                        log_likelihood,
                        information_score,
                        self.model.config(),
                    ));
                }
            }
        }

        // Special case: for sequences of exactly length 2, always try direct scoring
        if anomalies.is_empty() && sequence.len() == 2 {
            // Force scoring even if it didn't work above
            if let Some((likelihood, log_likelihood, information_score, anomaly_strength)) =
                self.compute_anomaly_metrics(sequence, true)
            {
                // Use the same threshold - no special treatment
                if anomaly_strength >= threshold {
                    anomalies.push(AnomalyScore::new(
                        sequence.to_vec(),
                        likelihood,
                        log_likelihood,
                        information_score,
                        self.model.config(),
                    ));
                }
            }
        }

        Ok(anomalies)
    }

    /// Compute anomaly metrics for a window.
    fn compute_anomaly_metrics(
        &self,
        window: &[String],
        _adaptive: bool,
    ) -> Option<(f64, f64, f64, f64)> {
        if window.len() < 2 {
            return None;
        }
        let likelihood = self.model.calculate_likelihood(window);
        let log_likelihood = if likelihood > 0.0 {
            likelihood.ln()
        } else {
            f64::NEG_INFINITY
        };
        let information_score = self.calculate_information_score(window);
        let anomaly_strength = AnomalyScore::calculate_anomaly_strength(
            likelihood,
            information_score,
            self.model.config(),
        );
        Some((likelihood, log_likelihood, information_score, anomaly_strength))
    }

    /// Compute anomaly metrics using precomputed StateIds.
    fn compute_anomaly_metrics_with_ids(
        &self,
        window: &[String],
        window_ids: &[StateId],
        _adaptive: bool,
    ) -> Option<(f64, f64, f64, f64)> {
        if window.len() < 2 {
            return None;
        }
        let likelihood = self.model.calculate_likelihood_ids(window_ids, window);
        let log_likelihood = if likelihood > 0.0 {
            likelihood.ln()
        } else {
            f64::NEG_INFINITY
        };
        let information_score = self.calculate_information_score_ids(window, window_ids);
        let anomaly_strength = AnomalyScore::calculate_anomaly_strength(
            likelihood,
            information_score,
            self.model.config(),
        );
        Some((likelihood, log_likelihood, information_score, anomaly_strength))
    }

    /// Average pointwise information content: mean of −log₂ P(x_i | context).
    fn calculate_information_score(&self, window: &[String]) -> f64 {
        let mut total = 0.0;
        for i in 1..window.len() {
            let prob = self
                .model
                .get_best_context_probability(&window[..i], &window[i]);
            total += -prob.log2();
        }
        if window.len() > 1 {
            total / (window.len() - 1) as f64
        } else {
            0.0
        }
    }

    /// Information score using precomputed StateIds.
    fn calculate_information_score_ids(&self, window: &[String], window_ids: &[StateId]) -> f64 {
        let mut total = 0.0;
        for i in 1..window.len() {
            let prob = self.model.get_best_context_probability_ids(
                &window_ids[..i],
                window_ids[i],
                &window[i],
            );
            total += -prob.log2();
        }
        if window.len() > 1 {
            total / (window.len() - 1) as f64
        } else {
            0.0
        }
    }
}

/// Batch process multiple sequences in parallel
///
/// # Complexity
/// - Time: O(k × n × max_order × |alphabet|) where k = number of sequences
/// - Space: O(k × |alphabet|^max_order) in worst case
///
/// # Performance Guarantees
/// - Processes sequences in parallel using Rayon
/// - Each sequence gets its own detector instance
/// - Failed sequences are handled gracefully
pub fn batch_process_sequences(
    sequences: &[Vec<String>],
    config: &AnomalyGridConfig,
    threshold: f64,
) -> AnomalyGridResult<Vec<Vec<AnomalyScore>>> {
    use rayon::prelude::*;

    // Validate threshold once for all sequences
    if !threshold.is_finite() || !(MIN_THRESHOLD..=MAX_THRESHOLD).contains(&threshold) {
        return Err(AnomalyGridError::invalid_threshold(threshold));
    }

    // Validate configuration
    config.validate()?;

    let results: Vec<Vec<AnomalyScore>> = sequences
        .par_iter()
        .map(|sequence| {
            if sequence.len() <= config.max_order {
                return Vec::new();
            }

            AnomalyDetector::with_config(config.clone()).map_or_else(
                |_| Vec::new(),
                |mut detector| {
                    detector
                        .train(sequence)
                        .ok()
                        .and_then(|()| detector.detect_anomalies(sequence, threshold).ok())
                        .unwrap_or_default()
                },
            )
        })
        .collect();

    Ok(results)
}
