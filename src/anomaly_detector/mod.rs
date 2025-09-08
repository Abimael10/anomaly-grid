//! Anomaly Detector module for Markov chain-based anomaly detection
//!
//! This module provides anomaly detection functionality using variable-order
//! Markov models with information-theoretic scoring.

use crate::config::AnomalyGridConfig;
use crate::constants::validation::*;
use crate::error::{AnomalyGridError, AnomalyGridResult};
use crate::markov_model::MarkovModel;
use crate::performance::{optimize_context_tree, OptimizationConfig, PerformanceMetrics};
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
    /// Create a new anomaly score
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

    /// Calculate combined anomaly strength using configuration parameters
    fn calculate_anomaly_strength(
        likelihood: f64,
        information_score: f64,
        config: &AnomalyGridConfig,
    ) -> f64 {
        // Combine likelihood and information score into normalized strength [0,1]
        let log_likelihood_component = if likelihood > 0.0 {
            (-likelihood.ln()).max(0.0)
        } else {
            config.normalization_factor
        };

        let combined_score = log_likelihood_component * config.likelihood_weight
            + information_score * config.information_weight;

        // Normalize to [0,1] using tanh with configurable factor
        (combined_score / config.normalization_factor).tanh()
    }
}

/// Anomaly detector using Markov chain analysis
#[derive(Debug, Clone)]
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
    pub fn train(&mut self, sequence: &[String]) -> AnomalyGridResult<()> {
        let start_time = Instant::now();

        let result = self.model.train(sequence);

        // Update performance metrics
        self.metrics.training_time_ms = start_time.elapsed().as_millis() as u64;
        self.metrics.context_count = self.model.context_tree().context_count();
        self.metrics.estimated_memory_bytes = self.model.context_tree().estimate_memory_usage();

        result
    }

    /// Detect anomalies in a sequence using sliding window analysis
    ///
    /// # Complexity
    /// - Time: O(m × max_order) where m = test sequence length
    /// - Space: O(1) for detection (excluding result storage)
    ///
    /// # Performance Guarantees
    /// - Validates threshold is in valid range \[0,1\]
    /// - Checks if model has been trained before detection
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

        // Handle short sequences gracefully
        if sequence.len() <= self.model.max_order() {
            return Ok(Vec::new());
        }

        let window_size = self.model.max_order() + 1;
        let mut anomalies = Vec::new();

        for window in sequence.windows(window_size) {
            if let Some(score) = self.calculate_anomaly_score(window) {
                // Filter by threshold
                if score.likelihood < threshold {
                    anomalies.push(score);
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

    /// Calculate anomaly score for a sequence window
    fn calculate_anomaly_score(&self, window: &[String]) -> Option<AnomalyScore> {
        if window.len() < 2 {
            return None;
        }

        // Calculate likelihood
        let likelihood = self.model.calculate_likelihood(window);
        let log_likelihood = if likelihood > 0.0 {
            likelihood.ln()
        } else {
            f64::NEG_INFINITY
        };

        // Calculate information-theoretic score
        let information_score = self.calculate_information_score(window);

        Some(AnomalyScore::new(
            window.to_vec(),
            likelihood,
            log_likelihood,
            information_score,
            self.model.config(),
        ))
    }

    /// Calculate information-theoretic anomaly score
    fn calculate_information_score(&self, window: &[String]) -> f64 {
        let mut total_information = 0.0;
        let mut count = 0;

        for i in 1..window.len() {
            let max_context_len = i.min(self.model.max_order());

            // Try different context lengths
            for context_len in 1..=max_context_len {
                let context = &window[i - context_len..i];
                let next_state = &window[i];

                let prob = self.model.get_best_context_probability(context, next_state);
                if prob > 0.0 {
                    // Information content: I(x) = -log₂(P(x))
                    total_information += -prob.log2();
                    count += 1;
                    break; // Use the first (longest) available context
                }
            }
        }

        if count > 0 {
            total_information / count as f64
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

            match AnomalyDetector::with_config(config.clone()) {
                Ok(mut detector) => match detector.train(sequence) {
                    Ok(()) => detector
                        .detect_anomalies(sequence, threshold)
                        .unwrap_or_default(),
                    Err(_) => Vec::new(),
                },
                Err(_) => Vec::new(),
            }
        })
        .collect();

    Ok(results)
}
