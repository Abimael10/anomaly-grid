//! Anomaly detection over finite-alphabet sequences.
//!
//! ## Score
//!
//! For each window of length `max_order + 1` we compute, in **bits**:
//!
//! - `H̄ = (1/(n−1)) · Σ −log₂ P_wb(xᵢ | x₁..xᵢ₋₁)` — the average
//!   per-symbol surprise under the trained Witten-Bell model. Identical
//!   to `information_score` because the contexts and probabilities used
//!   are identical; we expose both fields for backward compatibility.
//! - `anomaly_strength = tanh((w_l + w_i) · H̄ / normalization_factor)`
//!   — a smooth, monotonic squashing into `[0, 1)`. Tunable via
//!   [`AnomalyGridConfig::likelihood_weight`],
//!   [`AnomalyGridConfig::information_weight`], and
//!   [`AnomalyGridConfig::normalization_factor`].
//!
//! ## Correctness
//!
//! All quantities live in the same unit (bits) — the previous mix of
//! `−ln(L)` and `−log₂ P` silently injected a `ln 2` scale factor
//! before [`tanh`]. Likelihood is computed as the chain-rule joint via
//! `Σ log₂ P` then `exp2`, avoiding underflow on per-step products.

use crate::config::AnomalyGridConfig;
use crate::constants::validation::{MAX_THRESHOLD, MIN_THRESHOLD};
use crate::error::{AnomalyGridError, AnomalyGridResult};
use crate::markov_model::MarkovModel;
use crate::performance::{optimize_context_tree, ContextStatistics, OptimizationConfig, PerformanceMetrics};
use crate::string_interner::StateId;
use crate::validation::validate_training_data_quality;
use std::sync::Arc;
use std::time::Instant;

/// Anomaly score for a sequence window.
///
/// `information_score` and the internals of `anomaly_strength` are in
/// **bits**, removing the v0.5 unit mismatch where `−ln(L)` was added
/// to `−log₂ P` before `tanh`.
#[derive(Debug, Clone, PartialEq)]
pub struct AnomalyScore {
    /// The window that was analysed.
    pub sequence: Vec<String>,
    /// Joint chain-rule likelihood `∏ᵢ P_wb(xᵢ | context)` ∈ \[0, 1\].
    pub likelihood: f64,
    /// Natural-log of `likelihood`. `f64::NEG_INFINITY` when likelihood
    /// has underflowed.
    pub log_likelihood: f64,
    /// Average pointwise information content `−log₂ P(xᵢ | context)` (bits, ≥ 0).
    pub information_score: f64,
    /// Combined anomaly strength ∈ \[0, 1) via `tanh`.
    pub anomaly_strength: f64,
}

impl AnomalyScore {
    fn from_metrics(sequence: Vec<String>, m: WindowMetrics, config: &AnomalyGridConfig) -> Self {
        Self {
            sequence,
            likelihood: m.likelihood,
            log_likelihood: m.log_likelihood_nats,
            information_score: m.information_score_bits,
            anomaly_strength: Self::calculate_anomaly_strength(
                m.information_score_bits,
                config,
            ),
        }
    }

    /// Combined anomaly strength in \[0, 1).
    ///
    /// ```text
    /// surprise_bits = information_score                   // ≥ 0, in bits
    /// raw           = (w_l + w_i) · surprise_bits         // bits
    /// strength      = tanh(raw / normalization_factor)
    /// ```
    ///
    /// The two weights are kept as separate fields for backwards-compatibility
    /// of the config; their sum is what scales the surprise. Both
    /// components are in the same unit (bits) so combining them is sound.
    pub(crate) fn calculate_anomaly_strength(
        information_score_bits: f64,
        config: &AnomalyGridConfig,
    ) -> f64 {
        let surprise_bits = information_score_bits.max(0.0);
        let weight = config.likelihood_weight + config.information_weight;
        let raw = weight * surprise_bits;
        (raw / config.normalization_factor).tanh()
    }
}

/// Bundle of metrics for a single window.
#[derive(Debug, Clone, Copy)]
struct WindowMetrics {
    likelihood: f64,
    log_likelihood_nats: f64,
    information_score_bits: f64,
}

/// Anomaly detector using a variable-order Markov model.
#[derive(Debug)]
pub struct AnomalyDetector {
    model: MarkovModel,
    metrics: PerformanceMetrics,
}

impl AnomalyDetector {
    /// Create a new detector with the given maximum context order.
    pub fn new(max_order: usize) -> AnomalyGridResult<Self> {
        if max_order == 0 {
            return Err(AnomalyGridError::invalid_max_order(max_order));
        }
        Ok(Self {
            model: MarkovModel::new(max_order)?,
            metrics: PerformanceMetrics::new(),
        })
    }

    /// Create a detector with a custom configuration.
    pub fn with_config(config: AnomalyGridConfig) -> AnomalyGridResult<Self> {
        config.validate()?;
        Ok(Self {
            model: MarkovModel::with_config(config)?,
            metrics: PerformanceMetrics::new(),
        })
    }

    /// Train on a single sequence of known-normal data.
    pub fn train(&mut self, sequence: &[String]) -> AnomalyGridResult<()> {
        let start_time = Instant::now();
        // Surfaced via `training_warnings`, never silently swallowed.
        let _ = validate_training_data_quality(sequence);

        let result = self.model.train(sequence);

        self.metrics.training_time_ms = start_time.elapsed().as_millis() as u64;
        self.metrics.context_count = self.model.context_tree().context_count();
        self.metrics.estimated_memory_bytes = self.model.context_tree().estimate_memory_usage();
        result
    }

    /// Train on multiple independent sequences (no cross-sequence transitions).
    pub fn train_sequences(&mut self, sequences: &[Vec<String>]) -> AnomalyGridResult<()> {
        let start_time = Instant::now();

        if sequences.is_empty() {
            return Err(AnomalyGridError::invalid_configuration(
                "sequences",
                "empty",
                "at least one sequence",
            ));
        }

        self.model.prepare_state_mapping(sequences);

        for (i, sequence) in sequences.iter().enumerate() {
            if sequence.len() < self.model.config().min_sequence_length {
                return Err(AnomalyGridError::sequence_too_short(
                    self.model.config().min_sequence_length,
                    sequence.len(),
                    "sequence training",
                ));
            }
            self.model.train_with_existing_vocab(sequence).map_err(|e| {
                AnomalyGridError::invalid_configuration(
                    "sequence_training",
                    &format!("sequence {i} failed"),
                    &format!("valid sequence: {e}"),
                )
            })?;
        }

        self.metrics.training_time_ms = start_time.elapsed().as_millis() as u64;
        self.metrics.context_count = self.model.context_tree().context_count();
        self.metrics.estimated_memory_bytes = self.model.context_tree().estimate_memory_usage();
        Ok(())
    }

    /// Diagnostic warnings about a training sequence.
    pub fn training_warnings(&self, sequence: &[String]) -> Vec<String> {
        validate_training_data_quality(sequence)
    }

    /// Detect anomalies in a sequence using sliding-window analysis.
    pub fn detect_anomalies(
        &self,
        sequence: &[String],
        threshold: f64,
    ) -> AnomalyGridResult<Vec<AnomalyScore>> {
        if !threshold.is_finite() || !(MIN_THRESHOLD..=MAX_THRESHOLD).contains(&threshold) {
            return Err(AnomalyGridError::invalid_threshold(threshold));
        }
        if self.model.context_tree().context_count() == 0 {
            return Err(AnomalyGridError::empty_context_tree());
        }

        if sequence.len() <= self.model.max_order() {
            return Ok(self.detect_with_adaptive_order(sequence, threshold));
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
            if let Some(metrics) = self.compute_window_metrics_ids(window, window_ids) {
                let score = AnomalyScore::from_metrics(
                    window.to_vec(),
                    metrics,
                    self.model.config(),
                );
                if score.anomaly_strength >= threshold {
                    anomalies.push(score);
                }
            }
        }

        Ok(anomalies)
    }

    pub fn detect_anomalies_with_monitoring(
        &mut self,
        sequence: &[String],
        threshold: f64,
    ) -> AnomalyGridResult<Vec<AnomalyScore>> {
        let start_time = Instant::now();
        let result = self.detect_anomalies(sequence, threshold);

        let elapsed_micros = start_time.elapsed().as_micros() as u64;
        self.metrics.detection_time_ms = if elapsed_micros == 0 {
            1
        } else {
            elapsed_micros.div_ceil(1000)
        };
        result
    }

    pub fn max_order(&self) -> usize {
        self.model.max_order()
    }

    pub fn model(&self) -> &MarkovModel {
        &self.model
    }

    pub fn performance_metrics(&self) -> &PerformanceMetrics {
        &self.metrics
    }

    pub fn optimize(&mut self, optimization_config: &OptimizationConfig) -> AnomalyGridResult<()> {
        let context_tree = self.model.context_tree_mut();
        let optimization_metrics = optimize_context_tree(context_tree, optimization_config)?;
        self.metrics.context_count = optimization_metrics.context_count;
        self.metrics.estimated_memory_bytes = optimization_metrics.estimated_memory_bytes;
        Ok(())
    }

    pub fn context_statistics(&self) -> ContextStatistics {
        self.model.context_tree().get_context_statistics()
    }

    fn detect_with_adaptive_order(
        &self,
        sequence: &[String],
        threshold: f64,
    ) -> Vec<AnomalyScore> {
        if sequence.len() < 2 {
            return Vec::new();
        }

        let mut anomalies = Vec::new();
        let max_window_size = sequence.len().min(self.model.max_order() + 1);

        for window_size in (2..=max_window_size).rev() {
            if sequence.len() >= window_size {
                for window in sequence.windows(window_size) {
                    if let Some(metrics) = self.compute_window_metrics(window) {
                        let score = AnomalyScore::from_metrics(
                            window.to_vec(),
                            metrics,
                            self.model.config(),
                        );
                        if score.anomaly_strength >= threshold {
                            anomalies.push(score);
                        }
                    }
                }
                if !anomalies.is_empty() {
                    break;
                }
            }
        }

        if anomalies.is_empty() && sequence.len() >= 2 {
            if let Some(metrics) = self.compute_window_metrics(sequence) {
                let score = AnomalyScore::from_metrics(
                    sequence.to_vec(),
                    metrics,
                    self.model.config(),
                );
                if score.anomaly_strength >= threshold {
                    anomalies.push(score);
                }
            }
        }

        anomalies
    }

    fn compute_window_metrics(&self, window: &[String]) -> Option<WindowMetrics> {
        if window.len() < 2 {
            return None;
        }
        let mut log2_sum = 0.0;
        for i in 1..window.len() {
            let p = self
                .model
                .get_best_context_probability(&window[..i], &window[i]);
            log2_sum += p.log2();
        }
        let n_minus_1 = (window.len() - 1) as f64;
        let info_score_bits = -log2_sum / n_minus_1;
        let likelihood = log2_sum.exp2().clamp(0.0, 1.0);
        let log_likelihood_nats = if likelihood > 0.0 {
            likelihood.ln()
        } else {
            f64::NEG_INFINITY
        };
        Some(WindowMetrics {
            likelihood,
            log_likelihood_nats,
            information_score_bits: info_score_bits,
        })
    }

    fn compute_window_metrics_ids(
        &self,
        window: &[String],
        window_ids: &[StateId],
    ) -> Option<WindowMetrics> {
        if window.len() < 2 {
            return None;
        }
        let mut log2_sum = 0.0;
        for i in 1..window.len() {
            let p = self.model.get_best_context_probability_ids(
                &window_ids[..i],
                window_ids[i],
                &window[i],
            );
            log2_sum += p.log2();
        }
        let n_minus_1 = (window.len() - 1) as f64;
        let info_score_bits = -log2_sum / n_minus_1;
        let likelihood = log2_sum.exp2().clamp(0.0, 1.0);
        let log_likelihood_nats = if likelihood > 0.0 {
            likelihood.ln()
        } else {
            f64::NEG_INFINITY
        };
        Some(WindowMetrics {
            likelihood,
            log_likelihood_nats,
            information_score_bits: info_score_bits,
        })
    }
}

/// Score many sequences in parallel against a **pre-trained** detector.
///
/// This is the correct paradigm for anomaly detection: train once on
/// known-normal data, then score any number of unknown sequences
/// concurrently. Replaces v0.5's `batch_process_sequences`, which
/// trained a fresh detector on every input and was therefore degenerate.
///
/// # Errors
///
/// Returns [`AnomalyGridError::InvalidThreshold`] if `threshold` is
/// outside `[0, 1]` or non-finite. Per-sequence errors are propagated
/// — if any sequence fails (e.g. detector untrained), the call returns
/// the first error.
pub fn batch_score(
    detector: &AnomalyDetector,
    sequences: &[Vec<String>],
    threshold: f64,
) -> AnomalyGridResult<Vec<Vec<AnomalyScore>>> {
    use rayon::prelude::*;

    if !threshold.is_finite() || !(MIN_THRESHOLD..=MAX_THRESHOLD).contains(&threshold) {
        return Err(AnomalyGridError::invalid_threshold(threshold));
    }

    sequences
        .par_iter()
        .map(|seq| detector.detect_anomalies(seq, threshold))
        .collect()
}
