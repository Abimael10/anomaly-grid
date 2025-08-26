//! Configuration management for Anomaly Grid
//!
//! This module provides centralized configuration for all library parameters,
//! enabling fine-tuning of algorithm behavior and performance characteristics.

use crate::error::{AnomalyGridError, AnomalyGridResult};

/// Configuration parameters for Anomaly Grid components
#[derive(Debug, Clone, PartialEq)]
pub struct AnomalyGridConfig {
    /// Maximum context order for Markov model
    pub max_order: usize,

    /// Laplace smoothing parameter (alpha)
    pub smoothing_alpha: f64,

    /// Maximum number of contexts to store (None = unlimited)
    pub memory_limit: Option<usize>,

    /// Minimum probability for numerical stability
    pub min_probability: f64,

    /// Weight for likelihood component in anomaly strength calculation
    pub likelihood_weight: f64,

    /// Weight for information component in anomaly strength calculation
    pub information_weight: f64,

    /// Normalization factor for tanh scaling in anomaly strength
    pub normalization_factor: f64,

    /// Minimum sequence length for training
    pub min_sequence_length: usize,
}

impl Default for AnomalyGridConfig {
    fn default() -> Self {
        Self {
            max_order: 3,
            smoothing_alpha: 1.0,
            memory_limit: Some(1_000_000),
            min_probability: 1e-12,
            likelihood_weight: 0.7,
            information_weight: 0.3,
            normalization_factor: 10.0,
            min_sequence_length: 2,
        }
    }
}

impl AnomalyGridConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration optimized for small alphabets (≤ 10 states)
    pub fn for_small_alphabet() -> Self {
        Self {
            max_order: 4,
            memory_limit: Some(100_000),
            ..Self::default()
        }
    }

    /// Create a configuration optimized for large alphabets (> 20 states)
    pub fn for_large_alphabet() -> Self {
        Self {
            max_order: 2,
            memory_limit: Some(50_000),
            smoothing_alpha: 0.5, // Less aggressive smoothing
            ..Self::default()
        }
    }

    /// Create a configuration optimized for memory-constrained environments
    pub fn for_low_memory() -> Self {
        Self {
            max_order: 2,
            memory_limit: Some(10_000),
            ..Self::default()
        }
    }

    /// Create a configuration optimized for high accuracy
    pub fn for_high_accuracy() -> Self {
        Self {
            max_order: 5,
            smoothing_alpha: 0.1, // Minimal smoothing
            memory_limit: Some(5_000_000),
            ..Self::default()
        }
    }

    /// Validate the configuration parameters
    pub fn validate(&self) -> AnomalyGridResult<()> {
        // Validate max_order
        if self.max_order == 0 {
            return Err(AnomalyGridError::invalid_configuration(
                "max_order",
                &self.max_order.to_string(),
                "a positive integer > 0",
            ));
        }

        // Validate smoothing_alpha
        if !self.smoothing_alpha.is_finite() || self.smoothing_alpha <= 0.0 {
            return Err(AnomalyGridError::invalid_configuration(
                "smoothing_alpha",
                &self.smoothing_alpha.to_string(),
                "a positive finite number",
            ));
        }

        // Validate min_probability
        if !self.min_probability.is_finite()
            || self.min_probability <= 0.0
            || self.min_probability >= 1.0
        {
            return Err(AnomalyGridError::invalid_configuration(
                "min_probability",
                &self.min_probability.to_string(),
                "a value in (0, 1)",
            ));
        }

        // Validate weights
        if !self.likelihood_weight.is_finite() || self.likelihood_weight < 0.0 {
            return Err(AnomalyGridError::invalid_configuration(
                "likelihood_weight",
                &self.likelihood_weight.to_string(),
                "a non-negative finite number",
            ));
        }

        if !self.information_weight.is_finite() || self.information_weight < 0.0 {
            return Err(AnomalyGridError::invalid_configuration(
                "information_weight",
                &self.information_weight.to_string(),
                "a non-negative finite number",
            ));
        }

        // Validate weight sum
        let weight_sum = self.likelihood_weight + self.information_weight;
        if (weight_sum - 1.0).abs() > 1e-10 {
            return Err(AnomalyGridError::invalid_configuration(
                "weight_sum",
                &weight_sum.to_string(),
                "likelihood_weight + information_weight = 1.0",
            ));
        }

        // Validate normalization_factor
        if !self.normalization_factor.is_finite() || self.normalization_factor <= 0.0 {
            return Err(AnomalyGridError::invalid_configuration(
                "normalization_factor",
                &self.normalization_factor.to_string(),
                "a positive finite number",
            ));
        }

        // Validate min_sequence_length
        if self.min_sequence_length < 2 {
            return Err(AnomalyGridError::invalid_configuration(
                "min_sequence_length",
                &self.min_sequence_length.to_string(),
                "at least 2",
            ));
        }

        // Validate memory_limit if set
        if let Some(limit) = self.memory_limit {
            if limit == 0 {
                return Err(AnomalyGridError::invalid_configuration(
                    "memory_limit",
                    "0",
                    "None (unlimited) or a positive integer",
                ));
            }
        }

        Ok(())
    }

    /// Set max_order with validation
    pub fn with_max_order(mut self, max_order: usize) -> AnomalyGridResult<Self> {
        if max_order == 0 {
            return Err(AnomalyGridError::invalid_max_order(max_order));
        }
        self.max_order = max_order;
        Ok(self)
    }

    /// Set smoothing_alpha with validation
    pub fn with_smoothing_alpha(mut self, alpha: f64) -> AnomalyGridResult<Self> {
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(AnomalyGridError::invalid_configuration(
                "smoothing_alpha",
                &alpha.to_string(),
                "a positive finite number",
            ));
        }
        self.smoothing_alpha = alpha;
        Ok(self)
    }

    /// Set memory_limit with validation
    pub fn with_memory_limit(mut self, limit: Option<usize>) -> AnomalyGridResult<Self> {
        if let Some(limit_val) = limit {
            if limit_val == 0 {
                return Err(AnomalyGridError::invalid_configuration(
                    "memory_limit",
                    "0",
                    "None (unlimited) or a positive integer",
                ));
            }
        }
        self.memory_limit = limit;
        Ok(self)
    }

    /// Set anomaly strength weights with validation
    pub fn with_weights(
        mut self,
        likelihood_weight: f64,
        information_weight: f64,
    ) -> AnomalyGridResult<Self> {
        if !likelihood_weight.is_finite() || likelihood_weight < 0.0 {
            return Err(AnomalyGridError::invalid_configuration(
                "likelihood_weight",
                &likelihood_weight.to_string(),
                "a non-negative finite number",
            ));
        }

        if !information_weight.is_finite() || information_weight < 0.0 {
            return Err(AnomalyGridError::invalid_configuration(
                "information_weight",
                &information_weight.to_string(),
                "a non-negative finite number",
            ));
        }

        let weight_sum = likelihood_weight + information_weight;
        if (weight_sum - 1.0).abs() > 1e-10 {
            return Err(AnomalyGridError::invalid_configuration(
                "weight_sum",
                &weight_sum.to_string(),
                "likelihood_weight + information_weight = 1.0",
            ));
        }

        self.likelihood_weight = likelihood_weight;
        self.information_weight = information_weight;
        Ok(self)
    }

    /// Get estimated memory usage for given alphabet size
    pub fn estimate_memory_usage(&self, alphabet_size: usize) -> usize {
        let mut total_contexts = 0;
        for order in 1..=self.max_order {
            total_contexts += alphabet_size.pow(order as u32);
        }

        // Apply memory limit if set
        if let Some(limit) = self.memory_limit {
            total_contexts.min(limit)
        } else {
            total_contexts
        }
    }

    /// Check if configuration is suitable for given alphabet size
    pub fn is_suitable_for_alphabet(&self, alphabet_size: usize) -> bool {
        // Calculate actual memory needed (without limit capping)
        let mut actual_contexts = 0;
        for order in 1..=self.max_order {
            actual_contexts += alphabet_size.pow(order as u32);
        }

        // Consider suitable if actual contexts fit within limits
        match self.memory_limit {
            Some(limit) => actual_contexts <= limit,
            None => actual_contexts <= 10_000_000, // Reasonable default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AnomalyGridConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.max_order, 3);
        assert_eq!(config.smoothing_alpha, 1.0);
        assert_eq!(config.memory_limit, Some(1_000_000));
    }

    #[test]
    fn test_preset_configs() {
        assert!(AnomalyGridConfig::for_small_alphabet().validate().is_ok());
        assert!(AnomalyGridConfig::for_large_alphabet().validate().is_ok());
        assert!(AnomalyGridConfig::for_low_memory().validate().is_ok());
        assert!(AnomalyGridConfig::for_high_accuracy().validate().is_ok());
    }

    #[test]
    fn test_invalid_max_order() {
        let result = AnomalyGridConfig::default().with_max_order(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_smoothing_alpha() {
        let result = AnomalyGridConfig::default().with_smoothing_alpha(-1.0);
        assert!(result.is_err());

        let result = AnomalyGridConfig::default().with_smoothing_alpha(f64::NAN);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_weights() {
        let result = AnomalyGridConfig::default().with_weights(0.5, 0.6); // Sum > 1
        assert!(result.is_err());

        let result = AnomalyGridConfig::default().with_weights(-0.1, 1.1); // Negative weight
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_estimation() {
        let config = AnomalyGridConfig::default();

        // For alphabet size 2, order 3: 2^1 + 2^2 + 2^3 = 2 + 4 + 8 = 14
        let estimated = config.estimate_memory_usage(2);
        assert_eq!(estimated, 14);

        // Test with memory limit
        let config_limited = AnomalyGridConfig::default()
            .with_memory_limit(Some(10))
            .unwrap();
        let estimated_limited = config_limited.estimate_memory_usage(2);
        assert_eq!(estimated_limited, 10); // Capped by limit
    }

    #[test]
    fn test_alphabet_suitability() {
        let config = AnomalyGridConfig::for_small_alphabet();
        assert!(config.is_suitable_for_alphabet(5));
        assert!(config.is_suitable_for_alphabet(10));

        let config = AnomalyGridConfig::for_large_alphabet();
        assert!(config.is_suitable_for_alphabet(50));

        let config = AnomalyGridConfig::for_low_memory();
        assert!(config.is_suitable_for_alphabet(3));

        // For low memory config with max_order=2 and memory_limit=10_000:
        // 100 states would need 100^1 + 100^2 = 100 + 10_000 = 10_100 contexts
        // This exceeds the 10_000 limit, so should be rejected
        let estimated = config.estimate_memory_usage(100);
        assert_eq!(estimated, 10_000); // Should be capped at limit
        assert!(!config.is_suitable_for_alphabet(100)); // Too large for low memory
    }

    #[test]
    fn test_config_validation() {
        let mut config = AnomalyGridConfig::default();

        // Valid config should pass
        assert!(config.validate().is_ok());

        // Invalid max_order
        config.max_order = 0;
        assert!(config.validate().is_err());
        config.max_order = 3;

        // Invalid smoothing_alpha
        config.smoothing_alpha = -1.0;
        assert!(config.validate().is_err());
        config.smoothing_alpha = 1.0;

        // Invalid weights
        config.likelihood_weight = 0.8;
        config.information_weight = 0.3; // Sum > 1
        assert!(config.validate().is_err());
    }
}
