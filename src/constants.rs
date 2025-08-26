//! Constants and default values for Anomaly Grid
//!
//! This module centralizes all magic numbers and default parameters,
//! making them easily configurable and well-documented.

/// Performance-related constants
pub mod performance {
    /// Default Laplace smoothing parameter (alpha)
    ///
    /// This value provides a good balance between smoothing and preserving
    /// the original distribution. Higher values increase smoothing.
    pub const DEFAULT_SMOOTHING_ALPHA: f64 = 1.0;

    /// Default likelihood weight in anomaly strength calculation
    ///
    /// This determines how much the likelihood component contributes
    /// to the final anomaly strength score.
    pub const DEFAULT_LIKELIHOOD_WEIGHT: f64 = 0.7;

    /// Default information weight in anomaly strength calculation
    ///
    /// This determines how much the information content contributes
    /// to the final anomaly strength score.
    pub const DEFAULT_INFORMATION_WEIGHT: f64 = 0.3;

    /// Default normalization factor for tanh scaling
    ///
    /// This factor controls the steepness of the tanh normalization
    /// in anomaly strength calculation.
    pub const DEFAULT_NORMALIZATION_FACTOR: f64 = 10.0;

    /// Default minimum probability for numerical stability
    ///
    /// This prevents division by zero and log(0) operations.
    /// Should be much smaller than any realistic probability.
    pub const DEFAULT_MIN_PROBABILITY: f64 = 1e-12;

    /// Default memory limit (number of contexts)
    ///
    /// This prevents excessive memory usage in pathological cases.
    /// Set to None in config to disable the limit.
    pub const DEFAULT_MEMORY_LIMIT: usize = 1_000_000;

    /// Default maximum context order
    ///
    /// This provides a good balance between model expressiveness
    /// and computational efficiency for most use cases.
    pub const DEFAULT_MAX_ORDER: usize = 3;

    /// Default minimum sequence length for training
    ///
    /// Sequences shorter than this cannot provide meaningful
    /// transition information for Markov models.
    pub const DEFAULT_MIN_SEQUENCE_LENGTH: usize = 2;
}

/// Numerical precision constants
pub mod precision {
    /// Epsilon for floating-point comparisons
    ///
    /// Used for comparing probabilities and other floating-point values
    /// where exact equality is not reliable.
    pub const EPSILON: f64 = 1e-10;

    /// Relative epsilon for floating-point comparisons
    ///
    /// Used when the magnitude of values being compared varies significantly.
    pub const RELATIVE_EPSILON: f64 = 1e-9;

    /// Maximum allowed probability (slightly less than 1.0)
    ///
    /// Prevents numerical issues when probabilities are very close to 1.0.
    pub const MAX_PROBABILITY: f64 = 1.0 - 1e-15;

    /// Minimum entropy value (effectively zero)
    ///
    /// Used to handle numerical precision issues in entropy calculations.
    pub const MIN_ENTROPY: f64 = 1e-15;
}

/// Memory and performance limits
pub mod limits {
    /// Maximum recommended alphabet size for default configuration
    ///
    /// Beyond this size, consider using specialized configuration
    /// or reducing max_order to control memory usage.
    pub const MAX_RECOMMENDED_ALPHABET_SIZE: usize = 50;

    /// Maximum recommended context order for large alphabets
    ///
    /// For alphabets larger than MAX_RECOMMENDED_ALPHABET_SIZE,
    /// this order helps control exponential memory growth.
    pub const MAX_ORDER_FOR_LARGE_ALPHABET: usize = 2;

    /// Maximum recommended context order for small alphabets
    ///
    /// For small alphabets (≤ 10 states), higher orders can be used
    /// without excessive memory consumption.
    pub const MAX_ORDER_FOR_SMALL_ALPHABET: usize = 6;

    /// Threshold for considering an alphabet "small"
    ///
    /// Alphabets with this many or fewer unique states are considered
    /// small and can use higher context orders efficiently.
    pub const SMALL_ALPHABET_THRESHOLD: usize = 10;

    /// Threshold for considering an alphabet "large"
    ///
    /// Alphabets with more than this many unique states are considered
    /// large and should use lower context orders.
    pub const LARGE_ALPHABET_THRESHOLD: usize = 20;

    /// Maximum contexts for low-memory environments
    ///
    /// Suitable for embedded systems or memory-constrained applications.
    pub const LOW_MEMORY_CONTEXT_LIMIT: usize = 10_000;

    /// Maximum contexts for high-accuracy applications
    ///
    /// Allows larger memory usage for applications where accuracy
    /// is more important than memory efficiency.
    pub const HIGH_ACCURACY_CONTEXT_LIMIT: usize = 5_000_000;
}

/// Validation constants
pub mod validation {
    /// Minimum valid threshold value
    ///
    /// Thresholds below this value are considered invalid.
    pub const MIN_THRESHOLD: f64 = 0.0;

    /// Maximum valid threshold value
    ///
    /// Thresholds above this value are considered invalid.
    pub const MAX_THRESHOLD: f64 = 1.0;

    /// Minimum valid smoothing alpha
    ///
    /// Smoothing parameters below this value are considered invalid.
    pub const MIN_SMOOTHING_ALPHA: f64 = 1e-10;

    /// Maximum reasonable smoothing alpha
    ///
    /// Very large smoothing parameters may indicate configuration errors.
    pub const MAX_SMOOTHING_ALPHA: f64 = 1000.0;

    /// Minimum valid max_order
    ///
    /// Context orders below this value are invalid.
    pub const MIN_MAX_ORDER: usize = 1;

    /// Maximum reasonable max_order
    ///
    /// Very high orders may indicate configuration errors and will
    /// consume excessive memory.
    pub const MAX_REASONABLE_MAX_ORDER: usize = 10;
}

/// Information theory constants
pub mod information {
    /// Base for logarithms in information calculations
    ///
    /// Using base 2 gives information content in bits.
    pub const LOG_BASE: f64 = 2.0;

    /// Maximum reasonable information content
    ///
    /// Information content above this value may indicate numerical issues.
    pub const MAX_INFORMATION_CONTENT: f64 = 100.0;

    /// Minimum meaningful entropy difference
    ///
    /// Entropy differences below this value are considered negligible.
    pub const MIN_ENTROPY_DIFFERENCE: f64 = 1e-6;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_consistency() {
        // Weights should sum to 1.0
        let weight_sum =
            performance::DEFAULT_LIKELIHOOD_WEIGHT + performance::DEFAULT_INFORMATION_WEIGHT;
        assert!((weight_sum - 1.0).abs() < precision::EPSILON);
    }

    #[test]
    fn test_probability_bounds() {
        // Test probability bounds are consistent
        // These are compile-time constant checks for documentation
        let min_prob = performance::DEFAULT_MIN_PROBABILITY;
        let max_prob = precision::MAX_PROBABILITY;
        assert!(min_prob < max_prob);
        assert!(min_prob > 0.0);
        assert!(max_prob < 1.0);
    }

    #[test]
    fn test_threshold_bounds() {
        // Test threshold bounds are valid
        let min_thresh = validation::MIN_THRESHOLD;
        let max_thresh = validation::MAX_THRESHOLD;
        assert!(min_thresh >= 0.0);
        assert!(max_thresh <= 1.0);
        assert!(min_thresh < max_thresh);
    }

    #[test]
    fn test_alphabet_size_thresholds() {
        // Test alphabet size thresholds are ordered correctly
        let small = limits::SMALL_ALPHABET_THRESHOLD;
        let large = limits::LARGE_ALPHABET_THRESHOLD;
        let max_rec = limits::MAX_RECOMMENDED_ALPHABET_SIZE;
        assert!(small < large);
        assert!(large < max_rec);
    }

    #[test]
    fn test_memory_limits() {
        // Test memory limits are ordered correctly
        let low = limits::LOW_MEMORY_CONTEXT_LIMIT;
        let default = performance::DEFAULT_MEMORY_LIMIT;
        let high = limits::HIGH_ACCURACY_CONTEXT_LIMIT;
        assert!(low < default);
        assert!(default < high);
    }

    #[test]
    fn test_order_limits() {
        // Test order limits are valid
        let min_order = validation::MIN_MAX_ORDER;
        let max_large = limits::MAX_ORDER_FOR_LARGE_ALPHABET;
        let max_small = limits::MAX_ORDER_FOR_SMALL_ALPHABET;
        let max_reasonable = validation::MAX_REASONABLE_MAX_ORDER;
        assert!(min_order >= 1);
        assert!(max_large < max_small);
        assert!(max_small <= max_reasonable);
    }

    #[test]
    fn test_precision_constants() {
        // Test precision constants are ordered correctly
        let epsilon = precision::EPSILON;
        let min_entropy = precision::MIN_ENTROPY;
        let rel_epsilon = precision::RELATIVE_EPSILON;
        let max_prob = precision::MAX_PROBABILITY;
        assert!(epsilon > min_entropy);
        assert!(rel_epsilon > epsilon);
        assert!(max_prob > 0.99); // Should be very close to 1.0
    }

    #[test]
    fn test_smoothing_bounds() {
        // Test smoothing bounds are valid
        let min_alpha = validation::MIN_SMOOTHING_ALPHA;
        let max_alpha = validation::MAX_SMOOTHING_ALPHA;
        let default_alpha = performance::DEFAULT_SMOOTHING_ALPHA;
        assert!(min_alpha > 0.0);
        assert!(max_alpha > default_alpha);
        assert!(default_alpha >= min_alpha);
        assert!(default_alpha <= max_alpha);
    }
}
