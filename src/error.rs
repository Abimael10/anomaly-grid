//! Error types for the Anomaly Grid library
//!
//! This module provides structured error handling with detailed context
//! and actionable error messages for all library operations.

/// Custom error types for Anomaly Grid operations
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyGridError {
    /// Sequence is too short for the requested operation
    SequenceTooShort {
        /// Minimum required sequence length
        expected: usize,
        /// Actual sequence length provided
        actual: usize,
        /// Context about what operation failed
        operation: String,
    },

    /// Invalid max_order parameter
    InvalidMaxOrder {
        /// The invalid value provided
        value: usize,
        /// Additional context about valid range
        context: String,
    },

    /// Invalid threshold parameter
    InvalidThreshold {
        /// The invalid threshold value
        value: f64,
        /// Expected range description
        expected_range: String,
    },

    /// Memory limit exceeded during context tree building
    MemoryLimitExceeded {
        /// Current number of contexts
        current: usize,
        /// Maximum allowed contexts
        limit: usize,
        /// Suggested action
        suggestion: String,
    },

    /// Context tree is empty (no training data)
    EmptyContextTree {
        /// Suggested action to resolve
        suggestion: String,
    },

    /// Invalid configuration parameter
    InvalidConfiguration {
        /// Parameter name
        parameter: String,
        /// Invalid value
        value: String,
        /// Expected format or range
        expected: String,
    },
}

impl std::fmt::Display for AnomalyGridError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnomalyGridError::SequenceTooShort {
                expected,
                actual,
                operation,
            } => {
                write!(
                    f,
                    "Sequence too short for {operation}: expected at least {expected} elements, got {actual}"
                )
            }
            AnomalyGridError::InvalidMaxOrder { value, context } => {
                write!(f, "Invalid max_order {value}: {context}")
            }
            AnomalyGridError::InvalidThreshold {
                value,
                expected_range,
            } => {
                write!(f, "Invalid threshold {value}: expected {expected_range}")
            }
            AnomalyGridError::MemoryLimitExceeded {
                current,
                limit,
                suggestion,
            } => {
                write!(
                    f,
                    "Memory limit exceeded: {current} contexts created, limit is {limit}. {suggestion}"
                )
            }
            AnomalyGridError::EmptyContextTree { suggestion } => {
                write!(
                    f,
                    "Context tree is empty: no training data processed. {suggestion}"
                )
            }
            AnomalyGridError::InvalidConfiguration {
                parameter,
                value,
                expected,
            } => {
                write!(
                    f,
                    "Invalid configuration for '{parameter}': got '{value}', expected {expected}"
                )
            }
        }
    }
}

impl std::error::Error for AnomalyGridError {}

/// Result type alias for Anomaly Grid operations
pub type AnomalyGridResult<T> = std::result::Result<T, AnomalyGridError>;

impl AnomalyGridError {
    /// Create a sequence too short error with context
    pub fn sequence_too_short(expected: usize, actual: usize, operation: &str) -> Self {
        Self::SequenceTooShort {
            expected,
            actual,
            operation: operation.to_string(),
        }
    }

    /// Create an invalid max_order error
    pub fn invalid_max_order(value: usize) -> Self {
        Self::InvalidMaxOrder {
            value,
            context: "max_order must be greater than 0".to_string(),
        }
    }

    /// Create an invalid threshold error
    pub fn invalid_threshold(value: f64) -> Self {
        Self::InvalidThreshold {
            value,
            expected_range: "a value between 0.0 and 1.0 (inclusive)".to_string(),
        }
    }

    /// Create a memory limit exceeded error
    pub fn memory_limit_exceeded(current: usize, limit: usize) -> Self {
        Self::MemoryLimitExceeded {
            current,
            limit,
            suggestion: "Consider reducing max_order, alphabet size, or increasing memory_limit"
                .to_string(),
        }
    }

    /// Create an empty context tree error
    pub fn empty_context_tree() -> Self {
        Self::EmptyContextTree {
            suggestion: "Call train() with a valid sequence before detection".to_string(),
        }
    }

    /// Create an invalid configuration error
    pub fn invalid_configuration(parameter: &str, value: &str, expected: &str) -> Self {
        Self::InvalidConfiguration {
            parameter: parameter.to_string(),
            value: value.to_string(),
            expected: expected.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_too_short_error() {
        let error = AnomalyGridError::sequence_too_short(5, 3, "training");
        let message = error.to_string();

        assert!(message.contains("training"));
        assert!(message.contains("expected at least 5"));
        assert!(message.contains("got 3"));
    }

    #[test]
    fn test_invalid_max_order_error() {
        let error = AnomalyGridError::invalid_max_order(0);
        let message = error.to_string();

        assert!(message.contains("Invalid max_order 0"));
        assert!(message.contains("must be greater than 0"));
    }

    #[test]
    fn test_invalid_threshold_error() {
        let error = AnomalyGridError::invalid_threshold(1.5);
        let message = error.to_string();

        assert!(message.contains("Invalid threshold 1.5"));
        assert!(message.contains("between 0.0 and 1.0"));
    }

    #[test]
    fn test_memory_limit_exceeded_error() {
        let error = AnomalyGridError::memory_limit_exceeded(150000, 100000);
        let message = error.to_string();

        assert!(message.contains("150000 contexts"));
        assert!(message.contains("limit is 100000"));
        assert!(message.contains("Consider reducing"));
    }

    #[test]
    fn test_empty_context_tree_error() {
        let error = AnomalyGridError::empty_context_tree();
        let message = error.to_string();

        assert!(message.contains("Context tree is empty"));
        assert!(message.contains("Call train()"));
    }

    #[test]
    fn test_error_equality() {
        let error1 = AnomalyGridError::invalid_max_order(0);
        let error2 = AnomalyGridError::invalid_max_order(0);
        let error3 = AnomalyGridError::invalid_max_order(1);

        assert_eq!(error1, error2);
        assert_ne!(error1, error3);
    }

    #[test]
    fn test_error_debug() {
        let error = AnomalyGridError::sequence_too_short(5, 3, "testing");
        let debug_str = format!("{error:?}");

        assert!(debug_str.contains("SequenceTooShort"));
        assert!(debug_str.contains("expected: 5"));
        assert!(debug_str.contains("actual: 3"));
    }
}
