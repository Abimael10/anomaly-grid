//! Error types for the Anomaly Grid library.
//!
//! All fallible operations return [`AnomalyGridResult`]. Errors are
//! constructed via the inherent helper methods so call-sites remain
//! terse while [`thiserror`] generates the `Display`/`Error` plumbing.

use thiserror::Error;

/// Library-wide result alias.
pub type AnomalyGridResult<T> = core::result::Result<T, AnomalyGridError>;

/// Structured error type for Anomaly Grid.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum AnomalyGridError {
    #[error("Sequence too short for {operation}: expected at least {expected} elements, got {actual}")]
    SequenceTooShort {
        expected: usize,
        actual: usize,
        operation: String,
    },

    #[error("Invalid max_order {value}: {context}")]
    InvalidMaxOrder { value: usize, context: String },

    #[error("Invalid threshold {value}: expected {expected_range}")]
    InvalidThreshold { value: f64, expected_range: String },

    #[error("Memory limit exceeded: {current} contexts created, limit is {limit}. {suggestion}")]
    MemoryLimitExceeded {
        current: usize,
        limit: usize,
        suggestion: String,
    },

    #[error("Context tree is empty: no training data processed. {suggestion}")]
    EmptyContextTree { suggestion: String },

    #[error("Invalid configuration for '{parameter}': got '{value}', expected {expected}")]
    InvalidConfiguration {
        parameter: String,
        value: String,
        expected: String,
    },

    /// Catch-all for violated internal invariants (e.g. an index that the
    /// module itself just produced going out of range). These indicate a
    /// bug in the library, not in the caller.
    #[error("Internal invariant violated: {0}")]
    Internal(&'static str),
}

impl AnomalyGridError {
    pub fn sequence_too_short(expected: usize, actual: usize, operation: &str) -> Self {
        Self::SequenceTooShort {
            expected,
            actual,
            operation: operation.to_string(),
        }
    }

    pub fn invalid_max_order(value: usize) -> Self {
        Self::InvalidMaxOrder {
            value,
            context: "max_order must be greater than 0".to_string(),
        }
    }

    pub fn invalid_threshold(value: f64) -> Self {
        Self::InvalidThreshold {
            value,
            expected_range: "a value between 0.0 and 1.0 (inclusive)".to_string(),
        }
    }

    pub fn memory_limit_exceeded(current: usize, limit: usize) -> Self {
        Self::MemoryLimitExceeded {
            current,
            limit,
            suggestion: "Consider reducing max_order, alphabet size, or increasing memory_limit"
                .to_string(),
        }
    }

    pub fn empty_context_tree() -> Self {
        Self::EmptyContextTree {
            suggestion: "Call train() with a valid sequence before detection".to_string(),
        }
    }

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
    fn sequence_too_short_display() {
        let msg = AnomalyGridError::sequence_too_short(5, 3, "training").to_string();
        assert!(msg.contains("training"));
        assert!(msg.contains("expected at least 5"));
        assert!(msg.contains("got 3"));
    }

    #[test]
    fn invalid_max_order_display() {
        let msg = AnomalyGridError::invalid_max_order(0).to_string();
        assert!(msg.contains("Invalid max_order 0"));
        assert!(msg.contains("must be greater than 0"));
    }

    #[test]
    fn invalid_threshold_display() {
        let msg = AnomalyGridError::invalid_threshold(1.5).to_string();
        assert!(msg.contains("Invalid threshold 1.5"));
        assert!(msg.contains("between 0.0 and 1.0"));
    }

    #[test]
    fn memory_limit_display() {
        let msg = AnomalyGridError::memory_limit_exceeded(150_000, 100_000).to_string();
        assert!(msg.contains("150000 contexts"));
        assert!(msg.contains("limit is 100000"));
        assert!(msg.contains("Consider reducing"));
    }

    #[test]
    fn empty_context_tree_display() {
        let msg = AnomalyGridError::empty_context_tree().to_string();
        assert!(msg.contains("Context tree is empty"));
        assert!(msg.contains("Call train()"));
    }

    #[test]
    fn equality_and_debug() {
        let e1 = AnomalyGridError::invalid_max_order(0);
        let e2 = AnomalyGridError::invalid_max_order(0);
        let e3 = AnomalyGridError::invalid_max_order(1);
        assert_eq!(e1, e2);
        assert_ne!(e1, e3);
        let dbg = format!("{:?}", AnomalyGridError::sequence_too_short(5, 3, "t"));
        assert!(dbg.contains("SequenceTooShort"));
    }
}
