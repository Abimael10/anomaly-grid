#![deny(
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used,
    clippy::expect_used
)]
#![allow(
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::missing_const_for_fn,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::doc_markdown
)]

//! Anomaly Grid - Sequential Pattern Analysis Library
//!
//! A focused library for anomaly detection in finite-alphabet sequences using
//! variable-order Markov chains with hierarchical context selection.
//!
//! This library provides pattern-based anomaly detection through
//! information-theoretic measures and probability estimation.
//!
//! # Features
//!
//! - **Variable-Order Markov Models**: Hierarchical context selection with Witten-Bell interpolation
//! - **Information Theory**: Shannon entropy, KL divergence
//! - **Hierarchical Context Selection**: Automatic fallback from longer to shorter contexts
//! - **Parallel Processing**: Batch analysis using Rayon for multiple sequences
//!
//! # Quick Start
//!
//! ```rust
//! use anomaly_grid::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut detector = AnomalyDetector::new(3)?;
//! let normal_sequence = vec![
//!     "A".to_string(), "B".to_string(), "C".to_string(),
//!     "A".to_string(), "B".to_string(), "C".to_string(),
//! ];
//! detector.train(&normal_sequence)?;
//!
//! let test_sequence = vec![
//!     "A".to_string(), "X".to_string(), "Y".to_string(),
//! ];
//! let anomalies = detector.detect_anomalies(&test_sequence, 0.1)?;
//! for anomaly in anomalies {
//!     println!("anomaly {:?}", anomaly.sequence);
//! }
//! # Ok(()) }
//! ```
//!
//! # Architecture
//!
//! - [`context_tree`]: context storage and probability estimation
//! - [`markov_model`]: variable-order Markov chain implementation
//! - [`anomaly_detector`]: anomaly detection over a trained model

pub mod anomaly_detector;
pub mod config;
pub mod context_tree;
pub mod error;
pub mod markov_model;
pub mod performance;

pub(crate) mod constants;
pub(crate) mod context_trie;
pub(crate) mod string_interner;
pub(crate) mod transition_counts;
pub(crate) mod validation;

pub use anomaly_detector::{batch_process_sequences, AnomalyDetector, AnomalyScore};
pub use config::AnomalyGridConfig;
pub use context_tree::{ContextNode, ContextTree};
pub use error::{AnomalyGridError, AnomalyGridResult};
pub use markov_model::MarkovModel;
pub use performance::{
    optimize_context_tree, ContextStatistics, OptimizationConfig, PerformanceMetrics,
};

/// Library version string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_string_matches_cargo() {
        // VERSION is `env!("CARGO_PKG_VERSION")`, a compile-time non-empty string.
        assert!(VERSION.contains('.'));
    }

    #[test]
    fn basic_workflow() -> AnomalyGridResult<()> {
        let mut detector = AnomalyDetector::new(2)?;
        let sequence: Vec<String> = ["A", "B", "A", "B"].iter().map(|s| (*s).to_string()).collect();
        detector.train(&sequence)?;

        let test_sequence: Vec<String> =
            ["A", "X", "Y"].iter().map(|s| (*s).to_string()).collect();
        let anomalies = detector.detect_anomalies(&test_sequence, 0.5)?;
        for anomaly in anomalies {
            assert!((0.0..=1.0).contains(&anomaly.likelihood));
            assert!((0.0..=1.0).contains(&anomaly.anomaly_strength));
        }
        Ok(())
    }

    #[test]
    fn module_integration() -> AnomalyGridResult<()> {
        let mut tree = ContextTree::new(2)?;
        let sequence: Vec<String> = ["A", "B", "C"].iter().map(|s| (*s).to_string()).collect();
        let config = AnomalyGridConfig::default();

        tree.build_from_sequence(&sequence, &config)?;
        assert!(tree.context_count() > 0);

        let mut model = MarkovModel::new(2)?;
        model.train(&sequence)?;
        let likelihood = model.calculate_likelihood(&sequence);
        assert!(likelihood > 0.0);
        assert!(likelihood <= 1.0);

        let mut detector = AnomalyDetector::new(2)?;
        detector.train(&sequence)?;
        let anomalies = detector.detect_anomalies(&sequence, 0.1)?;
        assert!(anomalies.len() <= 1);
        Ok(())
    }
}
