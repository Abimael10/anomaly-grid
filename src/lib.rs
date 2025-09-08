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
//! - **Variable-Order Markov Models**: Hierarchical context selection with Laplace smoothing
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
//! // Create and train detector
//! let mut detector = AnomalyDetector::new(3)?;
//! let normal_sequence = vec![
//!     "A".to_string(), "B".to_string(), "C".to_string(),
//!     "A".to_string(), "B".to_string(), "C".to_string(),
//! ];
//! detector.train(&normal_sequence)?;
//!
//! // Detect anomalies
//! let test_sequence = vec![
//!     "A".to_string(), "X".to_string(), "Y".to_string(),
//! ];
//! let anomalies = detector.detect_anomalies(&test_sequence, 0.1)?;
//!
//! for anomaly in anomalies {
//!     println!("Anomaly: {:?}, Likelihood: {:.6}",
//!              anomaly.sequence, anomaly.likelihood);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Architecture
//!
//! The library is organized into three main modules:
//!
//! - [`context_tree`]: Context storage and probability estimation
//! - [`markov_model`]: Variable-order Markov chain implementation  
//! - [`anomaly_detector`]: Anomaly detection using Markov models
//!
//! # Use Cases
//!
//! - **Network Security**: Detecting unusual protocol sequences and attack patterns
//! - **User Behavior Analysis**: Identifying privilege escalation and suspicious activities
//! - **Financial Fraud**: Detecting unusual transaction patterns and velocity attacks
//! - **System Monitoring**: Identifying anomalous log sequences and security incidents
//! - **Bioinformatics**: Detecting mutations and unusual genetic sequences

pub mod anomaly_detector;
pub mod config;
pub mod constants;
pub mod context_tree;
pub mod error;
pub mod markov_model;
pub mod performance;
pub mod string_interner;
//pub mod collection_analysis;
pub mod transition_counts;

// Re-export main types for convenience
pub use anomaly_detector::{batch_process_sequences, AnomalyDetector, AnomalyScore};
pub use config::AnomalyGridConfig;
pub use context_tree::{ContextNode, ContextTree};
pub use error::{AnomalyGridError, AnomalyGridResult};
pub use markov_model::MarkovModel;
pub use performance::{
    optimize_context_tree, ContextStatistics, OptimizationConfig, PerformanceMetrics,
};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Get library information
pub fn info() -> String {
    format!("Anomaly Grid v{VERSION} - Markov Chain-based Sequence Anomaly Detection")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_info() {
        let info = info();
        assert!(info.contains("Anomaly Grid"));
        assert!(info.contains(VERSION));
    }

    #[test]
    fn test_basic_workflow() {
        let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
        let sequence = vec![
            "A".to_string(),
            "B".to_string(),
            "A".to_string(),
            "B".to_string(),
        ];

        // Training should succeed
        assert!(detector.train(&sequence).is_ok());

        // Detection should work
        let test_sequence = vec!["A".to_string(), "X".to_string(), "Y".to_string()];
        let anomalies = detector
            .detect_anomalies(&test_sequence, 0.5)
            .expect("Failed to detect anomalies");

        // Should detect some anomalies or handle gracefully
        for anomaly in anomalies {
            assert!(anomaly.likelihood >= 0.0);
            assert!(anomaly.likelihood <= 1.0);
            assert!(anomaly.anomaly_strength >= 0.0);
            assert!(anomaly.anomaly_strength <= 1.0);
        }
    }

    #[test]
    fn test_module_integration() {
        // Test that all modules work together
        let mut tree = ContextTree::new(2).expect("Failed to create context tree");
        let sequence = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let config = AnomalyGridConfig::default();

        assert!(tree.build_from_sequence(&sequence, &config).is_ok());
        assert!(!tree.contexts.is_empty());

        let mut model = MarkovModel::new(2).expect("Failed to create Markov model");
        assert!(model.train(&sequence).is_ok());

        let likelihood = model.calculate_likelihood(&sequence);
        assert!(likelihood > 0.0);
        assert!(likelihood <= 1.0);

        let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
        assert!(detector.train(&sequence).is_ok());

        let anomalies = detector
            .detect_anomalies(&sequence, 0.1)
            .expect("Failed to detect anomalies");
        // Normal sequence should have few anomalies
        assert!(anomalies.len() <= 1);
    }
}
