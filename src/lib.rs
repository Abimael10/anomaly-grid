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

//! # Anomaly Grid
//!
//! Sequence anomaly detection over finite alphabets using **variable-order
//! Markov chains** with **Witten-Bell interpolation** and an
//! **information-theoretic** anomaly score.
//!
//! ## Problem
//!
//! Given a corpus of *known-normal* sequences over a finite alphabet
//! (HTTP verbs, syscalls, protocol states, codon triplets, …), score each
//! window of an unknown sequence by how *surprising* it is under that
//! corpus. Surprise has two components, both measured in **bits**:
//!
//! 1. **Average surprise** = `−log₂ P(window)` per symbol, where the
//!    conditional probabilities come from the variable-order Markov model.
//! 2. **Information score** = `mean(−log₂ P(xᵢ | context))` summed across
//!    the window.
//!
//! These are combined and squashed by `tanh` into an *anomaly strength*
//! ∈ \[0, 1):
//!
//! ```text
//! s = w_l · (−log₂ P) / (n−1)  +  w_i · I
//! anomaly_strength = tanh(s / normalization_factor)
//! ```
//!
//! ## Smoothing
//!
//! Conditional probability is the Witten-Bell interpolation:
//!
//! ```text
//! P_wb(x | c) = λ(c)·P_ml(x|c) + (1−λ(c))·P_wb(x | suffix(c))
//! λ(c) = N(c) / (N(c) + T(c))
//! ```
//!
//! where `N(c)` is the number of observations of context `c` and `T(c)`
//! is the number of distinct continuations seen. The order-0 base case
//! is Add-α (Laplace) over the global alphabet:
//! `P(x) = (count(x) + α) / (N + α·|Σ|)`.
//!
//! ## Quick start: train once, score many in parallel
//!
//! ```no_run
//! use anomaly_grid::{AnomalyDetector, batch_score};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut detector = AnomalyDetector::new(3)?;
//!
//! // Train on a corpus of known-normal sequences.
//! let normal: Vec<Vec<String>> = vec![
//!     ["LOGIN", "AUTH", "READ", "LOGOUT"].iter().map(|s| s.to_string()).collect(),
//!     ["LOGIN", "AUTH", "WRITE", "LOGOUT"].iter().map(|s| s.to_string()).collect(),
//! ];
//! detector.train_sequences(&normal)?;
//!
//! // Score a stream of unknown sequences in parallel using rayon.
//! let unknown: Vec<Vec<String>> = vec![
//!     ["LOGIN", "AUTH", "READ", "LOGOUT"].iter().map(|s| s.to_string()).collect(),
//!     ["LOGIN", "PRIV_ESCALATE", "WRITE", "LOGOUT"].iter().map(|s| s.to_string()).collect(),
//! ];
//! let results = batch_score(&detector, &unknown, 0.5)?;
//! for (seq_index, scores) in results.iter().enumerate() {
//!     for s in scores {
//!         println!("seq {seq_index}: window {:?} strength {:.3}", s.sequence, s.anomaly_strength);
//!     }
//! }
//! # Ok(()) }
//! ```
//!
//! ## Use cases
//!
//! - **Network/protocol intrusion**: trains on benign session traces (states like
//!   `SYN_SENT → ESTABLISHED → DATA_XFER → FIN_WAIT1 → CLOSED`), flags sessions
//!   that diverge (e.g. `RESET` mid-stream, skipped handshake).
//! - **Syscall trace monitoring**: trains on normal trace prefixes
//!   (`open → read → close`, `socket → connect → send`), flags windows whose
//!   pointwise surprise spikes — fileless malware and shell escapes typically
//!   produce locally improbable transitions.
//! - **Bioinformatics motif scanning**: trains on canonical reading frames
//!   (codon triplets), flags codon windows whose Markov likelihood is low —
//!   useful for spotting frameshifts and rare splice variants in known taxa.
//!
//! See `examples/` for runnable code.
//!
//! ## v0.5 → v0.6 migration
//!
//! - `batch_process_sequences(seqs, config, threshold)` is **removed**. The
//!   old function trained a fresh detector on every input then scored that
//!   same input — degenerate. Use [`batch_score`] with a pre-trained
//!   [`AnomalyDetector`] instead.
//! - `AnomalyScore::log_likelihood` is now the *average per-symbol log₂*
//!   conditional probability (bits), not the natural-log of the joint.
//! - `AnomalyScore::likelihood` is the geometric-mean conditional
//!   probability `exp2(log_likelihood)` ∈ \[0, 1\].

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

pub use anomaly_detector::{batch_score, AnomalyDetector, AnomalyScore};
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
        // VERSION is `env!("CARGO_PKG_VERSION")`, a compile-time non-empty
        // string. Assert basic shape (a digit and a dot) instead.
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
