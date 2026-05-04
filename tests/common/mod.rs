//! Shared fixtures and helpers for integration tests.
//!
//! Each file in `tests/` is its own test binary, so common setup is
//! centralised here and pulled in via `mod common;`. Cargo recognises
//! `tests/common/mod.rs` as a module rather than a separate test binary.

#![allow(dead_code)] // each test file uses a different subset

use anomaly_grid::AnomalyDetector;

/// Convenience: `["A", "B"] → vec!["A".to_string(), "B".to_string()]`.
pub fn s(strs: &[&str]) -> Vec<String> {
    strs.iter().map(|x| (*x).to_string()).collect()
}

/// Detector trained on `units` repeated `reps` times at the given order.
/// Panics on failure — these are tests.
#[allow(clippy::missing_panics_doc)]
pub fn trained(units: &[&str], reps: usize, order: usize) -> AnomalyDetector {
    let mut detector = AnomalyDetector::new(order).expect("detector");
    let mut training = Vec::with_capacity(units.len() * reps);
    for _ in 0..reps {
        training.extend(units.iter().map(|s| (*s).to_string()));
    }
    detector.train(&training).expect("train");
    detector
}

/// Detector trained on the canonical "A, B, C" repeating pattern.
#[allow(clippy::missing_panics_doc)]
pub fn pattern_abc(reps: usize, order: usize) -> AnomalyDetector {
    trained(&["A", "B", "C"], reps, order)
}

/// Maximum anomaly strength found by scoring `seq` against `detector`.
#[allow(clippy::missing_panics_doc)]
pub fn max_strength(detector: &AnomalyDetector, seq: &[String]) -> f64 {
    detector
        .detect_anomalies(seq, 0.0)
        .expect("detect")
        .iter()
        .map(|s| s.anomaly_strength)
        .fold(0.0_f64, f64::max)
}
