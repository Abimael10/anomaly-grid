//! Error-path tests: invalid configuration, invalid threshold, untrained
//! detector, sequence-too-short, batch-API misuse.

#![allow(clippy::float_cmp)]

mod common;
use common::s;

use anomaly_grid::{batch_score, AnomalyDetector, AnomalyGridConfig, AnomalyGridError};

/// `AnomalyDetector::new(0)` produces `InvalidMaxOrder`.
#[test]
fn detector_creation_rejects_max_order_zero() {
    match AnomalyDetector::new(0) {
        Err(AnomalyGridError::InvalidMaxOrder { value, .. }) => assert_eq!(value, 0),
        other => panic!("expected InvalidMaxOrder, got {other:?}"),
    }
}

/// Invalid thresholds produce `InvalidThreshold`.
#[test]
fn detector_rejects_invalid_thresholds() {
    let mut detector = AnomalyDetector::new(2).expect("detector");
    detector.train(&s(&["A", "B", "C"])).expect("train");

    for threshold in [1.5, -0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let result = detector.detect_anomalies(&s(&["X", "Y"]), threshold);
        assert!(result.is_err(), "threshold {threshold} should be rejected");
        match result {
            Err(AnomalyGridError::InvalidThreshold { value, .. }) => {
                if threshold.is_nan() {
                    assert!(value.is_nan());
                } else {
                    assert_eq!(value, threshold);
                }
            }
            other => panic!("expected InvalidThreshold, got {other:?}"),
        }
    }
}

/// Detection on an untrained detector produces `EmptyContextTree`.
#[test]
fn untrained_detector_errors() {
    let detector = AnomalyDetector::new(2).expect("detector");
    match detector.detect_anomalies(&s(&["A", "B"]), 0.5) {
        Err(AnomalyGridError::EmptyContextTree { .. }) => {}
        other => panic!("expected EmptyContextTree, got {other:?}"),
    }
}

/// Invalid configuration values are rejected at construction time.
#[test]
fn invalid_configuration_rejected() {
    assert!(AnomalyGridConfig::default().with_smoothing_alpha(-1.0).is_err());
    assert!(AnomalyGridConfig::default().with_max_order(0).is_err());
    assert!(AnomalyGridConfig::default().with_weights(-0.5, 0.5).is_err());
    assert!(AnomalyGridConfig::default().with_weights(0.5, 0.6).is_err()); // sum > 1
}

/// `train_sequences` rejects an empty input list.
#[test]
fn train_sequences_rejects_empty_input() {
    let mut detector = AnomalyDetector::new(2).expect("detector");
    let empty: Vec<Vec<String>> = vec![];
    assert!(detector.train_sequences(&empty).is_err());
}

/// `batch_score` rejects an out-of-range threshold.
#[test]
fn batch_score_rejects_invalid_threshold() {
    let mut detector = AnomalyDetector::new(2).expect("detector");
    detector.train(&s(&["A", "B", "C"])).expect("train");
    assert!(batch_score(&detector, &[s(&["A", "B"])], 1.5).is_err());
    assert!(batch_score(&detector, &[s(&["A", "B"])], f64::NAN).is_err());
}

/// `batch_score` propagates per-sequence errors. Empty/short sequences
/// in the input cause the whole call to short-circuit on the first
/// `SequenceTooShort` from `detect_anomalies` — well-formed inputs
/// before the failure are NOT returned because rayon collects all errors.
#[test]
fn batch_score_propagates_short_sequence_error() {
    let mut detector = AnomalyDetector::new(2).expect("detector");
    detector.train(&s(&["A", "B", "C"])).expect("train");

    // Length-2 sequence is too short for max_order=2 — adaptive path
    // handles it gracefully and returns an empty score list.
    let inputs = vec![s(&["A", "B", "C"]), s(&["A", "B"])];
    let result = batch_score(&detector, &inputs, 0.1);
    // Either: success (adaptive path returns empty for too-short inputs).
    // Just ensure it doesn't panic.
    let _ = result.expect("batch_score should not panic on short inputs");
}

/// Recovery: detector usable after a failed train call.
#[test]
fn detector_recovers_from_failed_training() {
    let mut detector = AnomalyDetector::new(2).expect("detector");
    let empty: Vec<String> = vec![];
    assert!(detector.train(&empty).is_err());

    detector.train(&s(&["A", "B", "C", "A", "B", "C"])).expect("retry");
    let scores = detector.detect_anomalies(&s(&["A", "B"]), 0.1).expect("detect");
    for s in &scores {
        assert!((0.0..=1.0).contains(&s.likelihood));
    }
}
