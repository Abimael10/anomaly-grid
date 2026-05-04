//! Public-API smoke tests.
//!
//! Covers construction, training, configuration, performance metrics,
//! and optimisation. Mathematical properties live in `math.rs`;
//! detection-logic contract in `detection.rs`.

#![allow(clippy::float_cmp)]

mod common;
use common::{pattern_abc, s};

use anomaly_grid::{AnomalyDetector, AnomalyGridConfig, MarkovModel, OptimizationConfig};

/// Constructor accepts a positive `max_order` and rejects 0.
#[test]
fn detector_construction_validates_max_order() {
    assert!(AnomalyDetector::new(3).is_ok());
    assert!(AnomalyDetector::new(0).is_err());
    assert_eq!(AnomalyDetector::new(2).expect("ok").max_order(), 2);
}

/// `with_config` round-trips configuration values into the detector.
#[test]
fn detector_with_config_preserves_values() {
    let config = AnomalyGridConfig::default()
        .with_max_order(2)
        .expect("max_order")
        .with_smoothing_alpha(0.5)
        .expect("alpha");
    let detector = AnomalyDetector::with_config(config).expect("detector");
    assert_eq!(detector.max_order(), 2);
    assert_eq!(detector.model().config().smoothing_alpha, 0.5);
}

/// Training records non-trivial performance metrics.
#[test]
fn training_populates_performance_metrics() {
    let detector = pattern_abc(30, 2);
    let metrics = detector.performance_metrics();
    assert!(metrics.context_count > 0);
    assert!(metrics.estimated_memory_bytes > 0);
}

/// `detect_anomalies_with_monitoring` records detection time.
#[test]
fn detection_with_monitoring_records_time() {
    let mut detector = pattern_abc(30, 2);
    let _ = detector
        .detect_anomalies_with_monitoring(&s(&["X", "Y"]), 0.1)
        .expect("detect");
    assert!(detector.performance_metrics().detection_time_ms > 0);
}

/// `train_sequences` accepts multiple independent sequences.
#[test]
fn train_sequences_accepts_multiple_inputs() {
    let mut detector = AnomalyDetector::new(2).expect("detector");
    let sequences = vec![
        s(&["A", "B", "C"]),
        s(&["D", "E", "F"]),
        s(&["G", "H", "I"]),
    ];
    detector.train_sequences(&sequences).expect("train_sequences");
    assert!(detector.model().context_tree().context_count() > 0);
}

/// `train_sequences` rejects an empty input list.
#[test]
fn train_sequences_rejects_empty_input() {
    let mut detector = AnomalyDetector::new(2).expect("detector");
    let empty: Vec<Vec<String>> = vec![];
    assert!(detector.train_sequences(&empty).is_err());
}

/// `train_sequences` preserves earlier states' vocabulary across batches.
#[test]
fn train_sequences_preserves_vocabulary_across_batches() {
    let mut detector = AnomalyDetector::new(2).expect("detector");
    detector
        .train_sequences(&[s(&["A", "B", "A"]), s(&["B", "C", "B"])])
        .expect("train");

    let model = detector.model();
    assert!(model.state_mapping().contains_key("A"));
    assert!(model.state_mapping().contains_key("C"));
    assert!(model.get_best_context_probability(&s(&["A"]), "B") > 0.0);
}

/// Optimisation does not increase the context count and detection still
/// works after pruning.
#[test]
fn optimize_reduces_or_maintains_context_count() {
    let mut detector = AnomalyDetector::new(3).expect("detector");
    let mut training: Vec<String> = Vec::new();
    for i in 0..100 {
        training.push(format!("S{}", i % 10));
        training.push(format!("N{}", (i + 1) % 10));
    }
    detector.train(&training).expect("train");

    let initial = detector.performance_metrics().context_count;
    detector
        .optimize(&OptimizationConfig {
            enable_pruning: true,
            min_context_count: 2,
            min_entropy: 0.1,
            max_contexts: Some(50),
            enable_monitoring: true,
        })
        .expect("optimize");
    let after = detector.performance_metrics().context_count;
    assert!(after <= initial, "after={after} > initial={initial}");

    let scores = detector
        .detect_anomalies(&s(&["S0", "N1"]), 0.1)
        .expect("detect");
    for score in &scores {
        assert!(score.likelihood.is_finite());
        assert!(score.anomaly_strength.is_finite());
    }
}

/// `MarkovModel` standalone can be constructed and trained.
#[test]
fn markov_model_standalone_workflow() {
    let mut model = MarkovModel::new(2).expect("model");
    model.train(&s(&["A", "B", "C", "A", "B", "C"])).expect("train");

    assert!(!model.state_mapping().is_empty());
    let p = model.calculate_likelihood(&s(&["A", "B"]));
    assert!((0.0..=1.0).contains(&p));
}

/// Configuration access roundtrips through `MarkovModel`.
#[test]
fn markov_model_exposes_config() {
    let cfg = AnomalyGridConfig::default()
        .with_smoothing_alpha(2.0)
        .expect("alpha");
    let model = MarkovModel::with_config(cfg).expect("model");
    assert_eq!(model.config().smoothing_alpha, 2.0);
}
