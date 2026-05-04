//! Sequence-level behaviour: context windows, temporal dependencies,
//! pattern recognition, length handling, and alphabet scaling.

#![allow(clippy::float_cmp)]

mod common;
use common::{max_strength, pattern_abc, s};

use anomaly_grid::AnomalyDetector;

/// At order N, only the last N symbols of context affect predictions.
#[test]
fn context_window_truncates_to_max_order() {
    let mut detector = AnomalyDetector::new(2).expect("detector");
    let mut training = Vec::new();
    for _ in 0..50 {
        training.extend(s(&["A", "B", "C", "D"]));
    }
    detector.train(&training).expect("train");

    let with_2 = detector
        .model()
        .get_best_context_probability(&s(&["B", "C"]), "D");
    let with_5 = detector
        .model()
        .get_best_context_probability(&s(&["X", "Y", "Z", "B", "C"]), "D");
    assert!(
        (with_2 - with_5).abs() < 1e-9,
        "order-2 model used >2 symbols of context: {with_2} vs {with_5}"
    );
}

/// Higher-order contexts capture longer dependencies. For pattern
/// `ABCD ABCD …`, both `P(D|C)` (order 1) and `P(D|A,B,C)` (order 3)
/// approach 1 at sufficient training volume.
#[test]
fn temporal_dependencies_at_increasing_order() {
    let mut training = Vec::new();
    for _ in 0..100 {
        training.extend(s(&["A", "B", "C", "D"]));
    }

    let mut o1 = AnomalyDetector::new(1).expect("o1");
    o1.train(&training).expect("train1");
    let p_o1 = o1.model().get_best_context_probability(&s(&["C"]), "D");

    let mut o3 = AnomalyDetector::new(3).expect("o3");
    o3.train(&training).expect("train3");
    let p_o3 = o3
        .model()
        .get_best_context_probability(&s(&["A", "B", "C"]), "D");

    assert!(p_o1 > 0.9, "order-1 P(D|C) = {p_o1}");
    assert!(p_o3 > 0.9, "order-3 P(D|ABC) = {p_o3}");
}

/// Repeated patterns are recognised: trained detector flags a sequence
/// that breaks the pattern but accepts one that respects it.
#[test]
fn pattern_recognition_separates_normal_from_anomalous() {
    let detector = pattern_abc(40, 2);
    let normal = max_strength(&detector, &s(&["A", "B", "C", "A", "B", "C"]));
    let anomalous = max_strength(&detector, &s(&["A", "B", "C", "Z", "Y", "X"]));
    assert!(anomalous > normal, "anomalous={anomalous} not > normal={normal}");
}

/// Detector handles a 16-symbol alphabet without exceeding default
/// memory limits and produces finite scores.
#[test]
fn wide_alphabet_is_handled() {
    let alphabet: Vec<String> = (0..16).map(|i| format!("S{i}")).collect();
    let mut training = Vec::new();
    for i in 0..200 {
        training.push(alphabet[i % 16].clone());
    }

    let mut detector = AnomalyDetector::new(2).expect("detector");
    detector.train(&training).expect("train");

    let test: Vec<String> = (0..20).map(|i| alphabet[(i * 3) % 16].clone()).collect();
    let scores = detector.detect_anomalies(&test, 0.0).expect("detect");
    for score in &scores {
        assert!(score.anomaly_strength.is_finite());
        assert!(score.likelihood.is_finite());
    }
    assert!(detector.context_statistics().total_contexts > 0);
}

/// `train_sequences` keeps multiple sequences independent — no
/// transitions are learned across boundaries.
#[test]
fn train_sequences_does_not_cross_boundaries() {
    let mut detector = AnomalyDetector::new(2).expect("detector");
    detector
        .train_sequences(&[
            s(&["A", "B", "C", "A", "B", "C"]),
            s(&["X", "Y", "Z", "X", "Y", "Z"]),
        ])
        .expect("train");

    // Cross-sequence "C → X" transition should NOT be observed; "C → A"
    // (in-sequence) should dominate.
    let p_x_given_c = detector.model().get_best_context_probability(&s(&["C"]), "X");
    let p_a_given_c = detector.model().get_best_context_probability(&s(&["C"]), "A");
    assert!(
        p_a_given_c > p_x_given_c,
        "in-sequence P(A|C)={p_a_given_c} not > cross-sequence P(X|C)={p_x_given_c}"
    );
}

/// Long-sequence training and scoring stays numerically stable.
#[test]
fn long_sequence_score_stays_finite() {
    let mut detector = AnomalyDetector::new(4).expect("detector");
    let train: Vec<String> = (0..10_000).map(|i| format!("S{}", i % 5)).collect();
    detector.train(&train).expect("train");

    let test: Vec<String> = (0..1_000).map(|i| format!("S{}", i % 5)).collect();
    let scores = detector.detect_anomalies(&test, 0.0).expect("detect");
    for score in &scores {
        assert!(
            score.anomaly_strength.is_finite() && (0.0..=1.0).contains(&score.anomaly_strength),
            "anomaly_strength = {} on long sequence",
            score.anomaly_strength
        );
        assert!(
            score.information_score.is_finite() && score.information_score >= 0.0
        );
    }
}
