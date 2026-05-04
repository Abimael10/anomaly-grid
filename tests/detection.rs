//! Anomaly detection contract.
//!
//! Trains on a known-normal pattern, scores both normal and degraded
//! sequences, and pins the user-visible behaviour: scores ∈ [0, 1],
//! anomalies surface above thresholds, monotonicity, etc.

#![allow(clippy::float_cmp)]

mod common;
use common::{max_strength, pattern_abc, s};

use anomaly_grid::AnomalyDetector;

/// Every score lies in `[0, 1]` and `information_score ≥ 0`.
#[test]
fn scores_are_in_unit_interval() {
    let detector = pattern_abc(60, 2);
    let scores = detector
        .detect_anomalies(&s(&["A", "B", "C", "A", "X", "Y"]), 0.0)
        .expect("detect");
    assert!(!scores.is_empty(), "no scores produced");
    for score in &scores {
        assert!((0.0..=1.0).contains(&score.likelihood));
        assert!((0.0..=1.0).contains(&score.anomaly_strength));
        assert!(score.information_score >= 0.0);
        assert!(score.likelihood.is_finite());
        assert!(score.anomaly_strength.is_finite());
    }
}

/// `log_likelihood = ln(likelihood)` for non-zero likelihoods;
/// `−∞` otherwise.
#[test]
fn log_likelihood_matches_likelihood() {
    let detector = pattern_abc(40, 2);
    let scores = detector
        .detect_anomalies(&s(&["A", "B", "C", "X", "Y"]), 0.0)
        .expect("detect");
    for score in &scores {
        if score.likelihood > 0.0 {
            assert!(
                (score.log_likelihood - score.likelihood.ln()).abs() < 1e-10,
                "log_likelihood inconsistent: {} vs ln({}) = {}",
                score.log_likelihood,
                score.likelihood,
                score.likelihood.ln()
            );
        } else {
            assert!(
                score.log_likelihood.is_infinite() && score.log_likelihood < 0.0,
                "log_likelihood should be -∞ when likelihood = 0"
            );
        }
    }
}

/// A normal A→B→C window scores lower than a window with unseen symbols.
#[test]
fn rare_symbols_score_higher_than_normal() {
    let detector = pattern_abc(80, 2);
    let normal = max_strength(&detector, &s(&["A", "B", "C"]));
    let rare = max_strength(&detector, &s(&["A", "X", "Y"]));
    assert!(rare > normal, "rare={rare} not > normal={normal}");
}

/// Threshold is monotone: a higher cutoff cannot return more results.
#[test]
fn higher_threshold_yields_fewer_or_equal_results() {
    let detector = pattern_abc(60, 2);
    let test = s(&["A", "B", "C", "X", "Y", "Z", "A", "B", "C"]);

    let all = detector.detect_anomalies(&test, 0.0).expect("all");
    let strong = detector.detect_anomalies(&test, 0.8).expect("strong");

    assert!(strong.len() <= all.len());
    for s in &strong {
        assert!(s.anomaly_strength >= 0.8);
    }
}

/// Monotonicity: a window with strictly higher information_score also
/// has higher anomaly_strength (modulo numerical ties).
#[test]
fn information_and_strength_are_monotone() {
    let detector = pattern_abc(60, 2);
    let scores = detector
        .detect_anomalies(&s(&["A", "B", "C", "Z", "Y", "X", "A", "B", "C"]), 0.0)
        .expect("detect");
    for pair in scores.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        if a.information_score < b.information_score - 1e-9 {
            assert!(
                a.anomaly_strength <= b.anomaly_strength + 1e-9,
                "monotonicity broken: I_a={} S_a={} vs I_b={} S_b={}",
                a.information_score,
                a.anomaly_strength,
                b.information_score,
                b.anomaly_strength
            );
        }
    }
}

/// Untrained detector refuses to score (`EmptyContextTree` error).
#[test]
fn untrained_detector_errors_on_detect() {
    let detector = AnomalyDetector::new(2).expect("detector");
    assert!(detector.detect_anomalies(&s(&["A", "B"]), 0.5).is_err());
}

/// Empty / length-1 sequences return an empty score list, not an error.
#[test]
fn short_sequences_return_empty_scores() {
    let detector = pattern_abc(30, 2);
    assert!(detector.detect_anomalies(&[], 0.0).expect("empty").is_empty());
    assert!(detector.detect_anomalies(&s(&["A"]), 0.0).expect("len 1").is_empty());
}

/// Sequences shorter than `max_order + 1` use adaptive window sizing
/// and produce at least one score.
#[test]
fn short_sequences_use_adaptive_window() {
    let mut detector = AnomalyDetector::new(3).expect("detector");
    detector
        .train(&s(&["A", "B", "C", "D", "A", "B", "C", "D"]))
        .expect("train");
    let scores = detector.detect_anomalies(&s(&["A", "B"]), 0.0).expect("len 2");
    assert!(!scores.is_empty());
}
