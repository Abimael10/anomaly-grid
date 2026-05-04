//! Property-based tests for domain invariants.
//!
//! Uses proptest to verify that core mathematical properties hold across
//! randomly generated inputs.

use anomaly_grid::{AnomalyDetector, AnomalyGridConfig, ContextTree};
use proptest::prelude::*;

/// Generate a non-empty sequence over a small alphabet.
fn arb_sequence(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(prop::sample::select(vec!["A", "B", "C", "D"]), min_len..=max_len)
        .prop_map(|v| v.into_iter().map(|s| s.to_string()).collect())
}

/// Generate a training-length sequence (≥3 for default min_sequence_length).
fn arb_training_sequence() -> impl Strategy<Value = Vec<String>> {
    arb_sequence(4, 60)
}

// ── (a) Probability sum ≤ 1 + ε over global alphabet ──────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(80))]

    #[test]
    fn probability_sum_bounded(seq in arb_training_sequence()) {
        let config = AnomalyGridConfig::default();
        let mut tree = ContextTree::new(2).expect("tree");
        tree.build_from_sequence(&seq, &config).expect("build");

        let gv = tree.global_vocab_size();

        for (_, node) in tree.contexts() {
            // Sum smoothed probability over *every* symbol in the global alphabet
            let entries = tree.interner().entries();
            let total: f64 = entries
                .iter()
                .map(|(_, s)| node.get_probability(s, &config, gv))
                .sum();
            // Laplace smoothing: sum should be very close to 1.0
            prop_assert!((total - 1.0).abs() < 1e-9,
                "probability sum {} deviates from 1.0 (gv={}, node_count={})",
                total, gv, node.total_count());
        }
    }
}

// ── (b) Entropy ∈ [0, log₂|Σ|] ───────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(80))]

    #[test]
    fn entropy_in_bounds(seq in arb_training_sequence()) {
        let config = AnomalyGridConfig::default();
        let mut tree = ContextTree::new(2).expect("tree");
        tree.build_from_sequence(&seq, &config).expect("build");

        let gv = tree.global_vocab_size();
        let max_entropy = (gv as f64).log2();

        for (_, node) in tree.contexts() {
            let h = node.compute_entropy(&config, gv);
            prop_assert!(h >= 0.0, "entropy {} < 0", h);
            prop_assert!(h <= max_entropy + 1e-9,
                "entropy {} > log₂({}) = {}", h, gv, max_entropy);
        }
    }
}

// ── (c) anomaly_strength ∈ [0, 1] ────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(40))]

    #[test]
    fn anomaly_strength_bounded(
        train in arb_training_sequence(),
        test in arb_sequence(3, 20),
    ) {
        let mut detector = AnomalyDetector::new(2).expect("detector");
        detector.train(&train).expect("train");

        let scores = detector.detect_anomalies(&test, 0.0).expect("detect");
        for score in &scores {
            prop_assert!(score.anomaly_strength >= 0.0,
                "anomaly_strength {} < 0", score.anomaly_strength);
            prop_assert!(score.anomaly_strength <= 1.0,
                "anomaly_strength {} > 1", score.anomaly_strength);
        }
    }
}

// ── (d) Backoff monotonicity: longer context ≥ shorter context count ──
//
// If context [A, B] exists then context [B] must also exist, and the
// shorter context must have seen at least as many total transitions.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(80))]

    #[test]
    fn backoff_parent_context_exists(seq in arb_training_sequence()) {
        let config = AnomalyGridConfig::default();
        let mut tree = ContextTree::new(3).expect("tree");
        tree.build_from_sequence(&seq, &config).expect("build");

        let contexts = tree.contexts();
        for (ctx, node) in &contexts {
            if ctx.len() >= 2 {
                // The suffix (drop first element) must also be a context
                let suffix = ctx[1..].to_vec();
                let parent = contexts.get(&suffix);
                prop_assert!(parent.is_some(),
                    "context {:?} exists but suffix {:?} does not", ctx, suffix);

                // Parent must have at least as many total observations
                let parent_count = parent.expect("checked").total_count();
                prop_assert!(parent_count >= node.total_count(),
                    "parent {:?} count {} < child {:?} count {}",
                    suffix, parent_count, ctx, node.total_count());
            }
        }
    }
}

// ── (e) Training determinism: same input → same output ────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(40))]

    #[test]
    fn training_is_deterministic(seq in arb_training_sequence()) {
        let mut det1 = AnomalyDetector::new(2).expect("det1");
        let mut det2 = AnomalyDetector::new(2).expect("det2");
        det1.train(&seq).expect("train1");
        det2.train(&seq).expect("train2");

        // Same test sequence should produce identical scores
        let test: Vec<String> = vec!["A", "B", "C"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let scores1 = det1.detect_anomalies(&test, 0.0).expect("detect1");
        let scores2 = det2.detect_anomalies(&test, 0.0).expect("detect2");

        prop_assert_eq!(scores1.len(), scores2.len());
        for (s1, s2) in scores1.iter().zip(scores2.iter()) {
            prop_assert!((s1.likelihood - s2.likelihood).abs() < f64::EPSILON,
                "likelihood diverged: {} vs {}", s1.likelihood, s2.likelihood);
            prop_assert!((s1.anomaly_strength - s2.anomaly_strength).abs() < f64::EPSILON,
                "anomaly_strength diverged: {} vs {}", s1.anomaly_strength, s2.anomaly_strength);
        }
    }
}
