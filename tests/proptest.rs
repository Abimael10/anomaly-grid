//! Property-based tests for domain invariants.
//!
//! Properties are deliberately strong (sums to 1, monotonicity, parallel
//! determinism) rather than "doesn't panic" smoke tests. Edge-case
//! generators below stress the boundaries: empty / single-symbol /
//! single-symbol-alphabet / very-long / Unicode inputs.

use anomaly_grid::{batch_score, AnomalyDetector, AnomalyGridConfig, ContextTree};
use proptest::prelude::*;

/// Sequence over a small ASCII alphabet.
fn arb_sequence(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(prop::sample::select(vec!["A", "B", "C", "D"]), min_len..=max_len)
        .prop_map(|v| v.into_iter().map(str::to_string).collect())
}

fn arb_training_sequence() -> impl Strategy<Value = Vec<String>> {
    arb_sequence(4, 60)
}

/// Sequence over a Unicode alphabet (CJK + emoji + latin).
fn arb_unicode_sequence(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(
        prop::sample::select(vec!["α", "β", "中", "🦀", "Z"]),
        min_len..=max_len,
    )
    .prop_map(|v| v.into_iter().map(str::to_string).collect())
}

// ── (a) Smoothing axiom: Σ P(x | c) = 1 over global alphabet ──────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn probability_sum_equals_one(seq in arb_training_sequence()) {
        let config = AnomalyGridConfig::default();
        let mut tree = ContextTree::new(2).expect("tree");
        tree.build_from_sequence(&seq, &config).expect("build");

        let gv = tree.global_vocab_size();
        let alphabet = tree.alphabet();

        for (_, node) in tree.contexts() {
            let total: f64 = alphabet
                .iter()
                .map(|s| node.get_probability(s, &config, gv))
                .sum();
            prop_assert!((total - 1.0).abs() < 1e-9,
                "probability sum {} deviates from 1.0 (gv={}, count={})",
                total, gv, node.total_count());
        }
    }
}

// ── (b) Entropy ∈ [0, log₂|Σ|] ───────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

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
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn anomaly_strength_bounded(
        train in arb_training_sequence(),
        test in arb_sequence(3, 20),
    ) {
        let mut detector = AnomalyDetector::new(2).expect("detector");
        detector.train(&train).expect("train");

        let scores = detector.detect_anomalies(&test, 0.0).expect("detect");
        for score in &scores {
            prop_assert!((0.0..=1.0).contains(&score.anomaly_strength),
                "anomaly_strength {} not in [0,1]", score.anomaly_strength);
            prop_assert!((0.0..=1.0).contains(&score.likelihood),
                "likelihood {} not in [0,1]", score.likelihood);
            prop_assert!(score.information_score >= 0.0,
                "information_score {} < 0", score.information_score);
        }
    }
}

// ── (d) Backoff parent context exists & dominates child ──────────────
//
// For every context [a, b] there must also exist [b], because the trie
// stores all suborder windows. The shorter context aggregates more
// observations, so its total_count() is ≥ the longer one's.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn backoff_parent_context_exists(seq in arb_training_sequence()) {
        let config = AnomalyGridConfig::default();
        let mut tree = ContextTree::new(3).expect("tree");
        tree.build_from_sequence(&seq, &config).expect("build");

        let contexts = tree.contexts();
        for (ctx, node) in &contexts {
            if ctx.len() >= 2 {
                let suffix = ctx[1..].to_vec();
                let parent = contexts.get(&suffix);
                prop_assert!(parent.is_some(),
                    "context {:?} exists but suffix {:?} does not", ctx, suffix);

                if let Some(parent) = parent {
                    prop_assert!(parent.total_count() >= node.total_count(),
                        "parent {:?} count {} < child {:?} count {}",
                        suffix, parent.total_count(), ctx, node.total_count());
                }
            }
        }
    }
}

// ── (e) Training determinism ─────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn training_is_deterministic(seq in arb_training_sequence()) {
        let mut det1 = AnomalyDetector::new(2).expect("det1");
        let mut det2 = AnomalyDetector::new(2).expect("det2");
        det1.train(&seq).expect("train1");
        det2.train(&seq).expect("train2");

        let test: Vec<String> = ["A", "B", "C"].iter().map(|s| (*s).to_string()).collect();
        let scores1 = det1.detect_anomalies(&test, 0.0).expect("detect1");
        let scores2 = det2.detect_anomalies(&test, 0.0).expect("detect2");

        prop_assert_eq!(scores1.len(), scores2.len());
        for (s1, s2) in scores1.iter().zip(scores2.iter()) {
            prop_assert!((s1.likelihood - s2.likelihood).abs() < 1e-12);
            prop_assert!((s1.anomaly_strength - s2.anomaly_strength).abs() < 1e-12);
        }
    }
}

// ── (f) Empty / single-symbol / length-1 sequences ────────────────────

#[test]
fn detect_anomalies_on_empty_sequence_after_training() {
    let mut detector = AnomalyDetector::new(2).expect("detector");
    let train: Vec<String> = ["A", "B", "A", "B"].iter().map(|s| (*s).to_string()).collect();
    detector.train(&train).expect("train");

    let empty: Vec<String> = Vec::new();
    let scores = detector.detect_anomalies(&empty, 0.0).expect("detect");
    assert!(scores.is_empty());
}

#[test]
fn detect_anomalies_on_length_one() {
    let mut detector = AnomalyDetector::new(2).expect("detector");
    let train: Vec<String> = ["A", "B", "A", "B"].iter().map(|s| (*s).to_string()).collect();
    detector.train(&train).expect("train");

    let single = vec!["A".to_string()];
    let scores = detector.detect_anomalies(&single, 0.0).expect("detect");
    assert!(scores.is_empty());
}

#[test]
fn single_symbol_alphabet_is_handled() {
    // Training corpus of a single repeated symbol — every transition
    // P(A | A) is ~1.0. Anomaly strength on the same data should be 0.
    let mut detector = AnomalyDetector::new(2).expect("detector");
    let train: Vec<String> = std::iter::repeat_n("A".to_string(), 30).collect();
    detector.train(&train).expect("train");

    let test: Vec<String> = std::iter::repeat_n("A".to_string(), 6).collect();
    let scores = detector.detect_anomalies(&test, 0.0).expect("detect");
    for s in &scores {
        assert!(s.anomaly_strength < 0.1, "saw {} on uniform input", s.anomaly_strength);
        assert!(s.likelihood > 0.5, "saw likelihood {} on uniform input", s.likelihood);
    }
}

// ── (g) Unicode round-trip ────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn unicode_symbols_are_supported(seq in arb_unicode_sequence(6, 30)) {
        let mut detector = AnomalyDetector::new(2).expect("detector");
        detector.train(&seq).expect("train");

        // Score the same data: legitimate (non-anomalous) sequences should
        // not blow up and should produce finite values.
        let scores = detector.detect_anomalies(&seq, 0.0).expect("detect");
        for s in &scores {
            prop_assert!(s.anomaly_strength.is_finite());
            prop_assert!(s.likelihood.is_finite());
            prop_assert!(s.information_score.is_finite());
        }
    }
}

// ── (h) Long-sequence numerical stability ────────────────────────────

#[test]
fn long_sequence_log_likelihood_is_finite() {
    let mut detector = AnomalyDetector::new(4).expect("detector");
    // 10k-symbol training corpus (small alphabet, many contexts).
    let train: Vec<String> = (0..10_000).map(|i| format!("S{}", i % 5)).collect();
    detector.train(&train).expect("train");

    // 1k-symbol scoring window. Joint likelihood will underflow to 0 —
    // the anomaly_strength path uses the per-symbol bits metric, which
    // stays finite.
    let test: Vec<String> = (0..1_000).map(|i| format!("S{}", i % 5)).collect();
    let scores = detector.detect_anomalies(&test, 0.0).expect("detect");
    for s in &scores {
        assert!(
            s.anomaly_strength.is_finite() && (0.0..=1.0).contains(&s.anomaly_strength),
            "anomaly_strength = {} on long sequence",
            s.anomaly_strength
        );
        assert!(
            s.information_score.is_finite() && s.information_score >= 0.0,
            "information_score = {} on long sequence",
            s.information_score
        );
    }
}

// ── (i) Parallel batch determinism ────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn batch_score_matches_loop(
        train in arb_training_sequence(),
        seqs in prop::collection::vec(arb_sequence(3, 20), 5..40),
    ) {
        let mut detector = AnomalyDetector::new(2).expect("detector");
        detector.train(&train).expect("train");

        let parallel = batch_score(&detector, &seqs, 0.0).expect("batch_score");
        let serial: Vec<_> = seqs
            .iter()
            .map(|s| detector.detect_anomalies(s, 0.0).expect("detect"))
            .collect();

        prop_assert_eq!(parallel.len(), serial.len());
        for (par, ser) in parallel.iter().zip(serial.iter()) {
            prop_assert_eq!(par.len(), ser.len());
            for (a, b) in par.iter().zip(ser.iter()) {
                prop_assert!((a.anomaly_strength - b.anomaly_strength).abs() < 1e-12);
            }
        }
    }
}
