//! Mathematical correctness: variable-order Markov chain, Kolmogorov
//! probability axioms, Witten-Bell + Laplace smoothing, Shannon
//! entropy, and KL divergence.
//!
//! Concrete-input regressions for invariants that the property suite in
//! `proptest.rs` also covers in randomised form.

#![allow(clippy::float_cmp)]

mod common;
use common::{pattern_abc, s, trained};

use anomaly_grid::{AnomalyGridConfig, ContextTree, MarkovModel};

// ── Markov chain mathematics ────────────────────────────────────────

/// Markov property: `P(next | context)` depends only on the last
/// `max_order` symbols, not on what preceded them.
#[test]
fn markov_property_truncates_long_history() {
    let detector = pattern_abc(100, 2);

    let p_short = detector
        .model()
        .get_best_context_probability(&s(&["A", "B"]), "C");
    let p_long = detector
        .model()
        .get_best_context_probability(&s(&["X", "Y", "Z", "A", "B"]), "C");

    assert!(
        (p_short - p_long).abs() < 1e-9,
        "Markov property violated: short={p_short} long={p_long}"
    );
}

/// Chain rule: joint likelihood of `[A, B, A]` equals
/// `P(B|A) · P(A|B)` (the only two factors at order 1).
#[test]
fn chain_rule_factorisation_holds() {
    let detector = trained(&["A", "B"], 100, 1);
    let model = detector.model();

    let joint = model.calculate_likelihood(&s(&["A", "B", "A"]));
    let p_b_given_a = model.get_best_context_probability(&s(&["A"]), "B");
    let p_a_given_b = model.get_best_context_probability(&s(&["B"]), "A");
    let expected = p_b_given_a * p_a_given_b;

    assert!(
        (joint - expected).abs() < 1e-9,
        "chain rule violated: calculated={joint} expected={expected}"
    );
}

/// Stationary cycle A→B→C→A trains transitions to ≈1.0.
#[test]
fn stationary_pattern_yields_near_unit_transitions() {
    let detector = pattern_abc(100, 1);
    let model = detector.model();

    for (name, p) in [
        ("B|A", model.get_best_context_probability(&s(&["A"]), "B")),
        ("C|B", model.get_best_context_probability(&s(&["B"]), "C")),
        ("A|C", model.get_best_context_probability(&s(&["C"]), "A")),
    ] {
        assert!(p > 0.9, "{name} = {p}, expected > 0.9");
    }
}

/// Marginal probability does not depend on `max_order`.
#[test]
fn marginal_is_stable_across_orders() {
    let sequence = s(&["A", "B", "A", "A"]);

    let mut o1 = MarkovModel::new(1).expect("o1");
    o1.train(&sequence).expect("train1");
    let mut o3 = MarkovModel::new(3).expect("o3");
    o3.train(&sequence).expect("train3");

    let p1 = o1.get_marginal_probability("A");
    let p3 = o3.get_marginal_probability("A");
    assert!(
        (p1 - p3).abs() < 1e-9,
        "marginal P(A) differs across orders: o1={p1} o3={p3}"
    );
}

/// A common context produces higher likelihood than a rare one.
#[test]
fn likelihood_is_higher_for_common_patterns() {
    let mut training: Vec<String> = Vec::new();
    for _ in 0..50 {
        training.extend(s(&["A", "B"]));
    }
    training.extend(s(&["X", "Y"]));

    let mut model = MarkovModel::new(2).expect("model");
    model.train(&training).expect("train");

    let common = model.calculate_likelihood(&s(&["A", "B"]));
    let rare = model.calculate_likelihood(&s(&["X", "Y"]));
    let unknown = model.calculate_likelihood(&s(&["P", "Q"]));

    assert!(common >= rare, "common={common} rare={rare}");
    assert!(rare >= unknown, "rare={rare} unknown={unknown}");
}

// ── Kolmogorov probability axioms ───────────────────────────────────

/// (P≥0) and (P(Ω)=1): every per-context distribution is a valid measure.
#[test]
fn distribution_satisfies_kolmogorov_axioms() {
    let detector = pattern_abc(50, 2);
    let model = detector.model();
    let tree = model.context_tree();
    let alphabet = tree.alphabet();

    for (ctx, _) in tree.contexts() {
        for x in &alphabet {
            let p = model.get_best_context_probability(&ctx, x);
            assert!((0.0..=1.0).contains(&p), "P({x}|{ctx:?}) = {p} not in [0,1]");
        }
        let total: f64 = alphabet
            .iter()
            .map(|x| model.get_best_context_probability(&ctx, x))
            .sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "ΣP(x|{ctx:?}) = {total}, expected 1"
        );
    }
}

/// Bayes' theorem: `P(B|A)·P(A) ≈ P(A|B)·P(B)` (joint symmetry).
#[test]
fn bayes_joint_symmetry_within_smoothing_tolerance() {
    let detector = trained(&["A", "B", "B", "A"], 80, 1);
    let model = detector.model();

    let lhs = model.get_best_context_probability(&s(&["A"]), "B") * model.get_marginal_probability("A");
    let rhs = model.get_best_context_probability(&s(&["B"]), "A") * model.get_marginal_probability("B");

    let rel = (lhs - rhs).abs() / lhs.max(rhs).max(1e-12);
    assert!(rel < 0.05, "Bayes joint symmetry rel_err = {rel} > 0.05");
}

/// Law of total probability: `Σ_y P(x|y)·P(y) ≈ P(x)`.
#[test]
fn law_of_total_probability_holds() {
    let detector = trained(&["A", "B", "C", "A", "B"], 80, 1);
    let model = detector.model();
    let alphabet = model.context_tree().alphabet();

    for x in &alphabet {
        let marginal = model.get_marginal_probability(x);
        let total: f64 = alphabet
            .iter()
            .map(|y| {
                model.get_best_context_probability(std::slice::from_ref(y), x)
                    * model.get_marginal_probability(y)
            })
            .sum();
        assert!(
            (total - marginal).abs() < 0.05,
            "ΣP({x}|y)·P(y) = {total}, marginal = {marginal}"
        );
    }
}

// ── Laplace smoothing closed form ───────────────────────────────────

/// Verifies the closed-form `P(x|c) = (count + α) / (N + α·|Σ|)` on a
/// hand-counted example. With α=2.0 and global alphabet {A, B, C}:
/// after seeing A→B twice and A→C once,
/// `P(B|A) = (2+2)/(3+6) = 4/9`, `P(C|A) = (1+2)/(3+6) = 1/3`.
#[test]
fn laplace_smoothing_closed_form() {
    let config = AnomalyGridConfig::default()
        .with_smoothing_alpha(2.0)
        .expect("alpha");

    let mut tree = ContextTree::new(1).expect("tree");
    tree.build_from_sequence(&s(&["A", "B", "A", "B", "A", "C"]), &config)
        .expect("build");

    let node = tree.get_context_node(&s(&["A"])).expect("context A");
    let gv = tree.global_vocab_size();

    let p_b = node.get_probability("B", &config, gv);
    let p_c = node.get_probability("C", &config, gv);

    assert!(
        (p_b - 4.0 / 9.0).abs() < 1e-10,
        "P(B|A) = {p_b}, expected 4/9"
    );
    assert!(
        (p_c - 3.0 / 9.0).abs() < 1e-10,
        "P(C|A) = {p_c}, expected 1/3"
    );
}

// ── Information theory ─────────────────────────────────────────────

fn build(seq: &[&str], order: usize) -> ContextTree {
    let mut tree = ContextTree::new(order).expect("tree");
    let cfg = AnomalyGridConfig::default();
    tree.build_from_sequence(&s(seq), &cfg).expect("build");
    tree
}

/// Shannon entropy is non-negative and bounded by `log₂ |Σ|`.
#[test]
fn entropy_is_in_unit_log_alphabet() {
    let tree = build(&["A", "B", "C", "D", "A", "B", "C", "D"], 2);
    let cfg = AnomalyGridConfig::default();
    let gv = tree.global_vocab_size();
    let max_h = (gv as f64).log2();

    for (ctx, node) in tree.contexts() {
        let h = node.compute_entropy(&cfg, gv);
        assert!(h >= 0.0, "entropy {h} < 0 for {ctx:?}");
        assert!(h <= max_h + 1e-9, "entropy {h} > log₂({gv}) = {max_h}");
    }
}

/// Deterministic context (always same continuation) → near-zero entropy.
/// Uniform context (every continuation equally likely) → near-max entropy.
#[test]
fn deterministic_entropy_is_below_uniform_entropy() {
    let det = build(&["A", "B", "A", "B", "A", "B", "A", "B"], 1);
    let uni = build(&["A", "X", "A", "Y", "A", "Z", "A", "W"], 1);
    let cfg = AnomalyGridConfig::default();

    let det_h = det
        .get_context_node(&s(&["A"]))
        .expect("det A")
        .compute_entropy(&cfg, det.global_vocab_size());
    let uni_h = uni
        .get_context_node(&s(&["A"]))
        .expect("uni A")
        .compute_entropy(&cfg, uni.global_vocab_size());

    assert!(uni_h > det_h, "uniform entropy {uni_h} not > deterministic {det_h}");
}

/// `KL(P‖U) ≥ 0` (Gibbs' inequality) — non-negative for any P.
#[test]
fn kl_divergence_is_non_negative() {
    let tree = build(&["A", "A", "A", "B", "C", "A", "A"], 1);
    let cfg = AnomalyGridConfig::default();
    let gv = tree.global_vocab_size();

    for (ctx, node) in tree.contexts() {
        let kl = node.compute_kl_divergence(&cfg, gv);
        assert!(kl >= -1e-12, "KL({ctx:?}‖U) = {kl} < 0");
    }
}

/// Information content `−log₂ P(x)` is non-negative for every observed pair.
#[test]
fn information_content_is_non_negative() {
    let tree = build(&["A", "B", "A", "C", "A", "B", "A"], 1);
    let cfg = AnomalyGridConfig::default();
    let gv = tree.global_vocab_size();

    for (_, node) in tree.contexts() {
        for (sym, p) in node.get_all_probabilities(&cfg, gv) {
            let ic = -p.log2();
            assert!(ic >= -1e-12, "I({sym}) = {ic} (p = {p})");
        }
    }
}

/// `information_score` favours longest supported context. With training
/// `[A, B, D]·2` and `[X, B, C]`, the bigram `B → C` is ambiguous, but
/// `A,B → D` is highly preferred; scoring `[A, B, C]` should reflect
/// the longest-context surprise of `C` given `[A, B]`.
#[test]
fn information_score_uses_longest_supported_context() {
    let mut detector = anomaly_grid::AnomalyDetector::new(2).expect("detector");
    detector
        .train_sequences(&[
            s(&["A", "B", "D"]),
            s(&["A", "B", "D"]),
            s(&["B", "C"]),
            s(&["X", "B", "C"]),
            s(&["Y", "B", "C"]),
        ])
        .expect("train");

    let scores = detector.detect_anomalies(&s(&["A", "B", "C"]), 0.0).expect("detect");
    assert_eq!(scores.len(), 1);

    let model = detector.model();
    let p_b_given_a = model.get_best_context_probability(&s(&["A"]), "B");
    let p_c_given_ab = model.get_best_context_probability(&s(&["A", "B"]), "C");
    let expected = (-p_b_given_a.log2() - p_c_given_ab.log2()) / 2.0;

    assert!(
        (scores[0].information_score - expected).abs() < 1e-9,
        "info_score = {} expected {}",
        scores[0].information_score, expected
    );
}
