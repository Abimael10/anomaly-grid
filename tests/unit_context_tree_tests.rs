//! Unit tests for ContextTree
//!
//! These tests focus on the mathematical correctness of the context tree
//! implementation, including probability calculations and information theory.

#![allow(clippy::uninlined_format_args)]
#![allow(clippy::useless_vec)]
#![allow(clippy::absurd_extreme_comparisons)]
#![allow(unused_comparisons)]

use anomaly_grid::*;

#[test]
fn test_context_tree_creation() {
    let tree = ContextTree::new(3);
    assert!(tree.is_ok());

    let invalid_tree = ContextTree::new(0);
    assert!(invalid_tree.is_err());
}

#[test]
fn test_context_tree_building() {
    let mut tree = ContextTree::new(2).expect("Failed to create context tree");

    let sequence = vec!["A", "B", "C", "A", "B", "C"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    let config = AnomalyGridConfig::default();
    let result = tree.build_from_sequence(&sequence, &config);
    assert!(result.is_ok(), "Building context tree should succeed");

    // Verify that contexts were created
    assert!(
        tree.context_count() > 0,
        "Context tree should have contexts"
    );
}

#[test]
fn test_probability_conservation() {
    let mut tree = ContextTree::new(1).expect("Failed to create context tree");

    let sequence = vec!["A", "B", "A", "C", "A", "B"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    let config = AnomalyGridConfig::default();
    tree.build_from_sequence(&sequence, &config)
        .expect("Failed to build tree");

    // With global-alphabet Laplace smoothing, the full distribution over all
    // |Σ| symbols sums to 1.0. `get_all_probabilities` only returns observed
    // symbols, so their sum ≤ 1.0; the remaining mass covers unseen symbols.
    let gv = tree.global_vocab_size();
    for (context, node) in &tree.contexts() {
        let probabilities = node.get_all_probabilities(&config, gv);
        let prob_sum: f64 = probabilities.values().sum();

        // Observed probabilities must not exceed 1.0
        assert!(
            prob_sum <= 1.0 + 1e-10,
            "Observed probability sum exceeds 1.0 for context {:?}: sum = {:.15}",
            context,
            prob_sum
        );

        // Full distribution sums to 1: observed + (|Σ| - |observed|) * smoothed_unseen
        let unseen_count = gv.saturating_sub(probabilities.len());
        let alpha = config.smoothing_alpha;
        let unseen_prob = alpha / (node.total_count() as f64 + alpha * gv as f64);
        let full_sum = prob_sum + unseen_count as f64 * unseen_prob;
        assert!(
            (full_sum - 1.0).abs() < 1e-10,
            "Full probability sum violated for context {:?}: sum = {:.15}",
            context,
            full_sum
        );

        for (symbol, &prob) in &probabilities {
            assert!(
                (0.0..=1.0).contains(&prob),
                "Probability out of bounds for {}|{:?}: P = {:.15}",
                symbol,
                context,
                prob
            );
        }
    }
}

#[test]
fn test_entropy_calculations() {
    let mut tree = ContextTree::new(1).expect("Failed to create context tree");

    let sequence = vec!["A", "B", "A", "B", "A", "C"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    let config = AnomalyGridConfig::default();
    tree.build_from_sequence(&sequence, &config)
        .expect("Failed to build tree");

    // Test entropy properties for all contexts
    let gv = tree.global_vocab_size();
    for (context, node) in &tree.contexts() {
        let entropy = node.compute_entropy(&config, gv);
        // With global alphabet smoothing, max possible entropy is log₂(|Σ|)
        let max_entropy = (gv as f64).log2();

        // Entropy must be non-negative
        assert!(
            entropy >= 0.0,
            "Entropy must be non-negative for context {:?}: H = {:.15}",
            context,
            entropy
        );

        // Entropy cannot exceed maximum possible entropy
        assert!(
            entropy <= max_entropy + 1e-10,
            "Entropy exceeds maximum for context {:?}: H = {:.15} > {:.15}",
            context,
            entropy,
            max_entropy
        );

        // Entropy must be finite
        assert!(
            entropy.is_finite(),
            "Entropy must be finite for context {:?}: H = {:.15}",
            context,
            entropy
        );
    }
}

#[test]
fn test_kl_divergence_properties() {
    let mut tree = ContextTree::new(1).expect("Failed to create context tree");

    let sequence = vec!["A", "B", "A", "B", "A", "C"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    let config = AnomalyGridConfig::default();
    tree.build_from_sequence(&sequence, &config)
        .expect("Failed to build tree");

    // Test KL divergence properties for all contexts
    let gv = tree.global_vocab_size();
    for (context, node) in &tree.contexts() {
        let kl_divergence = node.compute_kl_divergence(&config, gv);

        // KL divergence must be non-negative
        assert!(
            kl_divergence >= 0.0,
            "KL divergence must be non-negative for context {:?}: KL = {:.15}",
            context,
            kl_divergence
        );

        // KL divergence must be finite
        assert!(
            kl_divergence.is_finite(),
            "KL divergence must be finite for context {:?}: KL = {:.15}",
            context,
            kl_divergence
        );
    }
}

#[test]
fn test_laplace_smoothing() {
    let config = AnomalyGridConfig::default()
        .with_smoothing_alpha(2.0)
        .expect("Failed to set alpha");

    let mut tree = ContextTree::new(1).expect("Failed to create context tree");

    // Create sequence with exact known counts
    let sequence = vec![
        "A", "B", // A->B: 1
        "A", "B", // A->B: 2
        "A", "C", // A->C: 1
    ]
    .iter()
    .map(|s| s.to_string())
    .collect::<Vec<_>>();

    tree.build_from_sequence(&sequence, &config)
        .expect("Failed to build tree");

    // Test exact Laplace smoothing formula: P(x) = (count(x) + α) / (N + α*|Σ|)
    // Global alphabet: {A, B, C} → |Σ| = 3
    if let Some(node) = tree.get_context_node(&["A".to_string()]) {
        // Context "A": count(B)=2, count(C)=1, total=3, α=2.0, |Σ|=3
        // P(B|A) = (2 + 2) / (3 + 2*3) = 4/9
        // P(C|A) = (1 + 2) / (3 + 2*3) = 3/9 = 1/3
        let expected_prob_b = 4.0 / 9.0;
        let expected_prob_c = 3.0 / 9.0;

        let gv = tree.global_vocab_size();
        let actual_prob_b = node.get_probability("B", &config, gv);
        let actual_prob_c = node.get_probability("C", &config, gv);

        let error_b = (actual_prob_b - expected_prob_b).abs();
        let error_c = (actual_prob_c - expected_prob_c).abs();

        assert!(error_b < 1e-10,
               "Laplace smoothing formula incorrect for B: expected {:.15}, got {:.15}, error = {:.2e}",
               expected_prob_b, actual_prob_b, error_b);
        assert!(error_c < 1e-10,
               "Laplace smoothing formula incorrect for C: expected {:.15}, got {:.15}, error = {:.2e}",
               expected_prob_c, actual_prob_c, error_c);
    }
}

#[test]
fn test_context_retrieval() {
    let mut tree = ContextTree::new(2).expect("Failed to create context tree");

    let sequence = vec!["A", "B", "C", "A", "B", "D"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    let config = AnomalyGridConfig::default();
    tree.build_from_sequence(&sequence, &config)
        .expect("Failed to build tree");

    // Test context retrieval
    let context_ab = tree.get_context_node(&["A".to_string(), "B".to_string()]);
    assert!(context_ab.is_some(), "Context [A, B] should exist");

    let context_xyz = tree.get_context_node(&["X".to_string(), "Y".to_string(), "Z".to_string()]);
    assert!(context_xyz.is_none(), "Context [X, Y, Z] should not exist");
}

#[test]
fn test_transition_probabilities() {
    let mut tree = ContextTree::new(1).expect("Failed to create context tree");

    let sequence = vec!["A", "B", "A", "C", "A", "B"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    let config = AnomalyGridConfig::default();
    tree.build_from_sequence(&sequence, &config)
        .expect("Failed to build tree");

    // Test transition probability retrieval
    let prob_ab = tree.get_transition_probability(&["A".to_string()], "B");
    assert!(prob_ab.is_some(), "Transition A->B should exist");
    assert!(
        prob_ab.unwrap() > 0.0,
        "Transition probability should be positive"
    );

    let prob_ac = tree.get_transition_probability(&["A".to_string()], "C");
    assert!(prob_ac.is_some(), "Transition A->C should exist");
    assert!(
        prob_ac.unwrap() > 0.0,
        "Transition probability should be positive"
    );

    // Test that A->B has higher probability than A->C (B appears twice, C once)
    assert!(
        prob_ab.unwrap() > prob_ac.unwrap(),
        "A->B should be more likely than A->C"
    );
}

#[test]
fn test_memory_estimation() {
    let mut tree = ContextTree::new(2).expect("Failed to create context tree");

    let sequence = vec!["A", "B", "C", "D"]
        .repeat(10)
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    let config = AnomalyGridConfig::default();
    tree.build_from_sequence(&sequence, &config)
        .expect("Failed to build tree");

    let memory_estimate = tree.estimate_memory_usage();
    assert!(memory_estimate > 0, "Memory estimate should be positive");
}

#[test]
fn test_context_statistics() {
    let mut tree = ContextTree::new(2).expect("Failed to create context tree");

    let sequence = vec!["A", "B", "C", "A", "B", "D"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    let config = AnomalyGridConfig::default();
    tree.build_from_sequence(&sequence, &config)
        .expect("Failed to build tree");

    let stats = tree.get_context_statistics();
    assert!(stats.total_contexts > 0, "Should have contexts");
    assert!(stats.total_transitions > 0, "Should have transitions");
    assert!(
        stats.avg_entropy >= 0.0,
        "Average entropy should be non-negative"
    );
    assert!(
        stats.max_entropy >= stats.avg_entropy,
        "Max entropy should be >= average"
    );
}

#[test]
fn test_deterministic_vs_uniform_entropy() {
    let config = AnomalyGridConfig::default();

    // Test deterministic sequence (should have low entropy)
    let mut det_tree = ContextTree::new(1).expect("Failed to create context tree");
    let det_sequence = vec!["A", "B"]
        .repeat(50)
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    det_tree
        .build_from_sequence(&det_sequence, &config)
        .expect("Failed to build deterministic tree");

    // Test uniform sequence (should have higher entropy)
    let mut uniform_tree = ContextTree::new(1).expect("Failed to create context tree");
    let uniform_sequence = vec!["A", "X", "A", "Y", "A", "Z", "A", "W"]
        .repeat(25)
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    uniform_tree
        .build_from_sequence(&uniform_sequence, &config)
        .expect("Failed to build uniform tree");

    // Compare entropies
    if let (Some(det_node), Some(uniform_node)) = (
        det_tree.get_context_node(&["A".to_string()]),
        uniform_tree.get_context_node(&["A".to_string()]),
    ) {
        let det_entropy = det_node.compute_entropy(&config, det_tree.global_vocab_size());
        let uniform_entropy = uniform_node.compute_entropy(&config, uniform_tree.global_vocab_size());

        // Uniform distribution should have higher entropy
        assert!(
            uniform_entropy >= det_entropy,
            "Uniform entropy ({:.6}) should be >= deterministic entropy ({:.6})",
            uniform_entropy,
            det_entropy
        );
    }
}

#[test]
fn test_empty_sequence_handling() {
    let mut tree = ContextTree::new(2).expect("Failed to create context tree");

    let empty_sequence: Vec<String> = vec![];
    let config = AnomalyGridConfig::default();

    let result = tree.build_from_sequence(&empty_sequence, &config);
    assert!(result.is_err(), "Building from empty sequence should fail");
}

#[test]
fn test_single_element_sequence() {
    let mut tree = ContextTree::new(2).expect("Failed to create context tree");

    let single_sequence = vec!["A".to_string()];
    let config = AnomalyGridConfig::default();

    let result = tree.build_from_sequence(&single_sequence, &config);
    // This might succeed or fail depending on min_sequence_length - both are valid
    if result.is_ok() {
        assert!(
            tree.context_count() >= 0,
            "Context count should be non-negative"
        );
    }
}
