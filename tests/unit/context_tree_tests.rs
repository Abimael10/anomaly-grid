//! Unit tests for Context Tree module
//! 
//! These tests define the expected behavior of the context tree implementation
//! and context management functionality with trie-based storage.

use anomaly_grid::context_tree::*;
use anomaly_grid::config::AnomalyGridConfig;
use anomaly_grid::error::AnomalyGridError;

/// Epsilon for floating-point comparisons
const EPSILON: f64 = 1e-10;

#[test]
fn test_context_node_creation() {
    let node = ContextNode::default();
    
    // Test initial state
    assert_eq!(node.total_count(), 0);
    assert_eq!(node.vocab_size(), 0);
}

#[test]
fn test_context_node_add_transition() {
    let mut node = ContextNode::default();
    
    // Add some transitions
    node.add_transition("A");
    node.add_transition("B");
    node.add_transition("A");
    
    // Check counts
    assert_eq!(node.get_count("A"), 2);
    assert_eq!(node.get_count("B"), 1);
    assert_eq!(node.get_count("C"), 0);
    assert_eq!(node.total_count(), 3);
    assert_eq!(node.vocab_size(), 2);
}

#[test]
fn test_context_node_probabilities() {
    let mut node = ContextNode::default();
    
    // Add some transitions
    node.add_transition("A");
    node.add_transition("A");
    node.add_transition("B");
    
    let config = AnomalyGridConfig::default(); // Uses alpha = 1.0
    
    // Test exact Laplace smoothing formula: P(x) = (count(x) + α) / (N + α*|V|)
    let alpha = config.smoothing_alpha;
    let total_count = 3.0; // A:2, B:1
    let vocab_size = 2.0;  // A and B
    
    let expected_prob_a = (2.0 + alpha) / (total_count + alpha * vocab_size);
    let expected_prob_b = (1.0 + alpha) / (total_count + alpha * vocab_size);
    
    let actual_prob_a = node.get_probability("A", &config);
    let actual_prob_b = node.get_probability("B", &config);
    
    const ULTRA_STRICT_TOLERANCE: f64 = 1e-15;
    
    let error_a = (actual_prob_a - expected_prob_a).abs();
    let error_b = (actual_prob_b - expected_prob_b).abs();
    
    assert!(error_a < ULTRA_STRICT_TOLERANCE,
           "Laplace smoothing formula incorrect for A: expected {:.15}, got {:.15}, error = {:.2e}",
           expected_prob_a, actual_prob_a, error_a);
    assert!(error_b < ULTRA_STRICT_TOLERANCE,
           "Laplace smoothing formula incorrect for B: expected {:.15}, got {:.15}, error = {:.2e}",
           expected_prob_b, actual_prob_b, error_b);
    
    // Test probability conservation
    let prob_sum = actual_prob_a + actual_prob_b;
    assert!((prob_sum - 1.0).abs() < ULTRA_STRICT_TOLERANCE,
           "Probability conservation violated: sum = {:.15}", prob_sum);
}

#[test]
fn test_context_node_entropy_calculation() {
    let mut node = ContextNode::default();
    let config = AnomalyGridConfig::default();
    
    // Add equal transitions (maximum entropy case)
    node.add_transition("A");
    node.add_transition("B");
    
    let entropy = node.calculate_entropy(&config);
    // For two equal outcomes with Laplace smoothing, entropy should be close to log2(2) = 1.0
    assert!((entropy - 1.0).abs() < 0.1); // Allow some tolerance due to smoothing
    
    // Test deterministic case (minimum entropy)
    let mut deterministic_node = ContextNode::default();
    deterministic_node.add_transition("A");
    deterministic_node.add_transition("A");
    
    let det_entropy = deterministic_node.calculate_entropy(&config);
    // For deterministic outcome, entropy should be low
    assert!(det_entropy < 0.5);
}

#[test]
fn test_context_node_kl_divergence() {
    let mut node = ContextNode::default();
    let config = AnomalyGridConfig::default();
    
    // Add transitions
    node.add_transition("A");
    node.add_transition("B");
    
    let kl_div = node.calculate_kl_divergence(&config);
    // KL divergence should be non-negative
    assert!(kl_div >= 0.0);
    
    // For uniform distribution, KL divergence from uniform should be close to 0
    assert!(kl_div < 0.1); // Allow some tolerance due to smoothing
}

#[test]
fn test_context_tree_creation() {
    let tree = ContextTree::new(3).expect("Failed to create context tree");
    
    assert_eq!(tree.max_order, 3);
    assert_eq!(tree.context_count(), 0);
}

#[test]
fn test_context_tree_creation_invalid_order() {
    let result = ContextTree::new(0);
    assert!(result.is_err(), "Should fail with invalid max_order");
    
    match result.unwrap_err() {
        AnomalyGridError::InvalidMaxOrder { value, .. } => {
            assert_eq!(value, 0);
        }
        _ => panic!("Expected InvalidMaxOrder error"),
    }
}

#[test]
fn test_context_tree_build_from_sequence() {
    let mut tree = ContextTree::new(2).expect("Failed to create context tree");
    
    // Use longer sequence for statistical validity
    let mut sequence = Vec::new();
    for _ in 0..20 {
        sequence.extend(vec!["A".to_string(), "B".to_string(), "C".to_string()]);
    }
    
    let config = AnomalyGridConfig::default();
    let result = tree.build_from_sequence(&sequence, &config);
    
    assert!(result.is_ok(), "Building from valid sequence should succeed");
    assert!(tree.context_count() > 0, "Context tree should not be empty after building");
    
    // Verify mathematical properties
    let contexts_map = tree.contexts();
    for (context, node) in &contexts_map {
        // Probability conservation
        let probabilities = node.get_all_probabilities(&config);
        let prob_sum: f64 = probabilities.values().sum();
        assert!((prob_sum - 1.0).abs() < 1e-10, 
               "Probability conservation violated for context {:?}: sum = {:.15}", 
               context, prob_sum);
        
        // Entropy bounds
        let n_outcomes = probabilities.len() as f64;
        let max_entropy = n_outcomes.log2();
        let entropy = node.calculate_entropy(&config);
        assert!(entropy >= 0.0, "Entropy must be non-negative for context {:?}", context);
        assert!(entropy <= max_entropy + 1e-10, "Entropy exceeds maximum for context {:?}", context);
    }
}

#[test]
fn test_context_tree_get_transition_probability() {
    let mut tree = ContextTree::new(2).expect("Failed to create context tree");
    
    // Use longer sequence for statistical validity
    let mut sequence = Vec::new();
    for _ in 0..10 {
        sequence.extend(vec![
            "A".to_string(), "B".to_string(), "C".to_string(),
            "A".to_string(), "B".to_string(), "D".to_string()
        ]);
    }
    
    let config = AnomalyGridConfig::default();
    tree.build_from_sequence(&sequence, &config).expect("Failed to build tree");
    
    // Test getting transition probabilities
    let prob = tree.get_transition_probability(&["A".to_string()], "B");
    assert!(prob.is_some(), "Transition probability should exist for trained pattern");
    
    let prob_value = prob.unwrap();
    assert!(prob_value > 0.0 && prob_value <= 1.0,
           "Probability should be in (0,1]: {:.15}", prob_value);
    assert!(prob_value.is_finite(), "Probability should be finite");
    
    // Test non-existent transition
    let no_prob = tree.get_transition_probability(&["X".to_string()], "Y");
    assert!(no_prob.is_none(), "Non-existent transition should return None");
    
    // Test mathematical consistency: all probabilities from a context should sum to 1
    if let Some(node) = tree.get_context_node(&["A".to_string()]) {
        let probabilities = node.get_all_probabilities(&config);
        let total_prob: f64 = probabilities.values().sum();
        assert!((total_prob - 1.0).abs() < 1e-15,
               "Probabilities from context A should sum to 1.0: {:.15}", total_prob);
    }
}

#[test]
fn test_context_tree_empty_sequence() {
    let mut tree = ContextTree::new(2).expect("Failed to create tree");
    let empty_sequence: Vec<String> = vec![];
    let config = AnomalyGridConfig::default();
    
    let result = tree.build_from_sequence(&empty_sequence, &config);
    assert!(result.is_err());
}

#[test]
fn test_context_tree_single_element_sequence() {
    let mut tree = ContextTree::new(2).expect("Failed to create tree");
    let single_sequence = vec!["A".to_string()];
    let config = AnomalyGridConfig::default();
    
    let result = tree.build_from_sequence(&single_sequence, &config);
    assert!(result.is_err());
}

#[test]
fn test_context_tree_probability_conservation() {
    let mut tree = ContextTree::new(3).expect("Failed to create tree");
    let sequence = vec![
        "A".to_string(), "B".to_string(), "C".to_string(), 
        "A".to_string(), "B".to_string(), "D".to_string(),
        "A".to_string(), "C".to_string(), "D".to_string()
    ];
    let config = AnomalyGridConfig::default();
    
    tree.build_from_sequence(&sequence, &config).unwrap();
    
    // Check that all context probabilities sum to 1.0
    let contexts_map = tree.contexts();
    for (context, node) in &contexts_map {
        let probabilities = node.get_all_probabilities(&config);
        let prob_sum: f64 = probabilities.values().sum();
        assert!(
            (prob_sum - 1.0).abs() < 1e-10,
            "Context {:?} violates probability conservation: sum = {:.12}",
            context, prob_sum
        );
    }
}

#[test]
fn test_context_tree_entropy_bounds() {
    let mut tree = ContextTree::new(2).expect("Failed to create tree");
    let sequence = vec![
        "A".to_string(), "B".to_string(), "C".to_string(), "D".to_string(),
        "A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()
    ];
    let config = AnomalyGridConfig::default();
    
    tree.build_from_sequence(&sequence, &config).unwrap();
    
    // Check entropy bounds for all contexts
    let contexts_map = tree.contexts();
    for (context, node) in &contexts_map {
        let probabilities = node.get_all_probabilities(&config);
        let n_outcomes = probabilities.len() as f64;
        let max_entropy = n_outcomes.log2();
        let entropy = node.calculate_entropy(&config);
        
        assert!(
            entropy >= -1e-10,
            "Entropy must be non-negative: H = {:.6} for context {:?}",
            entropy, context
        );
        
        assert!(
            entropy <= max_entropy + 1e-10,
            "Entropy {:.6} exceeds theoretical maximum {:.6} for context {:?}",
            entropy, max_entropy, context
        );
    }
}