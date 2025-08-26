//! Unit tests for Context Tree module
//! 
//! These tests define the expected behavior of the context tree implementation
//! and context management functionality.

use anomaly_grid::context_tree::*;

/// Epsilon for floating-point comparisons
const EPSILON: f64 = 1e-10;

#[test]
fn test_context_node_creation() {
    let node = ContextNode::new();
    
    // Test initial state
    assert!(node.counts.is_empty());
    assert!(node.probabilities.is_empty());
    assert_eq!(node.entropy, 0.0);
    assert_eq!(node.kl_divergence, 0.0);
}

#[test]
fn test_context_node_add_transition() {
    let mut node = ContextNode::new();
    
    // Add some transitions
    node.add_transition("A".to_string());
    node.add_transition("B".to_string());
    node.add_transition("A".to_string());
    
    // Check counts
    assert_eq!(node.counts.get("A"), Some(&2));
    assert_eq!(node.counts.get("B"), Some(&1));
    assert_eq!(node.counts.get("C"), None);
}

#[test]
fn test_context_node_probabilities() {
    let mut node = ContextNode::new();
    
    // Add some transitions
    node.add_transition("A".to_string());
    node.add_transition("A".to_string());
    node.add_transition("B".to_string());
    
    let config = AnomalyGridConfig::default(); // Uses alpha = 1.0
    node.calculate_probabilities(&config);
    
    // Test exact Laplace smoothing formula: P(x) = (count(x) + α) / (N + α*|V|)
    let alpha = config.smoothing_alpha;
    let total_count = 3.0; // A:2, B:1
    let vocab_size = 2.0;  // A and B
    
    let expected_prob_a = (2.0 + alpha) / (total_count + alpha * vocab_size);
    let expected_prob_b = (1.0 + alpha) / (total_count + alpha * vocab_size);
    
    let actual_prob_a = node.probabilities.get("A").copied().unwrap_or(0.0);
    let actual_prob_b = node.probabilities.get("B").copied().unwrap_or(0.0);
    
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
    let mut node = ContextNode::new();
    
    // Add equal transitions (maximum entropy case)
    node.add_transition("A".to_string());
    node.add_transition("B".to_string());
    node.calculate_probabilities();
    
    // For two equal outcomes, entropy should be log2(2) = 1.0
    assert!((node.entropy - 1.0).abs() < 1e-10);
    
    // Test deterministic case (minimum entropy)
    let mut deterministic_node = ContextNode::new();
    deterministic_node.add_transition("A".to_string());
    deterministic_node.add_transition("A".to_string());
    deterministic_node.calculate_probabilities();
    
    // For deterministic outcome, entropy should be 0
    assert!(deterministic_node.entropy.abs() < 1e-10);
}

#[test]
fn test_context_node_kl_divergence() {
    let mut node = ContextNode::new();
    
    // Add transitions
    node.add_transition("A".to_string());
    node.add_transition("B".to_string());
    node.calculate_probabilities();
    
    // KL divergence should be non-negative
    assert!(node.kl_divergence >= 0.0);
    
    // For uniform distribution, KL divergence from uniform should be 0
    assert!(node.kl_divergence.abs() < 1e-10);
}

#[test]
fn test_context_tree_creation() {
    let tree = ContextTree::new(3).expect("Failed to create context tree");
    
    assert_eq!(tree.max_order, 3);
    assert!(tree.contexts.is_empty());
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
    assert!(!tree.contexts.is_empty(), "Context tree should not be empty after building");
    
    // Verify mathematical properties
    for (context, node) in &tree.contexts {
        // Probability conservation
        let prob_sum: f64 = node.probabilities.values().sum();
        assert!((prob_sum - 1.0).abs() < 1e-10, 
               "Probability conservation violated for context {:?}: sum = {:.15}", 
               context, prob_sum);
        
        // Entropy bounds
        let n_outcomes = node.probabilities.len() as f64;
        let max_entropy = n_outcomes.log2();
        assert!(node.entropy >= 0.0, "Entropy must be non-negative for context {:?}", context);
        assert!(node.entropy <= max_entropy + 1e-10, "Entropy exceeds maximum for context {:?}", context);
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
        let total_prob: f64 = node.probabilities.values().sum();
        assert!((total_prob - 1.0).abs() < 1e-15,
               "Probabilities from context A should sum to 1.0: {:.15}", total_prob);
    }
}

#[test]
fn test_context_tree_empty_sequence() {
    let mut tree = ContextTree::new(2);
    let empty_sequence: Vec<String> = vec![];
    
    let result = tree.build_from_sequence(&empty_sequence);
    assert!(result.is_err());
}

#[test]
fn test_context_tree_single_element_sequence() {
    let mut tree = ContextTree::new(2);
    let single_sequence = vec!["A".to_string()];
    
    let result = tree.build_from_sequence(&single_sequence);
    assert!(result.is_err());
}

#[test]
fn test_context_tree_probability_conservation() {
    let mut tree = ContextTree::new(3);
    let sequence = vec![
        "A".to_string(), "B".to_string(), "C".to_string(), 
        "A".to_string(), "B".to_string(), "D".to_string(),
        "A".to_string(), "C".to_string(), "D".to_string()
    ];
    
    tree.build_from_sequence(&sequence).unwrap();
    
    // Check that all context probabilities sum to 1.0
    for (context, node) in &tree.contexts {
        let prob_sum: f64 = node.probabilities.values().sum();
        assert!(
            (prob_sum - 1.0).abs() < 1e-10,
            "Context {:?} violates probability conservation: sum = {:.12}",
            context, prob_sum
        );
    }
}

#[test]
fn test_context_tree_entropy_bounds() {
    let mut tree = ContextTree::new(2);
    let sequence = vec![
        "A".to_string(), "B".to_string(), "C".to_string(), "D".to_string(),
        "A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()
    ];
    
    tree.build_from_sequence(&sequence).unwrap();
    
    // Check entropy bounds for all contexts
    for (context, node) in &tree.contexts {
        let n_outcomes = node.probabilities.len() as f64;
        let max_entropy = n_outcomes.log2();
        
        assert!(
            node.entropy >= -1e-10,
            "Entropy must be non-negative: H = {:.6} for context {:?}",
            node.entropy, context
        );
        
        assert!(
            node.entropy <= max_entropy + 1e-10,
            "Entropy {:.6} exceeds theoretical maximum {:.6} for context {:?}",
            node.entropy, max_entropy, context
        );
    }
}