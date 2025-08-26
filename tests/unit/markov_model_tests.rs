//! Unit tests for Markov Model module
//! 
//! These tests define the expected behavior of the variable-order Markov chain
//! implementation and hierarchical context selection.

use anomaly_grid::markov_model::*;

#[test]
fn test_markov_model_creation() {
    let model = MarkovModel::new(3);
    
    assert_eq!(model.max_order(), 3);
    assert!(model.state_mapping().is_empty());
}

#[test]
fn test_markov_model_train() {
    let mut model = MarkovModel::new(2);
    let sequence = vec![
        "A".to_string(), "B".to_string(), "C".to_string(), 
        "A".to_string(), "B".to_string(), "D".to_string()
    ];
    
    let result = model.train(&sequence);
    assert!(result.is_ok());
    
    // Check that state mapping was created
    assert!(!model.state_mapping().is_empty());
    assert!(model.state_mapping().contains_key("A"));
    assert!(model.state_mapping().contains_key("B"));
    assert!(model.state_mapping().contains_key("C"));
    assert!(model.state_mapping().contains_key("D"));
}

#[test]
fn test_markov_model_calculate_likelihood() {
    let mut model = MarkovModel::new(2);
    let training_sequence = vec![
        "A".to_string(), "B".to_string(), "A".to_string(), "B".to_string(),
        "A".to_string(), "B".to_string(), "A".to_string(), "B".to_string()
    ];
    
    model.train(&training_sequence).unwrap();
    
    // Test likelihood of a sequence similar to training data
    let test_sequence = vec!["A".to_string(), "B".to_string(), "A".to_string()];
    let likelihood = model.calculate_likelihood(&test_sequence);
    
    assert!(likelihood > 0.0);
    assert!(likelihood <= 1.0);
    assert!(likelihood.is_finite());
}

#[test]
fn test_markov_model_hierarchical_context_selection() {
    let mut model = MarkovModel::new(3);
    let sequence = vec![
        "A".to_string(), "B".to_string(), "C".to_string(), "D".to_string(),
        "A".to_string(), "B".to_string(), "C".to_string(), "E".to_string(),
        "A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()
    ];
    
    model.train(&sequence).unwrap();
    
    // Test that longer contexts are preferred when available
    let context = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    let prob_d = model.get_best_context_probability(&context, "D");
    let prob_e = model.get_best_context_probability(&context, "E");
    
    // Both should have some probability
    assert!(prob_d > 0.0);
    assert!(prob_e > 0.0);
    
    // D appears twice after ABC, E appears once, so D should be more likely
    assert!(prob_d > prob_e);
}

#[test]
fn test_markov_model_unseen_transitions() {
    let mut model = MarkovModel::new(2);
    let sequence = vec!["A".to_string(), "B".to_string(), "A".to_string(), "B".to_string()];
    
    model.train(&sequence).unwrap();
    
    // Test probability for unseen transition
    let prob = model.get_best_context_probability(&vec!["A".to_string()], "Z");
    
    // Should return a small but non-zero probability for unseen transitions
    assert!(prob > 0.0);
    assert!(prob < 0.1); // Should be much smaller than seen transitions
}

#[test]
fn test_markov_model_empty_sequence_training() {
    let mut model = MarkovModel::new(2);
    let empty_sequence: Vec<String> = vec![];
    
    let result = model.train(&empty_sequence);
    assert!(result.is_err());
}

#[test]
fn test_markov_model_single_element_training() {
    let mut model = MarkovModel::new(2);
    let single_sequence = vec!["A".to_string()];
    
    let result = model.train(&single_sequence);
    assert!(result.is_err());
}

#[test]
fn test_markov_model_likelihood_bounds() {
    let mut model = MarkovModel::new(2);
    let sequence = vec![
        "A".to_string(), "B".to_string(), "C".to_string(),
        "A".to_string(), "B".to_string(), "C".to_string()
    ];
    
    model.train(&sequence).unwrap();
    
    // Test various sequences
    let test_sequences = vec![
        vec!["A".to_string(), "B".to_string()], // Seen pattern
        vec!["A".to_string(), "Z".to_string()], // Partially unseen
        vec!["X".to_string(), "Y".to_string()], // Completely unseen
    ];
    
    for test_seq in test_sequences {
        let likelihood = model.calculate_likelihood(&test_seq);
        
        // All likelihoods should be valid probabilities
        assert!(likelihood >= 0.0, "Likelihood should be non-negative: {}", likelihood);
        assert!(likelihood <= 1.0, "Likelihood should not exceed 1.0: {}", likelihood);
        assert!(likelihood.is_finite(), "Likelihood should be finite: {}", likelihood);
    }
}

#[test]
fn test_markov_model_context_fallback() {
    let mut model = MarkovModel::new(3);
    let sequence = vec![
        "A".to_string(), "B".to_string(), "C".to_string(), "D".to_string(),
        "E".to_string(), "F".to_string(), "G".to_string(), "H".to_string()
    ];
    
    model.train(&sequence).unwrap();
    
    // Test with a context that doesn't exist at max order but exists at lower orders
    let context = vec!["X".to_string(), "Y".to_string(), "A".to_string()];
    let prob = model.get_best_context_probability(&context, "B");
    
    // Should fall back to shorter contexts and still return a probability
    assert!(prob > 0.0);
    assert!(prob.is_finite());
}

#[test]
fn test_markov_model_deterministic_sequence() {
    let mut model = MarkovModel::new(2);
    let deterministic_sequence = vec![
        "A".to_string(), "B".to_string(), "A".to_string(), "B".to_string(),
        "A".to_string(), "B".to_string(), "A".to_string(), "B".to_string()
    ];
    
    model.train(&deterministic_sequence).unwrap();
    
    // Test likelihood of the same pattern
    let test_sequence = vec!["A".to_string(), "B".to_string(), "A".to_string()];
    let likelihood = model.calculate_likelihood(&test_sequence);
    
    // Should have high likelihood for the trained pattern
    assert!(likelihood > 0.5);
    
    // Test likelihood of a different pattern
    let different_sequence = vec!["A".to_string(), "A".to_string(), "A".to_string()];
    let different_likelihood = model.calculate_likelihood(&different_sequence);
    
    // Should have lower likelihood for different pattern
    assert!(different_likelihood < likelihood);
}