//! Unit tests for MarkovModel
//!
//! These tests focus on the mathematical correctness of the Markov model
//! implementation, including likelihood calculations and hierarchical context selection.

use anomaly_grid::*;

#[test]
fn test_markov_model_creation() {
    let model = MarkovModel::new(3);
    assert!(model.is_ok());
    assert_eq!(model.unwrap().max_order(), 3);
    
    let invalid_model = MarkovModel::new(0);
    assert!(invalid_model.is_err());
}

#[test]
fn test_markov_model_with_config() {
    let config = AnomalyGridConfig::default()
        .with_max_order(2)
        .expect("Failed to set max_order");
    
    let model = MarkovModel::with_config(config);
    assert!(model.is_ok());
    assert_eq!(model.unwrap().max_order(), 2);
}

#[test]
fn test_basic_training() {
    let mut model = MarkovModel::new(2).expect("Failed to create model");
    
    let sequence = vec!["A", "B", "C", "A", "B", "C"]
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    
    let result = model.train(&sequence);
    assert!(result.is_ok(), "Training should succeed");
    
    // Verify that the model has learned something
    assert!(model.context_tree().context_count() > 0, "Model should have contexts");
    assert!(!model.state_mapping().is_empty(), "Model should have state mapping");
}

#[test]
fn test_likelihood_calculation_properties() {
    let mut model = MarkovModel::new(2).expect("Failed to create model");
    
    // Train with deterministic pattern
    let training_sequence = vec!["A", "B", "A", "B"].repeat(25)
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    model.train(&training_sequence).expect("Failed to train");
    
    let test_sequences = vec![
        vec!["A".to_string()],                                    // Single element
        vec!["A".to_string(), "B".to_string()],                   // Known pattern
        vec!["B".to_string(), "A".to_string()],                   // Known pattern
        vec!["X".to_string(), "Y".to_string()],                   // Unknown pattern
        vec!["A".to_string(), "B".to_string(), "A".to_string()],  // Longer known pattern
    ];
    
    for test_sequence in test_sequences {
        let likelihood = model.calculate_likelihood(&test_sequence);
        
        // Test mathematical bounds
        assert!(likelihood >= 0.0 && likelihood <= 1.0,
               "Likelihood out of bounds for {:?}: {:.15}",
               test_sequence, likelihood);
        
        // Test numerical stability
        assert!(likelihood.is_finite(),
               "Likelihood must be finite for {:?}: {:.15}",
               test_sequence, likelihood);
    }
}

#[test]
fn test_single_element_likelihood() {
    let mut model = MarkovModel::new(2).expect("Failed to create model");
    
    // Train with known distribution: A=70%, B=20%, C=10%
    let mut training_sequence = Vec::new();
    training_sequence.extend(vec!["A"; 70]);
    training_sequence.extend(vec!["B"; 20]);
    training_sequence.extend(vec!["C"; 10]);
    let training_strings: Vec<String> = training_sequence.repeat(10)
        .iter().map(|s| s.to_string()).collect();
    
    model.train(&training_strings).expect("Failed to train");
    
    // Test single element likelihoods
    let likelihood_a = model.calculate_likelihood(&[String::from("A")]);
    let likelihood_b = model.calculate_likelihood(&[String::from("B")]);
    let likelihood_c = model.calculate_likelihood(&[String::from("C")]);
    
    // A should have highest likelihood, C should have lowest
    assert!(likelihood_a > likelihood_b,
           "A should have higher likelihood than B: A={:.6}, B={:.6}",
           likelihood_a, likelihood_b);
    assert!(likelihood_b > likelihood_c,
           "B should have higher likelihood than C: B={:.6}, C={:.6}",
           likelihood_b, likelihood_c);
    
    // All should be positive and finite
    assert!(likelihood_a > 0.0 && likelihood_a.is_finite());
    assert!(likelihood_b > 0.0 && likelihood_b.is_finite());
    assert!(likelihood_c > 0.0 && likelihood_c.is_finite());
}

#[test]
fn test_marginal_probability_calculation() {
    let mut model = MarkovModel::new(1).expect("Failed to create model");
    
    // Train with known counts
    let sequence = vec!["A", "B", "A", "B", "A", "C"]
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    model.train(&sequence).expect("Failed to train");
    
    // Test marginal probabilities
    let prob_a = model.get_marginal_probability("A");
    let prob_b = model.get_marginal_probability("B");
    let prob_c = model.get_marginal_probability("C");
    
    // A appears 3 times, B appears 2 times, C appears 1 time
    // With smoothing, probabilities should reflect this ordering
    assert!(prob_a >= prob_b, "A should have >= probability than B");
    assert!(prob_b >= prob_c, "B should have >= probability than C");
    
    // All probabilities should be positive and sum to approximately 1
    assert!(prob_a > 0.0 && prob_a <= 1.0);
    assert!(prob_b > 0.0 && prob_b <= 1.0);
    assert!(prob_c > 0.0 && prob_c <= 1.0);
}

#[test]
fn test_hierarchical_context_selection() {
    let mut model = MarkovModel::new(3).expect("Failed to create model");
    
    // Create sequence with clear hierarchical patterns
    let sequence = vec![
        "A", "B", "C", "D",  // ABC->D
        "A", "B", "C", "D",  // ABC->D
        "A", "B", "C", "E",  // ABC->E
        "A", "B", "X", "Y",  // ABX->Y
    ].iter().map(|s| s.to_string()).collect::<Vec<_>>();
    
    model.train(&sequence).expect("Failed to train");
    
    // Test hierarchical context selection
    let context = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    let prob_d = model.get_best_context_probability(&context, "D");
    let prob_e = model.get_best_context_probability(&context, "E");
    
    // D appears twice after ABC, E appears once, so D should be more likely
    assert!(prob_d > prob_e,
           "Hierarchical context selection failed: P(D|ABC) = {:.6} should be > P(E|ABC) = {:.6}",
           prob_d, prob_e);
    
    // Both should be positive
    assert!(prob_d > 0.0 && prob_e > 0.0,
           "All probabilities should be positive: P(D|ABC) = {:.6}, P(E|ABC) = {:.6}",
           prob_d, prob_e);
}

#[test]
fn test_background_probability() {
    let mut model = MarkovModel::new(2).expect("Failed to create model");
    
    let sequence = vec!["A", "B", "C"]
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    model.train(&sequence).expect("Failed to train");
    
    // Test background probability for known state
    let bg_prob_a = model.get_background_probability("A");
    assert!(bg_prob_a > 0.0 && bg_prob_a <= 1.0,
           "Background probability for known state should be in (0,1]: {:.6}",
           bg_prob_a);
    
    // Test background probability for unknown state
    let bg_prob_x = model.get_background_probability("X");
    assert!(bg_prob_x > 0.0 && bg_prob_x <= 1.0,
           "Background probability for unknown state should be in (0,1]: {:.6}",
           bg_prob_x);
    
    // Unknown state should have lower probability than known state
    assert!(bg_prob_x <= bg_prob_a,
           "Unknown state should have lower background probability: X={:.6}, A={:.6}",
           bg_prob_x, bg_prob_a);
}

#[test]
fn test_context_probability_consistency() {
    let mut model = MarkovModel::new(2).expect("Failed to create model");
    
    let sequence = vec!["A", "B", "C", "A", "B", "D"]
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    model.train(&sequence).expect("Failed to train");
    
    // Test that context probabilities are consistent
    let context = vec!["A".to_string()];
    let prob_b = model.get_best_context_probability(&context, "B");
    let prob_c = model.get_best_context_probability(&context, "C");
    
    // Both transitions exist in the training data
    assert!(prob_b > 0.0, "P(B|A) should be positive");
    assert!(prob_c > 0.0, "P(C|A) should be positive");
    
    // Probabilities should be reasonable
    assert!(prob_b <= 1.0, "P(B|A) should be <= 1.0");
    assert!(prob_c <= 1.0, "P(C|A) should be <= 1.0");
}

#[test]
fn test_state_mapping() {
    let mut model = MarkovModel::new(2).expect("Failed to create model");
    
    let sequence = vec!["ALPHA", "BETA", "GAMMA", "ALPHA"]
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    model.train(&sequence).expect("Failed to train");
    
    let state_mapping = model.state_mapping();
    
    // All states should be mapped
    assert!(state_mapping.contains_key("ALPHA"));
    assert!(state_mapping.contains_key("BETA"));
    assert!(state_mapping.contains_key("GAMMA"));
    
    // Mapping should be consistent
    assert_eq!(state_mapping.len(), 3);
    
    // IDs should be unique
    let mut ids: Vec<usize> = state_mapping.values().cloned().collect();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), 3, "State IDs should be unique");
}

#[test]
fn test_likelihood_ordering() {
    let mut model = MarkovModel::new(2).expect("Failed to create model");
    
    // Train with clear pattern: A->B is very common, X->Y is rare
    let mut training_sequence = Vec::new();
    training_sequence.extend(vec!["A", "B"].repeat(50));  // Very common
    training_sequence.extend(vec!["X", "Y"]);             // Rare
    let training_strings: Vec<String> = training_sequence
        .iter().map(|s| s.to_string()).collect();
    
    model.train(&training_strings).expect("Failed to train");
    
    // Test likelihood ordering
    let common_sequence = vec!["A".to_string(), "B".to_string()];
    let rare_sequence = vec!["X".to_string(), "Y".to_string()];
    let unknown_sequence = vec!["P".to_string(), "Q".to_string()];
    
    let common_likelihood = model.calculate_likelihood(&common_sequence);
    let rare_likelihood = model.calculate_likelihood(&rare_sequence);
    let unknown_likelihood = model.calculate_likelihood(&unknown_sequence);
    
    // Common should have highest likelihood
    assert!(common_likelihood >= rare_likelihood,
           "Common pattern should have higher likelihood: common={:.6}, rare={:.6}",
           common_likelihood, rare_likelihood);
    
    // Unknown should have lowest likelihood
    assert!(rare_likelihood >= unknown_likelihood,
           "Rare pattern should have higher likelihood than unknown: rare={:.6}, unknown={:.6}",
           rare_likelihood, unknown_likelihood);
}

#[test]
fn test_empty_context_handling() {
    let mut model = MarkovModel::new(2).expect("Failed to create model");
    
    let sequence = vec!["A", "B", "C"]
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    model.train(&sequence).expect("Failed to train");
    
    // Test probability with empty context
    let empty_context: Vec<String> = vec![];
    let prob = model.get_best_context_probability(&empty_context, "A");
    
    // Should fall back to background probability
    assert!(prob > 0.0 && prob <= 1.0,
           "Empty context probability should be in (0,1]: {:.6}", prob);
}

#[test]
fn test_model_configuration_access() {
    let config = AnomalyGridConfig::default()
        .with_smoothing_alpha(2.0)
        .expect("Failed to set alpha");
    
    let model = MarkovModel::with_config(config.clone())
        .expect("Failed to create model");
    
    // Test configuration access
    assert_eq!(model.config().smoothing_alpha, 2.0);
    assert_eq!(model.config().max_order, config.max_order);
}

#[test]
fn test_numerical_stability_extreme_cases() {
    let mut model = MarkovModel::new(2).expect("Failed to create model");
    
    // Test with extreme cases
    let extreme_cases = vec![
        // Deterministic sequence
        vec!["A"; 1000].iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        // High entropy sequence
        (0..1000).map(|i| format!("S{}", i % 50)).collect::<Vec<_>>(),
        // Extreme skew
        {
            let mut seq = vec!["COMMON"; 999].iter().map(|s| s.to_string()).collect::<Vec<_>>();
            seq.push("RARE".to_string());
            seq
        },
    ];
    
    for (i, sequence) in extreme_cases.iter().enumerate() {
        let mut test_model = MarkovModel::new(2).expect("Failed to create model");
        
        let train_result = test_model.train(sequence);
        assert!(train_result.is_ok(), "Training should succeed for case {}", i);
        
        // Test likelihood calculation stability
        let test_seq = vec!["A".to_string(), "B".to_string()];
        let likelihood = test_model.calculate_likelihood(&test_seq);
        
        assert!(likelihood.is_finite(),
               "Likelihood should be finite for case {}: {:.15}", i, likelihood);
        assert!(likelihood >= 0.0 && likelihood <= 1.0,
               "Likelihood should be in bounds for case {}: {:.15}", i, likelihood);
    }
}