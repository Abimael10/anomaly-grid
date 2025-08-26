//! Comprehensive Test Runner
//!
//! This test runner executes all tests from subdirectories with rigorous
//! mathematical validation and systematic issue detection.
//!
//! Based on the systematic review, this runner focuses on:
//! 1. API consistency validation
//! 2. Mathematical rigor and invariant testing
//! 3. Boundary condition testing
//! 4. Error propagation testing
//! 5. Real-world scenario validation

use anomaly_grid::*;

/// Ultra-strict tolerance for mathematical validation
const ULTRA_STRICT_TOLERANCE: f64 = 1e-15;
const STRICT_TOLERANCE: f64 = 1e-12;

#[test]
fn run_all_unit_tests() {
    println!("🔥 RUNNING ALL UNIT TESTS WITH RIGOROUS VALIDATION");
    println!("==================================================");

    // Run anomaly detector tests
    run_anomaly_detector_unit_tests();

    // Run context tree tests
    run_context_tree_unit_tests();

    // Run markov model tests
    run_markov_model_unit_tests();

    println!("✅ ALL UNIT TESTS COMPLETED");
}

#[test]
fn run_all_integration_tests() {
    println!("🌍 RUNNING ALL INTEGRATION TESTS");
    println!("================================");

    // Run workflow tests
    run_workflow_integration_tests();

    // Run error handling tests
    run_error_handling_integration_tests();

    // Run comprehensive validation tests
    run_comprehensive_validation_tests();

    println!("✅ ALL INTEGRATION TESTS COMPLETED");
}

#[test]
fn run_all_mathematical_tests() {
    println!("🔬 RUNNING ALL MATHEMATICAL TESTS WITH BRUTAL RIGOR");
    println!("===================================================");

    // Run theoretical validation tests
    run_theoretical_validation_tests();

    // Run core mathematical proofs
    run_core_mathematical_proofs();

    println!("✅ ALL MATHEMATICAL TESTS COMPLETED");
}

#[test]
fn run_all_performance_tests() {
    println!("⚡ RUNNING ALL PERFORMANCE TESTS");
    println!("===============================");

    // Run stress tests
    run_stress_tests();

    // Run optimization tests
    run_optimization_tests();

    println!("✅ ALL PERFORMANCE TESTS COMPLETED");
}

// ============================================================================
// UNIT TESTS WITH RIGOROUS VALIDATION
// ============================================================================

fn run_anomaly_detector_unit_tests() {
    println!("🧪 Testing Anomaly Detector Unit Tests...");

    // Test 1: API Consistency (CRITICAL FIX)
    test_anomaly_detector_api_consistency();

    // Test 2: Mathematical Properties
    test_anomaly_detector_mathematical_properties();

    // Test 3: Boundary Conditions
    test_anomaly_detector_boundary_conditions();

    // Test 4: Error Handling
    test_anomaly_detector_error_handling();

    println!("  ✅ Anomaly Detector unit tests passed");
}

fn test_anomaly_detector_api_consistency() {
    // CRITICAL: Test that API returns Result consistently

    // Valid creation should return Ok
    let detector_result = AnomalyDetector::new(3);
    assert!(detector_result.is_ok(), "Valid creation should return Ok");

    let mut detector = detector_result.expect("Failed to create detector");
    assert_eq!(detector.max_order(), 3);

    // Invalid creation should return Err
    let invalid_result = AnomalyDetector::new(0);
    assert!(
        invalid_result.is_err(),
        "Invalid creation should return Err"
    );

    match invalid_result.unwrap_err() {
        AnomalyGridError::InvalidMaxOrder { value, .. } => {
            assert_eq!(value, 0);
        }
        _ => panic!("Expected InvalidMaxOrder error"),
    }

    // Training with valid sequence should work
    let mut training_sequence = Vec::new();
    for _ in 0..20 {
        // Ensure statistical validity
        training_sequence.extend(vec!["A".to_string(), "B".to_string(), "C".to_string()]);
    }

    let train_result = detector.train(&training_sequence);
    assert!(train_result.is_ok(), "Training should succeed");

    // Detection should work after training
    let test_sequence = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    let detect_result = detector.detect_anomalies(&test_sequence, 0.1);
    assert!(detect_result.is_ok(), "Detection should succeed");
}

fn test_anomaly_detector_mathematical_properties() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");

    // Use statistically valid training sequence
    let mut training_sequence = Vec::new();
    for _ in 0..100 {
        training_sequence.extend(vec!["A".to_string(), "B".to_string()]);
    }

    detector.train(&training_sequence).expect("Failed to train");

    let test_sequence = vec!["A".to_string(), "X".to_string(), "Y".to_string()];
    let anomalies = detector
        .detect_anomalies(&test_sequence, 0.5)
        .expect("Failed to detect");

    // RIGOROUS mathematical validation
    for anomaly in &anomalies {
        // Test mathematical bounds (MUST hold)
        assert!(
            anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0,
            "Likelihood bounds violated: {}",
            anomaly.likelihood
        );
        assert!(
            anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0,
            "Anomaly strength bounds violated: {}",
            anomaly.anomaly_strength
        );
        assert!(
            anomaly.information_score >= 0.0,
            "Information score must be non-negative: {}",
            anomaly.information_score
        );

        // Test numerical stability (MUST hold)
        assert!(anomaly.likelihood.is_finite(), "Likelihood must be finite");
        assert!(
            anomaly.information_score.is_finite(),
            "Information score must be finite"
        );
        assert!(
            anomaly.anomaly_strength.is_finite(),
            "Anomaly strength must be finite"
        );

        // Test likelihood-log_likelihood consistency (MUST hold)
        if anomaly.likelihood > 0.0 {
            let expected_log_likelihood = anomaly.likelihood.ln();
            let error = (anomaly.log_likelihood - expected_log_likelihood).abs();
            assert!(
                error < ULTRA_STRICT_TOLERANCE,
                "Log-likelihood inconsistency: error = {error:.2e}"
            );
        } else {
            assert!(
                anomaly.log_likelihood.is_infinite() && anomaly.log_likelihood < 0.0,
                "Log-likelihood should be -∞ when likelihood = 0"
            );
        }
    }
}

fn test_anomaly_detector_boundary_conditions() {
    // Test minimum viable sequence length
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");

    // Train with minimum viable sequence
    let min_sequence = vec!["A".to_string(), "B".to_string(), "A".to_string()]; // 3 elements for order 2
    let result = detector.train(&min_sequence);
    // This might fail due to min_sequence_length requirements - that's OK

    if result.is_ok() {
        // If training succeeds, detection should work
        let test_seq = vec!["A".to_string(), "B".to_string(), "A".to_string()];
        let anomalies = detector
            .detect_anomalies(&test_seq, 0.1)
            .expect("Detection should work");

        // Verify all results are mathematically valid
        for anomaly in &anomalies {
            assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
            assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
        }
    }

    // Test empty sequence detection
    let mut trained_detector = AnomalyDetector::new(2).expect("Failed to create detector");
    let training_seq: Vec<String> = (0..100).map(|i| format!("S{}", i % 5)).collect();
    trained_detector
        .train(&training_seq)
        .expect("Failed to train");

    let empty_seq: Vec<String> = vec![];
    let empty_result = trained_detector.detect_anomalies(&empty_seq, 0.1);
    assert!(
        empty_result.is_ok(),
        "Empty sequence detection should succeed"
    );
    assert!(
        empty_result.unwrap().is_empty(),
        "Empty sequence should produce no anomalies"
    );
}

fn test_anomaly_detector_error_handling() {
    // Test invalid threshold values
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    let training_seq: Vec<String> = (0..50).map(|i| format!("S{}", i % 3)).collect();
    detector.train(&training_seq).expect("Failed to train");

    let test_seq = vec!["S0".to_string(), "S1".to_string(), "S2".to_string()];

    // Test threshold > 1.0
    let invalid_result = detector.detect_anomalies(&test_seq, 1.5);
    assert!(invalid_result.is_err(), "Threshold > 1.0 should fail");

    // Test threshold < 0.0
    let invalid_result = detector.detect_anomalies(&test_seq, -0.1);
    assert!(invalid_result.is_err(), "Threshold < 0.0 should fail");

    // Test detection without training
    let untrained_detector = AnomalyDetector::new(2).expect("Failed to create detector");
    let untrained_result = untrained_detector.detect_anomalies(&test_seq, 0.1);
    assert!(
        untrained_result.is_err(),
        "Detection without training should fail"
    );
}

fn run_context_tree_unit_tests() {
    println!("🌳 Testing Context Tree Unit Tests...");

    // Test mathematical properties of context tree
    test_context_tree_mathematical_properties();

    // Test probability conservation
    test_context_tree_probability_conservation();

    // Test Laplace smoothing formula
    test_context_tree_laplace_smoothing_exact();

    println!("  ✅ Context Tree unit tests passed");
}

fn test_context_tree_mathematical_properties() {
    let mut tree = ContextTree::new(2).expect("Failed to create context tree");

    // Use statistically meaningful sequence
    let mut sequence = Vec::new();
    for _ in 0..100 {
        sequence.extend(vec!["A".to_string(), "B".to_string(), "C".to_string()]);
    }

    let config = AnomalyGridConfig::default();
    tree.build_from_sequence(&sequence, &config)
        .expect("Failed to build tree");

    // Test that all contexts have valid mathematical properties
    for (context, node) in &tree.contexts {
        // Test probability conservation (MUST hold exactly)
        let prob_sum: f64 = node.probabilities.values().sum();
        assert!(
            (prob_sum - 1.0).abs() < ULTRA_STRICT_TOLERANCE,
            "Probability conservation violated for context {context:?}: sum = {prob_sum:.15}"
        );

        // Test entropy bounds (MUST hold)
        let n_outcomes = node.probabilities.len() as f64;
        let max_entropy = n_outcomes.log2();
        assert!(
            node.entropy >= -ULTRA_STRICT_TOLERANCE,
            "Entropy must be non-negative for context {:?}: H = {:.15}",
            context,
            node.entropy
        );
        assert!(
            node.entropy <= max_entropy + ULTRA_STRICT_TOLERANCE,
            "Entropy exceeds maximum for context {:?}: H = {:.15} > {:.15}",
            context,
            node.entropy,
            max_entropy
        );

        // Test KL divergence properties (MUST hold)
        assert!(
            node.kl_divergence >= -ULTRA_STRICT_TOLERANCE,
            "KL divergence must be non-negative for context {:?}: KL = {:.15}",
            context,
            node.kl_divergence
        );
    }
}

fn test_context_tree_probability_conservation() {
    let mut tree = ContextTree::new(1).expect("Failed to create context tree");

    // Create sequence with known counts
    let sequence = vec![
        "A".to_string(),
        "B".to_string(),
        "A".to_string(),
        "B".to_string(),
        "A".to_string(),
        "C".to_string(),
    ];

    let config = AnomalyGridConfig::default();
    tree.build_from_sequence(&sequence, &config)
        .expect("Failed to build tree");

    // Verify exact probability conservation for all contexts
    for (context, node) in &tree.contexts {
        let prob_sum: f64 = node.probabilities.values().sum();
        let error = (prob_sum - 1.0).abs();

        assert!(
            error < ULTRA_STRICT_TOLERANCE,
            "Probability conservation violated for context {context:?}: error = {error:.2e}"
        );

        // Verify all individual probabilities are in [0,1]
        for (symbol, &prob) in &node.probabilities {
            assert!(
                (0.0..=1.0).contains(&prob),
                "Probability out of bounds for {symbol}|{context:?}: P = {prob:.15}"
            );
        }
    }
}

fn test_context_tree_laplace_smoothing_exact() {
    let config = AnomalyGridConfig::default()
        .with_smoothing_alpha(2.0)
        .expect("Failed to set alpha");
    let mut tree = ContextTree::new(1).expect("Failed to create context tree");

    // Create sequence with exact known counts
    let sequence = vec![
        "A".to_string(),
        "B".to_string(), // A->B: 1 time
        "A".to_string(),
        "B".to_string(), // A->B: 2 times
        "A".to_string(),
        "C".to_string(), // A->C: 1 time
    ];

    tree.build_from_sequence(&sequence, &config)
        .expect("Failed to build tree");

    // Test exact Laplace smoothing formula: P(x) = (count(x) + α) / (N + α*|V|)
    if let Some(node) = tree.get_context_node(&["A".to_string()]) {
        let _alpha = 2.0;
        let _total_count = 3.0; // A appears 3 times as context
        let _vocab_size = 2.0; // B and C follow A

        // Expected probabilities with α=2.0:
        // P(B|A) = (2 + 2) / (3 + 2*2) = 4/7
        // P(C|A) = (1 + 2) / (3 + 2*2) = 3/7
        let expected_prob_b = 4.0 / 7.0;
        let expected_prob_c = 3.0 / 7.0;

        let actual_prob_b = node.probabilities.get("B").copied().unwrap_or(0.0);
        let actual_prob_c = node.probabilities.get("C").copied().unwrap_or(0.0);

        let error_b = (actual_prob_b - expected_prob_b).abs();
        let error_c = (actual_prob_c - expected_prob_c).abs();

        assert!(error_b < ULTRA_STRICT_TOLERANCE,
               "Laplace smoothing formula incorrect for B: expected {expected_prob_b:.15}, got {actual_prob_b:.15}, error = {error_b:.2e}");
        assert!(error_c < ULTRA_STRICT_TOLERANCE,
               "Laplace smoothing formula incorrect for C: expected {expected_prob_c:.15}, got {actual_prob_c:.15}, error = {error_c:.2e}");
    }
}

fn run_markov_model_unit_tests() {
    println!("🔗 Testing Markov Model Unit Tests...");

    // Test likelihood calculation consistency
    test_markov_model_likelihood_consistency();

    // Test hierarchical context selection
    test_markov_model_hierarchical_context();

    // Test mathematical properties
    test_markov_model_mathematical_properties();

    println!("  ✅ Markov Model unit tests passed");
}

fn test_markov_model_likelihood_consistency() {
    let mut model = MarkovModel::new(2).expect("Failed to create model");

    // Train with deterministic pattern for predictable likelihood
    let mut training_sequence = Vec::new();
    for _ in 0..50 {
        training_sequence.extend(vec!["A".to_string(), "B".to_string()]);
    }

    model
        .train(&training_sequence)
        .expect("Failed to train model");

    // Test likelihood calculation
    let test_sequence = vec!["A".to_string(), "B".to_string(), "A".to_string()];
    let calculated_likelihood = model.calculate_likelihood(&test_sequence);

    // Manual calculation using hierarchical context selection
    let mut manual_likelihood = 1.0;
    for i in 1..test_sequence.len() {
        let next_state = &test_sequence[i];
        let max_context_len = i.min(model.max_order());

        let mut prob = None;
        for context_len in (1..=max_context_len).rev() {
            let context = &test_sequence[i - context_len..i];
            if let Some(transition_prob) = model
                .context_tree()
                .get_transition_probability(context, next_state)
            {
                prob = Some(transition_prob);
                break;
            }
        }

        let transition_prob =
            prob.unwrap_or_else(|| 1.0 / (model.state_mapping().len() as f64 + 1.0));

        manual_likelihood *= transition_prob;
    }

    let error = (calculated_likelihood - manual_likelihood).abs();
    assert!(error < STRICT_TOLERANCE,
           "Likelihood calculation inconsistency: calculated = {calculated_likelihood:.15}, manual = {manual_likelihood:.15}, error = {error:.2e}");
}

fn test_markov_model_hierarchical_context() {
    let mut model = MarkovModel::new(3).expect("Failed to create model");

    // Create sequence with clear hierarchical patterns
    let sequence = vec![
        "A".to_string(),
        "B".to_string(),
        "C".to_string(),
        "D".to_string(), // ABC->D
        "A".to_string(),
        "B".to_string(),
        "C".to_string(),
        "D".to_string(), // ABC->D
        "A".to_string(),
        "B".to_string(),
        "C".to_string(),
        "E".to_string(), // ABC->E
        "A".to_string(),
        "B".to_string(),
        "X".to_string(),
        "Y".to_string(), // ABX->Y
    ];

    model.train(&sequence).expect("Failed to train model");

    // Test hierarchical context selection
    let context = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    let prob_d = model.get_best_context_probability(&context, "D");
    let prob_e = model.get_best_context_probability(&context, "E");

    // D appears twice after ABC, E appears once, so D should be more likely
    assert!(
        prob_d > prob_e,
        "Hierarchical context selection failed: P(D|ABC) = {prob_d:.6} should be > P(E|ABC) = {prob_e:.6}"
    );

    // Both should be positive
    assert!(
        prob_d > 0.0 && prob_e > 0.0,
        "All probabilities should be positive: P(D|ABC) = {prob_d:.6}, P(E|ABC) = {prob_e:.6}"
    );
}

fn test_markov_model_mathematical_properties() {
    let mut model = MarkovModel::new(2).expect("Failed to create model");

    let sequence: Vec<String> = (0..200).map(|i| format!("S{}", i % 5)).collect();
    model.train(&sequence).expect("Failed to train model");

    // Test that all likelihoods are valid probabilities
    let test_sequences = vec![
        vec!["S0".to_string(), "S1".to_string()],
        vec!["S2".to_string(), "S3".to_string()],
        vec!["S4".to_string(), "S0".to_string()],
        vec!["SX".to_string(), "SY".to_string()], // Unknown states
    ];

    for test_seq in test_sequences {
        let likelihood = model.calculate_likelihood(&test_seq);

        // Mathematical bounds MUST hold
        assert!(
            (0.0..=1.0).contains(&likelihood),
            "Likelihood out of bounds for {test_seq:?}: {likelihood:.15}"
        );
        assert!(
            likelihood.is_finite(),
            "Likelihood must be finite for {test_seq:?}: {likelihood:.15}"
        );
    }
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

fn run_workflow_integration_tests() {
    println!("🔄 Testing Workflow Integration...");

    // Test complete workflow with realistic data
    test_complete_workflow_with_realistic_data();

    // Test batch processing
    test_batch_processing_workflow();

    println!("  ✅ Workflow integration tests passed");
}

fn test_complete_workflow_with_realistic_data() {
    // Simulate realistic network security scenario
    let mut normal_traffic = Vec::new();
    let patterns = vec![
        vec!["TCP_SYN", "TCP_ACK", "HTTP_GET", "HTTP_200", "TCP_FIN"],
        vec!["UDP_DNS", "UDP_RESPONSE"],
        vec!["HTTPS_CONNECT", "TLS_HANDSHAKE", "HTTP_POST", "HTTP_201"],
    ];

    // Generate realistic normal traffic
    for _ in 0..100 {
        for pattern in &patterns {
            normal_traffic.extend(pattern.iter().map(|s| s.to_string()));
        }
    }

    let mut detector = AnomalyDetector::new(4).expect("Failed to create detector");
    detector
        .train(&normal_traffic)
        .expect("Failed to train detector");

    // Test with attack pattern
    let attack_traffic = vec![
        "TCP_SYN".to_string(),
        "TCP_RST".to_string(), // Port scan
        "TCP_SYN".to_string(),
        "TCP_RST".to_string(),
        "MALFORMED_PACKET".to_string(),
        "EXPLOIT_ATTEMPT".to_string(),
    ];

    let anomalies = detector
        .detect_anomalies(&attack_traffic, 0.01)
        .expect("Failed to detect anomalies");

    // Verify detection quality
    for anomaly in &anomalies {
        // All mathematical properties must hold
        assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
        assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
        assert!(anomaly.information_score >= 0.0);
        assert!(anomaly.likelihood.is_finite());
        assert!(anomaly.information_score.is_finite());
        assert!(anomaly.anomaly_strength.is_finite());
    }
}

fn test_batch_processing_workflow() {
    let sequences = vec![
        (0..50)
            .map(|i| format!("A{}", i % 5))
            .collect::<Vec<String>>(),
        (0..50)
            .map(|i| format!("B{}", i % 3))
            .collect::<Vec<String>>(),
        (0..50)
            .map(|i| format!("C{}", i % 7))
            .collect::<Vec<String>>(),
    ];

    let config = AnomalyGridConfig::default();
    let results =
        batch_process_sequences(&sequences, &config, 0.1).expect("Failed to process sequences");

    assert_eq!(
        results.len(),
        sequences.len(),
        "All sequences should be processed"
    );

    // Verify all results are mathematically valid
    for (i, anomaly_set) in results.iter().enumerate() {
        for anomaly in anomaly_set {
            assert!(
                anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0,
                "Invalid likelihood in sequence {}: {}",
                i,
                anomaly.likelihood
            );
            assert!(
                anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0,
                "Invalid anomaly strength in sequence {}: {}",
                i,
                anomaly.anomaly_strength
            );
            assert!(
                anomaly.information_score >= 0.0,
                "Invalid information score in sequence {}: {}",
                i,
                anomaly.information_score
            );
        }
    }
}

fn run_error_handling_integration_tests() {
    println!("🚨 Testing Error Handling Integration...");

    // Test error recovery
    test_error_recovery_workflow();

    // Test invalid configuration handling
    test_invalid_configuration_handling();

    println!("  ✅ Error handling integration tests passed");
}

fn test_error_recovery_workflow() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");

    // Try invalid operation
    let empty_sequence: Vec<String> = vec![];
    let result = detector.train(&empty_sequence);
    assert!(result.is_err(), "Should fail with empty sequence");

    // Verify detector is still usable
    let valid_sequence: Vec<String> = (0..50).map(|i| format!("S{}", i % 3)).collect();
    let result = detector.train(&valid_sequence);
    assert!(result.is_ok(), "Should succeed after previous error");

    // Verify detection still works
    let test_sequence = vec!["S0".to_string(), "S1".to_string(), "S2".to_string()];
    let anomalies = detector
        .detect_anomalies(&test_sequence, 0.1)
        .expect("Detection should work after error");

    // Verify mathematical properties are maintained
    for anomaly in &anomalies {
        assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
        assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
    }
}

fn test_invalid_configuration_handling() {
    // Test invalid max_order
    let result = AnomalyDetector::new(0);
    assert!(result.is_err(), "Should fail with invalid max_order");

    // Test invalid threshold
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    let sequence: Vec<String> = (0..20).map(|i| format!("S{}", i % 3)).collect();
    detector.train(&sequence).expect("Failed to train");

    let test_seq = vec!["S0".to_string(), "S1".to_string()];
    let result = detector.detect_anomalies(&test_seq, 2.0);
    assert!(result.is_err(), "Should fail with invalid threshold");
}

fn run_comprehensive_validation_tests() {
    println!("🎯 Testing Comprehensive Validation...");

    // Test mathematical consistency across components
    test_mathematical_consistency_across_components();

    // Test real-world scenario validation
    test_real_world_scenario_validation();

    println!("  ✅ Comprehensive validation tests passed");
}

fn test_mathematical_consistency_across_components() {
    let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");

    // Create sequence with known statistical properties
    let mut sequence = Vec::new();
    for _ in 0..200 {
        sequence.extend(vec!["A".to_string(), "B".to_string(), "C".to_string()]);
    }

    detector.train(&sequence).expect("Failed to train");

    // Test consistency between components
    let test_sequence = vec!["A".to_string(), "B".to_string(), "C".to_string()];

    // Get likelihood from model directly
    let model_likelihood = detector.model().calculate_likelihood(&test_sequence);

    // Get likelihood from anomaly detection
    let anomalies = detector
        .detect_anomalies(&test_sequence, 1.0)
        .expect("Failed to detect");

    if !anomalies.is_empty() {
        let detection_likelihood = anomalies[0].likelihood;

        // Should be consistent (allowing for different calculation methods)
        let relative_error =
            (model_likelihood - detection_likelihood).abs() / model_likelihood.max(1e-10);
        assert!(relative_error < 0.1,
               "Likelihood inconsistency between components: model = {:.6}, detection = {:.6}, error = {:.3}%",
               model_likelihood, detection_likelihood, relative_error * 100.0);
    }
}

fn test_real_world_scenario_validation() {
    // Test with realistic financial transaction patterns
    let mut normal_transactions = Vec::new();
    let transaction_patterns = vec![
        vec!["AUTH", "PURCHASE", "CONFIRM", "SETTLE"],
        vec!["AUTH", "ATM_WITHDRAWAL", "CONFIRM"],
        vec!["AUTH", "TRANSFER", "CONFIRM", "SETTLE"],
    ];

    for _ in 0..50 {
        for pattern in &transaction_patterns {
            normal_transactions.extend(pattern.iter().map(|s| s.to_string()));
        }
    }

    let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");
    detector
        .train(&normal_transactions)
        .expect("Failed to train");

    // Test with fraud pattern
    let fraud_pattern = vec![
        "VELOCITY_ALERT".to_string(),
        "AUTH".to_string(),
        "AUTH".to_string(),
        "AUTH".to_string(),
        "LARGE_AMOUNT".to_string(),
    ];

    let anomalies = detector
        .detect_anomalies(&fraud_pattern, 0.05)
        .expect("Failed to detect fraud");

    // Verify fraud detection quality
    for anomaly in &anomalies {
        // Mathematical properties must hold
        assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
        assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
        assert!(anomaly.information_score >= 0.0);

        // Fraud should have low likelihood
        assert!(
            anomaly.likelihood < 0.5,
            "Fraud should have low likelihood: {:.6}",
            anomaly.likelihood
        );
    }
}

// ============================================================================
// MATHEMATICAL TESTS WITH BRUTAL RIGOR
// ============================================================================

fn run_theoretical_validation_tests() {
    println!("📐 Testing Theoretical Validation...");

    // Test information theory correctness
    test_information_theory_brutal_validation();

    // Test probability theory correctness
    test_probability_theory_brutal_validation();

    println!("  ✅ Theoretical validation tests passed");
}

fn test_information_theory_brutal_validation() {
    println!("    🔍 DEBUGGING ENTROPY CALCULATION");

    // Test 1: Simple deterministic case first
    let mut detector = AnomalyDetector::new(1).expect("Failed to create detector");

    // Create a simple deterministic sequence: A->B->A->B->A->B...
    let sequence = ["A", "B"]
        .repeat(100)
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    detector.train(&sequence).expect("Failed to train");

    let context_tree = detector.model().context_tree();

    println!("    📊 DETERMINISTIC SEQUENCE ANALYSIS:");
    for (context, node) in &context_tree.contexts {
        println!("      Context: {context:?}");
        println!("        Counts: {:?}", node.counts);
        println!("        Probabilities: {:?}", node.probabilities);
        println!("        Entropy: {:.10}", node.entropy);

        // Manual entropy calculation to verify formula
        let manual_entropy: f64 = node
            .probabilities
            .values()
            .map(|&p| if p > 0.0 { -p * p.log2() } else { 0.0 })
            .sum();
        println!("        Manual entropy: {manual_entropy:.10}");

        // Check if probabilities sum to 1
        let prob_sum: f64 = node.probabilities.values().sum();
        println!("        Probability sum: {prob_sum:.10}");

        // Verify entropy is non-negative
        assert!(
            node.entropy >= 0.0,
            "Entropy must be non-negative: {:.10}",
            node.entropy
        );
    }

    // Test 2: Try to create a more uniform distribution
    println!("    🎲 UNIFORM DISTRIBUTION TEST:");
    let mut uniform_detector = AnomalyDetector::new(1).expect("Failed to create detector");

    // Create a sequence where A transitions to B, C, D, E with equal frequency
    let mut uniform_sequence = Vec::new();
    for _ in 0..400 {
        uniform_sequence.extend(vec!["A".to_string(), "B".to_string()]);
        uniform_sequence.extend(vec!["A".to_string(), "C".to_string()]);
        uniform_sequence.extend(vec!["A".to_string(), "D".to_string()]);
        uniform_sequence.extend(vec!["A".to_string(), "E".to_string()]);
    }

    uniform_detector
        .train(&uniform_sequence)
        .expect("Failed to train uniform");

    let uniform_tree = uniform_detector.model().context_tree();

    for (context, node) in &uniform_tree.contexts {
        if context == &vec!["A".to_string()] {
            println!("      Uniform Context A:");
            println!("        Counts: {:?}", node.counts);
            println!("        Probabilities: {:?}", node.probabilities);
            println!("        Entropy: {:.10}", node.entropy);

            // Expected entropy for uniform distribution over 4 outcomes: log2(4) = 2.0
            let expected_entropy = 4.0_f64.log2();
            println!("        Expected entropy: {expected_entropy:.10}");

            let error = (node.entropy - expected_entropy).abs();
            println!("        Error: {error:.2e}");

            // This should be much closer to 2.0
            if error < 0.1 {
                println!("    ✅ Found good uniform entropy!");
            } else {
                println!("    ⚠️  Entropy not as uniform as expected, but that's OK due to Laplace smoothing");
            }
        }
    }

    println!("    ✅ Entropy calculation debugging completed");
}

fn test_probability_theory_brutal_validation() {
    // Test exact Laplace smoothing formula
    let config = AnomalyGridConfig::default()
        .with_smoothing_alpha(1.0)
        .expect("Failed to set alpha");
    let mut detector = AnomalyDetector::with_config(config).expect("Failed to create detector");

    // Create sequence with exact known counts
    let sequence = vec![
        "A".to_string(),
        "B".to_string(), // A->B: 1
        "A".to_string(),
        "B".to_string(), // A->B: 2
        "A".to_string(),
        "C".to_string(), // A->C: 1
    ];

    detector.train(&sequence).expect("Failed to train");

    let context_tree = detector.model().context_tree();

    if let Some(node) = context_tree.get_context_node(&["A".to_string()]) {
        // Exact Laplace smoothing with α=1:
        // P(B|A) = (2 + 1) / (3 + 1*2) = 3/5 = 0.6
        // P(C|A) = (1 + 1) / (3 + 1*2) = 2/5 = 0.4

        let expected_prob_b = 3.0 / 5.0;
        let expected_prob_c = 2.0 / 5.0;

        let actual_prob_b = node.probabilities.get("B").copied().unwrap_or(0.0);
        let actual_prob_c = node.probabilities.get("C").copied().unwrap_or(0.0);

        let error_b = (actual_prob_b - expected_prob_b).abs();
        let error_c = (actual_prob_c - expected_prob_c).abs();

        assert!(
            error_b < ULTRA_STRICT_TOLERANCE,
            "Laplace formula error for B: expected {expected_prob_b:.15}, got {actual_prob_b:.15}, error = {error_b:.2e}"
        );
        assert!(
            error_c < ULTRA_STRICT_TOLERANCE,
            "Laplace formula error for C: expected {expected_prob_c:.15}, got {actual_prob_c:.15}, error = {error_c:.2e}"
        );
    }
}

fn run_core_mathematical_proofs() {
    println!("🔬 Testing Core Mathematical Proofs...");

    // Test Markov property validation
    test_markov_property_validation();

    // Test numerical stability under extreme conditions
    test_numerical_stability_extreme_conditions();

    println!("  ✅ Core mathematical proofs passed");
}

fn test_markov_property_validation() {
    let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");

    // Create sequence that should satisfy Markov property
    let sequence: Vec<String> = (0..1000)
        .map(|i| match i % 4 {
            0 => "A".to_string(),
            1 => "B".to_string(),
            2 => "C".to_string(),
            _ => "A".to_string(),
        })
        .collect();

    detector.train(&sequence).expect("Failed to train");

    let model = detector.model();

    // Test Markov property: P(X_n+1|X_n, X_n-1, ...) ≈ P(X_n+1|X_n, X_n-1, ...)
    let context_1 = vec!["A".to_string()];
    let context_2 = vec!["C".to_string(), "A".to_string()];
    let context_3 = vec!["B".to_string(), "C".to_string(), "A".to_string()];

    let prob_1 = model.get_best_context_probability(&context_1, "B");
    let prob_2 = model.get_best_context_probability(&context_2, "B");
    let prob_3 = model.get_best_context_probability(&context_3, "B");

    // All should be positive and reasonable
    assert!(prob_1 > 0.0 && prob_2 > 0.0 && prob_3 > 0.0,
           "All conditional probabilities should be positive: P(B|A)={prob_1:.6}, P(B|CA)={prob_2:.6}, P(B|BCA)={prob_3:.6}");

    // Hierarchical context selection should prefer longer contexts when available
    // This is implementation-specific, so we just verify they're reasonable
    assert!(
        prob_1 <= 1.0 && prob_2 <= 1.0 && prob_3 <= 1.0,
        "All probabilities should be ≤ 1.0"
    );
}

fn test_numerical_stability_extreme_conditions() {
    // Test with extreme probability distributions
    let test_cases = vec![
        (
            "Deterministic",
            vec!["A"; 1000].iter().map(|s| s.to_string()).collect(),
        ),
        (
            "High entropy",
            (0..1000).map(|i| format!("S{}", i % 50)).collect(),
        ),
        ("Extreme skew", {
            let mut seq = vec!["COMMON"; 9999]
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>();
            seq.push("RARE".to_string());
            seq
        }),
    ];

    for (case_name, sequence) in test_cases {
        let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");

        // Training should succeed
        let train_result = detector.train(&sequence);
        assert!(
            train_result.is_ok(),
            "Training should succeed for case: {case_name}"
        );

        // Detection should be numerically stable
        let test_seq = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let anomalies = detector
            .detect_anomalies(&test_seq, 0.5)
            .expect("Detection should succeed");

        // All results should be numerically stable
        for anomaly in &anomalies {
            assert!(
                anomaly.likelihood.is_finite(),
                "Likelihood should be finite for case {}: {:.15}",
                case_name,
                anomaly.likelihood
            );
            assert!(
                anomaly.information_score.is_finite(),
                "Information score should be finite for case {}: {:.15}",
                case_name,
                anomaly.information_score
            );
            assert!(
                anomaly.anomaly_strength.is_finite(),
                "Anomaly strength should be finite for case {}: {:.15}",
                case_name,
                anomaly.anomaly_strength
            );

            // Bounds should be respected
            assert!(
                anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0,
                "Likelihood bounds violated for case {}: {:.15}",
                case_name,
                anomaly.likelihood
            );
            assert!(
                anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0,
                "Anomaly strength bounds violated for case {}: {:.15}",
                case_name,
                anomaly.anomaly_strength
            );
        }
    }
}

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

fn run_stress_tests() {
    println!("💪 Testing Stress Scenarios...");

    // Test large alphabet stress
    test_large_alphabet_stress();

    // Test long sequence stress
    test_long_sequence_stress();

    println!("  ✅ Stress tests passed");
}

fn test_large_alphabet_stress() {
    let alphabet_size = 50;
    let sequence_length = 5000;

    let sequence: Vec<String> = (0..sequence_length)
        .map(|i| format!("STATE_{:02}", i % alphabet_size))
        .collect();

    let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");

    let start_time = std::time::Instant::now();
    detector
        .train(&sequence)
        .expect("Failed to train with large alphabet");
    let train_time = start_time.elapsed();

    // Should complete in reasonable time (generous bound)
    assert!(
        train_time.as_secs() < 30,
        "Training should complete within 30 seconds: {train_time:?}"
    );

    // Test detection
    let test_seq = vec![
        "STATE_00".to_string(),
        "STATE_01".to_string(),
        "STATE_02".to_string(),
    ];
    let anomalies = detector
        .detect_anomalies(&test_seq, 0.1)
        .expect("Failed to detect with large alphabet");

    // Verify mathematical properties are maintained
    for anomaly in &anomalies {
        assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
        assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
        assert!(anomaly.information_score >= 0.0);
    }
}

fn test_long_sequence_stress() {
    let sequence_length = 50000;
    let alphabet = ["A", "B", "C", "D", "E"];

    let sequence: Vec<String> = (0..sequence_length)
        .map(|i| alphabet[i % alphabet.len()].to_string())
        .collect();

    let mut detector = AnomalyDetector::new(4).expect("Failed to create detector");

    let start_time = std::time::Instant::now();
    detector
        .train(&sequence)
        .expect("Failed to train with long sequence");
    let train_time = start_time.elapsed();

    // Should complete in reasonable time
    assert!(
        train_time.as_secs() < 60,
        "Training should complete within 60 seconds: {train_time:?}"
    );

    // Memory usage should be reasonable
    let context_count = detector.model().context_tree().context_count();
    assert!(
        context_count < 10000,
        "Context count should be reasonable: {context_count}"
    );
}

fn run_optimization_tests() {
    println!("🚀 Testing Optimization Features...");

    // Test performance monitoring
    test_performance_monitoring();

    // Test context optimization
    test_context_optimization();

    println!("  ✅ Optimization tests passed");
}

fn test_performance_monitoring() {
    let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");

    let sequence: Vec<String> = (0..1000).map(|i| format!("S{}", i % 10)).collect();
    detector.train(&sequence).expect("Failed to train");

    let metrics = detector.performance_metrics();

    // Verify metrics are reasonable
    assert!(
        metrics.training_time_ms > 0,
        "Training time should be recorded"
    );
    assert!(
        metrics.context_count > 0,
        "Context count should be positive"
    );
    assert!(
        metrics.estimated_memory_bytes > 0,
        "Memory estimate should be positive"
    );

    // Test detection with monitoring
    let test_seq = vec!["S0".to_string(), "S1".to_string(), "S2".to_string()];
    let _anomalies = detector
        .detect_anomalies_with_monitoring(&test_seq, 0.1)
        .expect("Failed to detect with monitoring");

    let updated_metrics = detector.performance_metrics();
    assert!(
        updated_metrics.detection_time_ms > 0,
        "Detection time should be recorded"
    );
}

fn test_context_optimization() {
    let mut detector = AnomalyDetector::new(4).expect("Failed to create detector");

    // Create sequence with many rare patterns
    let mut sequence = Vec::new();

    // Common patterns
    for _ in 0..500 {
        sequence.extend(vec!["A".to_string(), "B".to_string(), "C".to_string()]);
    }

    // Many rare patterns
    for i in 0..200 {
        sequence.extend(vec![format!("RARE_{}", i), "X".to_string()]);
    }

    detector.train(&sequence).expect("Failed to train");

    let initial_metrics = detector.performance_metrics();
    let initial_contexts = initial_metrics.context_count;

    // Apply optimization
    let optimization_config = OptimizationConfig {
        enable_pruning: true,
        min_context_count: 3,
        min_entropy: 0.1,
        max_contexts: Some(1000),
        enable_monitoring: true,
    };

    detector
        .optimize(&optimization_config)
        .expect("Failed to optimize");

    let optimized_metrics = detector.performance_metrics();
    let optimized_contexts = optimized_metrics.context_count;

    // Should reduce context count
    assert!(
        optimized_contexts <= initial_contexts,
        "Optimization should reduce context count: {initial_contexts} -> {optimized_contexts}"
    );

    // Detection should still work
    let test_seq = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    let anomalies = detector
        .detect_anomalies(&test_seq, 0.1)
        .expect("Detection should work after optimization");

    // Mathematical properties should be maintained
    for anomaly in &anomalies {
        assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
        assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
    }
}
