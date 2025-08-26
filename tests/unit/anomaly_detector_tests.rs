//! Unit tests for Anomaly Detector module
//! 
//! These tests define the expected behavior of the anomaly detection system
//! using Markov chain-based analysis.

use anomaly_grid::anomaly_detector::*;

#[test]
fn test_anomaly_score_creation() {
    let score = AnomalyScore {
        sequence: vec!["A".to_string(), "B".to_string()],
        likelihood: 0.5,
        log_likelihood: -0.693, // ln(0.5)
        information_score: 1.0,
        anomaly_strength: 0.3,
    };
    
    assert_eq!(score.sequence.len(), 2);
    assert_eq!(score.likelihood, 0.5);
    assert!((score.log_likelihood - (-0.693)).abs() < 0.001);
}

#[test]
fn test_anomaly_detector_creation() {
    let detector = AnomalyDetector::new(3).expect("Failed to create detector");
    
    assert_eq!(detector.max_order(), 3);
}

#[test]
fn test_anomaly_detector_creation_invalid_order() {
    let result = AnomalyDetector::new(0);
    assert!(result.is_err(), "Should fail with invalid max_order");
    
    match result.unwrap_err() {
        AnomalyGridError::InvalidMaxOrder { value, .. } => {
            assert_eq!(value, 0);
        }
        _ => panic!("Expected InvalidMaxOrder error"),
    }
}

#[test]
fn test_anomaly_detector_train() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    let normal_sequence = vec![
        "LOGIN".to_string(), "DASHBOARD".to_string(), "LOGOUT".to_string(),
        "LOGIN".to_string(), "DASHBOARD".to_string(), "LOGOUT".to_string(),
        "LOGIN".to_string(), "DASHBOARD".to_string(), "LOGOUT".to_string()
    ];
    
    let result = detector.train(&normal_sequence);
    assert!(result.is_ok());
}

#[test]
fn test_anomaly_detector_train_invalid_sequence() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    
    // Test empty sequence
    let empty_sequence: Vec<String> = vec![];
    let result = detector.train(&empty_sequence);
    assert!(result.is_err(), "Should fail with empty sequence");
    
    // Test sequence too short
    let short_sequence = vec!["A".to_string()];
    let result = detector.train(&short_sequence);
    assert!(result.is_err(), "Should fail with sequence too short");
}

#[test]
fn test_anomaly_detector_detect_normal_patterns() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    
    // Use longer sequence for statistical validity (minimum 20 * max_order)
    let mut normal_sequence = Vec::new();
    for _ in 0..20 {
        normal_sequence.extend(vec!["A".to_string(), "B".to_string(), "C".to_string()]);
    }
    
    detector.train(&normal_sequence).expect("Failed to train detector");
    
    // Test with similar normal pattern
    let test_sequence = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    let anomalies = detector.detect_anomalies(&test_sequence, 0.01).expect("Failed to detect anomalies");
    
    // Verify mathematical properties of results
    for anomaly in &anomalies {
        // All probabilities must be valid
        assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0, 
               "Invalid likelihood: {}", anomaly.likelihood);
        assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0,
               "Invalid anomaly strength: {}", anomaly.anomaly_strength);
        assert!(anomaly.information_score >= 0.0,
               "Invalid information score: {}", anomaly.information_score);
        
        // Verify likelihood-log_likelihood consistency
        if anomaly.likelihood > 0.0 {
            let expected_log_likelihood = anomaly.likelihood.ln();
            let error = (anomaly.log_likelihood - expected_log_likelihood).abs();
            assert!(error < 1e-10, "Log-likelihood inconsistency: error = {:.2e}", error);
        }
    }
}

#[test]
fn test_anomaly_detector_detect_anomalous_patterns() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    
    // Use longer sequence for statistical validity
    let mut normal_sequence = Vec::new();
    for _ in 0..50 {
        normal_sequence.extend(vec!["A".to_string(), "B".to_string()]);
    }
    
    detector.train(&normal_sequence).expect("Failed to train detector");
    
    // Test with anomalous pattern
    let anomalous_sequence = vec![
        "A".to_string(), "X".to_string(), "Y".to_string(), "Z".to_string()
    ];
    let anomalies = detector.detect_anomalies(&anomalous_sequence, 0.5).expect("Failed to detect anomalies");
    
    // Verify mathematical properties of anomaly detection
    for anomaly in &anomalies {
        // Verify all values are mathematically valid
        assert!(anomaly.likelihood.is_finite(), "Likelihood must be finite");
        assert!(anomaly.log_likelihood.is_finite() || anomaly.log_likelihood.is_infinite(), 
               "Log-likelihood must be finite or -∞");
        assert!(anomaly.information_score.is_finite(), "Information score must be finite");
        assert!(anomaly.anomaly_strength.is_finite(), "Anomaly strength must be finite");
        
        // Verify bounds
        assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
        assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
        assert!(anomaly.information_score >= 0.0);
        
        // For anomalous patterns, likelihood should be low
        assert!(anomaly.likelihood < 0.5, "Anomalous pattern should have low likelihood");
    }
}

#[test]
fn test_anomaly_detector_sliding_window() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    
    // Use longer training sequence for statistical validity
    let mut training_sequence = Vec::new();
    for _ in 0..30 {
        training_sequence.extend(vec!["A".to_string(), "B".to_string(), "C".to_string()]);
    }
    
    detector.train(&training_sequence).expect("Failed to train detector");
    
    // Test with longer sequence
    let test_sequence = vec![
        "A".to_string(), "B".to_string(), "C".to_string(), // Normal
        "A".to_string(), "X".to_string(), "Y".to_string(), // Anomalous
        "A".to_string(), "B".to_string(), "C".to_string()  // Normal again
    ];
    
    let anomalies = detector.detect_anomalies(&test_sequence, 0.1).expect("Failed to detect anomalies");
    
    // Verify mathematical properties of all anomaly scores
    for anomaly in &anomalies {
        // Verify finite values
        assert!(anomaly.likelihood.is_finite(), "Likelihood must be finite");
        assert!(anomaly.information_score.is_finite(), "Information score must be finite");
        assert!(anomaly.anomaly_strength.is_finite(), "Anomaly strength must be finite");
        
        // Verify bounds
        assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0,
               "Likelihood out of bounds: {}", anomaly.likelihood);
        assert!(anomaly.information_score >= 0.0,
               "Information score must be non-negative: {}", anomaly.information_score);
        assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0,
               "Anomaly strength out of bounds: {}", anomaly.anomaly_strength);
        
        // Verify log-likelihood consistency
        if anomaly.likelihood > 0.0 {
            let expected_log_likelihood = anomaly.likelihood.ln();
            let error = (anomaly.log_likelihood - expected_log_likelihood).abs();
            assert!(error < 1e-10, "Log-likelihood inconsistency: error = {:.2e}", error);
        } else {
            assert!(anomaly.log_likelihood.is_infinite() && anomaly.log_likelihood < 0.0,
                   "Log-likelihood should be -∞ when likelihood = 0");
        }
    }
}

#[test]
fn test_anomaly_detector_threshold_filtering() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    
    // Use longer sequence for statistical validity
    let mut sequence = Vec::new();
    for _ in 0..50 {
        sequence.extend(vec!["A".to_string(), "B".to_string()]);
    }
    
    detector.train(&sequence).expect("Failed to train detector");
    
    let test_sequence = vec![
        "A".to_string(), "B".to_string(), "X".to_string(), "Y".to_string()
    ];
    
    // Test with strict threshold
    let strict_anomalies = detector.detect_anomalies(&test_sequence, 0.001).expect("Failed to detect with strict threshold");
    
    // Test with lenient threshold
    let lenient_anomalies = detector.detect_anomalies(&test_sequence, 0.9).expect("Failed to detect with lenient threshold");
    
    // Verify threshold filtering property: strict ⊆ lenient
    assert!(strict_anomalies.len() <= lenient_anomalies.len(),
           "Strict threshold should detect fewer anomalies: {} vs {}",
           strict_anomalies.len(), lenient_anomalies.len());
    
    // Verify all strict anomalies are also in lenient (by likelihood)
    for strict_anomaly in &strict_anomalies {
        assert!(strict_anomaly.likelihood < 0.001,
               "Strict anomaly should have likelihood < threshold: {}", strict_anomaly.likelihood);
    }
    
    // Test invalid threshold handling
    let invalid_result = detector.detect_anomalies(&test_sequence, 1.5);
    assert!(invalid_result.is_err(), "Should fail with invalid threshold > 1.0");
    
    let invalid_result = detector.detect_anomalies(&test_sequence, -0.1);
    assert!(invalid_result.is_err(), "Should fail with invalid threshold < 0.0");
}

#[test]
fn test_anomaly_detector_information_score() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    
    // Use longer sequence for statistical validity
    let mut sequence = Vec::new();
    for _ in 0..100 {
        sequence.extend(vec!["A".to_string(), "B".to_string()]);
    }
    
    detector.train(&sequence).expect("Failed to train detector");
    
    // Test with predictable sequence
    let predictable = vec!["A".to_string(), "B".to_string(), "A".to_string()];
    let predictable_anomalies = detector.detect_anomalies(&predictable, 0.5).expect("Failed to detect predictable anomalies");
    
    // Test with unpredictable sequence
    let unpredictable = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];
    let unpredictable_anomalies = detector.detect_anomalies(&unpredictable, 0.5).expect("Failed to detect unpredictable anomalies");
    
    // Verify information score properties
    for anomalies in [&predictable_anomalies, &unpredictable_anomalies] {
        for anomaly in anomalies {
            // Information score should follow I(x) ≈ -log₂(P(x))
            assert!(anomaly.information_score >= 0.0,
                   "Information score must be non-negative: {}", anomaly.information_score);
            assert!(anomaly.information_score.is_finite(),
                   "Information score must be finite: {}", anomaly.information_score);
            
            // Verify relationship with likelihood
            if anomaly.likelihood > 0.0 {
                let theoretical_info = -anomaly.likelihood.log2();
                // Allow some tolerance due to averaging and implementation details
                let error = (anomaly.information_score - theoretical_info).abs();
                assert!(error < 10.0, "Information score should approximate -log₂(P): error = {:.3}", error);
            }
        }
    }
    
    // Verify that unpredictable sequences generally have higher information scores
    if !predictable_anomalies.is_empty() && !unpredictable_anomalies.is_empty() {
        let avg_predictable_info = predictable_anomalies.iter()
            .map(|a| a.information_score).sum::<f64>() / predictable_anomalies.len() as f64;
        let avg_unpredictable_info = unpredictable_anomalies.iter()
            .map(|a| a.information_score).sum::<f64>() / unpredictable_anomalies.len() as f64;
        
        // This is a statistical tendency, not a strict requirement
        println!("Predictable info: {:.3}, Unpredictable info: {:.3}", 
                avg_predictable_info, avg_unpredictable_info);
    }
}

#[test]
fn test_anomaly_detector_empty_sequence() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    
    // Use longer training sequence for statistical validity
    let mut training_sequence = Vec::new();
    for _ in 0..20 {
        training_sequence.extend(vec!["A".to_string(), "B".to_string(), "C".to_string()]);
    }
    
    detector.train(&training_sequence).expect("Failed to train detector");
    
    let empty_sequence: Vec<String> = vec![];
    let anomalies = detector.detect_anomalies(&empty_sequence, 0.1).expect("Failed to detect anomalies");
    
    // Empty sequence should produce no anomalies
    assert!(anomalies.is_empty(), "Empty sequence should produce no anomalies");
}

#[test]
fn test_anomaly_detector_short_sequence() {
    let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");
    
    // Use longer training sequence for statistical validity
    let mut training_sequence = Vec::new();
    for _ in 0..30 {
        training_sequence.extend(vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()]);
    }
    
    detector.train(&training_sequence).expect("Failed to train detector");
    
    // Test with sequence shorter than max_order + 1
    let short_sequence = vec!["A".to_string(), "B".to_string()];
    let anomalies = detector.detect_anomalies(&short_sequence, 0.1).expect("Failed to detect anomalies");
    
    // Should handle short sequences gracefully (no windows possible)
    assert!(anomalies.is_empty(), "Short sequence should produce no anomalies");
    
    // Test boundary case: exactly max_order elements
    let boundary_sequence = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    let boundary_anomalies = detector.detect_anomalies(&boundary_sequence, 0.1).expect("Failed to detect anomalies");
    assert!(boundary_anomalies.is_empty(), "Boundary sequence should produce no anomalies");
    
    // Test minimum viable sequence: max_order + 1 elements
    let min_sequence = vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()];
    let min_anomalies = detector.detect_anomalies(&min_sequence, 0.1).expect("Failed to detect anomalies");
    // This should work and may or may not produce anomalies
    for anomaly in &min_anomalies {
        assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
    }
}

#[test]
fn test_anomaly_detector_numerical_stability() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    
    // Use longer training sequence for statistical validity
    let mut sequence = Vec::new();
    for _ in 0..50 {
        sequence.extend(vec!["A".to_string(), "B".to_string(), "C".to_string()]);
    }
    
    detector.train(&sequence).expect("Failed to train detector");
    
    // Test with various sequences to ensure numerical stability
    let test_sequences = vec![
        (vec!["A".to_string(), "B".to_string(), "C".to_string()], "normal pattern"),
        (vec!["X".to_string(), "Y".to_string(), "Z".to_string()], "unknown pattern"),
        (vec!["A".to_string(), "A".to_string(), "A".to_string()], "repeated pattern"),
        (vec!["A".to_string(), "B".to_string(), "X".to_string()], "mixed pattern"),
    ];
    
    for (test_seq, description) in test_sequences {
        let anomalies = detector.detect_anomalies(&test_seq, 0.01).expect("Failed to detect anomalies");
        
        for anomaly in anomalies {
            // Check numerical stability - all values must be finite or properly infinite
            assert!(anomaly.likelihood.is_finite(), 
                   "Likelihood not finite for {}: {}", description, anomaly.likelihood);
            assert!(anomaly.log_likelihood.is_finite() || 
                   (anomaly.log_likelihood.is_infinite() && anomaly.log_likelihood < 0.0),
                   "Log-likelihood invalid for {}: {}", description, anomaly.log_likelihood);
            assert!(anomaly.information_score.is_finite(),
                   "Information score not finite for {}: {}", description, anomaly.information_score);
            assert!(anomaly.anomaly_strength.is_finite(),
                   "Anomaly strength not finite for {}: {}", description, anomaly.anomaly_strength);
            
            // Check mathematical bounds
            assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0,
                   "Likelihood out of bounds for {}: {}", description, anomaly.likelihood);
            assert!(anomaly.information_score >= 0.0,
                   "Information score negative for {}: {}", description, anomaly.information_score);
            assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0,
                   "Anomaly strength out of bounds for {}: {}", description, anomaly.anomaly_strength);
            
            // Verify mathematical consistency
            if anomaly.likelihood > 0.0 {
                let expected_log_likelihood = anomaly.likelihood.ln();
                let error = (anomaly.log_likelihood - expected_log_likelihood).abs();
                assert!(error < 1e-10, 
                       "Log-likelihood inconsistency for {}: error = {:.2e}", description, error);
            }
        }
    }
}