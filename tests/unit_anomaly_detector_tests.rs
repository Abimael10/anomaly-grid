//! Unit tests for AnomalyDetector
//!
//! These tests focus on the core functionality of the AnomalyDetector,
//! ensuring mathematical correctness and API reliability.

use anomaly_grid::*;

#[test]
fn test_detector_creation() {
    // Valid creation
    let detector = AnomalyDetector::new(3);
    assert!(detector.is_ok());
    assert_eq!(detector.unwrap().max_order(), 3);

    // Invalid creation
    let invalid_detector = AnomalyDetector::new(0);
    assert!(invalid_detector.is_err());
}

#[test]
fn test_detector_with_config() {
    let config = AnomalyGridConfig::default()
        .with_max_order(2)
        .expect("Failed to set max_order");
    
    let detector = AnomalyDetector::with_config(config);
    assert!(detector.is_ok());
    assert_eq!(detector.unwrap().max_order(), 2);
}

#[test]
fn test_basic_training_and_detection() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    
    // Train with normal pattern
    let training_sequence = vec!["A", "B", "C", "A", "B", "C"].repeat(20)
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    
    let train_result = detector.train(&training_sequence);
    assert!(train_result.is_ok(), "Training should succeed");
    
    // Test normal sequence (should have low anomaly scores)
    let normal_sequence = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    let normal_anomalies = detector.detect_anomalies(&normal_sequence, 0.0)
        .expect("Failed to detect anomalies");
    
    // Test anomalous sequence (should have higher anomaly scores)
    let anomalous_sequence = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];
    let anomalous_anomalies = detector.detect_anomalies(&anomalous_sequence, 0.0)
        .expect("Failed to detect anomalies");
    
    // Verify mathematical properties
    for anomaly in &normal_anomalies {
        assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
        assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
        assert!(anomaly.information_score >= 0.0);
        assert!(anomaly.likelihood.is_finite());
        assert!(anomaly.anomaly_strength.is_finite());
        assert!(anomaly.information_score.is_finite());
    }
    
    for anomaly in &anomalous_anomalies {
        assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
        assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
        assert!(anomaly.information_score >= 0.0);
        assert!(anomaly.likelihood.is_finite());
        assert!(anomaly.anomaly_strength.is_finite());
        assert!(anomaly.information_score.is_finite());
    }
}

#[test]
fn test_threshold_behavior() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    
    // Train with simple pattern
    let training_sequence = vec!["NORMAL", "PATTERN"].repeat(50)
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    detector.train(&training_sequence).expect("Failed to train");
    
    let test_sequence = vec!["ANOMALY".to_string(), "SEQUENCE".to_string()];
    
    // Test threshold monotonicity: higher thresholds should give fewer results
    let low_threshold_anomalies = detector.detect_anomalies(&test_sequence, 0.0)
        .expect("Failed to detect with low threshold");
    let high_threshold_anomalies = detector.detect_anomalies(&test_sequence, 0.9)
        .expect("Failed to detect with high threshold");
    
    assert!(high_threshold_anomalies.len() <= low_threshold_anomalies.len(),
           "Higher thresholds should detect fewer anomalies");
}

#[test]
fn test_invalid_thresholds() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    
    let training_sequence = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    detector.train(&training_sequence).expect("Failed to train");
    
    let test_sequence = vec!["X".to_string(), "Y".to_string()];
    
    // Test invalid thresholds
    assert!(detector.detect_anomalies(&test_sequence, 1.5).is_err());
    assert!(detector.detect_anomalies(&test_sequence, -0.1).is_err());
    assert!(detector.detect_anomalies(&test_sequence, f64::NAN).is_err());
    assert!(detector.detect_anomalies(&test_sequence, f64::INFINITY).is_err());
}

#[test]
fn test_untrained_detector() {
    let detector = AnomalyDetector::new(2).expect("Failed to create detector");
    
    let test_sequence = vec!["A".to_string(), "B".to_string()];
    let result = detector.detect_anomalies(&test_sequence, 0.5);
    
    assert!(result.is_err(), "Detection without training should fail");
}

#[test]
fn test_sequence_length_handling() {
    let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");
    
    // Train with sufficient data
    let training_sequence = vec!["A", "B", "C", "D"].repeat(25)
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    detector.train(&training_sequence).expect("Failed to train");
    
    // Test empty sequence
    let empty_sequence: Vec<String> = vec![];
    let empty_result = detector.detect_anomalies(&empty_sequence, 0.1);
    assert!(empty_result.is_ok());
    assert!(empty_result.unwrap().is_empty());
    
    // Test single element
    let single_sequence = vec!["X".to_string()];
    let single_result = detector.detect_anomalies(&single_sequence, 0.1);
    assert!(single_result.is_ok());
    
    // Test short sequence
    let short_sequence = vec!["X".to_string(), "Y".to_string()];
    let short_result = detector.detect_anomalies(&short_sequence, 0.1);
    assert!(short_result.is_ok());
    
    // Test normal length sequence
    let normal_sequence = vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()];
    let normal_result = detector.detect_anomalies(&normal_sequence, 0.1);
    assert!(normal_result.is_ok());
}

#[test]
fn test_performance_monitoring() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    
    let training_sequence = vec!["A", "B", "C"].repeat(30)
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    detector.train(&training_sequence).expect("Failed to train");
    
    // Check that training metrics are recorded
    let metrics = detector.performance_metrics();
    assert!(metrics.training_time_ms >= 0);
    assert!(metrics.context_count > 0);
    assert!(metrics.estimated_memory_bytes > 0);
    
    // Test detection with monitoring
    let test_sequence = vec!["X".to_string(), "Y".to_string()];
    let _result = detector.detect_anomalies_with_monitoring(&test_sequence, 0.1)
        .expect("Failed to detect with monitoring");
    
    // Check that detection metrics are recorded
    let updated_metrics = detector.performance_metrics();
    assert!(updated_metrics.detection_time_ms > 0);
}

#[test]
fn test_sequence_training() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    
    let sequences = vec![
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
        vec!["D".to_string(), "E".to_string(), "F".to_string()],
        vec!["G".to_string(), "H".to_string(), "I".to_string()],
    ];
    
    let result = detector.train_sequences(&sequences);
    assert!(result.is_ok(), "Sequence training should succeed");
    
    // Test that the detector can detect anomalies after sequence training
    let test_sequence = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];
    let anomalies = detector.detect_anomalies(&test_sequence, 0.0)
        .expect("Failed to detect after sequence training");
    
    // Verify mathematical properties
    for anomaly in &anomalies {
        assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
        assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
        assert!(anomaly.information_score >= 0.0);
    }
}

#[test]
fn test_batch_processing() {
    let sequences = vec![
        vec!["A", "B", "C"].iter().map(|s| s.to_string()).collect(),
        vec!["D", "E", "F"].iter().map(|s| s.to_string()).collect(),
        vec!["G", "H", "I"].iter().map(|s| s.to_string()).collect(),
    ];
    
    let config = AnomalyGridConfig::default();
    let results = batch_process_sequences(&sequences, &config, 0.1);
    
    assert!(results.is_ok(), "Batch processing should succeed");
    let anomaly_sets = results.unwrap();
    assert_eq!(anomaly_sets.len(), sequences.len());
    
    // Verify mathematical properties for all results
    for anomaly_set in &anomaly_sets {
        for anomaly in anomaly_set {
            assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
            assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
            assert!(anomaly.information_score >= 0.0);
        }
    }
}

#[test]
fn test_optimization() {
    let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");
    
    // Train with data that will create many contexts
    let mut training_sequence = Vec::new();
    for i in 0..100 {
        training_sequence.extend(vec![
            format!("STATE_{}", i % 10),
            format!("NEXT_{}", (i + 1) % 10),
        ]);
    }
    detector.train(&training_sequence).expect("Failed to train");
    
    let initial_metrics = detector.performance_metrics();
    let initial_contexts = initial_metrics.context_count;
    
    // Apply optimization
    let optimization_config = OptimizationConfig {
        enable_pruning: true,
        min_context_count: 2,
        min_entropy: 0.1,
        max_contexts: Some(50),
        enable_monitoring: true,
    };
    
    let optimize_result = detector.optimize(&optimization_config);
    assert!(optimize_result.is_ok(), "Optimization should succeed");
    
    let optimized_metrics = detector.performance_metrics();
    let optimized_contexts = optimized_metrics.context_count;
    
    // Should reduce context count
    assert!(optimized_contexts <= initial_contexts,
           "Optimization should reduce or maintain context count");
    
    // Detection should still work after optimization
    let test_sequence = vec!["STATE_0".to_string(), "NEXT_1".to_string()];
    let anomalies = detector.detect_anomalies(&test_sequence, 0.1)
        .expect("Detection should work after optimization");
    
    // Verify mathematical properties are maintained
    for anomaly in &anomalies {
        assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
        assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
        assert!(anomaly.information_score >= 0.0);
    }
}

#[test]
fn test_mathematical_consistency() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    
    // Train with deterministic pattern
    let training_sequence = vec!["A", "B", "A", "B"].repeat(50)
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    detector.train(&training_sequence).expect("Failed to train");
    
    let test_sequences = vec![
        vec!["A".to_string(), "B".to_string()],  // Known pattern
        vec!["B".to_string(), "A".to_string()],  // Known pattern
        vec!["X".to_string(), "Y".to_string()],  // Unknown pattern
    ];
    
    for test_sequence in test_sequences {
        let anomalies = detector.detect_anomalies(&test_sequence, 0.0)
            .expect("Failed to detect anomalies");
        
        for anomaly in &anomalies {
            // Test mathematical bounds
            assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0,
                   "Likelihood bounds violated: {}", anomaly.likelihood);
            assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0,
                   "Anomaly strength bounds violated: {}", anomaly.anomaly_strength);
            assert!(anomaly.information_score >= 0.0,
                   "Information score must be non-negative: {}", anomaly.information_score);
            
            // Test numerical stability
            assert!(anomaly.likelihood.is_finite(), "Likelihood must be finite");
            assert!(anomaly.anomaly_strength.is_finite(), "Anomaly strength must be finite");
            assert!(anomaly.information_score.is_finite(), "Information score must be finite");
            
            // Test likelihood-log_likelihood consistency
            if anomaly.likelihood > 0.0 {
                let expected_log_likelihood = anomaly.likelihood.ln();
                let error = (anomaly.log_likelihood - expected_log_likelihood).abs();
                assert!(error < 1e-10,
                       "Log-likelihood inconsistency: error = {:.2e}", error);
            } else {
                assert!(anomaly.log_likelihood.is_infinite() && anomaly.log_likelihood < 0.0,
                       "Log-likelihood should be -∞ when likelihood = 0");
            }
        }
    }
}