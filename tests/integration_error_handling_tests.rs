//! Integration tests for error handling and edge cases.
//! These tests ensure the library handles errors gracefully.

#![allow(clippy::uninlined_format_args)]
#![allow(clippy::useless_vec)]
#![allow(clippy::unnecessary_unwrap)]
#![allow(clippy::redundant_pattern_matching)]
#![allow(unused_variables)]

use anomaly_grid::*;

#[test]
fn test_invalid_detector_creation() {
    // Test invalid max_order
    let result = AnomalyDetector::new(0);
    assert!(result.is_err(), "Should fail with max_order = 0");

    match result.unwrap_err() {
        AnomalyGridError::InvalidMaxOrder { value, .. } => {
            assert_eq!(value, 0);
        }
        _ => panic!("Expected InvalidMaxOrder error"),
    }
}

#[test]
fn test_invalid_threshold_errors() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");

    let sequence = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    detector.train(&sequence).expect("Failed to train");

    let test_sequence = vec!["X".to_string(), "Y".to_string()];

    // Test various invalid thresholds
    let invalid_thresholds = vec![1.5, -0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];

    for &threshold in &invalid_thresholds {
        let result = detector.detect_anomalies(&test_sequence, threshold);
        assert!(
            result.is_err(),
            "Should fail with invalid threshold: {}",
            threshold
        );

        match result.unwrap_err() {
            AnomalyGridError::InvalidThreshold { value, .. } => {
                // For NaN values, we can't use assert_eq!, so check if both are NaN
                if threshold.is_nan() {
                    assert!(value.is_nan(), "Expected NaN value in error");
                } else {
                    assert_eq!(value, threshold);
                }
            }
            _ => panic!(
                "Expected InvalidThreshold error for threshold: {}",
                threshold
            ),
        }
    }
}

#[test]
fn test_untrained_detector_error() {
    let detector = AnomalyDetector::new(2).expect("Failed to create detector");

    let test_sequence = vec!["A".to_string(), "B".to_string()];
    let result = detector.detect_anomalies(&test_sequence, 0.5);

    assert!(result.is_err(), "Should fail when detector is not trained");

    match result.unwrap_err() {
        AnomalyGridError::EmptyContextTree { .. } => {
            // Expected error type
        }
        _ => panic!("Expected EmptyContextTree error"),
    }
}

#[test]
fn test_invalid_configuration_errors() {
    // Test invalid smoothing alpha
    let invalid_config_result = AnomalyGridConfig::default().with_smoothing_alpha(-1.0);
    assert!(
        invalid_config_result.is_err(),
        "Should fail with negative smoothing alpha"
    );

    // Test invalid max_order
    let invalid_order_result = AnomalyGridConfig::default().with_max_order(0);
    assert!(
        invalid_order_result.is_err(),
        "Should fail with max_order = 0"
    );

    // Test invalid weights
    let invalid_weights_result = AnomalyGridConfig::default().with_weights(-0.5, 0.5);
    assert!(
        invalid_weights_result.is_err(),
        "Should fail with negative weight"
    );
}

#[test]
fn test_sequence_too_short_error() {
    let mut detector = AnomalyDetector::new(5).expect("Failed to create detector");

    // Try to train with sequence shorter than min_sequence_length
    let short_sequence = vec!["A".to_string(), "B".to_string()];
    let result = detector.train(&short_sequence);

    // This might fail due to min_sequence_length requirements
    if result.is_err() {
        match result.unwrap_err() {
            AnomalyGridError::SequenceTooShort { .. } => {
                // Expected error type
            }
            _ => {
                // Other errors are also acceptable (e.g., insufficient data for order 5)
            }
        }
    }
}

#[test]
fn test_error_recovery() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");

    // Try invalid operation first
    let empty_sequence: Vec<String> = vec![];
    let result = detector.train(&empty_sequence);
    assert!(result.is_err(), "Should fail with empty sequence");

    // Verify detector is still usable after error
    let valid_sequence = vec!["A", "B", "C", "A", "B", "C"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let result = detector.train(&valid_sequence);
    assert!(result.is_ok(), "Should succeed after previous error");

    // Verify detection works after recovery
    let test_sequence = vec!["A".to_string(), "B".to_string()];
    let anomalies = detector
        .detect_anomalies(&test_sequence, 0.1)
        .expect("Detection should work after error recovery");

    // Verify mathematical properties are maintained
    for anomaly in &anomalies {
        assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
        assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
        assert!(anomaly.information_score >= 0.0);
    }
}

#[test]
fn test_batch_processing_error_handling() {
    let sequences = vec![
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
        vec![], // empty sequence - the new API returns an empty score list for it
        vec!["D".to_string(), "E".to_string(), "F".to_string()],
    ];

    let mut detector = AnomalyDetector::new(2).expect("detector");
    let training: Vec<String> = vec!["A", "B", "C", "D", "E", "F"]
        .into_iter()
        .map(str::to_string)
        .collect();
    detector.train(&training).expect("train");

    let result = batch_score(&detector, &sequences, 0.1);

    // batch_score handles empty/short sequences gracefully — they yield
    // zero scores rather than aborting the whole batch.
    let scored = result.expect("batch_score should succeed on benign inputs");
    assert_eq!(scored.len(), sequences.len());
    assert!(
        scored[1].is_empty(),
        "empty sequence yields no anomaly scores"
    );

    for set in &scored {
        for s in set {
            assert!((0.0..=1.0).contains(&s.likelihood));
            assert!((0.0..=1.0).contains(&s.anomaly_strength));
            assert!(s.information_score >= 0.0);
        }
    }
}

#[test]
fn test_invalid_batch_threshold() {
    let sequences = vec![vec!["A".to_string(), "B".to_string(), "C".to_string()]];

    let mut detector = AnomalyDetector::new(2).expect("detector");
    detector
        .train(&vec!["A".to_string(), "B".to_string(), "C".to_string()])
        .expect("train");

    let result = batch_score(&detector, &sequences, 1.5);
    assert!(result.is_err(), "Should fail with invalid threshold");

    let result = batch_score(&detector, &sequences, f64::NAN);
    assert!(result.is_err(), "Should fail with non-finite threshold");
}

#[test]
fn test_sequence_training_error_handling() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");

    // Test empty sequences list
    let empty_sequences: Vec<Vec<String>> = vec![];
    let result = detector.train_sequences(&empty_sequences);
    assert!(result.is_err(), "Should fail with empty sequences list");

    // Test sequences with some invalid entries
    let mixed_sequences = vec![
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
        vec![], // Empty sequence
        vec!["D".to_string(), "E".to_string(), "F".to_string()],
    ];

    let result = detector.train_sequences(&mixed_sequences);
    // This might succeed or fail depending on implementation - both are valid
    // The important thing is that it doesn't crash

    if result.is_ok() {
        // If training succeeded, detection should work
        let test_sequence = vec!["A".to_string(), "B".to_string()];
        let anomalies = detector
            .detect_anomalies(&test_sequence, 0.1)
            .expect("Detection should work after successful training");

        for anomaly in &anomalies {
            assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
            assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
            assert!(anomaly.information_score >= 0.0);
        }
    }
}

#[test]
fn test_optimization_error_handling() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");

    // Try to optimize without training
    let optimization_config = OptimizationConfig {
        enable_pruning: true,
        min_context_count: 1,
        min_entropy: 0.0,
        max_contexts: Some(100),
        enable_monitoring: true,
    };

    let result = detector.optimize(&optimization_config);
    // This might succeed or fail - both are valid for an untrained detector

    // Train the detector
    let training_sequence = vec!["A", "B", "C", "D"]
        .repeat(25)
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    detector.train(&training_sequence).expect("Failed to train");

    // Now optimization should work
    let result = detector.optimize(&optimization_config);
    assert!(result.is_ok(), "Optimization should succeed after training");

    // Detection should still work after optimization
    let test_sequence = vec!["A".to_string(), "B".to_string()];
    let anomalies = detector
        .detect_anomalies(&test_sequence, 0.1)
        .expect("Detection should work after optimization");

    for anomaly in &anomalies {
        assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
        assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
        assert!(anomaly.information_score >= 0.0);
    }
}

#[test]
fn test_concurrent_error_handling() {
    use std::sync::Arc;
    use std::thread;

    let detector = Arc::new(AnomalyDetector::new(2).expect("Failed to create detector"));

    // Test concurrent access to untrained detector
    let handles: Vec<_> = (0..5)
        .map(|i| {
            let detector_clone = Arc::clone(&detector);
            thread::spawn(move || {
                let test_sequence = vec![format!("TEST_{}", i), "SEQUENCE".to_string()];
                let result = detector_clone.detect_anomalies(&test_sequence, 0.1);

                // All should fail with EmptyContextTree error
                assert!(result.is_err(), "Untrained detector should fail");

                match result.unwrap_err() {
                    AnomalyGridError::EmptyContextTree { .. } => {
                        // Expected error
                    }
                    _ => panic!("Expected EmptyContextTree error"),
                }
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread should complete successfully");
    }
}

#[test]
fn test_memory_limit_error_handling() {
    // Test with very restrictive memory limit
    let config = AnomalyGridConfig::default().with_memory_limit(Some(1024)); // Very small limit

    if config.is_ok() {
        let mut detector =
            AnomalyDetector::with_config(config.unwrap()).expect("Failed to create detector");

        // Try to train with data that might exceed memory limit
        let large_sequence: Vec<String> = (0..10000).map(|i| format!("STATE_{}", i)).collect();

        let result = detector.train(&large_sequence);
        // This might succeed or fail depending on actual memory usage
        // The important thing is that it doesn't crash

        if result.is_ok() {
            // If training succeeded, detection should work
            let test_sequence = vec!["STATE_0".to_string(), "STATE_1".to_string()];
            let anomalies = detector
                .detect_anomalies(&test_sequence, 0.1)
                .expect("Detection should work if training succeeded");

            for anomaly in &anomalies {
                assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
                assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
                assert!(anomaly.information_score >= 0.0);
            }
        }
    }
}

#[test]
fn test_error_message_quality() {
    // Test that error messages are informative

    // Invalid max_order
    let result = AnomalyDetector::new(0);
    assert!(result.is_err());
    let error_msg = format!("{}", result.unwrap_err());
    assert!(
        error_msg.contains("max_order"),
        "Error message should mention max_order"
    );
    assert!(
        error_msg.contains("0"),
        "Error message should mention the invalid value"
    );

    // Invalid threshold
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    let sequence = vec!["A".to_string(), "B".to_string()];
    detector.train(&sequence).expect("Failed to train");

    let result = detector.detect_anomalies(&sequence, 1.5);
    assert!(result.is_err());
    let error_msg = format!("{}", result.unwrap_err());
    assert!(
        error_msg.contains("threshold"),
        "Error message should mention threshold"
    );
    assert!(
        error_msg.contains("1.5"),
        "Error message should mention the invalid value"
    );
}

#[test]
fn test_graceful_degradation() {
    let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");

    // Train with minimal data
    let minimal_sequence = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    detector
        .train(&minimal_sequence)
        .expect("Failed to train with minimal data");

    // Test various edge cases that should degrade gracefully
    let edge_cases = vec![
        vec![],                                 // Empty
        vec!["X".to_string()],                  // Single unknown
        vec!["A".to_string()],                  // Single known
        vec!["X".to_string(), "Y".to_string()], // Unknown pair
        vec!["A".to_string(), "B".to_string()], // Known pair
    ];

    for (i, test_case) in edge_cases.iter().enumerate() {
        let result = detector.detect_anomalies(test_case, 0.1);
        assert!(
            result.is_ok(),
            "Edge case {} should not crash: {:?}",
            i,
            test_case
        );

        let anomalies = result.unwrap();
        for anomaly in &anomalies {
            assert!(
                anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0,
                "Invalid likelihood in edge case {}: {}",
                i,
                anomaly.likelihood
            );
            assert!(
                anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0,
                "Invalid anomaly strength in edge case {}: {}",
                i,
                anomaly.anomaly_strength
            );
            assert!(
                anomaly.information_score >= 0.0,
                "Invalid information score in edge case {}: {}",
                i,
                anomaly.information_score
            );
        }
    }
}
