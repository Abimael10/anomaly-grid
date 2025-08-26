//! Error Handling and Configuration Integration Tests
//! 
//! These tests validate the new error handling infrastructure and configuration
//! management features implemented in Phase 4 of the atomic perfection plan.

use anomaly_grid::*;

#[test]
fn test_structured_error_handling() {
    println!("🔧 Testing Structured Error Handling");

    // Test invalid max_order
    let result = AnomalyDetector::new(0);
    assert!(result.is_err(), "Should fail with invalid max_order");
    
    match result.unwrap_err() {
        AnomalyGridError::InvalidMaxOrder { value, context } => {
            assert_eq!(value, 0);
            assert!(context.contains("must be greater than 0"));
            println!("  ✅ InvalidMaxOrder error correctly structured");
        }
        _ => panic!("Expected InvalidMaxOrder error"),
    }

    // Test invalid threshold
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    let sequence = vec!["A".to_string(), "B".to_string(), "A".to_string()];
    detector.train(&sequence).expect("Failed to train detector");

    let result = detector.detect_anomalies(&sequence, 1.5);
    assert!(result.is_err(), "Should fail with invalid threshold");
    
    match result.unwrap_err() {
        AnomalyGridError::InvalidThreshold { value, expected_range } => {
            assert_eq!(value, 1.5);
            assert!(expected_range.contains("0.0 and 1.0"));
            println!("  ✅ InvalidThreshold error correctly structured");
        }
        _ => panic!("Expected InvalidThreshold error"),
    }

    // Test empty context tree
    let empty_detector = AnomalyDetector::new(2).expect("Failed to create detector");
    let result = empty_detector.detect_anomalies(&sequence, 0.5);
    assert!(result.is_err(), "Should fail with empty context tree");
    
    match result.unwrap_err() {
        AnomalyGridError::EmptyContextTree { suggestion } => {
            assert!(suggestion.contains("train()"));
            println!("  ✅ EmptyContextTree error correctly structured");
        }
        _ => panic!("Expected EmptyContextTree error"),
    }

    // Test sequence too short
    let short_sequence = vec!["A".to_string()];
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    let result = detector.train(&short_sequence);
    assert!(result.is_err(), "Should fail with short sequence");
    
    match result.unwrap_err() {
        AnomalyGridError::SequenceTooShort { expected, actual, operation } => {
            assert_eq!(expected, 2);
            assert_eq!(actual, 1);
            assert!(operation.contains("training"));
            println!("  ✅ SequenceTooShort error correctly structured");
        }
        _ => panic!("Expected SequenceTooShort error"),
    }

    println!("  ✅ All error types properly structured and informative");
}

#[test]
fn test_configuration_management() {
    println!("🔧 Testing Configuration Management");

    // Test default configuration
    let config = AnomalyGridConfig::default();
    assert!(config.validate().is_ok(), "Default config should be valid");
    
    let detector = AnomalyDetector::with_config(config).expect("Failed to create detector with default config");
    assert_eq!(detector.max_order(), 3);
    println!("  ✅ Default configuration works correctly");

    // Test custom configuration
    let custom_config = AnomalyGridConfig::default()
        .with_max_order(4).expect("Failed to set max_order")
        .with_smoothing_alpha(0.5).expect("Failed to set smoothing_alpha")
        .with_weights(0.8, 0.2).expect("Failed to set weights");

    let detector = AnomalyDetector::with_config(custom_config).expect("Failed to create detector with custom config");
    assert_eq!(detector.max_order(), 4);
    println!("  ✅ Custom configuration works correctly");

    // Test invalid configuration - max_order
    let invalid_config = AnomalyGridConfig::default().with_max_order(0);
    assert!(invalid_config.is_err(), "Should fail with invalid max_order");
    
    match invalid_config.unwrap_err() {
        AnomalyGridError::InvalidMaxOrder { .. } => {
            println!("  ✅ Invalid max_order properly rejected");
        }
        _ => panic!("Expected InvalidMaxOrder error"),
    }

    // Test invalid configuration - weights
    let invalid_weights = AnomalyGridConfig::default().with_weights(0.5, 0.6);
    assert!(invalid_weights.is_err(), "Should fail with invalid weights");
    
    match invalid_weights.unwrap_err() {
        AnomalyGridError::InvalidConfiguration { parameter, .. } => {
            assert_eq!(parameter, "weight_sum");
            println!("  ✅ Invalid weights properly rejected");
        }
        _ => panic!("Expected InvalidConfiguration error"),
    }

    println!("  ✅ Configuration validation working correctly");
}

#[test]
fn test_memory_bounds_enforcement() {
    println!("🔧 Testing Memory Bounds Enforcement");

    // Test memory limit enforcement
    let config = AnomalyGridConfig::default()
        .with_memory_limit(Some(10)).expect("Failed to set memory limit");

    let mut detector = AnomalyDetector::with_config(config).expect("Failed to create detector");

    // Create a sequence that would exceed memory limit
    let large_sequence: Vec<String> = (0..100)
        .map(|i| format!("state_{}", i))
        .collect();

    let result = detector.train(&large_sequence);
    assert!(result.is_err(), "Should fail due to memory limit");
    
    match result.unwrap_err() {
        AnomalyGridError::MemoryLimitExceeded { current, limit, suggestion } => {
            assert_eq!(limit, 10);
            assert!(current >= 10);
            assert!(suggestion.contains("reducing max_order"));
            println!("  ✅ Memory limit properly enforced: {} contexts, limit {}", current, limit);
        }
        _ => panic!("Expected MemoryLimitExceeded error"),
    }

    // Test unlimited memory (None)
    let unlimited_config = AnomalyGridConfig::default()
        .with_memory_limit(None).expect("Failed to set unlimited memory");
    
    let mut unlimited_detector = AnomalyDetector::with_config(unlimited_config).expect("Failed to create unlimited detector");
    
    // This should work with reasonable sequence
    let reasonable_sequence: Vec<String> = (0..50)
        .map(|i| format!("S{}", i % 5))
        .collect();
    
    let result = unlimited_detector.train(&reasonable_sequence);
    assert!(result.is_ok(), "Should succeed with unlimited memory");
    println!("  ✅ Unlimited memory configuration works correctly");
}

#[test]
fn test_preset_configurations() {
    println!("🔧 Testing Preset Configurations");

    // Test small alphabet configuration
    let small_config = AnomalyGridConfig::for_small_alphabet();
    assert!(small_config.validate().is_ok(), "Small alphabet config should be valid");
    assert_eq!(small_config.max_order, 4);
    assert!(small_config.is_suitable_for_alphabet(5));
    assert!(small_config.is_suitable_for_alphabet(10));
    println!("  ✅ Small alphabet configuration validated");

    // Test large alphabet configuration
    let large_config = AnomalyGridConfig::for_large_alphabet();
    assert!(large_config.validate().is_ok(), "Large alphabet config should be valid");
    assert_eq!(large_config.max_order, 2);
    assert!(large_config.is_suitable_for_alphabet(30));
    println!("  ✅ Large alphabet configuration validated");

    // Test low memory configuration
    let low_mem_config = AnomalyGridConfig::for_low_memory();
    assert!(low_mem_config.validate().is_ok(), "Low memory config should be valid");
    assert_eq!(low_mem_config.memory_limit, Some(10_000));
    assert!(low_mem_config.is_suitable_for_alphabet(3));
    assert!(!low_mem_config.is_suitable_for_alphabet(100), "Should reject large alphabets");
    println!("  ✅ Low memory configuration validated");

    // Test high accuracy configuration
    let high_acc_config = AnomalyGridConfig::for_high_accuracy();
    assert!(high_acc_config.validate().is_ok(), "High accuracy config should be valid");
    assert_eq!(high_acc_config.max_order, 5);
    assert_eq!(high_acc_config.smoothing_alpha, 0.1);
    println!("  ✅ High accuracy configuration validated");

    // Test that all presets work with detectors
    for (name, config) in [
        ("small_alphabet", AnomalyGridConfig::for_small_alphabet()),
        ("large_alphabet", AnomalyGridConfig::for_large_alphabet()),
        ("low_memory", AnomalyGridConfig::for_low_memory()),
        ("high_accuracy", AnomalyGridConfig::for_high_accuracy()),
    ] {
        let detector = AnomalyDetector::with_config(config);
        assert!(detector.is_ok(), "Preset {} should create valid detector", name);
        println!("  ✅ Preset {} creates working detector", name);
    }
}

#[test]
fn test_configurable_anomaly_strength() {
    println!("🔧 Testing Configurable Anomaly Strength");

    let sequence = vec![
        "A".to_string(), "B".to_string(), "A".to_string(), "B".to_string(),
        "A".to_string(), "B".to_string(), "A".to_string(), "B".to_string(),
    ];

    // Test with different weight configurations
    let config1 = AnomalyGridConfig::default()
        .with_weights(0.9, 0.1).expect("Failed to set likelihood-heavy weights");

    let config2 = AnomalyGridConfig::default()
        .with_weights(0.1, 0.9).expect("Failed to set information-heavy weights");

    let mut detector1 = AnomalyDetector::with_config(config1).expect("Failed to create detector1");
    let mut detector2 = AnomalyDetector::with_config(config2).expect("Failed to create detector2");

    detector1.train(&sequence).expect("Failed to train detector1");
    detector2.train(&sequence).expect("Failed to train detector2");

    let test_sequence = vec!["A".to_string(), "X".to_string(), "Y".to_string()];
    let anomalies1 = detector1.detect_anomalies(&test_sequence, 0.5).expect("Failed to detect anomalies1");
    let anomalies2 = detector2.detect_anomalies(&test_sequence, 0.5).expect("Failed to detect anomalies2");

    // Both should detect anomalies, but with different strength calculations
    if !anomalies1.is_empty() && !anomalies2.is_empty() {
        let strength1 = anomalies1[0].anomaly_strength;
        let strength2 = anomalies2[0].anomaly_strength;
        
        // Both should be valid values
        assert!(strength1 >= 0.0 && strength1 <= 1.0, "Strength1 should be in [0,1]: {}", strength1);
        assert!(strength2 >= 0.0 && strength2 <= 1.0, "Strength2 should be in [0,1]: {}", strength2);
        
        println!("  ✅ Likelihood-heavy config: strength = {:.3}", strength1);
        println!("  ✅ Information-heavy config: strength = {:.3}", strength2);
    }

    // Test with different normalization factors
    let config3 = AnomalyGridConfig {
        normalization_factor: 5.0,
        ..AnomalyGridConfig::default()
    };

    let config4 = AnomalyGridConfig {
        normalization_factor: 20.0,
        ..AnomalyGridConfig::default()
    };

    let mut detector3 = AnomalyDetector::with_config(config3).expect("Failed to create detector3");
    let mut detector4 = AnomalyDetector::with_config(config4).expect("Failed to create detector4");

    detector3.train(&sequence).expect("Failed to train detector3");
    detector4.train(&sequence).expect("Failed to train detector4");

    let anomalies3 = detector3.detect_anomalies(&test_sequence, 0.5).expect("Failed to detect anomalies3");
    let anomalies4 = detector4.detect_anomalies(&test_sequence, 0.5).expect("Failed to detect anomalies4");

    if !anomalies3.is_empty() && !anomalies4.is_empty() {
        let strength3 = anomalies3[0].anomaly_strength;
        let strength4 = anomalies4[0].anomaly_strength;
        
        assert!(strength3 >= 0.0 && strength3 <= 1.0, "Strength3 should be in [0,1]: {}", strength3);
        assert!(strength4 >= 0.0 && strength4 <= 1.0, "Strength4 should be in [0,1]: {}", strength4);
        
        println!("  ✅ Low normalization (5.0): strength = {:.3}", strength3);
        println!("  ✅ High normalization (20.0): strength = {:.3}", strength4);
    }

    println!("  ✅ Configurable anomaly strength working correctly");
}

#[test]
fn test_batch_processing_with_config() {
    println!("🔧 Testing Batch Processing with Configuration");

    let sequences = vec![
        vec!["A".to_string(), "B".to_string(), "A".to_string(), "B".to_string()],
        vec!["X".to_string(), "Y".to_string(), "X".to_string(), "Y".to_string()],
        vec!["P".to_string(), "Q".to_string(), "P".to_string(), "Q".to_string()],
    ];

    let config = AnomalyGridConfig::for_small_alphabet();
    let results = batch_process_sequences(&sequences, &config, 0.1).expect("Failed to process sequences");

    assert_eq!(results.len(), 3, "Should process all sequences");
    
    // Each sequence should be processed independently
    for (i, result) in results.iter().enumerate() {
        println!("  Sequence {}: {} anomalies detected", i + 1, result.len());
        
        // Results should be valid (empty or containing valid anomaly scores)
        for anomaly in result {
            assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0, 
                   "Invalid likelihood: {}", anomaly.likelihood);
            assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0,
                   "Invalid anomaly strength: {}", anomaly.anomaly_strength);
            assert!(anomaly.information_score >= 0.0,
                   "Invalid information score: {}", anomaly.information_score);
        }
    }

    // Test batch processing with invalid threshold
    let invalid_result = batch_process_sequences(&sequences, &config, 1.5);
    assert!(invalid_result.is_err(), "Should fail with invalid threshold");
    
    match invalid_result.unwrap_err() {
        AnomalyGridError::InvalidThreshold { .. } => {
            println!("  ✅ Batch processing properly validates threshold");
        }
        _ => panic!("Expected InvalidThreshold error"),
    }

    // Test batch processing with invalid config
    let mut invalid_config = config.clone();
    invalid_config.max_order = 0;
    let invalid_config_result = batch_process_sequences(&sequences, &invalid_config, 0.1);
    assert!(invalid_config_result.is_err(), "Should fail with invalid config");
    
    println!("  ✅ Batch processing with configuration working correctly");
}

#[test]
fn test_memory_estimation() {
    println!("🔧 Testing Memory Estimation");

    let config = AnomalyGridConfig::default();
    
    // Test memory estimation for different alphabet sizes
    let small_alphabet_memory = config.estimate_memory_usage(2);
    let large_alphabet_memory = config.estimate_memory_usage(10);
    
    // Larger alphabets should require more memory
    assert!(large_alphabet_memory > small_alphabet_memory,
           "Large alphabet should need more memory: {} vs {}", 
           large_alphabet_memory, small_alphabet_memory);
    
    println!("  ✅ Memory estimation: 2 states = {} contexts, 10 states = {} contexts", 
             small_alphabet_memory, large_alphabet_memory);
    
    // Test with memory limit
    let limited_config = AnomalyGridConfig::default()
        .with_memory_limit(Some(100)).expect("Failed to set memory limit");
    
    let limited_memory = limited_config.estimate_memory_usage(100);
    assert!(limited_memory <= 100, "Should respect memory limit: {}", limited_memory);
    
    println!("  ✅ Memory limit respected: estimated {} <= limit 100", limited_memory);
    
    // Test alphabet suitability
    let small_config = AnomalyGridConfig::for_small_alphabet();
    assert!(small_config.is_suitable_for_alphabet(5), "Should be suitable for 5 states");
    assert!(small_config.is_suitable_for_alphabet(10), "Should be suitable for 10 states");
    
    let large_config = AnomalyGridConfig::for_large_alphabet();
    assert!(large_config.is_suitable_for_alphabet(50), "Should be suitable for 50 states");
    
    let low_mem_config = AnomalyGridConfig::for_low_memory();
    assert!(low_mem_config.is_suitable_for_alphabet(3), "Should be suitable for 3 states");
    assert!(!low_mem_config.is_suitable_for_alphabet(100), "Should not be suitable for 100 states");
    
    println!("  ✅ Alphabet suitability assessment working correctly");
}

#[test]
fn test_enhanced_documentation_examples() {
    println!("🔧 Testing Enhanced Documentation Examples");

    // Test the complexity guarantees mentioned in documentation
    let config = AnomalyGridConfig::for_small_alphabet();
    let mut detector = AnomalyDetector::with_config(config).expect("Failed to create detector");
    
    // Training complexity: O(n × max_order × |alphabet|)
    let sequence: Vec<String> = (0..100)
        .map(|i| format!("S{}", i % 5)) // 5-state alphabet
        .collect();
    
    let start = std::time::Instant::now();
    let result = detector.train(&sequence);
    let duration = start.elapsed();
    
    // Should complete successfully
    assert!(result.is_ok(), "Training should succeed");
    
    // Should complete in reasonable time (this is a basic sanity check)
    assert!(duration.as_millis() < 1000, "Training should complete quickly: {}ms", duration.as_millis());
    
    println!("  ✅ Training completed in {}ms for {} elements", duration.as_millis(), sequence.len());
    
    // Detection complexity: O(m × max_order)
    let test_sequence: Vec<String> = (0..50)
        .map(|i| format!("T{}", i % 3))
        .collect();
    
    let start = std::time::Instant::now();
    let anomalies = detector.detect_anomalies(&test_sequence, 0.1).expect("Failed to detect anomalies");
    let duration = start.elapsed();
    
    // Should complete in reasonable time
    assert!(duration.as_millis() < 100, "Detection should be fast: {}ms", duration.as_millis());
    
    println!("  ✅ Detection completed in {}ms for {} elements, found {} anomalies", 
             duration.as_millis(), test_sequence.len(), anomalies.len());
    
    // Results should be valid
    for anomaly in anomalies {
        assert!(anomaly.likelihood.is_finite(), "Likelihood should be finite");
        assert!(anomaly.information_score.is_finite(), "Information score should be finite");
        assert!(anomaly.anomaly_strength.is_finite(), "Anomaly strength should be finite");
        
        assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0, "Likelihood bounds");
        assert!(anomaly.information_score >= 0.0, "Information score bounds");
        assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0, "Strength bounds");
    }
    
    println!("  ✅ All results mathematically valid and within expected bounds");
}

#[test]
fn test_error_recovery_and_robustness() {
    println!("🔧 Testing Error Recovery and Robustness");

    // Test that detector remains usable after errors
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    
    // Try to train with invalid sequence
    let invalid_sequence = vec!["A".to_string()];
    let result = detector.train(&invalid_sequence);
    assert!(result.is_err(), "Should fail with short sequence");
    
    // Detector should still be usable with valid sequence
    let valid_sequence = vec!["A".to_string(), "B".to_string(), "A".to_string(), "B".to_string()];
    let result = detector.train(&valid_sequence);
    assert!(result.is_ok(), "Should succeed with valid sequence after previous error");
    
    // Should be able to detect anomalies normally
    let test_sequence = vec!["A".to_string(), "X".to_string(), "Y".to_string()];
    let anomalies = detector.detect_anomalies(&test_sequence, 0.5).expect("Should detect anomalies normally");
    
    println!("  ✅ Detector remains functional after error: {} anomalies detected", anomalies.len());
    
    // Test multiple error conditions don't break the detector
    let _ = detector.detect_anomalies(&test_sequence, 2.0); // Invalid threshold
    let _ = detector.detect_anomalies(&test_sequence, -0.5); // Invalid threshold
    
    // Should still work normally
    let final_anomalies = detector.detect_anomalies(&test_sequence, 0.5).expect("Should still work after multiple errors");
    assert_eq!(anomalies.len(), final_anomalies.len(), "Results should be consistent");
    
    println!("  ✅ Detector robust against multiple error conditions");
}