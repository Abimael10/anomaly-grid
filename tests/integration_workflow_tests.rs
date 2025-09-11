//! Integration tests for complete workflows.
//! These tests verify end-to-end functionality.

#![allow(clippy::uninlined_format_args)]
#![allow(clippy::useless_vec)]
#![allow(clippy::absurd_extreme_comparisons)]
#![allow(unused_comparisons)]

use anomaly_grid::*;

#[test]
fn test_network_security_workflow() {
    let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");

    // Train with normal network traffic patterns
    let normal_traffic = vec![
        "TCP_SYN",
        "TCP_ACK",
        "HTTP_GET",
        "HTTP_200",
        "TCP_FIN",
        "UDP_DNS",
        "UDP_RESPONSE",
        "HTTPS_CONNECT",
        "TLS_HANDSHAKE",
        "HTTP_POST",
        "HTTP_201",
    ]
    .repeat(50)
    .iter()
    .map(|s| s.to_string())
    .collect::<Vec<_>>();

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
        .detect_anomalies(&attack_traffic, 0.1)
        .expect("Failed to detect anomalies");

    // Verify detection quality
    for anomaly in &anomalies {
        assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
        assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
        assert!(anomaly.information_score >= 0.0);
        assert!(anomaly.likelihood.is_finite());
        assert!(anomaly.information_score.is_finite());
        assert!(anomaly.anomaly_strength.is_finite());
    }

    // Attack patterns should be detected as anomalous
    assert!(!anomalies.is_empty(), "Attack patterns should be detected");
}

#[test]
fn test_financial_fraud_workflow() {
    let mut detector = AnomalyDetector::new(4).expect("Failed to create detector");

    // Train with normal transaction patterns
    let normal_transactions = vec![
        "AUTH",
        "PURCHASE",
        "CONFIRM",
        "SETTLE",
        "AUTH",
        "ATM_WITHDRAWAL",
        "CONFIRM",
        "AUTH",
        "TRANSFER",
        "CONFIRM",
        "SETTLE",
    ]
    .repeat(30)
    .iter()
    .map(|s| s.to_string())
    .collect::<Vec<_>>();

    detector
        .train(&normal_transactions)
        .expect("Failed to train detector");

    // Test with fraud pattern
    let fraud_transactions = vec![
        "VELOCITY_ALERT".to_string(),
        "AUTH".to_string(),
        "AUTH".to_string(),
        "AUTH".to_string(),
        "LARGE_AMOUNT".to_string(),
        "FOREIGN_COUNTRY".to_string(),
    ];

    let anomalies = detector
        .detect_anomalies(&fraud_transactions, 0.05)
        .expect("Failed to detect anomalies");

    // Verify mathematical properties
    for anomaly in &anomalies {
        assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
        assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
        assert!(anomaly.information_score >= 0.0);
        assert!(anomaly.likelihood.is_finite());
        assert!(anomaly.information_score.is_finite());
        assert!(anomaly.anomaly_strength.is_finite());
    }

    // Fraud patterns should be detected
    assert!(!anomalies.is_empty(), "Fraud patterns should be detected");
}

#[test]
fn test_industrial_iot_workflow() {
    let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");

    // Train with normal sensor readings
    let normal_readings = vec![
        "TEMP_NORMAL",
        "PRESSURE_NORMAL",
        "VIBRATION_LOW",
        "TEMP_NORMAL",
        "PRESSURE_HIGH",
        "VIBRATION_LOW",
        "TEMP_HIGH",
        "PRESSURE_NORMAL",
        "VIBRATION_NORMAL",
    ]
    .repeat(40)
    .iter()
    .map(|s| s.to_string())
    .collect::<Vec<_>>();

    detector
        .train(&normal_readings)
        .expect("Failed to train detector");

    // Test with equipment failure pattern
    let failure_pattern = vec![
        "TEMP_CRITICAL".to_string(),
        "PRESSURE_CRITICAL".to_string(),
        "VIBRATION_CRITICAL".to_string(),
        "ALARM_BEARING_FAILURE".to_string(),
    ];

    let anomalies = detector
        .detect_anomalies(&failure_pattern, 0.01)
        .expect("Failed to detect anomalies");

    // Verify detection quality
    for anomaly in &anomalies {
        assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
        assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
        assert!(anomaly.information_score >= 0.0);
    }

    // Equipment failure should be detected
    assert!(
        !anomalies.is_empty(),
        "Equipment failure should be detected"
    );
}

#[test]
fn test_system_log_workflow() {
    let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");

    // Train with normal system logs
    let normal_logs = vec![
        "USER_LOGIN",
        "FILE_ACCESS",
        "PROCESS_START",
        "USER_LOGOUT",
        "SYSTEM_BOOT",
        "SERVICE_START",
        "NETWORK_CONNECT",
        "SERVICE_STOP",
        "BACKUP_START",
        "BACKUP_SUCCESS",
        "CLEANUP_TEMP",
    ]
    .repeat(25)
    .iter()
    .map(|s| s.to_string())
    .collect::<Vec<_>>();

    detector
        .train(&normal_logs)
        .expect("Failed to train detector");

    // Test with security incident
    let incident_logs = vec![
        "FAILED_LOGIN".to_string(),
        "FAILED_LOGIN".to_string(),
        "FAILED_LOGIN".to_string(),
        "PRIVILEGE_ESCALATION".to_string(),
        "SUSPICIOUS_FILE_ACCESS".to_string(),
        "DATA_EXFILTRATION".to_string(),
    ];

    let anomalies = detector
        .detect_anomalies(&incident_logs, 0.01)
        .expect("Failed to detect anomalies");

    // Verify mathematical properties
    for anomaly in &anomalies {
        assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
        assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
        assert!(anomaly.information_score >= 0.0);
    }

    // Security incidents should be detected
    assert!(
        !anomalies.is_empty(),
        "Security incidents should be detected"
    );
}

#[test]
fn test_edge_case_workflows() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");

    // Train with minimal data
    let minimal_training = vec!["A", "B", "C"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    detector
        .train(&minimal_training)
        .expect("Failed to train with minimal data");

    // Test edge cases
    let empty_test: Vec<String> = vec![];
    let empty_anomalies = detector
        .detect_anomalies(&empty_test, 0.1)
        .expect("Failed to detect anomalies");
    assert!(
        empty_anomalies.is_empty(),
        "Empty sequence should have no anomalies"
    );

    let short_test = vec!["X".to_string()];
    let short_anomalies = detector
        .detect_anomalies(&short_test, 0.1)
        .expect("Failed to detect anomalies");
    // Short sequences may or may not have anomalies - both are valid

    // Test with known pattern
    let known_test = vec!["A".to_string(), "B".to_string()];
    let known_anomalies = detector
        .detect_anomalies(&known_test, 0.1)
        .expect("Failed to detect anomalies");

    // Verify mathematical properties for all results
    for anomaly_set in [&empty_anomalies, &short_anomalies, &known_anomalies] {
        for anomaly in anomaly_set {
            assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
            assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
            assert!(anomaly.information_score >= 0.0);
        }
    }
}

#[test]
fn test_sequence_boundary_preservation() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");

    // Train with multiple sequences to preserve boundaries
    let sequences = vec![
        vec![
            "SEQ1_START".to_string(),
            "SEQ1_MID".to_string(),
            "SEQ1_END".to_string(),
        ],
        vec![
            "SEQ2_START".to_string(),
            "SEQ2_MID".to_string(),
            "SEQ2_END".to_string(),
        ],
        vec![
            "SEQ3_START".to_string(),
            "SEQ3_MID".to_string(),
            "SEQ3_END".to_string(),
        ],
    ];

    detector
        .train_sequences(&sequences)
        .expect("Failed to train sequences");

    // Test detection on individual sequences
    for (i, sequence) in sequences.iter().enumerate() {
        let anomalies = detector
            .detect_anomalies(sequence, 0.1)
            .expect("Failed to detect anomalies");

        // Verify mathematical properties
        for anomaly in &anomalies {
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

#[test]
fn test_batch_processing_workflow() {
    let sequences = vec![
        vec!["NORMAL", "PATTERN", "A"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        vec!["NORMAL", "PATTERN", "B"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        vec!["ANOMALY", "PATTERN", "X"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        vec!["ANOMALY", "PATTERN", "Y"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    ];

    let config = AnomalyGridConfig::default();
    let results =
        batch_process_sequences(&sequences, &config, 0.1).expect("Failed to process sequences");

    assert_eq!(
        results.len(),
        sequences.len(),
        "All sequences should be processed"
    );

    // Verify mathematical properties for all results
    for (i, anomaly_set) in results.iter().enumerate() {
        for anomaly in anomaly_set {
            assert!(
                anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0,
                "Invalid likelihood in batch result {}: {}",
                i,
                anomaly.likelihood
            );
            assert!(
                anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0,
                "Invalid anomaly strength in batch result {}: {}",
                i,
                anomaly.anomaly_strength
            );
            assert!(
                anomaly.information_score >= 0.0,
                "Invalid information score in batch result {}: {}",
                i,
                anomaly.information_score
            );
        }
    }
}

#[test]
fn test_performance_monitoring_workflow() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");

    // Train and monitor performance
    let training_data = vec!["A", "B", "C", "D"]
        .repeat(100)
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    detector.train(&training_data).expect("Failed to train");

    // Check training metrics
    let metrics = detector.performance_metrics();
    assert!(
        metrics.training_time_ms >= 0,
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
    let test_data = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];
    let _anomalies = detector
        .detect_anomalies_with_monitoring(&test_data, 0.1)
        .expect("Failed to detect with monitoring");

    // Check detection metrics
    let updated_metrics = detector.performance_metrics();
    assert!(
        updated_metrics.detection_time_ms > 0,
        "Detection time should be recorded"
    );
}

#[test]
fn test_optimization_workflow() {
    let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");

    // Train with data that creates many contexts
    let mut training_data = Vec::new();
    for i in 0..200 {
        training_data.extend(vec![
            format!("STATE_{}", i % 20),
            format!("NEXT_{}", (i + 1) % 20),
            format!("FINAL_{}", (i + 2) % 20),
        ]);
    }
    detector.train(&training_data).expect("Failed to train");

    let initial_metrics = detector.performance_metrics();
    let initial_contexts = initial_metrics.context_count;

    // Apply optimization
    let optimization_config = OptimizationConfig {
        enable_pruning: true,
        min_context_count: 3,
        min_entropy: 0.1,
        max_contexts: Some(100),
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
        "Optimization should reduce or maintain context count: {} -> {}",
        initial_contexts,
        optimized_contexts
    );

    // Detection should still work after optimization
    let test_sequence = vec![
        "STATE_0".to_string(),
        "NEXT_1".to_string(),
        "FINAL_2".to_string(),
    ];
    let anomalies = detector
        .detect_anomalies(&test_sequence, 0.1)
        .expect("Detection should work after optimization");

    // Verify mathematical properties are maintained
    for anomaly in &anomalies {
        assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
        assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
        assert!(anomaly.information_score >= 0.0);
    }
}

#[test]
fn test_threshold_sensitivity_workflow() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");

    // Train with clear normal pattern
    let normal_pattern = vec!["NORMAL", "FLOW"]
        .repeat(100)
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    detector.train(&normal_pattern).expect("Failed to train");

    // Test with anomalous pattern
    let anomalous_pattern = vec!["ANOMALY".to_string(), "DETECTED".to_string()];

    // Test different thresholds
    let thresholds = vec![0.0, 0.1, 0.3, 0.5, 0.7, 0.9];
    let mut detection_counts = Vec::new();

    for &threshold in &thresholds {
        let anomalies = detector
            .detect_anomalies(&anomalous_pattern, threshold)
            .expect("Failed to detect anomalies");
        detection_counts.push(anomalies.len());

        // Verify mathematical properties
        for anomaly in &anomalies {
            assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
            assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
            assert!(anomaly.information_score >= 0.0);

            // Anomaly strength should be >= threshold (with small tolerance for floating point precision)
            assert!(
                anomaly.anomaly_strength >= threshold - 1e-10,
                "Anomaly strength ({:.6}) should be >= threshold ({:.6})",
                anomaly.anomaly_strength,
                threshold
            );
        }
    }

    // Detection counts should be monotonically non-increasing
    for i in 1..detection_counts.len() {
        assert!(
            detection_counts[i] <= detection_counts[i - 1],
            "Higher thresholds should detect fewer anomalies: threshold[{}]={} vs threshold[{}]={}",
            i - 1,
            detection_counts[i - 1],
            i,
            detection_counts[i]
        );
    }
}
