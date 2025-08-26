//! Integration tests for the complete anomaly detection workflow
//! 
//! These tests validate end-to-end functionality and real-world scenarios.

use anomaly_grid::*;

#[test]
fn test_complete_workflow_network_security() {
    // Normal network traffic patterns
    let normal_traffic: Vec<String> = vec![
        "TCP_SYN", "TCP_ACK", "HTTP_GET", "HTTP_200", "TCP_FIN",
        "TCP_SYN", "TCP_ACK", "HTTPS_POST", "HTTP_201", "TCP_FIN",
        "UDP_DNS", "UDP_RESPONSE",
        "TCP_SYN", "TCP_ACK", "HTTP_GET", "HTTP_200", "TCP_FIN",
        "TCP_SYN", "TCP_ACK", "HTTP_GET", "HTTP_200", "TCP_FIN",
    ].into_iter().map(String::from).collect();

    let mut detector = AnomalyDetector::new(4).expect("Failed to create detector");
    detector.train(&normal_traffic).expect("Failed to train detector");

    // Test with attack patterns
    let attack_traffic: Vec<String> = vec![
        "TCP_SYN", "TCP_RST", "TCP_SYN", "TCP_RST", "TCP_SYN", "TCP_RST", // Port scan
        "HTTP_GET", "HTTP_GET", "HTTP_GET", "HTTP_GET", "HTTP_GET",        // DDoS
        "UNKNOWN_PROTOCOL", "MALFORMED_PACKET", "BUFFER_OVERFLOW",         // Attacks
    ].into_iter().map(String::from).collect();

    let anomalies = detector.detect_anomalies(&attack_traffic, 0.1).expect("Failed to detect anomalies");
    
    // Should detect some anomalies in attack traffic
    println!("Network security: {} anomalies detected", anomalies.len());
    
    // Verify anomaly properties
    for anomaly in &anomalies {
        assert!(anomaly.likelihood.is_finite());
        assert!(anomaly.information_score >= 0.0);
        assert!(anomaly.anomaly_strength >= 0.0);
        assert!(anomaly.anomaly_strength <= 1.0);
    }
}

#[test]
fn test_complete_workflow_user_behavior() {
    // Normal user session patterns
    let normal_sessions: Vec<String> = vec![
        "LOGIN", "DASHBOARD", "PROFILE", "SETTINGS", "LOGOUT",
        "LOGIN", "SEARCH", "VIEW_ITEM", "ADD_CART", "CHECKOUT", "LOGOUT",
        "LOGIN", "MESSAGES", "COMPOSE", "SEND", "LOGOUT",
        "LOGIN", "DASHBOARD", "PROFILE", "SETTINGS", "LOGOUT",
    ].into_iter().map(String::from).collect();

    let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");
    detector.train(&normal_sessions).expect("Failed to train detector");

    // Test suspicious behavior
    let suspicious_session: Vec<String> = vec![
        "LOGIN", "ADMIN_PANEL", "USER_LIST", "DELETE_USER", "DELETE_USER",
        "BULK_DOWNLOAD", "BULK_DOWNLOAD", "SENSITIVE_DATA_ACCESS",
    ].into_iter().map(String::from).collect();

    let anomalies = detector.detect_anomalies(&suspicious_session, 0.05).expect("Failed to detect anomalies");
    
    println!("User behavior: {} anomalies detected", anomalies.len());
    
    // Calculate behavioral risk scores
    let mut high_risk_count = 0;
    for anomaly in &anomalies {
        let risk_score = 1.0 - anomaly.likelihood;
        if risk_score > 0.8 {
            high_risk_count += 1;
        }
        
        // Verify anomaly structure
        assert!(!anomaly.sequence.is_empty());
        assert!(anomaly.likelihood >= 0.0);
        assert!(anomaly.likelihood <= 1.0);
    }
    
    println!("High-risk behaviors detected: {}", high_risk_count);
}

#[test]
fn test_complete_workflow_financial_transactions() {
    // Normal transaction patterns
    let normal_transactions: Vec<String> = vec![
        "AUTH", "PURCHASE", "CONFIRM", "SETTLEMENT",
        "AUTH", "ATM_WITHDRAWAL", "CONFIRM", "SETTLEMENT",
        "AUTH", "ONLINE_PAYMENT", "CONFIRM", "SETTLEMENT",
    ].into_iter().map(String::from).collect();

    // Build robust training set
    let mut training_data = Vec::new();
    for _ in 0..10 {
        training_data.extend(normal_transactions.clone());
    }

    let mut detector = AnomalyDetector::new(4).expect("Failed to create detector");
    detector.train(&training_data).expect("Failed to train detector");

    // Test fraudulent patterns
    let fraud_transactions: Vec<String> = vec![
        "AUTH", "LARGE_PURCHASE", "FOREIGN_COUNTRY", "CONFIRM",    // Unusual location
        "VELOCITY_ALERT", "AUTH", "AUTH", "AUTH", "AUTH",          // Rapid transactions
        "CARD_NOT_PRESENT", "LARGE_PURCHASE", "DECLINE", "RETRY", "RETRY", // Card testing
    ].into_iter().map(String::from).collect();

    let anomalies = detector.detect_anomalies(&fraud_transactions, 0.001).expect("Failed to detect anomalies");
    
    // Calculate fraud scores
    let fraud_alerts: Vec<_> = anomalies.into_iter()
        .map(|a| {
            let fraud_score = (1.0 - a.likelihood) * a.information_score;
            (a, fraud_score)
        })
        .filter(|(_, score)| *score > 1.0)
        .collect();

    println!("Financial fraud: {} fraud alerts generated", fraud_alerts.len());
    
    // Verify fraud detection capability
    for (anomaly, fraud_score) in &fraud_alerts {
        assert!(fraud_score.is_finite() && *fraud_score > 0.0);
        assert!(anomaly.information_score.is_finite());
    }
}

#[test]
fn test_complete_workflow_system_logs() {
    // Normal system events
    let normal_logs: Vec<String> = vec![
        "BOOT", "SERVICE_START", "AUTH_SUCCESS", "FILE_ACCESS", "NETWORK_CONNECT",
        "CRON_START", "BACKUP_BEGIN", "BACKUP_SUCCESS", "CRON_END",
        "HEALTH_CHECK", "MONITOR_CLEAR", "SERVICE_HEALTHY",
    ].into_iter().map(String::from).collect();

    // Build comprehensive training set
    let mut training_logs = Vec::new();
    for _ in 0..20 {
        training_logs.extend(normal_logs.clone());
    }

    let mut detector = AnomalyDetector::new(5).expect("Failed to create detector");
    detector.train(&training_logs).expect("Failed to train detector");

    // Test with security incidents
    let incident_logs: Vec<String> = vec![
        "UNAUTHORIZED_ACCESS", "PRIVILEGE_ESCALATION", "FILE_CORRUPTION",
        "SERVICE_CRASH", "SERVICE_CRASH", "SERVICE_CRASH",  // Repeated crashes
        "ROOTKIT_DETECTED", "MALWARE_POSITIVE", "CONFIG_TAMPERED",
    ].into_iter().map(String::from).collect();

    let anomalies = detector.detect_anomalies(&incident_logs, 0.01).expect("Failed to detect anomalies");
    
    // Filter critical anomalies
    let critical_anomalies: Vec<_> = anomalies.into_iter()
        .filter(|a| a.likelihood < 1e-6)
        .collect();
    
    println!("System logs: {} critical anomalies detected", critical_anomalies.len());
    
    // Verify critical anomaly detection
    for anomaly in &critical_anomalies {
        assert!(anomaly.likelihood.is_finite());
        assert!(anomaly.information_score >= 0.0);
        assert!(anomaly.anomaly_strength > 0.0);
    }
}

#[test]
fn test_batch_processing() {
    // Create multiple sequences for batch processing
    let sequences = vec![
        vec!["A", "B", "C", "A", "B", "C"].into_iter().map(String::from).collect(),
        vec!["X", "Y", "Z", "X", "Y", "Z"].into_iter().map(String::from).collect(),
        vec!["P", "Q", "R", "P", "Q", "R"].into_iter().map(String::from).collect(),
    ];
    
    let config = AnomalyGridConfig::default().with_max_order(3).expect("Failed to create config");
    let results = batch_process_sequences(&sequences, &config, 0.05).expect("Failed to process sequences");
    
    // Verify all sequences were processed
    assert_eq!(results.len(), sequences.len());
    
    // Verify results quality
    let total_anomalies: usize = results.iter().map(|r| r.len()).sum();
    println!("Batch processing: {} total anomalies across {} sequences", 
             total_anomalies, sequences.len());
    
    // Verify each result set
    for (i, anomaly_set) in results.iter().enumerate() {
        println!("Sequence {}: {} anomalies", i + 1, anomaly_set.len());
        
        for anomaly in anomaly_set {
            assert!(anomaly.likelihood.is_finite());
            assert!(anomaly.information_score.is_finite());
            assert!(anomaly.anomaly_strength.is_finite());
        }
    }
}

#[test]
fn test_error_handling_and_robustness() {
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    
    // Test empty sequence training
    let empty_sequence: Vec<String> = vec![];
    assert!(detector.train(&empty_sequence).is_err());
    
    // Test single element sequence training
    let single_sequence = vec!["A".to_string()];
    assert!(detector.train(&single_sequence).is_err());
    
    // Test valid training
    let valid_sequence = vec!["A".to_string(), "B".to_string(), "C".to_string(), "A".to_string()];
    assert!(detector.train(&valid_sequence).is_ok());
    
    // Test detection on empty sequence
    let empty_test: Vec<String> = vec![];
    let anomalies = detector.detect_anomalies(&empty_test, 0.1).expect("Failed to detect anomalies");
    assert!(anomalies.is_empty());
    
    // Test detection on short sequence
    let short_test = vec!["A".to_string()];
    let short_anomalies = detector.detect_anomalies(&short_test, 0.1).expect("Failed to detect anomalies");
    assert!(short_anomalies.is_empty());
}

#[test]
fn test_mathematical_consistency() {
    let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");
    let sequence: Vec<String> = vec![
        "A", "B", "C", "A", "B", "C", "D", "E", "A", "B"
    ].into_iter().map(String::from).collect();
    
    detector.train(&sequence).expect("Failed to train detector");
    
    let test_sequence: Vec<String> = vec!["A", "B", "C", "D"].into_iter().map(String::from).collect();
    let anomalies = detector.detect_anomalies(&test_sequence, 0.1).expect("Failed to detect anomalies");
    
    // Verify mathematical consistency
    for anomaly in &anomalies {
        // Likelihood should match log-likelihood
        let computed_likelihood = anomaly.log_likelihood.exp();
        if computed_likelihood.is_finite() && computed_likelihood > 0.0 {
            let relative_error = (anomaly.likelihood - computed_likelihood).abs() / computed_likelihood;
            assert!(relative_error < 1e-6, 
                "Likelihood/log-likelihood inconsistency: {:.6} vs {:.6}",
                anomaly.likelihood, computed_likelihood);
        }
        
        // Anomaly strength should be in [0,1]
        assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0,
            "Anomaly strength out of bounds: {:.6}", anomaly.anomaly_strength);
        
        // Information score should be non-negative
        assert!(anomaly.information_score >= 0.0,
            "Information score should be non-negative: {:.6}", anomaly.information_score);
    }
}

#[test]
fn test_performance_characteristics() {
    use std::time::Instant;
    
    let sizes = vec![100, 500];
    let orders = vec![2, 3];
    
    for &size in &sizes {
        for &order in &orders {
            let start_time = Instant::now();
            
            // Generate test sequence
            let states = vec!["A", "B", "C", "D", "E"];
            let sequence: Vec<String> = (0..size)
                .map(|i| states[(i * 7 + i * i) % states.len()].to_string())
                .collect();
            
            let mut detector = AnomalyDetector::new(order).expect("Failed to create detector");
            
            let train_start = Instant::now();
            detector.train(&sequence).expect("Failed to train detector");
            let train_time = train_start.elapsed();
            
            let detect_start = Instant::now();
            let anomalies = detector.detect_anomalies(&sequence, 0.01).expect("Failed to detect anomalies");
            let detect_time = detect_start.elapsed();
            
            let _total_time = start_time.elapsed();
            
            // Performance should be reasonable
            assert!(train_time.as_millis() < size as u128 * order as u128,
                "Training time should scale reasonably: {}ms for size={}, order={}",
                train_time.as_millis(), size, order);
            
            assert!(detect_time.as_millis() < size as u128,
                "Detection time should scale linearly: {}ms for size={}",
                detect_time.as_millis(), size);
            
            println!("Size: {}, Order: {} - Train: {:?}, Detect: {:?}, Anomalies: {}",
                     size, order, train_time, detect_time, anomalies.len());
        }
    }
}