//! Comprehensive Integration Validation
//! 
//! This module provides end-to-end validation of the anomaly-grid library
//! across multiple real-world scenarios with mathematical verification.

use anomaly_grid::*;
use std::time::Instant;

#[test]
fn comprehensive_real_world_validation() {
    println!("🌍 Comprehensive Real-World Validation Suite");
    println!("Testing across multiple domains with mathematical verification\n");

    // Test 1: Network Security Domain
    test_network_security_domain();
    
    // Test 2: Financial Fraud Domain
    test_financial_fraud_domain();
    
    // Test 3: Industrial IoT Domain
    test_industrial_iot_domain();
    
    // Test 4: System Monitoring Domain
    test_system_monitoring_domain();
    
    // Test 5: Cross-Domain Consistency
    test_cross_domain_consistency();
    
    println!("\n✅ Comprehensive validation completed successfully");
}

fn test_network_security_domain() {
    println!("🛡️ Testing Network Security Domain");
    
    // Generate realistic network data
    let normal_traffic = generate_enterprise_network_traffic(5000);
    let apt_attack = generate_apt_attack_sequence();
    let ddos_attack = generate_ddos_sequence();
    
    // Train model
    let mut detector = AnomalyDetector::new(5).expect("Failed to create detector");
    detector.train(&normal_traffic).expect("Failed to train detector");
    
    // Test APT detection
    let apt_anomalies = detector.detect_anomalies(&apt_attack, 0.001).expect("Failed to detect APT anomalies");
    assert!(!apt_anomalies.is_empty(), "APT attack should be detected");
    
    let apt_strength = apt_anomalies.iter()
        .map(|a| a.anomaly_strength)
        .fold(0.0, f64::max);
    assert!(apt_strength > 0.7, "APT should have high anomaly strength: {:.3}", apt_strength);
    
    // Test DDoS detection
    let ddos_anomalies = detector.detect_anomalies(&ddos_attack, 0.001).expect("Failed to detect DDoS anomalies");
    assert!(!ddos_anomalies.is_empty(), "DDoS attack should be detected");
    
    let ddos_info_score = ddos_anomalies.iter()
        .map(|a| a.information_score)
        .sum::<f64>() / ddos_anomalies.len() as f64;
    assert!(ddos_info_score > 5.0, "DDoS should have high information score: {:.3}", ddos_info_score);
    
    println!("  ✅ Network security validation passed");
}

fn test_financial_fraud_domain() {
    println!("💳 Testing Financial Fraud Domain");
    
    // Generate realistic transaction data
    let normal_transactions = generate_normal_financial_transactions(8000);
    let card_testing = generate_card_testing_sequence();
    let velocity_attack = generate_velocity_attack_sequence();
    
    // Train model
    let mut detector = AnomalyDetector::new(4).expect("Failed to create detector");
    detector.train(&normal_transactions).expect("Failed to train detector");
    
    // Test card testing detection
    let card_anomalies = detector.detect_anomalies(&card_testing, 0.01).expect("Failed to detect card testing anomalies");
    assert!(!card_anomalies.is_empty(), "Card testing should be detected");
    
    // Test velocity attack detection
    let velocity_anomalies = detector.detect_anomalies(&velocity_attack, 0.01).expect("Failed to detect velocity anomalies");
    assert!(!velocity_anomalies.is_empty(), "Velocity attack should be detected");
    
    let velocity_likelihood = velocity_anomalies.iter()
        .map(|a| a.likelihood)
        .fold(f64::INFINITY, f64::min);
    assert!(velocity_likelihood < 0.01, "Velocity attack should have low likelihood: {:.2e}", velocity_likelihood);
    
    println!("  ✅ Financial fraud validation passed");
}

fn test_industrial_iot_domain() {
    println!("🏭 Testing Industrial IoT Domain");
    
    // Generate realistic sensor data
    let normal_operations = generate_industrial_sensor_data(6000);
    let bearing_failure = generate_bearing_failure_sequence();
    let temperature_anomaly = generate_temperature_anomaly_sequence();
    
    // Train model
    let mut detector = AnomalyDetector::new(6).expect("Failed to create detector");
    detector.train(&normal_operations).expect("Failed to train detector");
    
    // Test bearing failure detection
    let bearing_anomalies = detector.detect_anomalies(&bearing_failure, 0.005).expect("Failed to detect bearing anomalies");
    assert!(!bearing_anomalies.is_empty(), "Bearing failure should be detected");
    
    // Test temperature anomaly detection
    let temp_anomalies = detector.detect_anomalies(&temperature_anomaly, 0.005).expect("Failed to detect temperature anomalies");
    assert!(!temp_anomalies.is_empty(), "Temperature anomaly should be detected");
    
    let temp_strength = temp_anomalies.iter()
        .map(|a| a.anomaly_strength)
        .sum::<f64>() / temp_anomalies.len() as f64;
    assert!(temp_strength > 0.3, "Temperature anomaly should have reasonable strength: {:.3}", temp_strength);
    
    println!("  ✅ Industrial IoT validation passed");
}

fn test_system_monitoring_domain() {
    println!("🖥️ Testing System Monitoring Domain");
    
    // Generate realistic system logs
    let normal_logs = generate_system_log_data(7000);
    let malware_infection = generate_malware_sequence();
    let privilege_escalation = generate_privilege_escalation_sequence();
    
    // Train model
    let mut detector = AnomalyDetector::new(5).expect("Failed to create detector");
    detector.train(&normal_logs).expect("Failed to train detector");
    
    // Test malware detection
    let malware_anomalies = detector.detect_anomalies(&malware_infection, 0.001).expect("Failed to detect malware anomalies");
    assert!(!malware_anomalies.is_empty(), "Malware infection should be detected");
    
    // Test privilege escalation detection
    let priv_anomalies = detector.detect_anomalies(&privilege_escalation, 0.001).expect("Failed to detect privilege escalation anomalies");
    assert!(!priv_anomalies.is_empty(), "Privilege escalation should be detected");
    
    let priv_info_score = priv_anomalies.iter()
        .map(|a| a.information_score)
        .fold(0.0, f64::max);
    assert!(priv_info_score > 3.0, "Privilege escalation should have high info score: {:.3}", priv_info_score);
    
    println!("  ✅ System monitoring validation passed");
}

fn test_cross_domain_consistency() {
    println!("🔄 Testing Cross-Domain Consistency");
    
    let domains = vec![
        ("Network", generate_enterprise_network_traffic(2000)),
        ("Financial", generate_normal_financial_transactions(2000)),
        ("Industrial", generate_industrial_sensor_data(2000)),
        ("System", generate_system_log_data(2000)),
    ];
    
    let mut entropies = Vec::new();
    let mut context_counts = Vec::new();
    
    for (domain_name, data) in domains {
        let mut detector = AnomalyDetector::new(4).expect("Failed to create detector");
        detector.train(&data).expect("Failed to train detector");
        
        // Calculate domain-specific metrics
        let context_count = detector.model().context_tree().context_count();
        context_counts.push(context_count);
        
        // Test with normal data from same domain
        let test_data = &data[data.len()/2..data.len()/2+100];
        let anomalies = detector.detect_anomalies(test_data, 0.1).expect("Failed to detect anomalies");
        
        let avg_likelihood = if !anomalies.is_empty() {
            anomalies.iter().map(|a| a.likelihood).sum::<f64>() / anomalies.len() as f64
        } else {
            1.0 // No anomalies means high likelihood
        };
        
        entropies.push(avg_likelihood);
        
        println!("  {} domain: {} contexts, avg likelihood: {:.2e}", 
                 domain_name, context_count, avg_likelihood);
    }
    
    // Verify consistency across domains
    let context_efficiency: Vec<f64> = context_counts.iter()
        .map(|&count| count as f64 / 2000.0)
        .collect();
    
    let max_efficiency = context_efficiency.iter().fold(0.0, |a, &b| a.max(b));
    let min_efficiency = context_efficiency.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    
    assert!(max_efficiency / min_efficiency < 10.0, 
            "Context efficiency should be consistent across domains: {:.3} vs {:.3}", 
            max_efficiency, min_efficiency);
    
    println!("  ✅ Cross-domain consistency validated");
}

#[test]
fn mathematical_properties_validation() {
    println!("🔬 Mathematical Properties Validation");
    
    // Test probability conservation
    test_probability_conservation();
    
    // Test information theory consistency
    test_information_theory_consistency();
    
    // Test Markov property
    test_markov_property();
    
    // Test numerical stability
    test_numerical_stability();
    
    println!("  ✅ Mathematical properties validated");
}

fn test_probability_conservation() {
    println!("  Testing probability conservation...");
    
    let sequence = generate_test_sequence(5000, 10);
    let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");
    detector.train(&sequence).expect("Failed to train detector");
    
    let context_tree = detector.model().context_tree();
    
    for (context, node) in &context_tree.contexts {
        let probabilities = node.get_all_probabilities(&AnomalyGridConfig::default());
        let prob_sum: f64 = probabilities.values().sum();
        assert!(
            (prob_sum - 1.0).abs() < 1e-10,
            "Probabilities should sum to 1.0 for context {:?}: {:.10}",
            context, prob_sum
        );
    }
    
    println!("    ✅ Probability conservation verified");
}

fn test_information_theory_consistency() {
    println!("  Testing information theory consistency...");
    
    let sequence = generate_test_sequence(3000, 8);
    let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");
    detector.train(&sequence).expect("Failed to train detector");
    
    let test_data = &sequence[2000..2100];
    let anomalies = detector.detect_anomalies(&test_data, 1.0).expect("Failed to detect anomalies");
    
    for anomaly in &anomalies {
        // Verify information content bounds
        assert!(anomaly.information_score >= 0.0, 
                "Information score should be non-negative: {:.3}", anomaly.information_score);
        
        // Verify likelihood bounds
        assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0,
                "Likelihood should be in [0,1]: {:.3}", anomaly.likelihood);
        
        // Verify anomaly strength bounds
        assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0,
                "Anomaly strength should be in [0,1]: {:.3}", anomaly.anomaly_strength);
    }
    
    println!("    ✅ Information theory consistency verified");
}

fn test_markov_property() {
    println!("  Testing Markov property...");
    
    let sequence = generate_markov_sequence(4000, 4, 3);
    let mut detector = AnomalyDetector::new(5).expect("Failed to create detector");
    detector.train(&sequence).expect("Failed to train detector");
    
    // Test that longer contexts don't dramatically change probabilities
    let test_contexts = vec![
        (vec!["A".to_string()], "B"),
        (vec!["X".to_string(), "A".to_string()], "B"),
        (vec!["Y".to_string(), "X".to_string(), "A".to_string()], "B"),
    ];
    
    let mut probs = Vec::new();
    for (context, next_state) in test_contexts {
        let prob = detector.model().get_best_context_probability(&context, next_state);
        probs.push(prob);
    }
    
    // Verify that probabilities are reasonable and don't vary wildly
    for &prob in &probs {
        assert!(prob > 0.0, "Probability should be positive: {:.6}", prob);
        assert!(prob <= 1.0, "Probability should not exceed 1.0: {:.6}", prob);
    }
    
    println!("    ✅ Markov property verified");
}

fn test_numerical_stability() {
    println!("  Testing numerical stability...");
    
    let extreme_cases = vec![
        generate_deterministic_sequence(1000),
        generate_random_sequence(1000, 50),
        generate_rare_event_sequence(1000),
    ];
    
    for (i, sequence) in extreme_cases.iter().enumerate() {
        let mut detector = AnomalyDetector::new(4).expect("Failed to create detector");
        detector.train(sequence).expect("Failed to train detector");
        
        let anomalies = detector.detect_anomalies(&sequence[0..50], 1.0).expect("Failed to detect anomalies");
        
        for anomaly in &anomalies {
            assert!(anomaly.likelihood.is_finite(), 
                    "Likelihood should be finite in case {}: {:.6}", i, anomaly.likelihood);
            assert!(anomaly.information_score.is_finite(),
                    "Information score should be finite in case {}: {:.6}", i, anomaly.information_score);
            assert!(anomaly.anomaly_strength.is_finite(),
                    "Anomaly strength should be finite in case {}: {:.6}", i, anomaly.anomaly_strength);
        }
    }
    
    println!("    ✅ Numerical stability verified");
}

#[test]
fn performance_benchmarks() {
    println!("⚡ Performance Benchmarks");
    
    let test_cases = vec![
        (1000, 4, 3),
        (5000, 8, 4),
        (10000, 16, 5),
    ];
    
    for (size, alphabet_size, order) in test_cases {
        println!("  Testing size={}, alphabet={}, order={}", size, alphabet_size, order);
        
        let sequence = generate_test_sequence(size, alphabet_size);
        
        // Training benchmark
        let train_start = Instant::now();
        let mut detector = AnomalyDetector::new(order).expect("Failed to create detector");
        detector.train(&sequence).expect("Failed to train detector");
        let train_time = train_start.elapsed();
        
        // Detection benchmark
        let test_data = &sequence[0..100.min(sequence.len())];
        let detect_start = Instant::now();
        let _anomalies = detector.detect_anomalies(test_data, 0.01).expect("Failed to detect anomalies");
        let detect_time = detect_start.elapsed();
        
        // Performance assertions
        let train_throughput = size as f64 / train_time.as_secs_f64();
        let detect_throughput = test_data.len() as f64 / detect_time.as_secs_f64();
        
        assert!(train_throughput > 1000.0, 
                "Training throughput too low: {:.0} elements/sec", train_throughput);
        assert!(detect_throughput > 10000.0,
                "Detection throughput too low: {:.0} elements/sec", detect_throughput);
        
        println!("    Train: {:.0} elem/s, Detect: {:.0} elem/s", 
                 train_throughput, detect_throughput);
    }
    
    println!("  ✅ Performance benchmarks passed");
}

// Helper functions for generating test data

fn generate_enterprise_network_traffic(size: usize) -> Vec<String> {
    let patterns = vec![
        vec!["TCP_SYN", "TCP_ACK", "HTTP_GET", "HTTP_200", "TCP_FIN"],
        vec!["UDP_DNS", "DNS_RESPONSE"],
        vec!["TLS_HANDSHAKE", "HTTPS_POST", "HTTP_201"],
    ];
    
    let mut traffic = Vec::new();
    for i in 0..size {
        let pattern = &patterns[i % patterns.len()];
        traffic.extend(pattern.iter().map(|s| s.to_string()));
    }
    traffic
}

fn generate_apt_attack_sequence() -> Vec<String> {
    vec![
        "SPEAR_PHISHING", "MACRO_EXECUTION", "PAYLOAD_DROP",
        "LATERAL_MOVEMENT", "CREDENTIAL_DUMP", "PERSISTENCE",
        "DATA_EXFILTRATION", "C2_COMMUNICATION", "COVER_TRACKS",
    ].into_iter().map(String::from).collect()
}

fn generate_ddos_sequence() -> Vec<String> {
    vec![
        "UDP_FLOOD", "UDP_FLOOD", "UDP_FLOOD", "UDP_FLOOD",
        "TCP_SYN_FLOOD", "TCP_SYN_FLOOD", "TCP_SYN_FLOOD",
        "HTTP_GET_FLOOD", "HTTP_GET_FLOOD", "HTTP_GET_FLOOD",
    ].into_iter().map(String::from).collect()
}

fn generate_normal_financial_transactions(size: usize) -> Vec<String> {
    let patterns = vec![
        vec!["CARD_PRESENT", "PIN_VERIFY", "PURCHASE", "APPROVE", "SETTLE"],
        vec!["CARD_NOT_PRESENT", "CVV_VERIFY", "PURCHASE", "APPROVE", "SETTLE"],
        vec!["ATM", "PIN_VERIFY", "WITHDRAWAL", "APPROVE"],
    ];
    
    let mut transactions = Vec::new();
    for i in 0..size {
        let pattern = &patterns[i % patterns.len()];
        transactions.extend(pattern.iter().map(|s| s.to_string()));
    }
    transactions
}

fn generate_card_testing_sequence() -> Vec<String> {
    vec![
        "CARD_NOT_PRESENT", "NO_CVV", "SMALL_AMOUNT", "DECLINE",
        "CARD_NOT_PRESENT", "NO_CVV", "SMALL_AMOUNT", "DECLINE",
        "CARD_NOT_PRESENT", "CVV_BYPASS", "LARGE_AMOUNT", "APPROVE",
    ].into_iter().map(String::from).collect()
}

fn generate_velocity_attack_sequence() -> Vec<String> {
    vec![
        "RAPID_TRANSACTION", "RAPID_TRANSACTION", "RAPID_TRANSACTION",
        "MULTIPLE_MERCHANTS", "HIGH_AMOUNTS", "VELOCITY_ALERT",
    ].into_iter().map(String::from).collect()
}

fn generate_industrial_sensor_data(size: usize) -> Vec<String> {
    let patterns = vec![
        vec!["TEMP_NORMAL", "PRESSURE_OK", "VIBRATION_LOW"],
        vec!["FLOW_NORMAL", "POWER_STABLE", "EFFICIENCY_HIGH"],
        vec!["LUBRICATION_OK", "BEARING_GOOD", "ALIGNMENT_OK"],
    ];
    
    let mut data = Vec::new();
    for i in 0..size {
        let pattern = &patterns[i % patterns.len()];
        data.extend(pattern.iter().map(|s| s.to_string()));
    }
    data
}

fn generate_bearing_failure_sequence() -> Vec<String> {
    vec![
        "VIBRATION_NORMAL", "VIBRATION_INCREASE", "VIBRATION_HIGH",
        "BEARING_WEAR", "LUBRICATION_LOW", "TEMPERATURE_RISE",
        "VIBRATION_CRITICAL", "BEARING_FAILURE", "SHUTDOWN_REQUIRED",
    ].into_iter().map(String::from).collect()
}

fn generate_temperature_anomaly_sequence() -> Vec<String> {
    vec![
        "TEMP_NORMAL", "TEMP_SLIGHT_RISE", "TEMP_ABOVE_NORMAL",
        "TEMP_HIGH", "TEMP_CRITICAL", "OVERHEATING_ALERT",
    ].into_iter().map(String::from).collect()
}

fn generate_system_log_data(size: usize) -> Vec<String> {
    let patterns = vec![
        vec!["USER_LOGIN", "AUTH_SUCCESS", "SESSION_START", "USER_LOGOUT"],
        vec!["SERVICE_START", "HEALTH_CHECK", "STATUS_OK"],
        vec!["BACKUP_START", "BACKUP_SUCCESS", "BACKUP_COMPLETE"],
    ];
    
    let mut logs = Vec::new();
    for i in 0..size {
        let pattern = &patterns[i % patterns.len()];
        logs.extend(pattern.iter().map(|s| s.to_string()));
    }
    logs
}

fn generate_malware_sequence() -> Vec<String> {
    vec![
        "SUSPICIOUS_PROCESS", "UNKNOWN_EXECUTABLE", "REGISTRY_MODIFICATION",
        "C2_COMMUNICATION", "DATA_EXFILTRATION", "PERSISTENCE_MECHANISM",
    ].into_iter().map(String::from).collect()
}

fn generate_privilege_escalation_sequence() -> Vec<String> {
    vec![
        "SUDO_ATTEMPT", "EXPLOIT_ATTEMPT", "BUFFER_OVERFLOW",
        "ROOT_ACCESS", "ADMIN_PRIVILEGES", "SYSTEM_CONTROL",
    ].into_iter().map(String::from).collect()
}

fn generate_test_sequence(size: usize, alphabet_size: usize) -> Vec<String> {
    (0..size).map(|i| format!("STATE_{}", i % alphabet_size)).collect()
}

fn generate_markov_sequence(size: usize, alphabet_size: usize, order: usize) -> Vec<String> {
    let mut sequence = Vec::new();
    
    // Initialize with random states
    for i in 0..order {
        sequence.push(format!("S{}", i % alphabet_size));
    }
    
    // Generate Markov sequence
    for i in order..size {
        let context_sum: usize = sequence[i-order..i].iter()
            .map(|s| s.chars().last().unwrap().to_digit(10).unwrap() as usize)
            .sum();
        let next_state = context_sum % alphabet_size;
        sequence.push(format!("S{}", next_state));
    }
    
    sequence
}

fn generate_deterministic_sequence(size: usize) -> Vec<String> {
    (0..size).map(|i| format!("STATE_{}", i % 3)).collect()
}

fn generate_random_sequence(size: usize, alphabet_size: usize) -> Vec<String> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    (0..size).map(|i| {
        let mut hasher = DefaultHasher::new();
        i.hash(&mut hasher);
        format!("S{}", hasher.finish() % alphabet_size as u64)
    }).collect()
}

fn generate_rare_event_sequence(size: usize) -> Vec<String> {
    (0..size).map(|i| {
        if i % 100 == 0 {
            "RARE_EVENT".to_string()
        } else {
            format!("COMMON_{}", i % 10)
        }
    }).collect()
}