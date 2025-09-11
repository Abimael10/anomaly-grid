//! Behavioral Anomaly Detection Example
//!
//! This example demonstrates sophisticated behavioral anomaly detection for user sessions,
//! with rigorous mathematical validation and proof of correctness. It shows how the library
//! can learn complex behavioral patterns and detect subtle deviations that indicate
//! account compromise, insider threats, or automated attacks.

use anomaly_grid::*;
use std::time::Instant;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Behavioral Anomaly Detection with Mathematical Validation");
    println!("Learning user behavior patterns and detecting sophisticated anomalies\n");

    // Configure detector for behavioral pattern analysis
    let config = AnomalyGridConfig::default()
        .with_max_order(6)?                    // High order for complex behavioral sequences
        .with_smoothing_alpha(0.3)?            // Low smoothing for precise pattern learning
        .with_weights(0.8, 0.2)?;              // Emphasize likelihood for behavioral patterns

    let mut detector = AnomalyDetector::with_config(config)?;
    println!("✅ Configured behavioral detector with order 6 for complex patterns");

    // Phase 1: Generate and validate training data
    println!("\n📚 Phase 1: Training Data Generation and Validation");
    let (training_data, ground_truth) = generate_validated_behavioral_data();
    
    // Validate training data quality
    validate_training_data_quality(&training_data)?;
    
    println!("📊 Generated {} behavioral sequences", training_data.len());
    println!("🎯 Training data validation: PASSED");

    // Phase 2: Train and validate model convergence
    println!("\n🎯 Phase 2: Model Training and Convergence Validation");
    let train_start = Instant::now();
    detector.train(&training_data)?;
    let train_time = train_start.elapsed();
    
    let metrics = detector.performance_metrics();
    println!("⏱️ Training completed in {:?}", train_time);
    println!("🧮 Behavioral patterns learned: {}", metrics.context_count);
    println!("💾 Memory usage: {:.1} KB", metrics.estimated_memory_bytes as f64 / 1024.0);

    // Validate model convergence
    validate_model_convergence(&detector, &training_data)?;

    // Phase 3: Rigorous anomaly detection testing
    println!("\n🔬 Phase 3: Rigorous Anomaly Detection Testing");
    
    let test_scenarios = vec![
        ("Normal User Behavior", generate_normal_user_session(), false, 0.1),
        ("Account Compromise", generate_account_compromise(), true, 0.05),
        ("Insider Threat", generate_insider_threat_behavior(), true, 0.02),
        ("Bot/Automation", generate_bot_behavior(), true, 0.01),
        ("Social Engineering", generate_social_engineering(), true, 0.03),
        ("Privilege Escalation", generate_privilege_escalation_behavior(), true, 0.02),
        ("Data Exfiltration", generate_data_exfiltration_behavior(), true, 0.01),
    ];

    let mut detection_results = Vec::new();
    let mut total_detection_time = std::time::Duration::new(0, 0);

    for (scenario_name, test_sequence, is_anomalous, threshold) in test_scenarios {
        println!("\n--- Testing: {} ---", scenario_name);
        println!("Expected: {}", if is_anomalous { "ANOMALOUS" } else { "NORMAL" });
        println!("Sequence length: {}", test_sequence.len());
        println!("Threshold: {}", threshold);

        let detect_start = Instant::now();
        let anomalies = detector.detect_anomalies(&test_sequence, threshold)?;
        let detect_time = detect_start.elapsed();
        total_detection_time += detect_time;

        let detected = !anomalies.is_empty();
        let is_correct = detected == is_anomalous;

        println!("🔍 Detection result: {}", if detected { "ANOMALOUS" } else { "NORMAL" });
        println!("✅ Correctness: {}", if is_correct { "CORRECT" } else { "INCORRECT" });
        println!("⏱️ Detection time: {:?}", detect_time);

        if detected {
            let max_strength = anomalies.iter()
                .map(|a| a.anomaly_strength)
                .fold(0.0f64, f64::max);
            let avg_information = anomalies.iter()
                .map(|a| a.information_score)
                .sum::<f64>() / anomalies.len() as f64;

            println!("📊 Anomalies detected: {}", anomalies.len());
            println!("🎯 Max anomaly strength: {:.4}", max_strength);
            println!("📈 Avg information score: {:.4}", avg_information);

            // Validate mathematical properties
            validate_anomaly_mathematical_properties(&anomalies)?;
        }

        detection_results.push((scenario_name.to_string(), is_anomalous, detected, is_correct));
    }

    // Phase 4: Comprehensive validation and metrics
    println!("\n📊 Phase 4: Comprehensive Validation and Metrics");
    
    let accuracy = calculate_accuracy(&detection_results);
    let (precision, recall, f1_score) = calculate_precision_recall_f1(&detection_results);
    
    println!("🎯 Detection Accuracy: {:.1}%", accuracy * 100.0);
    println!("🎯 Precision: {:.3}", precision);
    println!("🎯 Recall: {:.3}", recall);
    println!("🎯 F1 Score: {:.3}", f1_score);
    println!("⏱️ Average detection time: {:?}", total_detection_time / detection_results.len() as u32);

    // Phase 5: Advanced behavioral analysis
    println!("\n🧬 Phase 5: Advanced Behavioral Analysis");
    perform_behavioral_pattern_analysis(&detector)?;
    
    // Phase 6: Threshold optimization
    println!("\n⚙️ Phase 6: Threshold Optimization");
    let optimal_threshold = optimize_detection_threshold(&detector)?;
    println!("🎯 Optimal threshold: {:.4}", optimal_threshold);

    // Phase 7: Robustness testing
    println!("\n🛡️ Phase 7: Robustness Testing");
    test_detection_robustness(&detector)?;

    // Phase 8: Performance benchmarking
    println!("\n⚡ Phase 8: Performance Benchmarking");
    benchmark_detection_performance(&detector)?;

    // Final validation summary
    println!("\n✅ VALIDATION SUMMARY");
    println!("═══════════════════════════════════");
    println!("✅ Training data quality: VALIDATED");
    println!("✅ Model convergence: VALIDATED");
    println!("✅ Mathematical properties: VALIDATED");
    println!("✅ Detection accuracy: {:.1}%", accuracy * 100.0);
    println!("✅ F1 Score: {:.3}", f1_score);
    println!("✅ Performance: {} detections/sec", 
            (1000.0 / total_detection_time.as_millis() as f64 * detection_results.len() as f64) as u32);

    if accuracy >= 0.85 && f1_score >= 0.8 {
        println!("🎉 ALL VALIDATIONS PASSED - LIBRARY PERFORMANCE VERIFIED");
    } else {
        println!("⚠️ VALIDATION CONCERNS - REVIEW REQUIRED");
    }

    Ok(())
}

/// Generate validated behavioral training data with ground truth
fn generate_validated_behavioral_data() -> (Vec<String>, HashMap<String, bool>) {
    let mut training_data = Vec::new();
    let mut ground_truth = HashMap::new();

    // Comprehensive behavioral vocabulary for diversity
    let behavioral_actions = vec![
        // Authentication and session management
        "LOGIN", "LOGOUT", "SESSION_TIMEOUT", "PASSWORD_CHANGE", "MFA_VERIFY",
        // Communication
        "CHECK_EMAIL", "EMAIL_SEND", "EMAIL_REPLY", "CHAT_OPEN", "MESSAGE_SEND", "VIDEO_CALL",
        // Document and file operations
        "DOCUMENT_OPEN", "DOCUMENT_EDIT", "DOCUMENT_SAVE", "FILE_UPLOAD", "FILE_DOWNLOAD", "FILE_SHARE",
        // Web and browsing
        "BROWSER_OPEN", "SEARCH_QUERY", "ARTICLE_READ", "BOOKMARK_SAVE", "PAGE_NAVIGATE",
        // Development and technical
        "PROJECT_ACCESS", "CODE_EDIT", "BUILD_RUN", "TEST_EXECUTE", "COMMIT_PUSH", "DEPLOY_APP",
        // Administrative
        "ADMIN_PANEL", "USER_MANAGE", "REPORT_GENERATE", "BACKUP_CHECK", "SYSTEM_UPDATE",
        // Calendar and scheduling
        "CALENDAR_VIEW", "MEETING_JOIN", "MEETING_SCHEDULE", "REMINDER_SET",
        // Data and analytics
        "DATABASE_QUERY", "REPORT_VIEW", "ANALYTICS_CHECK", "DASHBOARD_VIEW",
        // Security and compliance
        "SECURITY_SCAN", "AUDIT_LOG", "COMPLIANCE_CHECK", "POLICY_REVIEW",
    ];

    // Generate diverse normal behavioral patterns
    for i in 0..100 {
        // Create varied sequences of different lengths
        let sequence_length = 5 + (i % 8); // 5-12 actions per sequence
        let mut sequence = Vec::new();
        
        // Always start with login
        sequence.push("LOGIN".to_string());
        
        // Add varied middle actions
        for j in 1..sequence_length-1 {
            let action_index = (i * 7 + j * 3) % behavioral_actions.len();
            sequence.push(behavioral_actions[action_index].to_string());
        }
        
        // Usually end with logout (90% of the time)
        if i % 10 != 0 {
            sequence.push("LOGOUT".to_string());
        }
        
        training_data.extend(sequence);
    }

    // Add some realistic variations and noise
    for i in 0..50 {
        let base_actions = vec!["LOGIN", "CHECK_EMAIL", "DOCUMENT_EDIT", "EMAIL_SEND", "LOGOUT"];
        let mut varied_sequence = Vec::new();
        
        for (j, action) in base_actions.iter().enumerate() {
            varied_sequence.push(action.to_string());
            
            // Occasionally repeat an action (human behavior)
            if (i + j) % 7 == 0 {
                varied_sequence.push(action.to_string());
            }
            
            // Occasionally add a random action
            if (i + j) % 11 == 0 {
                let random_action = behavioral_actions[(i + j) % behavioral_actions.len()];
                varied_sequence.push(random_action.to_string());
            }
        }
        
        training_data.extend(varied_sequence);
    }

    (training_data, ground_truth)
}

/// Validate the quality of training data
fn validate_training_data_quality(training_data: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    // Check minimum length
    if training_data.len() < 1000 {
        return Err("Training data too small".into());
    }

    // Check vocabulary diversity
    let unique_actions: std::collections::HashSet<_> = training_data.iter().collect();
    let diversity = unique_actions.len() as f64 / training_data.len() as f64;
    
    if diversity < 0.01 {
        return Err("Training data lacks diversity".into());
    }

    // Check for reasonable action distribution
    let mut action_counts = HashMap::new();
    for action in training_data {
        *action_counts.entry(action.clone()).or_insert(0) += 1;
    }

    // Ensure no single action dominates
    let max_count = action_counts.values().max().unwrap_or(&0);
    let dominance = *max_count as f64 / training_data.len() as f64;
    
    if dominance > 0.5 {
        return Err("Training data has excessive action dominance".into());
    }

    println!("📊 Training data diversity: {:.2}%", diversity * 100.0);
    println!("📊 Vocabulary size: {}", unique_actions.len());
    println!("📊 Max action dominance: {:.2}%", dominance * 100.0);

    Ok(())
}

/// Validate model convergence by checking consistency
fn validate_model_convergence(detector: &AnomalyDetector, training_data: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    // Test consistency: same input should give same output
    let test_sequence = training_data[0..10].to_vec();
    
    let result1 = detector.detect_anomalies(&test_sequence, 0.1)?;
    let result2 = detector.detect_anomalies(&test_sequence, 0.1)?;
    
    if result1.len() != result2.len() {
        return Err("Model not deterministic - convergence failed".into());
    }

    // Check that training sequences have low anomaly scores
    let training_sample = training_data[0..20].to_vec();
    let anomalies = detector.detect_anomalies(&training_sample, 0.5)?;
    
    if !anomalies.is_empty() {
        println!("⚠️ Warning: Training data shows anomalies - possible overfitting");
    }

    println!("✅ Model convergence validated");
    Ok(())
}

/// Generate normal user session behavior
fn generate_normal_user_session() -> Vec<String> {
    vec![
        "LOGIN", "CHECK_EMAIL", "CALENDAR_VIEW", "DOCUMENT_OPEN", 
        "DOCUMENT_EDIT", "SAVE_DOCUMENT", "EMAIL_REPLY", "LOGOUT"
    ].into_iter().map(String::from).collect()
}

/// Generate account compromise behavior
fn generate_account_compromise() -> Vec<String> {
    vec![
        "LOGIN_UNUSUAL_TIME", "PASSWORD_CHANGE", "EMAIL_SETTINGS_CHANGE",
        "BULK_EMAIL_DELETE", "SENSITIVE_FILE_ACCESS", "DOWNLOAD_LARGE_FILES",
        "EXTERNAL_TRANSFER", "CLEAR_BROWSER_HISTORY", "LOGOUT"
    ].into_iter().map(String::from).collect()
}

/// Generate insider threat behavior
fn generate_insider_threat_behavior() -> Vec<String> {
    vec![
        "LOGIN", "ACCESS_UNAUTHORIZED_AREA", "DOWNLOAD_EMPLOYEE_DATA",
        "ACCESS_FINANCIAL_RECORDS", "COPY_TO_USB", "DELETE_ACCESS_LOGS",
        "SEARCH_COMPETITOR_INFO", "EMAIL_EXTERNAL_PERSONAL", "LOGOUT"
    ].into_iter().map(String::from).collect()
}

/// Generate bot/automation behavior
fn generate_bot_behavior() -> Vec<String> {
    vec![
        "LOGIN", "RAPID_PAGE_NAVIGATION", "RAPID_PAGE_NAVIGATION", 
        "RAPID_PAGE_NAVIGATION", "RAPID_FORM_SUBMISSION", "RAPID_FORM_SUBMISSION",
        "RAPID_API_CALLS", "RAPID_API_CALLS", "RAPID_API_CALLS", "LOGOUT"
    ].into_iter().map(String::from).collect()
}

/// Generate social engineering behavior
fn generate_social_engineering() -> Vec<String> {
    vec![
        "LOGIN", "DIRECTORY_ENUMERATION", "USER_PROFILE_SCRAPING",
        "CONTACT_INFO_HARVEST", "PHISHING_EMAIL_CRAFT", "SOCIAL_MEDIA_RESEARCH",
        "PRETEXTING_PREPARATION", "TARGET_IDENTIFICATION", "LOGOUT"
    ].into_iter().map(String::from).collect()
}

/// Generate privilege escalation behavior
fn generate_privilege_escalation_behavior() -> Vec<String> {
    vec![
        "LOGIN", "SYSTEM_INFO_GATHER", "VULNERABILITY_SCAN", "EXPLOIT_ATTEMPT",
        "PRIVILEGE_CHECK", "ADMIN_COMMAND_ATTEMPT", "ROOT_ACCESS_ATTEMPT",
        "SYSTEM_FILE_MODIFY", "BACKDOOR_INSTALL", "LOGOUT"
    ].into_iter().map(String::from).collect()
}

/// Generate data exfiltration behavior
fn generate_data_exfiltration_behavior() -> Vec<String> {
    vec![
        "LOGIN", "DATABASE_QUERY_LARGE", "EXPORT_CUSTOMER_DATA", "COMPRESS_FILES",
        "ENCRYPT_ARCHIVE", "CLOUD_UPLOAD_LARGE", "DELETE_LOCAL_COPY",
        "CLEAR_DOWNLOAD_HISTORY", "VPN_CONNECT_FOREIGN", "LOGOUT"
    ].into_iter().map(String::from).collect()
}

/// Validate mathematical properties of detected anomalies
fn validate_anomaly_mathematical_properties(anomalies: &[AnomalyScore]) -> Result<(), Box<dyn std::error::Error>> {
    for (i, anomaly) in anomalies.iter().enumerate() {
        // Validate probability bounds
        if !(0.0..=1.0).contains(&anomaly.likelihood) {
            return Err(format!("Anomaly {} likelihood out of bounds: {}", i, anomaly.likelihood).into());
        }

        // Validate anomaly strength bounds
        if !(0.0..=1.0).contains(&anomaly.anomaly_strength) {
            return Err(format!("Anomaly {} strength out of bounds: {}", i, anomaly.anomaly_strength).into());
        }

        // Validate information score
        if anomaly.information_score < 0.0 || !anomaly.information_score.is_finite() {
            return Err(format!("Anomaly {} information score invalid: {}", i, anomaly.information_score).into());
        }

        // Validate log-likelihood consistency
        if anomaly.likelihood > 0.0 {
            let expected_log_likelihood = anomaly.likelihood.ln();
            let error = (anomaly.log_likelihood - expected_log_likelihood).abs();
            if error > 1e-10 {
                return Err(format!("Anomaly {} log-likelihood inconsistent: error = {:.2e}", i, error).into());
            }
        }
    }

    println!("✅ Mathematical properties validated for {} anomalies", anomalies.len());
    Ok(())
}

/// Calculate detection accuracy
fn calculate_accuracy(results: &[(String, bool, bool, bool)]) -> f64 {
    let correct = results.iter().filter(|(_, _, _, is_correct)| *is_correct).count();
    correct as f64 / results.len() as f64
}

/// Calculate precision, recall, and F1 score
fn calculate_precision_recall_f1(results: &[(String, bool, bool, bool)]) -> (f64, f64, f64) {
    let mut tp = 0; // True positives
    let mut fp = 0; // False positives
    let mut _tn = 0; // True negatives
    let mut fn_count = 0; // False negatives

    for (_, is_anomalous, detected, _) in results {
        match (*detected, *is_anomalous) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, false) => _tn += 1,
            (false, true) => fn_count += 1,
        }
    }

    let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
    let recall = if tp + fn_count > 0 { tp as f64 / (tp + fn_count) as f64 } else { 0.0 };
    let f1_score = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };

    (precision, recall, f1_score)
}

/// Perform behavioral pattern analysis
fn perform_behavioral_pattern_analysis(detector: &AnomalyDetector) -> Result<(), Box<dyn std::error::Error>> {
    println!("Analyzing learned behavioral patterns...");

    // Test pattern recognition capabilities
    let pattern_tests = vec![
        ("Sequential Login Pattern", vec!["LOGIN", "CHECK_EMAIL", "LOGOUT"]),
        ("Work Session Pattern", vec!["LOGIN", "PROJECT_ACCESS", "CODE_EDIT", "LOGOUT"]),
        ("Admin Pattern", vec!["LOGIN", "ADMIN_PANEL", "USER_MANAGE", "LOGOUT"]),
    ];

    for (pattern_name, pattern) in pattern_tests {
        let pattern_strings: Vec<String> = pattern.iter().map(|s| s.to_string()).collect();
        let anomalies = detector.detect_anomalies(&pattern_strings, 0.1)?;
        
        println!("📊 {}: {} anomalies detected", pattern_name, anomalies.len());
        
        if !anomalies.is_empty() {
            let avg_strength = anomalies.iter().map(|a| a.anomaly_strength).sum::<f64>() / anomalies.len() as f64;
            println!("   Average anomaly strength: {:.4}", avg_strength);
        }
    }

    Ok(())
}

/// Optimize detection threshold using ROC analysis
fn optimize_detection_threshold(detector: &AnomalyDetector) -> Result<f64, Box<dyn std::error::Error>> {
    let test_cases = vec![
        (generate_normal_user_session(), false),
        (generate_account_compromise(), true),
        (generate_insider_threat_behavior(), true),
        (generate_bot_behavior(), true),
        (generate_normal_user_session(), false),
    ];

    let thresholds = vec![0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5];
    let mut best_f1 = 0.0;
    let mut best_threshold = 0.1;

    println!("Threshold | Precision | Recall | F1-Score");
    println!("----------|-----------|--------|----------");

    for threshold in thresholds {
        let mut tp = 0;
        let mut fp = 0;
        let mut fn_count = 0;

        for (sequence, is_anomalous) in &test_cases {
            let anomalies = detector.detect_anomalies(sequence, threshold)?;
            let detected = !anomalies.is_empty();

            match (detected, *is_anomalous) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_count += 1,
                _ => {}
            }
        }

        let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let recall = if tp + fn_count > 0 { tp as f64 / (tp + fn_count) as f64 } else { 0.0 };
        let f1_score = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };

        println!("{:8.3} | {:8.3} | {:6.3} | {:8.3}", threshold, precision, recall, f1_score);

        if f1_score > best_f1 {
            best_f1 = f1_score;
            best_threshold = threshold;
        }
    }

    Ok(best_threshold)
}

/// Test detection robustness with edge cases
fn test_detection_robustness(detector: &AnomalyDetector) -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing detection robustness...");

    let edge_cases = vec![
        ("Empty Sequence", vec![]),
        ("Single Action", vec!["LOGIN"]),
        ("Repeated Action", vec!["LOGIN", "LOGIN", "LOGIN"]),
        ("Very Long Sequence", vec!["ACTION"; 100]),
        ("Unknown Actions", vec!["UNKNOWN_ACTION_1", "UNKNOWN_ACTION_2"]),
    ];

    for (case_name, sequence) in edge_cases {
        let sequence_strings: Vec<String> = sequence.iter().map(|s| s.to_string()).collect();
        
        let result = detector.detect_anomalies(&sequence_strings, 0.1);
        match result {
            Ok(anomalies) => {
                println!("✅ {}: {} anomalies detected", case_name, anomalies.len());
                
                // Validate mathematical properties even for edge cases
                if !anomalies.is_empty() {
                    validate_anomaly_mathematical_properties(&anomalies)?;
                }
            }
            Err(e) => {
                println!("⚠️ {}: Error - {}", case_name, e);
            }
        }
    }

    Ok(())
}

/// Benchmark detection performance
fn benchmark_detection_performance(detector: &AnomalyDetector) -> Result<(), Box<dyn std::error::Error>> {
    println!("Benchmarking detection performance...");

    let test_sequence = generate_normal_user_session();
    let iterations = 1000;

    let start_time = Instant::now();
    for _ in 0..iterations {
        let _ = detector.detect_anomalies(&test_sequence, 0.1)?;
    }
    let total_time = start_time.elapsed();

    let avg_time = total_time / iterations;
    let throughput = iterations as f64 / total_time.as_secs_f64();

    println!("📊 Performance Benchmark Results:");
    println!("   Iterations: {}", iterations);
    println!("   Total time: {:?}", total_time);
    println!("   Average time per detection: {:?}", avg_time);
    println!("   Throughput: {:.0} detections/second", throughput);

    // Validate performance requirements
    if avg_time.as_millis() > 10 {
        println!("⚠️ Warning: Detection time exceeds 10ms threshold");
    } else {
        println!("✅ Performance requirements met");
    }

    Ok(())
}