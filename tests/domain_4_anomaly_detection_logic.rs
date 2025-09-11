//! Domain 4: Anomaly Detection Logic
//!
//! This module implements comprehensive domain-driven tests for anomaly detection
//! logic, ensuring our implementation follows sound principles for identifying
//! deviations and anomalies in finite alphabets.

use anomaly_grid::*;
use std::collections::HashMap;

#[test]
fn domain_4_anomaly_detection_logic() {
    println!("🔬 DOMAIN 4: ANOMALY DETECTION LOGIC");
    println!("====================================");
    println!();
    
    let mut test_results = Vec::new();
    
    // Test 4.1: Anomaly Definition Consistency
    println!("Test 4.1: Anomaly Definition Consistency");
    println!("----------------------------------------");
    let anomaly_def_result = test_anomaly_definition_consistency_comprehensive();
    test_results.push(("Anomaly Definition", anomaly_def_result));
    println!();
    
    // Test 4.2: Likelihood-Based Detection
    println!("Test 4.2: Likelihood-Based Detection");
    println!("------------------------------------");
    let likelihood_result = test_likelihood_based_detection_comprehensive();
    test_results.push(("Likelihood Detection", likelihood_result));
    println!();
    
    // Test 4.3: Information-Theoretic Scoring
    println!("Test 4.3: Information-Theoretic Scoring");
    println!("---------------------------------------");
    let info_scoring_result = test_information_theoretic_scoring_comprehensive();
    test_results.push(("Information Scoring", info_scoring_result));
    println!();
    
    // Test 4.4: Threshold Semantics
    println!("Test 4.4: Threshold Semantics");
    println!("-----------------------------");
    let threshold_result = test_threshold_semantics_comprehensive();
    test_results.push(("Threshold Semantics", threshold_result));
    println!();
    
    // Test 4.5: Anomaly Strength Consistency
    println!("Test 4.5: Anomaly Strength Consistency");
    println!("--------------------------------------");
    let strength_result = test_anomaly_strength_consistency();
    test_results.push(("Anomaly Strength", strength_result));
    println!();
    
    // Domain 4 Summary
    println!("🏆 DOMAIN 4 SUMMARY");
    println!("===================");
    let passed_tests = test_results.iter().filter(|(_, result)| result.passed).count();
    let total_tests = test_results.len();
    
    for (test_name, result) in &test_results {
        let status = if result.passed { "✅" } else { "❌" };
        println!("  {} {}: {}", status, test_name, result.evidence);
    }
    
    println!();
    println!("Domain 4 Result: {}/{} tests passed", passed_tests, total_tests);
    
    assert_eq!(passed_tests, total_tests, 
               "Domain 4 (Anomaly Detection Logic) failed: {}/{} tests passed", 
               passed_tests, total_tests);
}

#[derive(Debug)]
struct DomainTestResult {
    passed: bool,
    evidence: String,
    details: Vec<String>,
}

impl DomainTestResult {
    fn pass(evidence: String) -> Self {
        Self {
            passed: true,
            evidence,
            details: Vec::new(),
        }
    }
    
    fn fail(evidence: String) -> Self {
        Self {
            passed: false,
            evidence,
            details: Vec::new(),
        }
    }
    
    fn with_details(mut self, details: Vec<String>) -> Self {
        self.details = details;
        self
    }
}

/// Test 4.1: Anomaly Definition Consistency
/// 
/// Tests that anomalies are consistently defined as:
/// - Low likelihood events (rare in training data)
/// - High information content events (surprising)
/// - Events that deviate from learned patterns
fn test_anomaly_definition_consistency_comprehensive() -> DomainTestResult {
    println!("  Testing anomaly definition consistency...");
    
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    
    // Train on a clear pattern: A->B->C->A->B->C...
    let training_sequence = vec!["A", "B", "C"].repeat(100)
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    detector.train(&training_sequence).expect("Failed to train");
    
    let mut violations = 0;
    let mut details = Vec::new();
    
    // Test 1: Normal patterns should have low anomaly scores
    println!("    Testing normal pattern detection");
    let normal_sequences = vec![
        vec!["A", "B", "C"],
        vec!["B", "C", "A"],
        vec!["C", "A", "B"],
    ];
    
    let mut normal_scores = Vec::new();
    for (i, sequence) in normal_sequences.iter().enumerate() {
        let seq_strings: Vec<String> = sequence.iter().map(|s| s.to_string()).collect();
        let anomalies = detector.detect_anomalies(&seq_strings, 0.0).unwrap_or_default();
        
        let max_score = anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
        normal_scores.push(max_score);
        
        println!("      Normal sequence {}: max anomaly score = {:.6}", i, max_score);
    }
    
    // Test 2: Anomalous patterns should have high anomaly scores
    println!("    Testing anomalous pattern detection");
    let anomalous_sequences = vec![
        vec!["X", "Y", "Z"],  // Completely unknown
        vec!["A", "X", "B"],  // Partially unknown
        vec!["C", "B", "A"],  // Reverse pattern
    ];
    
    let mut anomalous_scores = Vec::new();
    for (i, sequence) in anomalous_sequences.iter().enumerate() {
        let seq_strings: Vec<String> = sequence.iter().map(|s| s.to_string()).collect();
        let anomalies = detector.detect_anomalies(&seq_strings, 0.0).unwrap_or_default();
        
        let max_score = anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
        anomalous_scores.push(max_score);
        
        println!("      Anomalous sequence {}: max anomaly score = {:.6}", i, max_score);
    }
    
    // Test 3: Anomalous scores should be higher than normal scores
    println!("    Testing score ordering");
    let normal_max = normal_scores.iter().fold(0.0f64, |a, &b| a.max(b));
    let anomalous_min = anomalous_scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    
    println!("      Normal max score: {:.6}", normal_max);
    println!("      Anomalous min score: {:.6}", anomalous_min);
    
    if anomalous_min <= normal_max {
        violations += 1;
        details.push(format!("Score ordering violation: anomalous_min ({:.6}) <= normal_max ({:.6})", 
                           anomalous_min, normal_max));
    }
    
    // Test 4: Consistency across different thresholds
    println!("    Testing threshold consistency");
    let test_sequence = vec!["X".to_string(), "Y".to_string()];
    
    let thresholds = vec![0.0, 0.1, 0.5, 0.9];
    let mut detection_counts = Vec::new();
    
    for &threshold in &thresholds {
        let anomalies = detector.detect_anomalies(&test_sequence, threshold).unwrap_or_default();
        detection_counts.push(anomalies.len());
        println!("      Threshold {:.1}: {} anomalies detected", threshold, anomalies.len());
    }
    
    // Detection counts should be monotonically non-increasing
    let monotonic_violations = detection_counts.windows(2)
        .filter(|w| w[1] > w[0])
        .count();
    
    if monotonic_violations > 0 {
        violations += 1;
        details.push(format!("Threshold monotonicity violations: {}", monotonic_violations));
    }
    
    details.push(format!("Normal scores: {:?}", normal_scores));
    details.push(format!("Anomalous scores: {:?}", anomalous_scores));
    details.push(format!("Detection counts: {:?}", detection_counts));
    
    if violations == 0 {
        DomainTestResult::pass("Anomaly definition consistent".to_string())
            .with_details(details)
    } else {
        DomainTestResult::fail(format!("{} anomaly definition violations", violations))
            .with_details(details)
    }
}

/// Test 4.2: Likelihood-Based Detection
/// 
/// Tests that anomaly detection properly uses likelihood:
/// - Lower likelihood events should have higher anomaly scores
/// - Likelihood calculation should be mathematically sound
/// - Detection should be sensitive to probability differences
fn test_likelihood_based_detection_comprehensive() -> DomainTestResult {
    println!("  Testing likelihood-based detection...");
    
    let mut detector = AnomalyDetector::new(1).expect("Failed to create detector");
    
    // Create a training set with known probability distribution
    // A: 70%, B: 20%, C: 10%
    let mut training_sequence = Vec::new();
    training_sequence.extend(vec!["A"; 70]);
    training_sequence.extend(vec!["B"; 20]);
    training_sequence.extend(vec!["C"; 10]);
    
    // Repeat to get more data
    let training_strings: Vec<String> = training_sequence.repeat(10)
        .iter().map(|s| s.to_string()).collect();
    detector.train(&training_strings).expect("Failed to train");
    
    let mut violations = 0;
    let mut details = Vec::new();
    
    // Test 1: Likelihood calculation accuracy
    println!("    Testing likelihood calculation");
    let test_cases = vec![
        (vec!["A"], "High probability"),
        (vec!["B"], "Medium probability"),
        (vec!["C"], "Low probability"),
    ];
    
    let mut likelihoods = Vec::new();
    for (sequence, description) in &test_cases {
        let seq_strings: Vec<String> = sequence.iter().map(|s| s.to_string()).collect();
        let likelihood = detector.model().calculate_likelihood(&seq_strings);
        likelihoods.push(likelihood);
        
        println!("      {}: likelihood = {:.6}", description, likelihood);
    }
    
    // Test 2: Likelihood ordering (A > B > C)
    println!("    Testing likelihood ordering");
    if likelihoods.len() >= 3 {
        if likelihoods[0] <= likelihoods[1] || likelihoods[1] <= likelihoods[2] {
            violations += 1;
            details.push("Likelihood ordering violation: expected A > B > C".to_string());
        }
    }
    
    // Test 3: Anomaly score inverse relationship with likelihood
    println!("    Testing anomaly score vs likelihood relationship");
    let mut anomaly_scores = Vec::new();
    
    // Use longer sequences to ensure anomaly detection works
    let longer_test_cases = vec![
        (vec!["A", "A"], "High probability sequence"),
        (vec!["B", "B"], "Medium probability sequence"),
        (vec!["C", "C"], "Low probability sequence"),
    ];
    
    for (sequence, description) in &longer_test_cases {
        let seq_strings: Vec<String> = sequence.iter().map(|s| s.to_string()).collect();
        let anomalies = detector.detect_anomalies(&seq_strings, 0.0).unwrap_or_default();
        
        let max_score = anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
        anomaly_scores.push(max_score);
        
        println!("      {}: anomaly score = {:.6}", description, max_score);
    }
    
    // Anomaly scores should be inversely related to likelihood
    if anomaly_scores.len() >= 3 {
        // C (lowest likelihood) should have highest anomaly score
        // A (highest likelihood) should have lowest anomaly score
        if anomaly_scores[2] <= anomaly_scores[0] {
            violations += 1;
            details.push("Inverse likelihood relationship violation".to_string());
        }
    }
    
    // Test 4: Sensitivity to probability differences
    println!("    Testing sensitivity to probability differences");
    
    // Create sequences with very different likelihoods (use longer sequences)
    let high_likelihood_seq = vec!["A".to_string(), "A".to_string()]; // High probability
    let low_likelihood_seq = vec!["C".to_string(), "C".to_string()];   // Low probability
    
    let high_anomalies = detector.detect_anomalies(&high_likelihood_seq, 0.0).unwrap_or_default();
    let low_anomalies = detector.detect_anomalies(&low_likelihood_seq, 0.0).unwrap_or_default();
    
    let high_score = high_anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
    let low_score = low_anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
    
    let score_difference = low_score - high_score;
    println!("      Score difference (low - high likelihood): {:.6}", score_difference);
    
    if score_difference < 0.001 { // Should be significant difference (adjusted threshold)
        violations += 1;
        details.push(format!("Insufficient sensitivity: score difference = {:.6}", score_difference));
    }
    
    details.push(format!("Likelihoods: {:?}", likelihoods));
    details.push(format!("Anomaly scores: {:?}", anomaly_scores));
    
    println!("    Likelihood-based detection violations: {}", violations);
    for detail in &details {
        println!("      {}", detail);
    }
    
    if violations == 0 {
        DomainTestResult::pass("Likelihood-based detection working correctly".to_string())
            .with_details(details)
    } else {
        DomainTestResult::fail(format!("{} likelihood-based detection violations", violations))
            .with_details(details)
    }
}

/// Test 4.3: Information-Theoretic Scoring
/// 
/// Tests that information theory is properly applied in anomaly scoring:
/// - High information content events should have higher scores
/// - Information measures should be mathematically consistent
/// - Scoring should reflect surprise/unexpectedness
fn test_information_theoretic_scoring_comprehensive() -> DomainTestResult {
    println!("  Testing information-theoretic scoring...");
    
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    
    // Create training data with clear information content differences
    let training_sequence = vec!["COMMON", "COMMON", "COMMON", "RARE"].repeat(100)
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    detector.train(&training_sequence).expect("Failed to train");
    
    let mut violations = 0;
    let mut details = Vec::new();
    
    // Test 1: Information content calculation
    println!("    Testing information content in anomaly scores");
    
    let test_cases = vec![
        (vec!["COMMON", "COMMON"], "Low information"),
        (vec!["COMMON", "RARE"], "Medium information"),
        (vec!["UNKNOWN", "UNKNOWN"], "High information"),
    ];
    
    let mut info_scores = Vec::new();
    for (sequence, description) in &test_cases {
        let seq_strings: Vec<String> = sequence.iter().map(|s| s.to_string()).collect();
        let anomalies = detector.detect_anomalies(&seq_strings, 0.0).unwrap_or_default();
        
        // Get the information score component
        let max_info = anomalies.iter()
            .map(|a| a.information_score)
            .fold(0.0f64, f64::max);
        
        info_scores.push(max_info);
        println!("      {}: max information score = {:.6}", description, max_info);
    }
    
    // Test 2: Information score ordering
    println!("    Testing information score ordering");
    
    // Unknown sequences should have higher information scores than common ones
    if info_scores.len() >= 3 {
        if info_scores[2] <= info_scores[0] {
            violations += 1;
            details.push("Information score ordering violation".to_string());
        }
    }
    
    // Test 3: Information score contribution to anomaly strength
    println!("    Testing information score contribution to anomaly strength");
    
    let mut anomaly_strengths = Vec::new();
    for (sequence, description) in &test_cases {
        let seq_strings: Vec<String> = sequence.iter().map(|s| s.to_string()).collect();
        let anomalies = detector.detect_anomalies(&seq_strings, 0.0).unwrap_or_default();
        
        let max_strength = anomalies.iter()
            .map(|a| a.anomaly_strength)
            .fold(0.0f64, f64::max);
        
        anomaly_strengths.push(max_strength);
        println!("      {}: max anomaly strength = {:.6}", description, max_strength);
    }
    
    // Test 4: Consistency between information and anomaly strength
    println!("    Testing consistency between information and anomaly strength");
    
    // Higher information should generally lead to higher anomaly strength
    if anomaly_strengths.len() >= 3 {
        if anomaly_strengths[2] <= anomaly_strengths[0] {
            violations += 1;
            details.push("Information-strength consistency violation".to_string());
        }
    }
    
    // Test 5: Mathematical properties of information scores
    println!("    Testing mathematical properties of information scores");
    
    // All information scores should be non-negative
    let negative_info_scores = info_scores.iter().filter(|&&score| score < 0.0).count();
    if negative_info_scores > 0 {
        violations += 1;
        details.push(format!("Negative information scores: {}", negative_info_scores));
    }
    
    // Information scores should be finite
    let infinite_info_scores = info_scores.iter().filter(|&&score| !score.is_finite()).count();
    if infinite_info_scores > 0 {
        violations += 1;
        details.push(format!("Infinite information scores: {}", infinite_info_scores));
    }
    
    details.push(format!("Information scores: {:?}", info_scores));
    details.push(format!("Anomaly strengths: {:?}", anomaly_strengths));
    
    if violations == 0 {
        DomainTestResult::pass("Information-theoretic scoring working correctly".to_string())
            .with_details(details)
    } else {
        DomainTestResult::fail(format!("{} information-theoretic scoring violations", violations))
            .with_details(details)
    }
}

/// Test 4.4: Threshold Semantics
/// 
/// Tests that thresholds work as expected:
/// - Higher thresholds should result in fewer detections
/// - Threshold behavior should be monotonic
/// - Threshold values should have clear semantic meaning
fn test_threshold_semantics_comprehensive() -> DomainTestResult {
    println!("  Testing threshold semantics...");
    
    let mut detector = AnomalyDetector::new(1).expect("Failed to create detector");
    
    // Train with a simple pattern
    let training_sequence = vec!["NORMAL", "NORMAL", "ANOMALY"].repeat(100)
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    detector.train(&training_sequence).expect("Failed to train");
    
    let mut violations = 0;
    let mut details = Vec::new();
    
    // Test 1: Threshold monotonicity
    println!("    Testing threshold monotonicity");
    
    let test_sequence = vec!["UNKNOWN".to_string(), "STRANGE".to_string()];
    let thresholds = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    
    let mut detection_counts = Vec::new();
    for &threshold in &thresholds {
        let anomalies = detector.detect_anomalies(&test_sequence, threshold).unwrap_or_default();
        detection_counts.push(anomalies.len());
        println!("      Threshold {:.1}: {} detections", threshold, anomalies.len());
    }
    
    // Check monotonicity (non-increasing)
    let monotonic_violations = detection_counts.windows(2)
        .filter(|w| w[1] > w[0])
        .count();
    
    if monotonic_violations > 0 {
        violations += 1;
        details.push(format!("Threshold monotonicity violations: {}", monotonic_violations));
    }
    
    // Test 2: Threshold effectiveness
    println!("    Testing threshold effectiveness");
    
    let low_threshold_count = detection_counts[0]; // threshold = 0.0
    let high_threshold_count = detection_counts[detection_counts.len() - 1]; // threshold = 1.0
    
    if low_threshold_count == high_threshold_count {
        violations += 1;
        details.push("Threshold has no effect on detection count".to_string());
    }
    
    // Test 3: Threshold precision
    println!("    Testing threshold precision");
    
    // Get all anomaly scores for the test sequence
    let all_anomalies = detector.detect_anomalies(&test_sequence, 0.0).unwrap_or_default();
    let all_scores: Vec<f64> = all_anomalies.iter().map(|a| a.anomaly_strength).collect();
    
    println!("      All anomaly scores: {:?}", all_scores);
    
    // Test specific thresholds against actual scores
    for &threshold in &[0.1, 0.5, 0.9] {
        let expected_count = all_scores.iter().filter(|&&score| score >= threshold).count();
        let actual_anomalies = detector.detect_anomalies(&test_sequence, threshold).unwrap_or_default();
        let actual_count = actual_anomalies.len();
        
        if expected_count != actual_count {
            violations += 1;
            details.push(format!("Threshold precision violation at {:.1}: expected {}, got {}", 
                               threshold, expected_count, actual_count));
        }
    }
    
    // Test 4: Semantic meaning of thresholds
    println!("    Testing semantic meaning of thresholds");
    
    // Very low threshold (0.01) should catch almost everything
    let very_low_anomalies = detector.detect_anomalies(&test_sequence, 0.01).unwrap_or_default();
    
    // Very high threshold (0.99) should catch almost nothing
    let very_high_anomalies = detector.detect_anomalies(&test_sequence, 0.99).unwrap_or_default();
    
    println!("      Very low threshold (0.01): {} detections", very_low_anomalies.len());
    println!("      Very high threshold (0.99): {} detections", very_high_anomalies.len());
    
    if very_low_anomalies.len() < very_high_anomalies.len() {
        violations += 1;
        details.push("Semantic threshold meaning violation".to_string());
    }
    
    details.push(format!("Detection counts: {:?}", detection_counts));
    details.push(format!("All scores: {:?}", all_scores));
    
    if violations == 0 {
        DomainTestResult::pass("Threshold semantics working correctly".to_string())
            .with_details(details)
    } else {
        DomainTestResult::fail(format!("{} threshold semantic violations", violations))
            .with_details(details)
    }
}

/// Test 4.5: Anomaly Strength Consistency
/// 
/// Tests that anomaly strength values are consistent and meaningful:
/// - Strength values should be in valid range [0,1]
/// - Stronger anomalies should have higher strength values
/// - Strength should correlate with likelihood and information content
fn test_anomaly_strength_consistency() -> DomainTestResult {
    println!("  Testing anomaly strength consistency...");
    
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    
    // Train with varied patterns
    let training_sequence = vec!["A", "B", "A", "C", "A", "B", "A", "D"].repeat(50)
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    detector.train(&training_sequence).expect("Failed to train");
    
    let mut violations = 0;
    let mut details = Vec::new();
    
    // Test 1: Strength value range
    println!("    Testing anomaly strength value range");
    
    let test_sequences = vec![
        vec!["A", "B"],      // Normal
        vec!["A", "X"],      // Partially anomalous
        vec!["X", "Y"],      // Fully anomalous
        vec!["Z", "W", "Q"], // Very anomalous
    ];
    
    let mut all_strengths = Vec::new();
    for (i, sequence) in test_sequences.iter().enumerate() {
        let seq_strings: Vec<String> = sequence.iter().map(|s| s.to_string()).collect();
        let anomalies = detector.detect_anomalies(&seq_strings, 0.0).unwrap_or_default();
        
        for anomaly in &anomalies {
            all_strengths.push(anomaly.anomaly_strength);
            
            // Check range [0,1]
            if anomaly.anomaly_strength < 0.0 || anomaly.anomaly_strength > 1.0 {
                violations += 1;
                details.push(format!("Strength out of range in sequence {}: {:.6}", 
                                   i, anomaly.anomaly_strength));
            }
        }
        
        let max_strength = anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
        println!("      Sequence {}: max strength = {:.6}", i, max_strength);
    }
    
    // Test 2: Strength ordering
    println!("    Testing strength ordering");
    
    let normal_seq = vec!["A".to_string(), "B".to_string()];
    let anomalous_seq = vec!["X".to_string(), "Y".to_string()];
    
    let normal_anomalies = detector.detect_anomalies(&normal_seq, 0.0).unwrap_or_default();
    let anomalous_anomalies = detector.detect_anomalies(&anomalous_seq, 0.0).unwrap_or_default();
    
    let normal_max = normal_anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
    let anomalous_max = anomalous_anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
    
    println!("      Normal sequence max strength: {:.6}", normal_max);
    println!("      Anomalous sequence max strength: {:.6}", anomalous_max);
    
    if anomalous_max <= normal_max {
        violations += 1;
        details.push(format!("Strength ordering violation: anomalous ({:.6}) <= normal ({:.6})", 
                           anomalous_max, normal_max));
    }
    
    // Test 3: Strength components consistency
    println!("    Testing strength components consistency");
    
    let test_seq = vec!["UNKNOWN".to_string()];
    let test_anomalies = detector.detect_anomalies(&test_seq, 0.0).unwrap_or_default();
    
    for (i, anomaly) in test_anomalies.iter().enumerate() {
        println!("      Anomaly {}: strength={:.6}, likelihood={:.6}, info={:.6}", 
                i, anomaly.anomaly_strength, anomaly.likelihood, anomaly.information_score);
        
        // Strength should be related to likelihood and information
        // Lower likelihood should contribute to higher strength
        if anomaly.likelihood > 0.9 && anomaly.anomaly_strength > 0.5 {
            violations += 1;
            details.push(format!("High likelihood ({:.6}) with high strength ({:.6})", 
                               anomaly.likelihood, anomaly.anomaly_strength));
        }
        
        // Information score should be non-negative
        if anomaly.information_score < 0.0 {
            violations += 1;
            details.push(format!("Negative information score: {:.6}", anomaly.information_score));
        }
    }
    
    // Test 4: Strength distribution
    println!("    Testing strength distribution");
    
    if !all_strengths.is_empty() {
        let min_strength = all_strengths.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_strength = all_strengths.iter().fold(0.0f64, |a, &b| a.max(b));
        let mean_strength = all_strengths.iter().sum::<f64>() / all_strengths.len() as f64;
        
        println!("      Strength distribution: min={:.6}, max={:.6}, mean={:.6}", 
                min_strength, max_strength, mean_strength);
        
        // Should have reasonable distribution
        if max_strength - min_strength < 0.01 {
            violations += 1;
            details.push("Insufficient strength variation".to_string());
        }
        
        details.push(format!("Strength stats: min={:.6}, max={:.6}, mean={:.6}", 
                           min_strength, max_strength, mean_strength));
    }
    
    details.push(format!("Total strengths tested: {}", all_strengths.len()));
    
    if violations == 0 {
        DomainTestResult::pass("Anomaly strength consistency verified".to_string())
            .with_details(details)
    } else {
        DomainTestResult::fail(format!("{} anomaly strength violations", violations))
            .with_details(details)
    }
}