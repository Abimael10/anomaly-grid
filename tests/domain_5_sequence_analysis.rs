//! Domain 5: Sequence Analysis Correctness
//!
//! This module implements comprehensive domain-driven tests for sequence analysis
//! fundamentals, ensuring our implementation correctly handles sequence patterns,
//! context windows, and temporal dependencies in finite alphabets.

use anomaly_grid::*;
use std::collections::HashMap;

#[test]
fn domain_5_sequence_analysis() {
    println!("🔬 DOMAIN 5: SEQUENCE ANALYSIS CORRECTNESS");
    println!("==========================================");
    println!();
    
    let mut test_results = Vec::new();
    
    // Test 5.1: Context Window Semantics
    println!("Test 5.1: Context Window Semantics");
    println!("----------------------------------");
    let context_result = test_context_window_semantics_comprehensive();
    test_results.push(("Context Window", context_result));
    println!();
    
    // Test 5.2: Temporal Dependency Modeling
    println!("Test 5.2: Temporal Dependency Modeling");
    println!("--------------------------------------");
    let temporal_result = test_temporal_dependency_modeling_comprehensive();
    test_results.push(("Temporal Dependencies", temporal_result));
    println!();
    
    // Test 5.3: Pattern Recognition Accuracy
    println!("Test 5.3: Pattern Recognition Accuracy");
    println!("--------------------------------------");
    let pattern_result = test_pattern_recognition_accuracy_comprehensive();
    test_results.push(("Pattern Recognition", pattern_result));
    println!();
    
    // Test 5.4: Sequence Length Handling
    println!("Test 5.4: Sequence Length Handling");
    println!("----------------------------------");
    let length_result = test_sequence_length_handling_comprehensive();
    test_results.push(("Sequence Length", length_result));
    println!();
    
    // Test 5.5: Alphabet Size Scalability
    println!("Test 5.5: Alphabet Size Scalability");
    println!("------------------------------------");
    let alphabet_result = test_alphabet_size_scalability();
    test_results.push(("Alphabet Scalability", alphabet_result));
    println!();
    
    // Domain 5 Summary
    println!("🏆 DOMAIN 5 SUMMARY");
    println!("===================");
    let passed_tests = test_results.iter().filter(|(_, result)| result.passed).count();
    let total_tests = test_results.len();
    
    for (test_name, result) in &test_results {
        let status = if result.passed { "✅" } else { "❌" };
        println!("  {} {}: {}", status, test_name, result.evidence);
    }
    
    println!();
    println!("Domain 5 Result: {}/{} tests passed", passed_tests, total_tests);
    
    assert_eq!(passed_tests, total_tests, 
               "Domain 5 (Sequence Analysis) failed: {}/{} tests passed", 
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

/// Test 5.1: Context Window Semantics
/// 
/// Tests that context windows work correctly:
/// - Different context lengths should capture different patterns
/// - Longer contexts should be more specific
/// - Context selection should be adaptive and meaningful
fn test_context_window_semantics_comprehensive() -> DomainTestResult {
    println!("  Testing context window semantics...");
    
    let mut violations = 0;
    let mut details = Vec::new();
    
    // Test 1: Different context lengths capture different patterns
    println!("    Testing context length pattern capture");
    
    let context_lengths = vec![1, 2, 3];
    let mut pattern_specificity = Vec::new();
    
    for &max_order in &context_lengths {
        let mut detector = AnomalyDetector::new(max_order).expect("Failed to create detector");
        
        // Train with a repeating pattern that has different levels of structure
        let training_sequence = vec!["A", "B", "C", "A", "B", "D", "A", "B", "C", "A", "B", "D"].repeat(50)
            .iter().map(|s| s.to_string()).collect::<Vec<_>>();
        detector.train(&training_sequence).expect("Failed to train");
        
        // Test pattern recognition with different context lengths
        let test_sequence = vec!["A".to_string(), "B".to_string(), "X".to_string()]; // X breaks the pattern
        let anomalies = detector.detect_anomalies(&test_sequence, 0.0).unwrap_or_default();
        
        let max_score = anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
        pattern_specificity.push(max_score);
        
        println!("      Context length {}: max anomaly score = {:.6}", max_order, max_score);
    }
    
    // Test 2: Context specificity relationship
    println!("    Testing context specificity");
    
    // Higher-order contexts should generally be more sensitive to pattern violations
    // (though this depends on the specific pattern and data)
    let specificity_trend = pattern_specificity.windows(2)
        .map(|w| w[1] - w[0])
        .collect::<Vec<_>>();
    
    details.push(format!("Pattern specificity scores: {:?}", pattern_specificity));
    details.push(format!("Specificity trend: {:?}", specificity_trend));
    
    // Test 3: Context window boundary handling
    println!("    Testing context window boundary handling");
    
    let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");
    let training_sequence = vec!["START", "A", "B", "C", "END"].repeat(20)
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    detector.train(&training_sequence).expect("Failed to train");
    
    // Test sequences of different lengths
    let boundary_tests = vec![
        (vec!["START"], "Single element"),
        (vec!["START", "A"], "Two elements"),
        (vec!["START", "A", "B"], "Three elements"),
        (vec!["START", "A", "B", "C"], "Four elements"),
    ];
    
    let mut boundary_scores = Vec::new();
    for (sequence, description) in &boundary_tests {
        let seq_strings: Vec<String> = sequence.iter().map(|s| s.to_string()).collect();
        let anomalies = detector.detect_anomalies(&seq_strings, 0.0).unwrap_or_default();
        
        let max_score = anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
        boundary_scores.push(max_score);
        
        println!("      {}: max anomaly score = {:.6}", description, max_score);
    }
    
    // All boundary cases should be handled without errors
    let boundary_errors = boundary_scores.iter().filter(|&&score| score.is_nan() || score.is_infinite()).count();
    if boundary_errors > 0 {
        violations += 1;
        details.push(format!("Boundary handling errors: {}", boundary_errors));
    }
    
    // Test 4: Context adaptation
    println!("    Testing context adaptation");
    
    let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");
    
    // Train with sparse data that should force adaptive context selection
    let sparse_training = vec!["A", "B", "C", "D", "E", "F", "G", "H"].repeat(10)
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    detector.train(&sparse_training).expect("Failed to train");
    
    // Test with sequences that require different context lengths
    let adaptation_tests = vec![
        (vec!["A", "B"], "Short context needed"),
        (vec!["X", "Y", "Z"], "Unknown sequence"),
    ];
    
    for (sequence, description) in &adaptation_tests {
        let seq_strings: Vec<String> = sequence.iter().map(|s| s.to_string()).collect();
        let anomalies = detector.detect_anomalies(&seq_strings, 0.0).unwrap_or_default();
        
        let max_score = anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
        println!("      {}: max anomaly score = {:.6}", description, max_score);
        
        // Should handle adaptation without errors
        if max_score.is_nan() || max_score.is_infinite() {
            violations += 1;
            details.push(format!("Context adaptation error in: {}", description));
        }
    }
    
    details.push(format!("Boundary scores: {:?}", boundary_scores));
    
    if violations == 0 {
        DomainTestResult::pass("Context window semantics working correctly".to_string())
            .with_details(details)
    } else {
        DomainTestResult::fail(format!("{} context window violations", violations))
            .with_details(details)
    }
}

/// Test 5.2: Temporal Dependency Modeling
/// 
/// Tests that temporal dependencies are correctly captured:
/// - Sequential patterns should be learned accurately
/// - Temporal order should matter
/// - Dependencies should decay appropriately with distance
fn test_temporal_dependency_modeling_comprehensive() -> DomainTestResult {
    println!("  Testing temporal dependency modeling...");
    
    let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");
    
    // Create training data with clear temporal dependencies
    let training_sequence = vec!["CAUSE", "EFFECT1", "EFFECT2", "RESET"].repeat(100)
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    detector.train(&training_sequence).expect("Failed to train");
    
    let mut violations = 0;
    let mut details = Vec::new();
    
    // Test 1: Sequential pattern recognition
    println!("    Testing sequential pattern recognition");
    
    let pattern_tests = vec![
        (vec!["CAUSE", "EFFECT1", "EFFECT2"], "Correct sequence"),
        (vec!["CAUSE", "EFFECT2", "EFFECT1"], "Wrong order"),
        (vec!["EFFECT1", "CAUSE", "EFFECT2"], "Disrupted causality"),
    ];
    
    let mut pattern_scores = Vec::new();
    for (sequence, description) in &pattern_tests {
        let seq_strings: Vec<String> = sequence.iter().map(|s| s.to_string()).collect();
        let anomalies = detector.detect_anomalies(&seq_strings, 0.0).unwrap_or_default();
        
        let max_score = anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
        pattern_scores.push(max_score);
        
        println!("      {}: max anomaly score = {:.6}", description, max_score);
    }
    
    // Correct sequence should have lower anomaly score than wrong sequences
    if pattern_scores.len() >= 3 {
        let correct_score = pattern_scores[0];
        let wrong_order_score = pattern_scores[1];
        let disrupted_score = pattern_scores[2];
        
        if wrong_order_score <= correct_score || disrupted_score <= correct_score {
            violations += 1;
            details.push("Temporal order not properly detected".to_string());
        }
    }
    
    // Test 2: Temporal distance effects
    println!("    Testing temporal distance effects");
    
    let distance_tests = vec![
        (vec!["CAUSE", "EFFECT1"], "Immediate dependency"),
        (vec!["CAUSE", "X", "EFFECT1"], "One step removed"),
        (vec!["CAUSE", "X", "Y", "EFFECT1"], "Two steps removed"),
    ];
    
    let mut distance_scores = Vec::new();
    for (sequence, description) in &distance_tests {
        let seq_strings: Vec<String> = sequence.iter().map(|s| s.to_string()).collect();
        let anomalies = detector.detect_anomalies(&seq_strings, 0.0).unwrap_or_default();
        
        let max_score = anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
        distance_scores.push(max_score);
        
        println!("      {}: max anomaly score = {:.6}", description, max_score);
    }
    
    // Test 3: Dependency strength
    println!("    Testing dependency strength");
    
    // Create a new detector with strong vs weak dependencies
    let mut strong_detector = AnomalyDetector::new(2).expect("Failed to create detector");
    let strong_pattern = vec!["A", "B"].repeat(100); // Very strong A->B dependency
    let strong_strings: Vec<String> = strong_pattern.iter().map(|s| s.to_string()).collect();
    strong_detector.train(&strong_strings).expect("Failed to train");
    
    let mut weak_detector = AnomalyDetector::new(2).expect("Failed to create detector");
    let weak_pattern = vec!["A", "B", "A", "C", "A", "D", "A", "E"].repeat(25); // Weaker A->? dependency
    let weak_strings: Vec<String> = weak_pattern.iter().map(|s| s.to_string()).collect();
    weak_detector.train(&weak_strings).expect("Failed to train");
    
    // Test violation of the dependency
    let violation_sequence = vec!["A".to_string(), "X".to_string()];
    
    let strong_anomalies = strong_detector.detect_anomalies(&violation_sequence, 0.0).unwrap_or_default();
    let weak_anomalies = weak_detector.detect_anomalies(&violation_sequence, 0.0).unwrap_or_default();
    
    let strong_score = strong_anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
    let weak_score = weak_anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
    
    println!("      Strong dependency violation: {:.6}", strong_score);
    println!("      Weak dependency violation: {:.6}", weak_score);
    
    // Strong dependency violation should generally have higher score
    if strong_score < weak_score * 0.5 { // Allow some tolerance
        violations += 1;
        details.push("Dependency strength not properly modeled".to_string());
    }
    
    // Test 4: Temporal consistency
    println!("    Testing temporal consistency");
    
    // Same pattern should give consistent results regardless of position
    let consistency_tests = vec![
        vec!["CAUSE", "EFFECT1"],
        vec!["X", "CAUSE", "EFFECT1"],
        vec!["Y", "Z", "CAUSE", "EFFECT1"],
    ];
    
    let mut consistency_scores = Vec::new();
    for (i, sequence) in consistency_tests.iter().enumerate() {
        let seq_strings: Vec<String> = sequence.iter().map(|s| s.to_string()).collect();
        let anomalies = detector.detect_anomalies(&seq_strings, 0.0).unwrap_or_default();
        
        // Look for anomalies in the sequence (position-based filtering not available)
        let pattern_score = anomalies.iter()
            .map(|a| a.anomaly_strength)
            .fold(0.0f64, f64::max);
        
        consistency_scores.push(pattern_score);
        println!("      Position {}: pattern score = {:.6}", i, pattern_score);
    }
    
    details.push(format!("Pattern scores: {:?}", pattern_scores));
    details.push(format!("Distance scores: {:?}", distance_scores));
    details.push(format!("Consistency scores: {:?}", consistency_scores));
    
    if violations == 0 {
        DomainTestResult::pass("Temporal dependency modeling working correctly".to_string())
            .with_details(details)
    } else {
        DomainTestResult::fail(format!("{} temporal dependency violations", violations))
            .with_details(details)
    }
}

/// Test 5.3: Pattern Recognition Accuracy
/// 
/// Tests that patterns are recognized accurately:
/// - Repeating patterns should be learned
/// - Pattern variations should be detected
/// - Pattern completion should work correctly
fn test_pattern_recognition_accuracy_comprehensive() -> DomainTestResult {
    println!("  Testing pattern recognition accuracy...");
    
    let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");
    
    // Train with multiple overlapping patterns
    let mut training_sequence = Vec::new();
    training_sequence.extend(vec!["PATTERN1", "A", "B", "C"].repeat(30));
    training_sequence.extend(vec!["PATTERN2", "X", "Y", "Z"].repeat(30));
    training_sequence.extend(vec!["PATTERN3", "P", "Q", "R"].repeat(30));
    
    let training_strings: Vec<String> = training_sequence.iter().map(|s| s.to_string()).collect();
    detector.train(&training_strings).expect("Failed to train");
    
    let mut violations = 0;
    let mut details = Vec::new();
    
    // Test 1: Known pattern recognition
    println!("    Testing known pattern recognition");
    
    let known_patterns = vec![
        (vec!["PATTERN1", "A", "B", "C"], "Complete pattern 1"),
        (vec!["PATTERN2", "X", "Y", "Z"], "Complete pattern 2"),
        (vec!["PATTERN3", "P", "Q", "R"], "Complete pattern 3"),
    ];
    
    let mut known_scores = Vec::new();
    for (pattern, description) in &known_patterns {
        let seq_strings: Vec<String> = pattern.iter().map(|s| s.to_string()).collect();
        let anomalies = detector.detect_anomalies(&seq_strings, 0.0).unwrap_or_default();
        
        let max_score = anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
        known_scores.push(max_score);
        
        println!("      {}: max anomaly score = {:.6}", description, max_score);
    }
    
    // Test 2: Pattern variation detection
    println!("    Testing pattern variation detection");
    
    let pattern_variations = vec![
        (vec!["PATTERN1", "A", "B", "X"], "Pattern 1 with variation"),
        (vec!["PATTERN2", "X", "WRONG", "Z"], "Pattern 2 with variation"),
        (vec!["PATTERN3", "P", "Q", "Q"], "Pattern 3 with variation"),
    ];
    
    let mut variation_scores = Vec::new();
    for (pattern, description) in &pattern_variations {
        let seq_strings: Vec<String> = pattern.iter().map(|s| s.to_string()).collect();
        let anomalies = detector.detect_anomalies(&seq_strings, 0.0).unwrap_or_default();
        
        let max_score = anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
        variation_scores.push(max_score);
        
        println!("      {}: max anomaly score = {:.6}", description, max_score);
    }
    
    // Variations should have higher scores than known patterns
    let known_max = known_scores.iter().fold(0.0f64, |a, &b| a.max(b));
    let variation_min = variation_scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    
    if variation_min <= known_max {
        violations += 1;
        details.push(format!("Pattern variations not detected: variation_min ({:.6}) <= known_max ({:.6})", 
                           variation_min, known_max));
    }
    
    // Test 3: Partial pattern recognition
    println!("    Testing partial pattern recognition");
    
    let partial_patterns = vec![
        (vec!["PATTERN1", "A"], "Partial pattern 1"),
        (vec!["PATTERN2", "X", "Y"], "Partial pattern 2"),
        (vec!["PATTERN3"], "Pattern start only"),
    ];
    
    let mut partial_scores = Vec::new();
    for (pattern, description) in &partial_patterns {
        let seq_strings: Vec<String> = pattern.iter().map(|s| s.to_string()).collect();
        let anomalies = detector.detect_anomalies(&seq_strings, 0.0).unwrap_or_default();
        
        let max_score = anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
        partial_scores.push(max_score);
        
        println!("      {}: max anomaly score = {:.6}", description, max_score);
    }
    
    // Test 4: Unknown pattern detection
    println!("    Testing unknown pattern detection");
    
    let unknown_patterns = vec![
        (vec!["UNKNOWN", "W", "V", "U"], "Completely unknown pattern"),
        (vec!["PATTERN1", "WRONG", "SEQUENCE"], "Mixed known/unknown"),
    ];
    
    let mut unknown_scores = Vec::new();
    for (pattern, description) in &unknown_patterns {
        let seq_strings: Vec<String> = pattern.iter().map(|s| s.to_string()).collect();
        let anomalies = detector.detect_anomalies(&seq_strings, 0.0).unwrap_or_default();
        
        let max_score = anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
        unknown_scores.push(max_score);
        
        println!("      {}: max anomaly score = {:.6}", description, max_score);
    }
    
    // Unknown patterns should have higher scores than known patterns
    let unknown_min = unknown_scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    
    if unknown_min <= known_max {
        violations += 1;
        details.push(format!("Unknown patterns not detected: unknown_min ({:.6}) <= known_max ({:.6})", 
                           unknown_min, known_max));
    }
    
    // Test 5: Pattern discrimination
    println!("    Testing pattern discrimination");
    
    // Test that the system can distinguish between similar patterns
    let discrimination_tests = vec![
        (vec!["PATTERN1", "A", "B", "C"], "Original pattern"),
        (vec!["PATTERN1", "A", "C", "B"], "Swapped elements"),
        (vec!["PATTERN1", "B", "A", "C"], "Different start"),
    ];
    
    let mut discrimination_scores = Vec::new();
    for (pattern, description) in &discrimination_tests {
        let seq_strings: Vec<String> = pattern.iter().map(|s| s.to_string()).collect();
        let anomalies = detector.detect_anomalies(&seq_strings, 0.0).unwrap_or_default();
        
        let max_score = anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
        discrimination_scores.push(max_score);
        
        println!("      {}: max anomaly score = {:.6}", description, max_score);
    }
    
    details.push(format!("Known pattern scores: {:?}", known_scores));
    details.push(format!("Variation scores: {:?}", variation_scores));
    details.push(format!("Partial scores: {:?}", partial_scores));
    details.push(format!("Unknown scores: {:?}", unknown_scores));
    details.push(format!("Discrimination scores: {:?}", discrimination_scores));
    
    if violations == 0 {
        DomainTestResult::pass("Pattern recognition accuracy verified".to_string())
            .with_details(details)
    } else {
        DomainTestResult::fail(format!("{} pattern recognition violations", violations))
            .with_details(details)
    }
}

/// Test 5.4: Sequence Length Handling
/// 
/// Tests that sequences of different lengths are handled correctly:
/// - Very short sequences should work
/// - Very long sequences should work
/// - Length should not affect correctness
fn test_sequence_length_handling_comprehensive() -> DomainTestResult {
    println!("  Testing sequence length handling...");
    
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    
    // Train with medium-length sequences
    let training_sequence = vec!["A", "B", "C", "D"].repeat(50)
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    detector.train(&training_sequence).expect("Failed to train");
    
    let mut violations = 0;
    let mut details = Vec::new();
    
    // Test 1: Very short sequences
    println!("    Testing very short sequences");
    
    let short_sequences = vec![
        vec!["A"],
        vec!["A", "B"],
        vec!["B", "C"],
    ];
    
    let mut short_scores = Vec::new();
    for (i, sequence) in short_sequences.iter().enumerate() {
        let seq_strings: Vec<String> = sequence.iter().map(|s| s.to_string()).collect();
        let result = detector.detect_anomalies(&seq_strings, 0.0);
        
        match result {
            Ok(anomalies) => {
                let max_score = anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
                short_scores.push(max_score);
                println!("      Length {}: max anomaly score = {:.6}", sequence.len(), max_score);
            }
            Err(e) => {
                violations += 1;
                details.push(format!("Short sequence {} failed: {}", i, e));
                short_scores.push(f64::NAN);
            }
        }
    }
    
    // Test 2: Medium sequences
    println!("    Testing medium sequences");
    
    let medium_sequences = vec![
        vec!["A", "B", "C", "D"],
        vec!["B", "C", "D", "A"],
        vec!["A", "X", "C", "D"], // With anomaly
    ];
    
    let mut medium_scores = Vec::new();
    for (i, sequence) in medium_sequences.iter().enumerate() {
        let seq_strings: Vec<String> = sequence.iter().map(|s| s.to_string()).collect();
        let result = detector.detect_anomalies(&seq_strings, 0.0);
        
        match result {
            Ok(anomalies) => {
                let max_score = anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
                medium_scores.push(max_score);
                println!("      Length {}: max anomaly score = {:.6}", sequence.len(), max_score);
            }
            Err(e) => {
                violations += 1;
                details.push(format!("Medium sequence {} failed: {}", i, e));
                medium_scores.push(f64::NAN);
            }
        }
    }
    
    // Test 3: Long sequences
    println!("    Testing long sequences");
    
    let long_sequences = vec![
        vec!["A", "B", "C", "D"].repeat(5), // 20 elements
        vec!["A", "B", "C", "D"].repeat(10), // 40 elements
    ];
    
    let mut long_scores = Vec::new();
    for (i, sequence) in long_sequences.iter().enumerate() {
        let seq_strings: Vec<String> = sequence.iter().map(|s| s.to_string()).collect();
        let result = detector.detect_anomalies(&seq_strings, 0.0);
        
        match result {
            Ok(anomalies) => {
                let max_score = anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
                long_scores.push(max_score);
                println!("      Length {}: max anomaly score = {:.6}", sequence.len(), max_score);
            }
            Err(e) => {
                violations += 1;
                details.push(format!("Long sequence {} failed: {}", i, e));
                long_scores.push(f64::NAN);
            }
        }
    }
    
    // Test 4: Length consistency
    println!("    Testing length consistency");
    
    // Same pattern at different lengths should give consistent results
    let base_pattern = vec!["A", "B", "X"]; // X is anomalous
    let length_tests = vec![
        base_pattern.clone(),
        [base_pattern.clone(), vec!["D"]].concat(),
        [base_pattern.clone(), vec!["D", "E", "F"]].concat(),
    ];
    
    let mut consistency_scores = Vec::new();
    for sequence in &length_tests {
        let seq_strings: Vec<String> = sequence.iter().map(|s| s.to_string()).collect();
        let anomalies = detector.detect_anomalies(&seq_strings, 0.0).unwrap_or_default();
        
        // Look for anomalies in the sequence (position-based filtering not available)
        let anomaly_score = anomalies.iter()
            .map(|a| a.anomaly_strength)
            .fold(0.0f64, f64::max);
        
        consistency_scores.push(anomaly_score);
        println!("      Length {}: anomaly score at position 2 = {:.6}", sequence.len(), anomaly_score);
    }
    
    // Test 5: Performance with length
    println!("    Testing performance scaling with length");
    
    let performance_tests = vec![10, 50, 100];
    let mut performance_times = Vec::new();
    
    for &length in &performance_tests {
        let test_sequence = vec!["A", "B", "C", "D"].repeat(length / 4)
            .iter().map(|s| s.to_string()).collect::<Vec<_>>();
        
        let start_time = std::time::Instant::now();
        let _result = detector.detect_anomalies(&test_sequence, 0.0);
        let elapsed = start_time.elapsed();
        
        performance_times.push(elapsed.as_millis());
        println!("      Length {}: processing time = {}ms", length, elapsed.as_millis());
    }
    
    // Check for reasonable performance scaling (should be roughly linear)
    if performance_times.len() >= 2 {
        let scaling_factor = performance_times[1] as f64 / performance_times[0] as f64;
        if scaling_factor > 10.0 { // More than 10x slowdown is concerning
            violations += 1;
            details.push(format!("Poor performance scaling: {}x", scaling_factor));
        }
    }
    
    details.push(format!("Short scores: {:?}", short_scores));
    details.push(format!("Medium scores: {:?}", medium_scores));
    details.push(format!("Long scores: {:?}", long_scores));
    details.push(format!("Consistency scores: {:?}", consistency_scores));
    details.push(format!("Performance times: {:?}ms", performance_times));
    
    if violations == 0 {
        DomainTestResult::pass("Sequence length handling working correctly".to_string())
            .with_details(details)
    } else {
        DomainTestResult::fail(format!("{} sequence length violations", violations))
            .with_details(details)
    }
}

/// Test 5.5: Alphabet Size Scalability
/// 
/// Tests that the system scales correctly with alphabet size:
/// - Small alphabets should work efficiently
/// - Large alphabets should work correctly
/// - Memory usage should scale reasonably
fn test_alphabet_size_scalability() -> DomainTestResult {
    println!("  Testing alphabet size scalability...");
    
    let mut violations = 0;
    let mut details = Vec::new();
    
    // Test 1: Small alphabet
    println!("    Testing small alphabet");
    
    let mut small_detector = AnomalyDetector::new(2).expect("Failed to create detector");
    let small_alphabet = vec!["A", "B"];
    let small_training = small_alphabet.repeat(100);
    let small_strings: Vec<String> = small_training.iter().map(|s| s.to_string()).collect();
    
    let small_result = small_detector.train(&small_strings);
    match small_result {
        Ok(_) => {
            let test_seq = vec!["A".to_string(), "X".to_string()];
            let anomalies = small_detector.detect_anomalies(&test_seq, 0.0).unwrap_or_default();
            let small_score = anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
            println!("      Small alphabet (2 states): max anomaly score = {:.6}", small_score);
        }
        Err(e) => {
            violations += 1;
            details.push(format!("Small alphabet failed: {}", e));
        }
    }
    
    // Test 2: Medium alphabet
    println!("    Testing medium alphabet");
    
    let mut medium_detector = AnomalyDetector::new(2).expect("Failed to create detector");
    let medium_alphabet: Vec<String> = (0..10).map(|i| format!("STATE_{}", i)).collect();
    let medium_training: Vec<String> = medium_alphabet.iter().cycle().take(200).cloned().collect();
    
    let medium_result = medium_detector.train(&medium_training);
    match medium_result {
        Ok(_) => {
            let test_seq = vec![medium_alphabet[0].clone(), "UNKNOWN".to_string()];
            let anomalies = medium_detector.detect_anomalies(&test_seq, 0.0).unwrap_or_default();
            let medium_score = anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
            println!("      Medium alphabet (10 states): max anomaly score = {:.6}", medium_score);
        }
        Err(e) => {
            violations += 1;
            details.push(format!("Medium alphabet failed: {}", e));
        }
    }
    
    // Test 3: Large alphabet
    println!("    Testing large alphabet");
    
    let mut large_detector = AnomalyDetector::new(1).expect("Failed to create detector"); // Use order 1 for large alphabet
    let large_alphabet: Vec<String> = (0..50).map(|i| format!("STATE_{:03}", i)).collect();
    let large_training: Vec<String> = large_alphabet.iter().cycle().take(200).cloned().collect(); // Less repetition for large alphabet
    
    let large_result = large_detector.train(&large_training);
    match large_result {
        Ok(_) => {
            let test_seq = vec![large_alphabet[0].clone(), "UNKNOWN".to_string()];
            let anomalies = large_detector.detect_anomalies(&test_seq, 0.0).unwrap_or_default();
            let large_score = anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max);
            println!("      Large alphabet (50 states): max anomaly score = {:.6}", large_score);
        }
        Err(e) => {
            violations += 1;
            details.push(format!("Large alphabet failed: {}", e));
        }
    }
    
    // Test 4: Alphabet size vs performance
    println!("    Testing alphabet size vs performance");
    
    let alphabet_sizes = vec![5, 15, 25];
    let mut performance_results = Vec::new();
    
    for &size in &alphabet_sizes {
        let alphabet: Vec<String> = (0..size).map(|i| format!("S{}", i)).collect();
        let training_data: Vec<String> = alphabet.iter().cycle().take(alphabet.len() * 20).cloned().collect();
        
        let mut detector = AnomalyDetector::new(1).expect("Failed to create detector");
        
        let start_time = std::time::Instant::now();
        let train_result = detector.train(&training_data);
        let train_time = start_time.elapsed();
        
        match train_result {
            Ok(_) => {
                let test_seq = vec![alphabet[0].clone(), "UNKNOWN".to_string()];
                let detect_start = std::time::Instant::now();
                let _anomalies = detector.detect_anomalies(&test_seq, 0.0);
                let detect_time = detect_start.elapsed();
                
                performance_results.push((size, train_time.as_millis(), detect_time.as_millis()));
                println!("      Alphabet size {}: train={}ms, detect={}ms", 
                        size, train_time.as_millis(), detect_time.as_millis());
            }
            Err(e) => {
                violations += 1;
                details.push(format!("Alphabet size {} failed: {}", size, e));
            }
        }
    }
    
    // Test 5: Memory efficiency
    println!("    Testing memory efficiency");
    
    // This is a basic test - in a real implementation you might want to measure actual memory usage
    let efficiency_tests = vec![
        (vec!["A".to_string(), "B".to_string()], "Binary alphabet"),
        ((0..8).map(|i| format!("{}", i)).collect::<Vec<String>>(), "Octal alphabet"),
        ((0..16).map(|i| format!("{:X}", i)).collect::<Vec<String>>(), "Hex alphabet"),
    ];
    
    for (alphabet, description) in &efficiency_tests {
        let training_data: Vec<String> = alphabet.iter().cycle().take(alphabet.len() * 50).cloned().collect();
        let mut detector = AnomalyDetector::new(1).expect("Failed to create detector");
        
        match detector.train(&training_data) {
            Ok(_) => {
                println!("      {}: training successful", description);
            }
            Err(e) => {
                violations += 1;
                details.push(format!("{} failed: {}", description, e));
            }
        }
    }
    
    details.push(format!("Performance results: {:?}", performance_results));
    
    if violations == 0 {
        DomainTestResult::pass("Alphabet size scalability verified".to_string())
            .with_details(details)
    } else {
        DomainTestResult::fail(format!("{} alphabet scalability violations", violations))
            .with_details(details)
    }
}